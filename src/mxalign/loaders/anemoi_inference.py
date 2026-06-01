from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from .registry import register_loader
from ..properties.properties import Space, Time, Uncertainty
from .base import BaseLoader

DEFAULTS_NETCDF = {"chunks": "auto", "engine": "h5netcdf", "parallel": True, "identical_layout": True}

# Variables that are static spatial fields, not per-timestep forecasts.
_SPATIAL_VARS = frozenset({"latitude", "longitude"})

DEFAULTS_ZARR = {
    "chunks": "auto",
    "storage_options": {"anon": True},
}


@register_loader
class AnemoiInferenceLoader(BaseLoader):
    name = "anemoi-inference"

    space = Space.GRID
    time = Time.FORECAST
    uncertainty = Uncertainty.DETERMINISTIC

    def _load(self):

        kwargs = self.kwargs.copy()

        if isinstance(self.files, str):
            if Path(self.files).suffix.lower() == ".zarr":
                files = self.files

                for k, v in DEFAULTS_ZARR.items():
                    kwargs[k] = self.kwargs.get(k, v)

                loader = _open_zarr
            else:
                files = [self.files]

                for k, v in DEFAULTS_NETCDF.items():
                    kwargs[k] = self.kwargs.get(k, v)

                loader = _open_mf_dataset
        else:
            files = self.files
            if Path(files[0]).suffix.lower() == ".zarr":
                for k, v in DEFAULTS_ZARR.items():
                    kwargs[k] = self.kwargs.get(k, v)
                kwargs["engine"] = "zarr"

            else:
                for k, v in DEFAULTS_NETCDF.items():
                    kwargs[k] = self.kwargs.get(k, v)

            loader = _open_mf_dataset

        # Pass reference_times hint to fast path so it doesn't need to parse
        # filenames.  Consumed (popped) inside _open_mf_dataset; ignored by
        # _open_zarr.
        if loader is _open_mf_dataset and self.reference_times is not None:
            kwargs["_reference_times"] = np.asarray(self.reference_times)

        ds = loader(files, **kwargs)
        return ds


def _load_nc_vars(path, var_names, engine):
    """Load all named variables from one NC file.

    Executed by dask workers at compute-time (not graph-build time), so 358
    files are opened in parallel across dask threads rather than serially
    during graph construction.

    Returns a dict {var_name: np.ndarray shape (n_lt, n_grid)}.
    """
    ds = xr.open_dataset(path, engine=engine)
    result = {v: ds[v].values for v in var_names}
    ds.close()
    return result


def _load_nc_var(path, var_name, engine):
    """Load a single variable from one NC file.

    One delayed task per (file, variable) pair: each result is ~23 MB instead
    of ~1.5 GB per file.  Without a shared intermediate dict there is no
    dependency forcing all 65 variable results to stay in memory at once,
    so peak worker memory scales with concurrency (O(n_threads × chunk_size))
    rather than with n_files × file_size.
    """
    ds = xr.open_dataset(path, engine=engine)
    result = ds[var_name].values
    ds.close()
    return result


def _open_mf_dataset(files, **kwargs):
    identical_layout = kwargs.pop("identical_layout", True)
    # Reference times from the blueprint config (sorted datetime64 array,
    # index-aligned with the sorted files list).  Preferred over filename
    # parsing; absent when the loader is called outside the blueprint system.
    reference_times_hint = kwargs.pop("_reference_times", None)
    engine = kwargs.get("engine", "h5netcdf")

    # Always open file 0: needed for lead_times (and schema in fast path).
    ds0 = xr.open_dataset(files[0], engine=engine)
    times0 = ds0["time"].values
    lead_times = times0 - times0[0]

    if not identical_layout or len(files) == 1:
        ds0.close()
        ds = xr.open_mfdataset(files, preprocess=_preprocess, **kwargs)
        return (
            ds.assign_coords({"lead_time": ("time", lead_times)})
            .rename_dims({"values": "grid_index"})
            .swap_dims({"time": "lead_time"})
        )

    # ------------------------------------------------------------------
    # Fast path: identical_layout=True
    # Build a lazy dataset without opening files[1:].  Each file's data
    # becomes a dask.delayed task executed at compute-time.  Only the
    # schema (shape, dtype, coords) is read here, from file 0 only.
    # ------------------------------------------------------------------
    import dask
    import dask.array as dsa

    data_vars = tuple(v for v in ds0.data_vars if v not in _SPATIAL_VARS)
    lat = ds0["latitude"].values
    lon = ds0["longitude"].values
    n_lt, n_grid = ds0[data_vars[0]].shape   # (time, values)
    dtype = ds0[data_vars[0]].dtype
    ds0.close()

    # Resolve a reference_time for each file.
    # Primary: use the blueprint-provided array (format-agnostic, no I/O).
    # Fallback: parse from filename stem (ISO-8601: 2023-01-01T00.nc).
    # If neither works, abort the fast path.
    if reference_times_hint is not None and len(reference_times_hint) == len(files):
        ref_time_list = [np.datetime64(rt, "ns") for rt in reference_times_hint]
    else:
        ref_time_list = []
        for f in files:
            try:
                ref_time_list.append(
                    np.datetime64(datetime.strptime(Path(f).stem, "%Y-%m-%dT%H"), "ns")
                )
            except ValueError:
                import warnings
                warnings.warn(
                    f"identical_layout=True: cannot parse reference_time from "
                    f"{Path(f).name!r}; falling back to open_mfdataset",
                    stacklevel=2,
                )
                return _open_mf_dataset(files, identical_layout=False, **kwargs)

    individual_dss = []
    for f, ref_time in zip(files, ref_time_list):
        # One delayed task per (file, variable): no shared intermediate dict,
        # so dask can free each ~23 MB result immediately after its consumer
        # finishes instead of holding a ~1.5 GB per-file dict until all 65
        # getitem tasks complete.
        ds_vars = {
            v: xr.DataArray(
                dsa.from_delayed(
                    dask.delayed(_load_nc_var)(f, v, engine),
                    shape=(n_lt, n_grid),
                    dtype=dtype,
                ),
                dims=["lead_time", "grid_index"],
            )
            for v in data_vars
        }

        ds_individual = (
            xr.Dataset(ds_vars)
            .assign_coords({
                "lead_time": lead_times,
                "latitude": ("grid_index", lat),
                "longitude": ("grid_index", lon),
            })
            .expand_dims({"reference_time": [ref_time]})
        )
        individual_dss.append(ds_individual)

    return xr.concat(individual_dss, dim="reference_time", coords="minimal", join="override")


def _open_zarr(files, **kwargs):

    ds = xr.open_zarr(files, **kwargs)
    times = ds["time"].values
    lead_times = times - times[0]

    ds_out = _preprocess(ds)

    ds_out = (
        ds_out.assign_coords({"lead_time": ("time", lead_times)})
        .rename_dims({"values": "grid_index"})
        .swap_dims({"time": "lead_time"})
    )

    return ds_out


def _preprocess(ds):
    ds_out = (
        ds.set_coords(["longitude", "latitude"])
        .expand_dims("reference_time")
        .assign_coords({"reference_time": ("reference_time", [ds["time"].values[0]])})
        .drop_vars("time")
    )

    return ds_out
