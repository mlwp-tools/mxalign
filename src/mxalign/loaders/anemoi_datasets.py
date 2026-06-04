import numpy as np
import xarray as xr

from .registry import register_loader
from ..properties.properties import Space, Time, Uncertainty
from .base import BaseLoader

DROP_VARS = [
    "latitude",
    "longitude",
    "time",
    "cos_julian_day",
    "cos_latitude",
    "cos_local_time",
    "cos_longitude",
    "insolation",
    "sin_julian_day",
    "sin_latitude",
    "sin_local_time",
    "sin_longitude",
]

COORDS = dict(longitude="longitudes", latitude="latitudes", valid_time="dates")

DEFAULTS = {"chunks": "auto"}


@register_loader
class AnemoiDatasetsLoader(BaseLoader):
    name = "anemoi-datasets"

    space = Space.GRID
    time = Time.OBSERVATION
    uncertainty = Uncertainty.DETERMINISTIC

    def _load(self):

        if isinstance(self.files, list):
            dss = [xr.open_zarr(file, consolidated=False) for file in self.files]
            dss_postproc = [_postprocess(ds) for ds in dss]
            ds_postproc = xr.concat(dss_postproc, dim="valid_time")
        else:
            ds = xr.open_zarr(self.files, consolidated=False)
            ds_postproc = _postprocess(ds)

        if self.variables:
            ds_selected = ds_postproc.sel(variable=self.variables)
        else:
            ds_selected = ds_postproc
            if len(ds_selected["variable"]) > 10:
                print(
                    f"Transforming anemoi-datasets xr.DataArray with {len(ds_postproc['variable'])} variables to xr.Dataset, this might take some time. Consider selecting the relevant variables during loading"
                )
        return ds_selected.to_dataset(dim="variable")

    def fast_slice_recipe(self):
        """Recipe for per-rt direct zarr region read (fused engine).

        Only the single-file zarr path is supported in v1. The leaf
        computes valid_times = rt + lead_times and uses zarr's vectorised
        indexing to fetch one slice; no xarray/dask lazy graph involved.
        """
        if isinstance(self.files, list):
            if len(self.files) != 1:
                return None
            path = self.files[0]
        else:
            path = self.files
        return {
            "kind": "anemoi-datasets-zarr",
            "path": path,
            "consolidated": False,
            "drop_vars": list(DROP_VARS),
        }

    def slice(self, reference_time, lead_times, variables):
        """Eagerly read one (reference_time, lead_times, variables) slice.

        See ``BaseLoader.slice`` for the contract. Only the single-file
        zarr path is supported; multi-file returns ``None``.
        """
        if isinstance(self.files, list):
            if len(self.files) != 1:
                return None
            path = self.files[0]
        else:
            path = self.files

        src = xr.open_zarr(path, consolidated=False)

        # 'dates' coord on the 'time' dim is the canonical valid_time array.
        valid_times = src["dates"].astype("datetime64[ns]").load().values
        var_attr = list(src.attrs["variables"])
        try:
            var_idx = np.array(
                [var_attr.index(v) for v in variables], dtype=np.int64
            )
        except ValueError as e:
            raise ValueError(
                f"{type(self).__name__}.slice: variable not found in {path}: {e}"
            ) from None

        rt = np.datetime64(reference_time, "ns")
        requested_vts = np.array(
            [rt + np.timedelta64(lt, "ns") for lt in lead_times],
            dtype="datetime64[ns]",
        )
        pos = np.searchsorted(valid_times, requested_vts)
        if pos.max() >= valid_times.size or not np.all(
            valid_times[pos] == requested_vts
        ):
            bad = requested_vts[
                (pos >= valid_times.size)
                | (valid_times[pos.clip(max=valid_times.size - 1)] != requested_vts)
            ]
            raise ValueError(
                f"{type(self).__name__}.slice: missing valid_times in {path}: "
                f"{bad[:5]}... (reference_time={reference_time})"
            )

        arr = src["data"].isel(ensemble=0)
        # If `pos` is strictly contiguous (the common case: 1 h cadence
        # lead_times), issue a single slice read instead of a fancy index.
        # Fancy indexing along `time` triggers one chunk read per requested
        # step per variable, which on finely-time-chunked zarrs blows up
        # into thousands of small reads. A slice is a single contiguous
        # request and avoids that amplification entirely.
        pos_arr = np.asarray(pos)
        if pos_arr.size == 0:
            contiguous = False
        elif pos_arr.size == 1:
            contiguous = True
        else:
            contiguous = bool(np.all(np.diff(pos_arr) == 1))
        if contiguous:
            start = int(pos_arr[0])
            stop = int(pos_arr[-1]) + 1
            arr_sel = arr.isel(
                time=slice(start, stop),
                variable=xr.DataArray(var_idx, dims="variable_out"),
            )
        else:
            arr_sel = arr.isel(
                time=xr.DataArray(pos_arr, dims="lead_time"),
                variable=xr.DataArray(var_idx, dims="variable_out"),
            )
        vals = np.asarray(arr_sel.load().values)  # (n_lt, n_var, n_grid)
        return xr.Dataset(
            {
                v: (("lead_time", "grid_index"), vals[:, i, :])
                for i, v in enumerate(variables)
            }
        )


def _postprocess(dataset: xr.Dataset) -> xr.Dataset:
    """Post-process the dataset to add coordinates and drop unused variables.

    Args:
        dataset (xr.Dataset): The input dataset to be processed.

    Returns:
        xr.Dataset: The processed dataset with assigned coordinates and
            attributes.
    """

    # Add coordinates
    coords = {
        key: dataset[value].astype("datetime64[ns]").load()
        if key == "valid_time"
        else dataset[value].load()
        for key, value in COORDS.items()
    }
    for key in ("latitude", "longitude"):
        coords[key] = coords[key].astype(np.float32)

    coords["variable"] = dataset.attrs["variables"]
    coords["valid_time"] = coords["valid_time"].astype("datetime64[ns]")
    ds_coords = dataset.assign_coords(coords)

    # Drop unused variables and remove ensemble dimension
    drop_vars = [var for var in DROP_VARS if var in coords["variable"]]

    ds_pruned = (
        ds_coords["data"]
        .isel(ensemble=0)
        .drop_sel(variable=drop_vars)
        .swap_dims({"time": "valid_time"})
        .rename({"cell": "grid_index"})
    )
    return ds_pruned
