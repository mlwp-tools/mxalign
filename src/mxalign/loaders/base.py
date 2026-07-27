from abc import ABC, abstractmethod

import numpy as np

from .registry import register_loader
from ..properties.properties import Properties, Space, Time, Uncertainty
from ..properties.validation import validate_dataset
from ..properties.utils import set_properties_attrs


class BaseLoader(ABC):
    """Base class for all loaders."""

    name: str = "base"

    space: Space | None = None
    time: Time | None = None
    uncertainty: Uncertainty | None = None

    def __init__(self, files, variables=None, grid_mapping=None,
                 valid_times=None, reference_times=None, lead_times=None,
                 **kwargs):
        self.files = files
        self.variables = [variables] if isinstance(variables, str) else variables
        self.grid_mapping = grid_mapping
        # Optional pre-pruning hints; consumed in load(), not forwarded
        # to backend kwargs (which would explode for unknown args).
        #   - valid_times: 1D datetime64 set, used to prune observation
        #     datasets carrying a `valid_time` dim.
        #   - reference_times / lead_times: 1D arrays defining the
        #     allowed rectangular (rt, lt) window for forecast datasets.
        self.valid_times = valid_times
        self.reference_times = reference_times
        self.lead_times = lead_times
        self.kwargs = kwargs

    def load(self):
        ds = self._load()
        if self.variables:
            ds = self._select_variables(ds)

        # Generic time pre-pruning. Applied here (after _load / variable
        # selection, before properties/validation) so every loader benefits
        # without needing to know about the time hints. Dask's culling will
        # drop the unused upstream chunks at execution time.
        #
        # Two cases:
        #   - Observation datasets carry `valid_time` as a 1D dim, pruned
        #     against `self.valid_times`.
        #   - Forecast datasets carry `reference_time` and `lead_time` as
        #     dims; pruned rectangularly against `self.reference_times` and
        #     `self.lead_times`. This enforces the blueprint's `dates.range`
        #     (max lead time) and `dates.period` (rt spacing) without the
        #     spurious over-keep that an axis-independent mask derived from
        #     `valid_times` would produce for commensurate spacings.
        if "valid_time" in ds.dims and self.valid_times is not None:
            wanted = np.asarray(self.valid_times)
            keep = np.intersect1d(wanted, ds["valid_time"].values)
            if keep.size and keep.size < ds["valid_time"].size:
                all_vt = ds["valid_time"].values
                positions = np.searchsorted(all_vt, keep)
                if positions[-1] - positions[0] == len(positions) - 1:
                    # contiguous block — isel with a slice keeps the dask graph
                    # small (only the needed chunks, not all 403K zarr tasks)
                    ds = ds.isel(valid_time=slice(int(positions[0]), int(positions[-1]) + 1))
                else:
                    ds = ds.sel(valid_time=keep)
        elif {"reference_time", "lead_time"} <= set(ds.dims):
            if self.lead_times is not None:
                lt = ds["lead_time"].values
                wanted_lt = np.asarray(self.lead_times).astype(lt.dtype)
                keep_lt = np.isin(lt, wanted_lt)
                if keep_lt.any() and keep_lt.sum() < lt.size:
                    ds = ds.isel(lead_time=keep_lt)
            if self.reference_times is not None:
                rt = ds["reference_time"].values
                wanted_rt = np.asarray(self.reference_times).astype(rt.dtype)
                keep_rt = np.isin(rt, wanted_rt)
                if keep_rt.any() and keep_rt.sum() < rt.size:
                    ds = ds.isel(reference_time=keep_rt)

        properties = self._get_properties(ds)
        validate_dataset(ds, properties)

        ds = set_properties_attrs(ds, properties)

        if self.grid_mapping:
            ds = self._add_grid_mapping(ds)

        # Make sure all the coordinates are loaded
        for coord in ds.coords:
            ds[coord] = ds[coord].compute()

        return ds

    @abstractmethod
    def _load(self): ...

    def _select_variables(self, ds):
        return ds[self.variables]

    def _add_grid_mapping(self, ds):
        ds = ds.space.add_crs(self.grid_mapping)
        ds = ds.space.add_grid_mapping(self.grid_mapping)
        return ds

    def _get_properties(self, ds):
        properties = Properties(
            space=self.space, time=self.time, uncertainty=self.uncertainty
        )
        return properties

    def slice(self, reference_time, lead_times, variables):
        """Eagerly load a single per-reference_time slice.

        Returns an in-memory ``xr.Dataset`` with dims ``(lead_time, grid_index)``
        and one data variable per name in ``variables``. No ``reference_time``
        dim (callers iterate reference_times). Bypasses xarray/dask lazy
        graphs by reading directly from the underlying store.

        Parameters
        ----------
        reference_time
            datetime64-coercible. Forecast initial time to read.
        lead_times
            Sequence of timedelta64-coercible offsets from reference_time.
        variables
            List of variable names to read. Required.

        Returns
        -------
        xr.Dataset | None
            None if this loader cannot serve a per-rt slice (e.g. unsupported
            file layout). The fused engine treats None as a hard error.
        """
        return None


@register_loader
class MxAlignLoader(BaseLoader):
    name = "mxalign"

    space = None
    time = None
    uncertainty = None

    def _load(self):
        import xarray as xr

        files = [self.files] if isinstance(self.files, str) else self.files

        ds = xr.open_mfdataset(files, chunks="auto", **self.kwargs)
        if "code" in ds.dims:
            ds = ds.rename_dims({"code": "point_index"}).transpose(
                "valid_time", "point_index"
            )
        return ds

    def _get_properties(self, ds):
        if "reference_time" in ds.dims and "lead_time" in ds.dims:
            time = Time.FORECAST
        elif "valid_time" in ds.dims:
            time = Time.OBSERVATION
        else:
            raise ValueError("Unknown temporal dimensions")

        if "grid_index" in ds.dims or "xc" in ds.dims or "latitude" in ds.dims:
            space = Space.GRID
        elif "point_index" in ds.dims:
            space = Space.POINT
        else:
            raise ValueError("Unknown spatial dimensions")

        if "member" in ds.dims:
            uncertainty = Uncertainty.ENSEMBLE
        elif "quantile" in ds.dims:
            uncertainty = Uncertainty.QUANTILE
        else:
            uncertainty = Uncertainty.DETERMINISTIC

        return Properties(space=space, time=time, uncertainty=uncertainty)
