import numpy as np
import xarray as xr

from .registry import register_loader
from ..properties.properties import Space, Time, Uncertainty
from .base import BaseLoader


@register_loader
class ZarrUWCWForecastsLoader(BaseLoader):
    """Loader for UWC-WEST forecasts zarr v2 files."""

    name = "zarr_uwcw_forecasts"

    space = Space.GRID
    time = Time.FORECAST
    uncertainty = Uncertainty.DETERMINISTIC

    def _load(self):
        files = [self.files] if isinstance(self.files, str) else self.files

        if len(files) == 1:
            ds = self._load_single_zarr(files[0])
        else:
            dss = [self._load_single_zarr(f) for f in files]
            ds = xr.merge(dss)

        # Filter to requested reference_times if provided by config dates section.
        # Without this, the full multi-year zarr is kept in the dask graph, causing OOM.
        if "filter_reference_times" in self.kwargs:
            ref_filter = np.array(self.kwargs["filter_reference_times"], dtype="datetime64[ns]")
            ds = ds.sel(reference_time=ref_filter)

        return ds

    def _load_single_zarr(self, zarr_path):
        # mask_and_scale=False: prevent xarray's CFMaskCoder from replacing value=0 with NaT.
        # zarr stores integer arrays with default fill_value=0, so reference_time[0] (epoch)
        # and lead_time[0] (0-hour lead) would otherwise be silently masked to NaT.
        # CF datetime decoding (decode_cf=True, default) still runs normally.
        ds = xr.open_zarr(zarr_path, chunks="auto", consolidated=False, mask_and_scale=False)
        ds = ds.chunk({"grid_index": -1})

        # xr.open_zarr CF-decodes reference_time as datetime64[s] (newer xarray default).
        # Cast to datetime64[ns] to match the observation datasets and avoid sel() mismatches.
        # lead_time has units='hours' (no 'since'), so CF decode is skipped — convert manually.
        ds = ds.assign_coords(
            reference_time=ds["reference_time"].values.astype("datetime64[ns]"),
            # Cast to timedelta64[h] first (in case xr.open_zarr returned int64 with units='hours'),
            # then to timedelta64[ns] so that datetime64[ns] + timedelta64[ns] works in numpy >= 2.0.
            # Mixed-resolution arithmetic (datetime64[ns] + timedelta64[h]) silently returns NaT in numpy 2.0.
            lead_time=ds["lead_time"].values.astype("timedelta64[h]").astype("timedelta64[ns]"),
        )

        # Drop the zarr-stored valid_time (CF-decoded as datetime64[s]).
        # add_valid_time() will recompute it from reference_time + lead_time as datetime64[ns].
        if "valid_time" in ds.coords:
            ds = ds.drop_vars("valid_time")

        return ds
