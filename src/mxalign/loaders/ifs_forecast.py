import numpy as np
import xarray as xr

from .registry import register_loader
from ..properties.properties import Properties, Space, Time, Uncertainty
from .base import BaseLoader


@register_loader
class IFSForecastLoader(BaseLoader):
    try:
        import cfgrib  
    except Exception:
        raise ImportError("Please install the cfgrib package to load IFS-Forecasts")

    name = "ifs-forecast"

    space = Space.GRID
    time = Time.FORECAST
    uncertainty = None

    def _load(self):
        kwargs = self.kwargs.copy()
        files = [self.files] if isinstance(self.files, str) else self.files

        ds = xr.open_mfdataset(
            files,
            combine="nested",
            concat_dim="time",
            chunks={
                "time": 1,
                "step": -1,
                "values": -1,
            },
            **kwargs,
        )

        ds.coords["longitude"] = (ds.coords["longitude"] + 180.0) % 360.0 - 180.0

        rename_dims = {
            "time": "reference_time",
            "step": "lead_time",
            "values": "grid_index",
        }
        rename_vars = {
            "time": "reference_time",
            "step": "lead_time",
        }

        if "number" in ds.dims:
            rename_dims["number"] = "ensemble_member"
        if "number" in ds.coords:
            rename_vars["number"] = "ensemble_member"

        ds = ds.rename_dims({k: v for k, v in rename_dims.items() if k in ds.dims})
        ds = ds.rename_vars({k: v for k, v in rename_vars.items() if k in ds.variables})

        if "surface" in ds.variables:
            ds = ds.drop_vars("surface")

        if "lead_time" in ds.coords and np.issubdtype(ds["lead_time"].dtype, np.timedelta64):
            ds = ds.assign_coords(
                lead_time=(ds["lead_time"].values / np.timedelta64(1, "h")).astype(int)
            )
            ds["lead_time"].attrs["units"] = "h"

        return ds

    def _get_properties(self, ds):
        if "member" in ds.dims:
            uncertainty = Uncertainty.ENSEMBLE
        elif "quantile" in ds.dims:
            uncertainty = Uncertainty.QUANTILE
        else:
            uncertainty = Uncertainty.DETERMINISTIC

        return Properties(
            space=Space.GRID,
            time=Time.FORECAST,
            uncertainty=uncertainty,
        )