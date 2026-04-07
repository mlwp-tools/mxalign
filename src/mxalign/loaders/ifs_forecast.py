from pathlib import Path
import xarray as xr

from .registry import register_loader
from ..properties.properties import Space, Time, Uncertainty
from .base import BaseLoader

@register_loader
class IFSForecastLoader(BaseLoader):
    try:
        import cfgrib
    except:
        ImportError("Please install the cfgrib package to load IFS-Forecasts")
    
    name = "ifs-forecast"

    space = Space.GRID
    time = Time.FORECAST
    uncertainty = Uncertainty.DETERMINISTIC

    def _load(self):
        
        kwargs = self.kwargs.copy()

        files = [self.files] if isinstance(self.files, str) else self.files
        
        ds = xr.open_mfdataset(
            files,
            combine="nested",
            concat_dim="time",
            chunks={
                "time" : 1,
                "step": -1,
                "values": -1
            },
            **kwargs
        )
        
        ds.coords["longitude"] = (ds.coords["longitude"] + 180.) % 360. -180.

        ds_out = ds.rename_dims(
            time="reference_time",
            step="lead_time",
            values="grid_index"
        ).rename_vars(
            time="reference_time",
            step="lead_time"
        ).drop_vars(
            ["number","surface"]
        )

        return ds_out