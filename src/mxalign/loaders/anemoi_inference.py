from .registry import register_loader
from ..properties.properties import Space, Time, Uncertainty
from .base import BaseLoader

DEFAULTS={
    "chunks": "auto",
    "engine": "h5netcdf",
    "parallel": True
}

@register_loader
class AnemoiInferenceLoader(BaseLoader):

    name = "anemoi-inference"
    
    space = Space.GRID
    time=Time.FORECAST
    uncertainty=Uncertainty.DETERMINISTIC

    def _load(self):
        import xarray as xr
        
        files = [self.files] if isinstance(self.files, str) else self.files
        times = xr.open_dataset(files[0])["time"].values
        lead_times = times - times[0]    

        kwargs = self.kwargs.copy()
        for k, v in DEFAULTS.items():
            kwargs[k] = self.kwargs.get(k,v)

        ds = xr.open_mfdataset(
            files, 
            preprocess=_preprocess,
            **kwargs
        )

        ds_out = ds.\
            assign_coords({"lead_time": ("time", lead_times)}).\
            rename_dims({"values": "grid_index"}).\
            swap_dims({"time": "lead_time"})

        return ds_out

def _preprocess(ds):
    ds_out = ds.\
        set_coords(["longitude", "latitude"]).\
        expand_dims("reference_time").\
        assign_coords(
            {"reference_time": ("reference_time", [ds["time"].values[0]])}
        ).\
        drop_vars("time")
    
    return ds_out