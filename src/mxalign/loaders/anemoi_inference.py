from pathlib import Path
import xarray as xr

from .registry import register_loader
from ..properties.properties import Space, Time, Uncertainty
from .base import BaseLoader

DEFAULTS_NETCDF = {
    "chunks": "auto",
    "engine": "h5netcdf",
    "parallel": True
}

DEFAULTS_ZARR = {
    "chunks": "auto",
    "storage_options": {"anon": True},
}

@register_loader
class AnemoiInferenceLoader(BaseLoader):

    name = "anemoi-inference"
    
    space = Space.GRID
    time=Time.FORECAST
    uncertainty=Uncertainty.DETERMINISTIC

    def _load(self):


        kwargs = self.kwargs.copy()
        
        if isinstance(self.files,str):
            if Path(self.files).suffix.lower() == ".zarr":
                files = self.files


                for k, v in DEFAULTS_ZARR.items():
                    kwargs[k] = self.kwargs.get(k,v)

                loader = _open_zarr
            else:
                files = [self.files]

                for k, v in DEFAULTS_NETCDF.items():
                    kwargs[k] = self.kwargs.get(k,v)

                loader = _open_mf_dataset
        else:
            files = self.files
            if Path(files[0]).suffix.lower() == ".zarr":          
                for k, v in DEFAULTS_ZARR.items():
                    kwargs[k] = self.kwargs.get(k,v)
                kwargs["engine"] = "zarr"

            else: 
                for k, v in DEFAULTS_NETCDF.items():
                    kwargs[k] = self.kwargs.get(k,v)

            loader = _open_mf_dataset


        ds = loader(files, **kwargs)
        return ds

def _open_mf_dataset(files, **kwargs):

    times = xr.open_dataset(files[0], engine=kwargs["engine"], chunks=kwargs["chunks"])["time"].values
    lead_times = times - times[0]    

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

def _open_zarr(files, **kwargs):

    ds = xr.open_zarr(files, **kwargs)
    times = ds["time"].values
    lead_times = times - times[0] 
    
    ds_out = _preprocess(ds)
        
    ds_out = ds_out.\
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