import numpy as np
import xarray as xr

from .registry import register_loader
from ..properties.properties import Properties, Space, Time, Uncertainty
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

COORDS = dict(
    longitude="longitudes",
    latitude="latitudes",
    valid_time="dates"
)

DEFAULTS={
    "chunks": "auto"
}


@register_loader
class AnemoiDatasetsLoader(BaseLoader):

    name = "anemoi-datasets"
    
    space=Space.GRID
    time=Time.OBSERVATION
    uncertainty=Uncertainty.DETERMINISTIC
    
    def _load(self):

        if isinstance(self.files, list):
            dss = [
                xr.open_zarr(file, consolidated=False) 
                for file in self.files
            ]
            dss_postproc = [_postprocess(ds) for ds in dss]
            ds_postproc = xr.concat(dss_postproc, dim="valid_time")
        else:
            ds = xr.open_zarr(self.files,consolidated=False)
            ds_postproc = _postprocess(ds)
             
        if self.variables:
            ds_selected = ds_postproc.sel(variable=self.variables)
        else:
            ds_selected = ds_postproc
            if len(ds_selected["variable"]) > 10:
                print(f"Transforming anemoi-datasets xr.DataArray with {len(ds_postproc['variable'])} variables to xr.Dataset, this might take some time. Consider selecting the relevant variables during loading")
        return ds_selected.to_dataset(dim="variable")

def _postprocess(dataset : xr.Dataset) -> xr.Dataset:
    """Post-process the dataset to add coordinates and drop unused variables.

    Args:
        dataset (xr.Dataset): The input dataset to be processed.

    Returns:
        xr.Dataset: The processed dataset with assigned coordinates and
            attributes.
    """
    
    # Add coordinates
    coords = {key: dataset[value].astype("datetime64[ns]").load() if key == "valid_time" else dataset[value].load() for key, value in COORDS.items()}
    for key in ("latitude","longitude"):
        coords[key] = coords[key].astype(np.float32)

    coords["variable"] = dataset.attrs["variables"]
    coords["valid_time"] = coords["valid_time"].astype("datetime64[ns]")
    ds_coords = dataset.assign_coords(coords)

    # Drop unused variables and remove ensemble dimension
    drop_vars = [var for var in DROP_VARS if var in coords["variable"]]
    
    ds_pruned = ds_coords["data"].isel(
        ensemble=0
    ).drop_sel(
        variable=drop_vars
    ).swap_dims(
        {"time":"valid_time"}
    ).rename(
        {"cell":"grid_index"}
    )
    return ds_pruned
