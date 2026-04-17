from .registry import register_loader
from ..properties.properties import Space, Time, Uncertainty
from .base import BaseLoader
import re
from pathlib import Path
import numpy as np
import xarray as xr
    # "chunks": "auto",
DEFAULTS={
    "engine": "h5netcdf",
    "parallel": True,
}

DROP_DIMS = [
    "height1",
    "height_above_msl",
    "height",
    "height2",
]


@register_loader
class BrisInferenceLoader(BaseLoader):

    name = "bris-inference"
    
    space = Space.GRID
    time=Time.FORECAST
    uncertainty=Uncertainty.DETERMINISTIC

    def __init__(self, files, variables=None, grid_mapping=None, ens_size=None, **kwargs):
        super().__init__(files, variables, grid_mapping, **kwargs)
        self.ens_size = ens_size
        # Detect uncertainty based on member presence
        # self._has_members = None  # Will be determined in _load()

    def _load(self):
        
        files = [self.files] if isinstance(self.files, str) else self.files
        
        # Check if we have ensemble members
        # member_indices = [_extract_member_index(f) for f in files]
        # has_members = any(idx is not None for idx in member_indices)
        # self._has_members = has_members
        # and all(idx is not None for idx in member_indices)
        if self.ens_size is not None and self.ens_size > 1:
            # Load with member dimension
            ds = self._load_with_members(files)
        else:
            if self.ens_size is None:
                print("Warning: ens_size not set, defaulting to deterministic loading.")
            # Load without member dimension (original behavior)
            ds = self._load_deterministic(files)
            # raise ValueError(
            #     "Cannot mix files with and without member indices. "
            #     f"Member indices found: {member_indices}"
            # )
        
        return ds

    def _load_deterministic(self, files):
        """Load forecast data without member dimension (original behavior)."""
        import xarray as xr
        import pandas as pd
        
        times = xr.open_dataset(files[0])["time"].values
        lead_times = pd.to_timedelta(times - times[0]).to_pytimedelta()

        kwargs = self.kwargs.copy()
        for k, v in DEFAULTS.items():
            kwargs[k] = self.kwargs.get(k, v)

        ds = xr.open_mfdataset(
            files, 
            preprocess=_preprocess_deterministic,
            **kwargs
        )

        ds_out = ds.\
            assign_coords({"lead_time": ("time", lead_times)}).\
            rename_dims({"values": "grid_index"}).\
            swap_dims({"time": "lead_time"}).\
            chunk({"grid_index": -1})

        return ds_out

    def _load_with_members(self, files):
        """Load ensemble forecast data with member dimension."""
        import xarray as xr
        import pandas as pd

        engine = self.kwargs.get("engine", DEFAULTS["engine"])
        # Get lead times from the first file
        ds = xr.open_dataset(files[0], engine=engine)
        print("first dataset", ds)
        times = ds["time"].values
        lead_times = pd.to_timedelta(times - times[0])

        kwargs = self.kwargs.copy()
        for k, v in DEFAULTS.items():
            kwargs[k] = self.kwargs.get(k, v)

        ds = xr.open_mfdataset(
            files,
            preprocess=_preprocess_with_member,
             concat_dim="forecast_reference_time",
             combine="nested", 
             join="override",
            **kwargs,
        )
        print("dataset after opening", ds)
        print("lead times", lead_times)

        # ds = ds.sel(lead_time=slice(0,self.max_lead_time))
        # ds = ds.rename({"location":"grid_index"})
        print("ds before stacking", ds)
        ds = ds.stack(grid_index=('x', 'y')).reset_index("grid_index").drop_vars(["x", "y"])
        print("ds after stacking", ds)
        ds_coords = ds.assign_coords(
            {
                "lead_time": ("lead_time", np.array(lead_times)),
                "valid_time": (
                    ["forecast_reference_time", "lead_time"],
                    ds["forecast_reference_time"].data[:,np.newaxis] + \
                        np.array(lead_times)[np.newaxis,:]
                ),
                "ensemble_member": ("ensemble_member", ds["ensemble_member"].data[:self.ens_size+1])
                }
            )
        ds_coords = ds_coords.rename({'forecast_reference_time': 'reference_time', "ensemble_member": "member"}).chunk({"member": -1, "grid_index": -1})

        return ds_coords

    # def _get_properties(self, ds):
    #     """Override to set uncertainty based on member presence."""
    #     from ..properties.properties import Properties
        
    #     # Determine uncertainty based on whether members were detected
    #     uncertainty = Uncertainty.ENSEMBLE if self._has_members else Uncertainty.DETERMINISTIC
        
    #     return Properties(
    #         space=self.space,
    #         time=self.time,
    #         uncertainty=uncertainty
    #     )


def _preprocess_deterministic(ds):
    """Preprocess a single forecast file without member dimension."""
    print("dataset before preprocessing", ds)
    for var in DROP_DIMS:
        if var in ds.coords:
            ds = ds.squeeze(var)
            ds = ds.drop(var)

    ds_renamed = ds.rename_dims(
        {
            "time":"lead_time",
        }
    )

    return ds_renamed


def _preprocess_with_member(ds):
    """Preprocess and add member dimension with explicit member index."""
    print("dataset before preprocessing", ds)
    for var in DROP_DIMS:
        if var in ds.coords:
            ds = ds.squeeze(var)
            ds = ds.drop(var)

    ds_renamed = ds.rename_dims(
        {
            "time":"lead_time",
        }
    )

    return ds_renamed