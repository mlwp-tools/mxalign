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
    "parallel": True
}

# def _extract_member_index(filename):
#     """
#     Extract member index from filename.
#     Supports patterns like:
#     - mbr000, mbr001, etc. (anemoi convention)
#     - member_0, member_1, etc.
#     - _m000, _m001, etc.
    
#     Returns None if no member index is found.
#     """
#     fname = Path(filename).stem  # Get filename without extension
    
#     # Try different patterns
#     patterns = [
#         r'_mbr(\d+)',        # _mbr000
#         r'_member(\d+)',     # _member0
#         r'_m(\d+)(?:_|\.)',  # _m000_
#         r'mbr(\d+)',         # mbr000
#         r'member(\d+)',      # member0
#     ]
    
#     for pattern in patterns:
#         match = re.search(pattern, fname)
#         if match:
#             return int(match.group(1))
    
#     return None

@register_loader
class AnemoiInferenceLoader(BaseLoader):

    name = "anemoi-inference"
    
    space = Space.GRID
    time=Time.FORECAST
    uncertainty=Uncertainty.DETERMINISTIC

    def __init__(self, files, variables=None, grid_mapping=None, ens_size=None, start_from=0,**kwargs):
        super().__init__(files, variables, grid_mapping, **kwargs)
        self.ens_size = ens_size
        print(f"Initialized AnemoiInferenceLoader with ens_size: {self.ens_size}")
        print("base ens size:", ens_size)
        print("start from:", start_from)
        self.start_from = start_from
        # Detect uncertainty based on member presence
        # self._has_members = None  # Will be determined in _load()

    def _load(self):
        
        files = [self.files] if isinstance(self.files, str) else self.files
        
        # Check if we have ensemble members
        # member_indices = [_extract_member_index(f) for f in files]
        # has_members = any(idx is not None for idx in member_indices)
        # self._has_members = has_members
        # and all(idx is not None for idx in member_indices)
        if self.ens_size > 1:
            # Load with member dimension
            ds = self._load_with_members(files)
        elif self.ens_size == 1:

            # Load without member dimension (original behavior)
            ds = self._load_deterministic(files)
        else:
            raise ValueError(
                "Invalid ensemble configuration. Ensure that ens_size is set correctly and that member indices are present in filenames if ens_size > 1."
            )
            # raise ValueError(
            #     "Cannot mix files with and without member indices. "
            #     f"Member indices found: {member_indices}"
            # )
        
        return ds

    def _load_deterministic(self, files):
        """Load forecast data without member dimension (original behavior)."""
        import xarray as xr
        
        times = xr.open_dataset(files[0])["time"].values
        lead_times = times - times[0]    

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
            swap_dims({"time": "lead_time"})

        return ds_out

    def _load_with_members(self, files):
        """Load ensemble forecast data with member dimension."""
        import xarray as xr
        
        kwargs = self.kwargs.copy()
        for k, v in DEFAULTS.items():
            kwargs[k] = self.kwargs.get(k, v)

        # Load each member separately  
        # datasets = []
        # for filepath, member_idx in zip(files, member_indices):
        #     ds = xr.open_dataset(filepath, engine=kwargs.get("engine", "h5netcdf"))
        #     ds = _preprocess_with_member(ds, member_idx)
        #     datasets.append(ds)
        
        # Concatenate along member dimension
        # ds = xr.concat(datasets, dim="member")
        
        # Get lead times from the first member
        times = xr.open_dataset(files[0])["time"].values
        lead_times = times - times[0]
        if len(files) % (self.ens_size) != 0:
            raise ValueError(f"Number of files ({len(files)}) must be divisible by ens_size ({self.ens_size}).")
        files = [files[i:i+(self.ens_size)] for i in range(0, len(files), self.ens_size)]
        print(f"Loading {len(files)} files with {self.ens_size} ensemble members each.")
        print(f"Files: {files}")
        ds = xr.open_mfdataset(
            files, 
            preprocess=_preprocess_with_member,
            concat_dim=["reference_time", "member"],
            combine="nested",
            chunks={
                    "reference_time" : "auto",
                    "time": "auto",
                    "values": "auto",
                    "member": -1
                },
            **kwargs
        )
        
        # Remove duplicate reference_times and sort to ensure unique index
        ds = ds.drop_duplicates(dim="reference_time")
        ds = ds.sortby("reference_time")
        print("dataset", ds)
        ds_out = ds.\
            assign_coords({"lead_time": ("time", lead_times), "member": ("member", np.arange(self.start_from,self.ens_size+self.start_from))}).\
            rename_dims({"values": "grid_index"}).\
            swap_dims({"time": "lead_time"}).\
            chunk({"member": -1, "lead_time": "auto", "grid_index": "auto", "reference_time": "auto"})

        return ds_out

    def _get_properties(self, ds):
        """Override to set uncertainty based on member presence."""
        from ..properties.properties import Properties
        
        # Determine uncertainty based on whether members were detected
        uncertainty = Uncertainty.ENSEMBLE if "member" in ds.dims else Uncertainty.DETERMINISTIC
        
        return Properties(
            space=self.space,
            time=self.time,
            uncertainty=uncertainty
        )


def _preprocess_deterministic(ds):
    """Preprocess a single forecast file without member dimension."""
    ds_out = ds.\
        set_coords(["longitude", "latitude"]).\
        expand_dims("reference_time").\
        assign_coords(
            {"reference_time": ("reference_time", [ds["time"].values[0]])}
        ).\
        drop_vars("time")
    
    return ds_out


def _preprocess_with_member(ds):
    """Preprocess and add member dimension."""
    ds_out = ds.\
        set_coords(["longitude", "latitude"]).\
        expand_dims("reference_time").\
        assign_coords(
            {"reference_time": ("reference_time", [ds["time"].values[0]])}
        ).\
        expand_dims("member").\
        drop_vars("time")
    
    return ds_out