import xarray as xr
import zarr
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
        """Load zarr v2 files using zarr directly and convert to xarray."""
        files = [self.files] if isinstance(self.files, str) else self.files
        
        # Load all zarr files
        if len(files) == 1:
            ds = self._load_single_zarr(files[0])
        else: #not supposed to be used
            # Load multiple zarr files and concatenate along reference_time
            dss = [self._load_single_zarr(file) for file in files]
            ds = xr.concat(dss, dim="reference_time")
        
        return ds
    
    def _load_single_zarr(self, zarr_path):
        """Load a single zarr v2 file with lazy loading and time decoding."""
        import dask.array as da
        import numpy as np
        
        # Open zarr group
        zarr_group = zarr.open_group(zarr_path, mode='r')
        
        # Dictionary to hold data variables and coordinates
        data_vars = {}
        coords = {}
        
        # Load all arrays from the zarr group
        for name in zarr_group:
            zarr_array = zarr_group[name]
            
            # Auto-detect dimensions based on array name and shape
            dims = self._get_dims_for_array(name, zarr_array.shape)
            
            # For small 1D arrays, load immediately
            if len(dims) == 1:
                data = zarr_array[:]
                
                # Convert time coordinates from hours to proper types
                if name == 'reference_time':
                    # Convert from hours since epoch to datetime
                    # Get units from attributes if available
                    units = zarr_array.attrs.get('units', 'hours since 1970-01-01')
                    # Parse "hours since YYYY-MM-DD" format
                    if 'since' in units:
                        base_time_str = units.split('since ')[-1].strip()
                        base_time = np.datetime64(base_time_str, 's')
                        data = base_time + data.astype('timedelta64[h]')
                    data_array = xr.DataArray(data, dims=dims, name=name)
                elif name == 'lead_time':
                    # Convert from hours to timedelta
                    data = data.astype('timedelta64[h]')
                    data_array = xr.DataArray(data, dims=dims, name=name)
                else:
                    data_array = xr.DataArray(data, dims=dims, name=name)
                
                # Remove grid_index as a coordinate
                if name != 'grid_index':
                    coords[name] = data_array
            else:
                # For large multi-dimensional arrays, use dask for lazy loading
                try:
                    # Convert zarr array to dask array (lazy loading)
                    dask_array = da.from_zarr(zarr_array, chunks='auto')
                    data_array = xr.DataArray(dask_array, dims=dims, name=name)
                    data_vars[name] = data_array
                except Exception:
                    # Fallback: use regular array if dask fails
                    data = zarr_array[:]
                    data_array = xr.DataArray(data, dims=dims, name=name)
                    data_vars[name] = data_array
        
        # Create Dataset with proper dimensions
        ds = xr.Dataset(data_vars, coords=coords)
        
        return ds
    
    def _get_dims_for_array(self, array_name, shape):
        """Determine dimension names for an array based on its name and shape."""
        
        # 1D arrays - map to their dimension names
        if len(shape) == 1:
            if array_name == 'reference_time':
                return ('reference_time',)
            elif array_name == 'lead_time':
                return ('lead_time',)
            elif array_name == 'valid_time':
                return ('valid_time',)
            elif array_name in ['latitude', 'longitude', 'grid_index']:
                return ('grid_index',)
            else:
                return (f'dim_0',)
        
        # 3D arrays - typically (reference_time, lead_time, grid_index) for UWC-West fields
        elif len(shape) == 3:
            return ('reference_time', 'lead_time', 'grid_index')
        
        # 2D arrays - typically (reference_time, lead_time) for valid_time
        elif len(shape) == 2:
            if array_name == 'valid_time':
                return ('reference_time', 'lead_time')
            else:
                return (f'dim_0', f'dim_1')
        
        # Fallback for other dimensions
        else:
            return tuple(f'dim_{i}' for i in range(len(shape)))
