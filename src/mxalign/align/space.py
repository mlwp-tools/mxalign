import xarray as xr

from ..properties.properties import Properties, Time, Space, Uncertainty
from ..properties.utils import properties_to_attrs

def align_space(datasets, reference, **kwargs):
    if isinstance(datasets, (xr.Dataset, xr.DataArray)):
        datasets = [datasets]
    if isinstance(datasets, dict):
        keys = datasets.keys()
        datasets = datasets.items()
    else:
        keys = None

    aligned = [ds.space.align_with(reference, **kwargs) for ds in datasets]
    # align_with returns (aligned_ds, filtered_reference); keep the last filtered reference
    datasets = [ds for ds, _ in aligned]
    reference = aligned[-1][1]

    if keys is None:
        if len(datasets) == 1:
            return datasets[0], reference
        else:
            return datasets, reference
    else:
        return {key: value for (key, value) in zip(keys, datasets)}, reference
        
