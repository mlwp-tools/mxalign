import xarray as xr

from .registry import get_transformation


def transform(
    name: str,
    datasets: xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset],
    *args,
    **kwargs,
) -> xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset]:
    """Apply the named transformation to one or more datasets; returns the same container type."""
    func = get_transformation(name)
    if isinstance(datasets, dict):
        return {key: func(ds.copy(), *args, **kwargs) for key, ds in datasets.items()}
    if isinstance(datasets, list):
        return [func(ds.copy(), *args, **kwargs) for ds in datasets]
    return func(datasets.copy(), *args, **kwargs)
