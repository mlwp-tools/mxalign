import xarray as xr

from .registry import get_transformation


def transform(
    name: str,
    datasets: xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset],
    *args,
    **kwargs,
) -> xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset]:
    """Apply the named transformation to one or more datasets; returns the same container type."""
    transform = get_transformation(name)
    if isinstance(datasets, dict):
        keys = list(datasets.keys())
        datasets = list(datasets.values())
    else:
        if not isinstance(datasets, list):
            datasets = [datasets]
        keys = None

    if keys:
        transformed_datasets = dict()
        for key, ds in zip(keys, datasets):
            transformed_datasets[key] = transform(ds.copy(), *args, **kwargs)
    else:
        transformed_datasets = []
        for ds in datasets:
            transformed_datasets.append(transform(ds.copy(), *args, **kwargs))

    transformed_datasets = (
        transformed_datasets[0]
        if len(transformed_datasets) == 1
        else transformed_datasets
    )
    return transformed_datasets
