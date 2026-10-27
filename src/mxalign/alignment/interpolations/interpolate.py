import xarray as xr

from .registry import get_interpolation


def interpolate(
    source_datasets: xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset],
    target_dataset: xr.Dataset,
    method: str,
    **kwargs,
) -> xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset]:
    """Interpolate one or more source datasets to ``target_dataset``; returns the same container type."""
    interp_cls = get_interpolation(method)
    interpolator = interp_cls(target_dataset, **kwargs)

    if isinstance(source_datasets, dict):
        return {
            key: interpolator.interpolate(ds.copy())
            for key, ds in source_datasets.items()
        }
    if isinstance(source_datasets, list):
        return [interpolator.interpolate(ds.copy()) for ds in source_datasets]
    return interpolator.interpolate(source_datasets.copy())
