"""Convenience function to interpolate datasets with a named interpolator."""

import xarray as xr

from .registry import get_interpolation


def interpolate(
    source_datasets: xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset],
    target_dataset: xr.Dataset,
    method: str,
    **kwargs: object,
) -> xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset]:
    """Interpolate one or more source datasets to ``target_dataset``.

    Parameters
    ----------
    source_datasets : xr.Dataset or list of xr.Dataset or dict of str to xr.Dataset
        Dataset(s) to interpolate. Each is copied before interpolation.
    target_dataset : xr.Dataset
        Dataset whose spatial locations the sources are interpolated to.
    method : str
        Name of a registered interpolator, e.g. ``"xarray"`` or ``"delaunay"``.
    **kwargs : object
        Options passed to the interpolator constructor.

    Returns
    -------
    xr.Dataset or list of xr.Dataset or dict of str to xr.Dataset
        The interpolated dataset(s), in the same container type as
        ``source_datasets``.

    Raises
    ------
    ValueError
        If ``method`` is not a registered interpolator.
    """
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
