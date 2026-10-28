"""Entry point for applying a registered transformation to datasets."""

import xarray as xr

from .registry import get_transformation


def transform(
    name: str,
    datasets: xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset],
    *args: object,
    **kwargs: object,
) -> xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset]:
    """Apply the named transformation to one or more datasets.

    Each dataset is copied before being passed to the transformation, so the
    inputs are not modified.

    Parameters
    ----------
    name : str
        Name of a registered transformation (see
        :func:`available_transformations`).
    datasets : xr.Dataset or list of xr.Dataset or dict of str to xr.Dataset
        Dataset(s) to transform.
    *args
        Positional arguments passed on to the transformation.
    **kwargs
        Keyword arguments passed on to the transformation.

    Returns
    -------
    xr.Dataset or list of xr.Dataset or dict of str to xr.Dataset
        The transformed dataset(s), in the same container type as
        ``datasets``.

    Raises
    ------
    ValueError
        If ``name`` is not a registered transformation.
    """
    func = get_transformation(name)
    if isinstance(datasets, dict):
        return {key: func(ds.copy(), *args, **kwargs) for key, ds in datasets.items()}
    if isinstance(datasets, list):
        return [func(ds.copy(), *args, **kwargs) for ds in datasets]
    return func(datasets.copy(), *args, **kwargs)
