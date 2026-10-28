"""Built-in transformations: renaming, unit conversion and wind speed."""

import xarray as xr

from .registry import register_transformation


@register_transformation("rename")
def transform_rename(ds: xr.Dataset, rename_dict: dict[str, list[str]]) -> xr.Dataset:
    """Rename variables in ``ds`` according to a mapping of new to old names.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset whose variables are renamed.
    rename_dict : dict of str to list of str
        Mapping of new names to lists of old names; any variable in ``ds``
        whose name is in a list is renamed to the corresponding key.

    Returns
    -------
    xr.Dataset
        Dataset with the variables renamed.
    """
    new_dict = {}
    for new_name, old_names in rename_dict.items():
        for name in ds.keys():
            if name in old_names:
                new_dict[name] = new_name
    return ds.rename(new_dict)


@register_transformation("kelvin_to_celcius")
def transform_kelvin_to_celcius(
    ds: xr.Dataset,
    variables: str | list[str],
    inverse: bool = False,
) -> xr.Dataset:
    """Convert ``variables`` between Kelvin and Celsius.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset holding the variables; modified in place.
    variables : str or list of str
        Name(s) of the variables to convert.
    inverse : bool, default False
        Convert from Celsius to Kelvin instead of Kelvin to Celsius.

    Returns
    -------
    xr.Dataset
        ``ds`` with the variables converted.
    """
    T_C2K = 273.15
    if isinstance(variables, str):
        variables = [variables]
    t = T_C2K if inverse else -T_C2K
    for var in variables:
        ds[var] = ds[var] + t
    return ds


@register_transformation("uv_to_speed")
def transform(ds: xr.Dataset, u: str, v: str, speed: str) -> xr.Dataset:
    """Compute wind speed from ``u`` and ``v`` components and store it as ``speed``.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset holding the wind components; modified in place.
    u : str
        Name of the eastward wind component variable.
    v : str
        Name of the northward wind component variable.
    speed : str
        Name of the wind speed variable to write.

    Returns
    -------
    xr.Dataset
        ``ds`` with the ``speed`` variable added.
    """
    import numpy as np

    result = np.sqrt(ds[u] ** 2 + ds[v] ** 2)
    ds[speed] = result
    return ds
