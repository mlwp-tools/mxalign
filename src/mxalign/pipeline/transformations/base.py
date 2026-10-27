import xarray as xr

from .registry import register_transformation


@register_transformation("rename")
def transform_rename(ds: xr.Dataset, rename_dict: dict[str, list[str]]) -> xr.Dataset:
    """Rename variables in ``ds``; ``rename_dict`` maps new names to lists of old names."""
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
    """Convert ``variables`` between Kelvin and Celsius; set ``inverse=True`` for °C → K."""
    T_C2K = 273.15
    if isinstance(variables, str):
        variables = [variables]
    t = T_C2K if inverse else -T_C2K
    for var in variables:
        ds[var] = ds[var] + t
    return ds


@register_transformation("uv_to_speed")
def transform(ds: xr.Dataset, u: str, v: str, speed: str) -> xr.Dataset:
    """Compute wind speed from ``u`` and ``v`` components and store it as ``speed``."""
    import numpy as np

    result = np.sqrt(ds[u] ** 2 + ds[v] ** 2)
    ds[speed] = result
    return ds
