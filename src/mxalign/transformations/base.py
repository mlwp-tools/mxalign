from .registry import register_transformation

@register_transformation("rename")
def transform(ds, rename_dict):
    new_dict = {}
    for new_name, old_names in rename_dict.items():
        for name in ds.keys():
            if name in old_names:
                new_dict[name]= new_name
            else:
                pass
    return ds.rename(new_dict)

@register_transformation("kelvin_to_celcius")
def transform(ds, variables , inverse=False):
    T_C2K = 273.15
    if isinstance(variables, str):
        variables = [variables]
    if inverse:
        t = T_C2K
    else:
        t = -T_C2K

    for var in variables:
        ds[var] = ds[var] + t

    return(ds)

@register_transformation("pa_to_hpa")
def transform(ds, variables, inverse=False):
    if isinstance(variables, str):
        variables = [variables]
    factor = 100.0 if inverse else 0.01
    for var in variables:
        ds[var] = ds[var] * factor
    return ds

@ register_transformation("uv_to_speed")
def transform(ds, u, v, speed):
    import numpy as np
    result = np.sqrt(ds[u]**2 + ds[v]**2)
    ds[speed] = result
    return ds

@register_transformation("scale")
def transform(ds, variables, factor):
    if isinstance(variables, str):
        variables = [variables]
    for var in variables:
        if var in ds:
            ds[var] = ds[var] * factor
    return ds