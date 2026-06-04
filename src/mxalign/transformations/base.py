from .registry import register_transformation


def _sig_rename(rename_dict):
    outputs = list(rename_dict.keys())
    inputs: list[str] = []
    for v in rename_dict.values():
        inputs.extend(v if isinstance(v, list) else [v])
    return inputs, outputs


@register_transformation("rename", signature=_sig_rename)
def transform_rename(ds, rename_dict):
    new_dict = {}
    for new_name, old_names in rename_dict.items():
        for name in ds.keys():
            if name in old_names:
                new_dict[name] = new_name
            else:
                pass
    return ds.rename(new_dict)


def _sig_kelvin_to_celcius(variables, inverse=False):
    v = [variables] if isinstance(variables, str) else list(variables)
    return v, v  # in-place


@register_transformation("kelvin_to_celcius", signature=_sig_kelvin_to_celcius)
def transform_kelvin_to_celcius(ds, variables, inverse=False):
    T_C2K = 273.15
    if isinstance(variables, str):
        variables = [variables]
    if inverse:
        t = T_C2K
    else:
        t = -T_C2K

    for var in variables:
        ds[var] = ds[var] + t

    return ds


def _sig_uv_to_speed(u, v, speed):
    us = [u] if isinstance(u, str) else list(u)
    vs = [v] if isinstance(v, str) else list(v)
    ss = [speed] if isinstance(speed, str) else list(speed)
    return us + vs, ss


@register_transformation("uv_to_speed", signature=_sig_uv_to_speed)
def transform(ds, u, v, speed):
    import numpy as np

    us = [u] if isinstance(u, str) else u
    vs = [v] if isinstance(v, str) else v
    speeds = [speed] if isinstance(speed, str) else speed
    for u_var, v_var, s_var in zip(us, vs, speeds):
        ds[s_var] = np.sqrt(ds[u_var] ** 2 + ds[v_var] ** 2)
    return ds
