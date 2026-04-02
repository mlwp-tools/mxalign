from .registry import register_transformation


@register_transformation("external")
def transform(ds, func_path, inputs, output, **kwargs):
    func = _resolve_function(func_path)

    input_kwargs = {arg_name: ds[var_name] for arg_name, var_name in inputs.items()}

    all_kwargs = {**input_kwargs, **kwargs}
    result = func(**all_kwargs)
    # print(result)
    ds[output] = (ds.dims, result)
    return ds


def _resolve_function(func_path):
    import importlib

    module_path, func_name = func_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Could not import module '{module_path}' required for transform '{func_path}'. "
            f"Make sure it is installed. Original error: {e}"
        )
    try:
        return getattr(module, func_name)
    except AttributeError:
        raise AttributeError(
            f"Module '{module_path}' has no function '{func_name}'. "
            f"Check the function name in your config."
        )
