from collections.abc import Callable

import xarray as xr

from .registry import register_transformation


@register_transformation("external")
def transform(
    ds: xr.Dataset,
    func_path: str,
    inputs: dict[str, str],
    output: str,
    **kwargs,
) -> xr.Dataset:
    """Call an external function and store its result as ``output`` in ``ds``.

    Parameters
    ----------
    func_path:
        Dotted path to the function, e.g. ``"mypackage.module.func"``.
    inputs:
        Mapping of function argument names to variable names in ``ds``.
    output:
        Name of the new variable written back into ``ds``.
    """
    func = _resolve_function(func_path)
    input_kwargs = {arg_name: ds[var_name] for arg_name, var_name in inputs.items()}
    result = func(**{**input_kwargs, **kwargs})
    ds[output] = (ds.dims, result)
    return ds


def _resolve_function(func_path: str) -> Callable:
    """Import and return the callable at ``func_path`` (e.g. ``"pkg.module.func"``)."""
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
