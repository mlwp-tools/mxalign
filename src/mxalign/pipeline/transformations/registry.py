"""Registry mapping transformation names to transformation functions."""

from collections.abc import Callable

_TRANSFORMATION_REGISTRY: dict[str, Callable] = {}


def register_transformation(name: str) -> Callable:
    """Return a decorator that registers a transformation function under ``name``.

    Parameters
    ----------
    name : str
        Name under which the decorated function is registered; an existing
        registration with the same name is replaced.

    Returns
    -------
    Callable
        Decorator that registers the function and returns it unchanged.
    """

    def decorator(func: Callable) -> Callable:
        """Register ``func`` under ``name``.

        Parameters
        ----------
        func : Callable
            Transformation function taking a dataset as first argument.

        Returns
        -------
        Callable
            ``func`` itself, unchanged.
        """
        _TRANSFORMATION_REGISTRY[name] = func
        return func

    return decorator


def available_transformations() -> list[str]:
    """Return the names of all registered transformations.

    Returns
    -------
    list of str
        Registered transformation names, in registration order.
    """
    return list(_TRANSFORMATION_REGISTRY.keys())


def get_transformation(name: str) -> Callable:
    """Return the transformation function registered under ``name``.

    Parameters
    ----------
    name : str
        Name of the registered transformation.

    Returns
    -------
    Callable
        The registered transformation function.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    try:
        return _TRANSFORMATION_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown transformation: {name}")
