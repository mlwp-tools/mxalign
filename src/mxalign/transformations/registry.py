from collections.abc import Callable

_TRANSFORMATION_REGISTRY: dict[str, Callable] = {}


def register_transformation(name: str) -> Callable:
    """Decorator that registers a transformation function under ``name``."""
    def decorator(func: Callable) -> Callable:
        _TRANSFORMATION_REGISTRY[name] = func
        return func

    return decorator


def available_transformations() -> list[str]:
    """Return the names of all registered transformations."""
    return list(_TRANSFORMATION_REGISTRY.keys())


def get_transformation(name: str) -> Callable:
    """Return the transformation function registered under ``name``.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    try:
        return _TRANSFORMATION_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown transformation: {name}")
