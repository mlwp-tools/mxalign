"""Registry mapping interpolator names to interpolator classes."""

from .base import BaseInterpolator

_INTERPOLATORS: dict[str, type[BaseInterpolator]] = {}


def register_interpolator(cls: type[BaseInterpolator]) -> type[BaseInterpolator]:
    """Register an interpolator class under ``cls.name``.

    Intended for use as a class decorator.

    Parameters
    ----------
    cls : type of BaseInterpolator
        Interpolator class to register; must define a ``name`` attribute.

    Returns
    -------
    type of BaseInterpolator
        The class, unchanged.
    """
    _INTERPOLATORS[cls.name] = cls
    return cls


def available_interpolations() -> list[str]:
    """Return the names of all registered interpolators.

    Returns
    -------
    list of str
        Names of the registered interpolators.
    """
    return list(_INTERPOLATORS.keys())


def get_interpolation(name: str) -> type[BaseInterpolator]:
    """Return the interpolator class registered under ``name``.

    Parameters
    ----------
    name : str
        Name of a registered interpolator, e.g. ``"xarray"`` or ``"delaunay"``.

    Returns
    -------
    type of BaseInterpolator
        The registered interpolator class.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    try:
        return _INTERPOLATORS[name]
    except KeyError:
        raise ValueError(f"Unknown interpolation: {name}")
