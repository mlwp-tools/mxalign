_INTERPOLATORS: dict[str, type] = {}


def register_interpolator(cls: type) -> type:
    """Register an interpolator class under ``cls.name``; returns the class unchanged."""
    _INTERPOLATORS[cls.name] = cls
    return cls


def available_interpolations() -> list[str]:
    """Return the names of all registered interpolators."""
    return list(_INTERPOLATORS.keys())


def get_interpolation(name: str) -> type:
    """Return the interpolator class registered under ``name``.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    try:
        return _INTERPOLATORS[name]
    except KeyError:
        raise ValueError(f"Unknown interpolation: {name}")
