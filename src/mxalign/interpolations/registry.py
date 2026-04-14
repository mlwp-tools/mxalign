_INTERPOLATORS = {}


def register_interpolator(cls):
    _INTERPOLATORS[cls.name] = cls
    return cls


def available_interpolations():
    return list(_INTERPOLATORS.keys())


def get_interpolation(name):
    try:
        return _INTERPOLATORS[name]
    except KeyError:
        raise ValueError(f"Unknown interpolation: {name}")
