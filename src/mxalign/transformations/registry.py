_TRANSFORMATION_REGISTRY = {}


def register_transformation(name):
    def decorator(func):
        _TRANSFORMATION_REGISTRY[name] = func
        return func

    return decorator


def available_transformations():
    return list(_TRANSFORMATION_REGISTRY.keys())


def get_transformation(name):
    try:
        return _TRANSFORMATION_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown transformation: {name}")
