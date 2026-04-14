_LOADERS = {}


def register_loader(cls):
    _LOADERS[cls.name] = cls
    return cls


def available_loaders():
    return list(_LOADERS.keys())


def get_loader(name):
    try:
        return _LOADERS[name]
    except KeyError:
        raise ValueError(f"Unknown loader: {name}")
