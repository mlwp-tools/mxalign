from .registry import get_loader


def load(name, files, variables=None, grid_mapping=None, **kwargs):
    loader_cls = get_loader(name)
    loader = loader_cls(files, variables, grid_mapping, **kwargs)

    return loader.load()
