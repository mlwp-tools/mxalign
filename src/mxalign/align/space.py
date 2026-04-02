import xarray as xr


def align_space(datasets, reference, **kwargs):
    if isinstance(datasets, (xr.Dataset, xr.DataArray)):
        datasets = [datasets]
    if isinstance(datasets, dict):
        keys = datasets.keys()
        datasets = datasets.items()
    else:
        keys = None

    datasets = [ds.space.align_with(reference, **kwargs)[0] for ds in datasets]

    if keys is None:
        if len(datasets) == 1:
            return datasets[0]
        else:
            return datasets
    else:
        return {key: value for (key, value) in zip(keys, datasets)}
