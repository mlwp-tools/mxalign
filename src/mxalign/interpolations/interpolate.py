from .registry import get_interpolation


def interpolate(source_datasets, target_dataset, method, **kwargs):
    interp_cls = get_interpolation(method)
    interpolator = interp_cls(target_dataset, **kwargs)

    if isinstance(source_datasets, dict):
        keys = list(source_datasets.keys())
        datasets = list(source_datasets.values())
    else:
        if not isinstance(source_datasets, list):
            datasets = [source_datasets]
        keys = None

    if keys:
        interpolated_datasets = dict()
        for key, ds in zip(keys, datasets):
            interpolated_datasets[key] = interpolator.interpolate(ds.copy())
    else:
        interpolated_datasets = []
        for ds in datasets:
            interpolated_datasets.append(interpolator.interpolate(ds.copy()))
    interpolated_datasets = (
        interpolated_datasets[0]
        if len(interpolated_datasets) == 1
        else interpolated_datasets
    )
    return interpolated_datasets
