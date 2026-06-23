import xarray as xr


def align_space(datasets, reference, **kwargs):
    """Align all datasets spatially to a reference dataset.

    Each non-reference dataset is aligned by calling ``ds.mx.align_space_with(ref_ds)``.
    Extra kwargs are forwarded to ``align_space_with`` (e.g. ``method``).

    Parameters
    ----------
    datasets : xr.Dataset, list, or dict of xr.Dataset
    reference : str or xr.Dataset
        Key into *datasets* dict, or an xr.Dataset to align to.
    """
    if isinstance(datasets, dict):
        keys = list(datasets.keys())
        ds_list = list(datasets.values())
        ref_ds = datasets[reference] if isinstance(reference, str) else reference
    else:
        ds_list = [datasets] if isinstance(datasets, (xr.Dataset, xr.DataArray)) else list(datasets)
        keys = None
        ref_ds = reference

    aligned = [
        ds if ds is ref_ds else ds.mx.align_space_with(ref_ds, **kwargs)
        for ds in ds_list
    ]

    if keys is not None:
        return dict(zip(keys, aligned))
    return aligned[0] if len(aligned) == 1 else aligned
