import xarray as xr


def align_time(
    datasets: list[xr.Dataset] | dict[str, xr.Dataset],
    reference: str | xr.Dataset,
    **kwargs,
):
    """Align all datasets temporally to a reference dataset.

    Each non-reference dataset is aligned by calling ``ds.mx.align_time_with(ref_ds)``.
    Extra kwargs are forwarded to ``align_time_with`` (e.g. ``lead_time``, ``join``).

    Parameters
    ----------
    datasets : list or dict of xr.Dataset
    reference : str or xr.Dataset
        Key into *datasets* dict, or an xr.Dataset to align to.
    """
    if isinstance(datasets, dict):
        keys = list(datasets.keys())
        ds_list = list(datasets.values())
        ref_ds = datasets[reference] if isinstance(reference, str) else reference
    else:
        ds_list = [datasets] if isinstance(datasets, xr.Dataset) else list(datasets)
        keys = None
        ref_ds = reference

    aligned = [
        ds if ds is ref_ds else ds.mx.align_time_with(ref_ds, **kwargs)
        for ds in ds_list
    ]

    if keys is not None:
        return dict(zip(keys, aligned))
    return aligned[0] if len(aligned) == 1 else aligned
