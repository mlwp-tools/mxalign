import xarray as xr


def align_space(
    datasets: xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset],
    reference: str | xr.Dataset,
    **kwargs,
) -> xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset]:
    """Align all datasets spatially to a reference dataset.

    Each non-reference dataset is aligned via ``ds.mx.align_space_with(ref_ds)``.
    Extra kwargs are forwarded to ``align_space_with`` (e.g. ``method``).

    Parameters
    ----------
    datasets:
        Single dataset, list, or dict of datasets to align.
    reference:
        Key into *datasets* dict, or a dataset to align to.
    """
    if isinstance(datasets, dict):
        keys = list(datasets.keys())
        ds_list = list(datasets.values())
        ref_ds = datasets[reference] if isinstance(reference, str) else reference
    else:
        ds_list = (
            [datasets]
            if isinstance(datasets, (xr.Dataset, xr.DataArray))
            else list(datasets)
        )
        keys = None
        ref_ds = reference

    aligned = [
        ds if ds is ref_ds else ds.mx.align_space_with(ref_ds, **kwargs)
        for ds in ds_list
    ]

    if keys is not None:
        return dict(zip(keys, aligned))
    return aligned[0] if len(aligned) == 1 else aligned
