"""Propagation of NaN masks across a collection of datasets."""

import itertools

import xarray as xr


def broadcast_nans(
    datasets: xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset],
) -> xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset]:
    """Propagate NaN masks so a NaN at a coordinate in any dataset is NaN in all.

    Works pairwise on the variables two datasets share, at the coordinate
    values they share; values outside that intersection are left as they are.

    The uncertainty dimension (``member`` for an ensemble, ``quantile`` for a
    quantile dataset, taken from the uncertainty trait) is never matched
    between datasets. Instead, a value counts as missing when any of its
    members or quantiles is, and the mask is then applied to all of them. This
    keeps the ensemble size constant over the verification sample.

    Parameters
    ----------
    datasets : xr.Dataset or list of xr.Dataset or dict of str to xr.Dataset
        Datasets to propagate NaNs across. A single dataset is returned as is.

    Returns
    -------
    xr.Dataset or list of xr.Dataset or dict of str to xr.Dataset
        Copies of the input datasets with NaN masks propagated, in the same
        container type as ``datasets``.

    Raises
    ------
    KeyError
        If a dataset has no uncertainty trait attribute.
    ValueError
        If a shared variable has different dimensions in two datasets, other
        than their uncertainty dimensions.
    """
    if isinstance(datasets, xr.Dataset):
        return datasets
    elif isinstance(datasets, dict):
        keys = list(datasets.keys())
        working = [ds.copy() for ds in datasets.values()]
    else:
        keys = None
        working = [ds.copy() for ds in datasets]

    uncertainty_dims = [_uncertainty_dims(ds) for ds in working]

    for i, j in itertools.combinations(range(len(working)), 2):
        dsA, dsB = working[i], working[j]
        for var in dsA.data_vars:
            if var not in dsB:
                continue
            maskA = _missing(dsA[var], uncertainty_dims[i])
            maskB = _missing(dsB[var], uncertainty_dims[j])
            if set(maskA.dims) != set(maskB.dims):
                raise ValueError(
                    f"Cannot broadcast NaNs for variable '{var}': dimensions "
                    f"{sorted(maskA.dims)} and {sorted(maskB.dims)} differ. Only "
                    "the uncertainty dimension (member, quantile) may differ "
                    "between datasets; align them in time and space first."
                )
            # Masks over the shared coordinate values, applied back to each
            # dataset; outside the intersection nothing is masked
            maskA, maskB = xr.align(maskA, maskB, join="inner")
            mask = maskA | maskB
            dsA[var] = dsA[var].where(~mask.reindex_like(dsA[var], fill_value=False))
            dsB[var] = dsB[var].where(~mask.reindex_like(dsB[var], fill_value=False))

    return dict(zip(keys, working)) if keys else working


def _uncertainty_dims(ds: xr.Dataset) -> list[str]:
    """Return the uncertainty dimensions of ``ds`` according to its trait.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with an uncertainty trait attribute.

    Returns
    -------
    list of str
        ``["member"]`` for an ensemble, ``["quantile"]`` for a quantile
        dataset and ``[]`` for a deterministic one.
    """
    if ds.mx.is_ensemble():
        return ["member"]
    if ds.mx.is_quantile():
        return ["quantile"]
    return []


def _missing(da: xr.DataArray, uncertainty_dims: list[str]) -> xr.DataArray:
    """Return where ``da`` is missing, reduced over its uncertainty dimensions.

    Parameters
    ----------
    da : xr.DataArray
        Variable to build the mask for.
    uncertainty_dims : list of str
        Dimensions over which a single missing value marks the whole
        ensemble (or set of quantiles) as missing.

    Returns
    -------
    xr.DataArray
        Boolean mask without the uncertainty dimensions and without non-index
        coordinates, so they cannot conflict when masks are combined.
    """
    mask = da.isnull().any([d for d in uncertainty_dims if d in da.dims])
    return mask.reset_coords(drop=True)
