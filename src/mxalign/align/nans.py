import xarray as xr
import itertools


def broadcast_nans(datasets: dict | list) -> None:
    """
    Broadcasts NaN values across a list of xarray Datasets by ensuring that if a value is NaN
    in one dataset at a specific coordinate, it becomes NaN in all datasets at that coordinate.

    Parameters
    ----------
    datasets : list[xr.Dataset] | dict[str, xr.Dataset]
        A list of xarray Datasets to process. The datasets should share some common
        coordinates and variables.

    Returns
    -------
    list[xr.Dataset] | dict[str, xr.Dataset]


    Notes
    -----
    - The function operates on pairs of datasets, comparing each dataset with every other dataset
      in the list.
    - Only coordinate values that exist in both datasets of a pair are considered.
    - Only variables that exist in both datasets of a pair are processed.
    - The NaN broadcasting is performed at the intersection of coordinates between each pair
      of datasets.

    Examples
    --------
    >>> ds1 = xr.Dataset(...)
    >>> ds2 = xr.Dataset(...)
    >>> ds3 = xr.Dataset(...)
    >>> broadcast_nans([ds1, ds2, ds3])
    """

    if isinstance(datasets, xr.Dataset):
        return datasets
    elif isinstance(datasets, dict):
        keys = list(datasets.keys())
        working = [ds.copy(deep=True) for ds in datasets.values()]
    else:
        keys = None
        working = [ds.copy(deep=True) for ds in datasets]

    # Iterate over all pairs of datasets
    for dsA, dsB in itertools.combinations(working, 2):
        # Find the shared coordinates for all dimensions
        common_coords = {
            dim: sorted(set(dsA[dim].values) & set(dsB[dim].values)) for dim in dsA.dims
        }

        # Iterate over all variables
        for var in dsA.data_vars:
            if var in dsB:  # Ensure both datasets have the variable
                # Select the data at common coordinates
                selA = dsA[var].sel(**common_coords)
                selB = dsB[var].sel(**common_coords)

                # Compute NaN mask for shared coordinates
                nan_mask = selA.isnull() | selB.isnull()

                # Apply NaN mask back to both datasets
                dsA[var].loc[common_coords] = (
                    dsA[var].sel(**common_coords).where(~nan_mask)
                )
                dsB[var].loc[common_coords] = (
                    dsB[var].sel(**common_coords).where(~nan_mask)
                )

    return dict(zip(keys, working)) if keys else working
