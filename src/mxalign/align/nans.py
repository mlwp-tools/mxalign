import xarray as xr
import itertools


def broadcast_nans(
    datasets: xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset],
) -> xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset]:
    """Propagate NaN masks across all datasets so a NaN at any coordinate is NaN in all.

    Operates pairwise over deep copies; only shared coordinates and variables are considered.
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
