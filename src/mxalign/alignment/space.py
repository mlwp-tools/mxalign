import numpy as np
import xarray as xr

# Tolerance in degrees that the coordinates of two grids can differ while still
# being interpreted as the same grid. 0.0001 degrees ~ 10m at 45 deg latitude.
COORD_TOLERANCE = 0.0001


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


# --- Pairwise alignment (used by ``ds.mx.align_space_with``) ---


def align_grid_grid(ds1: xr.Dataset, ds2: xr.Dataset, **kwargs) -> xr.Dataset:
    """Return ``ds1`` unchanged if its grid matches ``ds2``'s within ``COORD_TOLERANCE``; raises otherwise."""
    if np.array_equal(
        ds1["longitude"].values, ds2["longitude"].values
    ) and np.array_equal(ds1["latitude"].values, ds2["latitude"].values):
        return ds1
    elif np.allclose(
        ds1["longitude"].values, ds2["longitude"].values, atol=COORD_TOLERANCE
    ) and np.allclose(
        ds1["latitude"].values, ds2["latitude"].values, atol=COORD_TOLERANCE
    ):
        print(
            f"Some lat-lon coordinates differ but within {COORD_TOLERANCE}°, treating as equal"
        )
        return ds1
    else:
        raise NotImplementedError("Regridding not implemented")


def align_grid_point(
    ds1: xr.Dataset, ds2: xr.Dataset, method: str = "xarray", **kwargs
) -> xr.Dataset:
    """Interpolate grid ``ds1`` to the point locations of ``ds2`` using the named interpolator."""
    from .interpolations.registry import get_interpolation

    interp_cls = get_interpolation(method)
    return interp_cls(ds2, **kwargs).interpolate(ds1.copy())
