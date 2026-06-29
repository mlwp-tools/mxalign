import numpy as np
import xarray as xr

# Tolerance in degrees that the coordinates of two grids can differ while still
# being interpreted as the same grid. 0.0001 degrees ~ 10m at 45 deg latitude.
COORD_TOLERANCE = 0.0001


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
    from ..interpolations.registry import get_interpolation

    interp_cls = get_interpolation(method)
    return interp_cls(ds2, **kwargs).interpolate(ds1.copy())
