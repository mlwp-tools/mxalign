"""Helpers for longitudes that may use either the 0–360 or the [-180, 180) convention."""

import numpy as np
from numpy.typing import ArrayLike


def wrap_longitude(longitude: ArrayLike, start: float) -> np.ndarray:
    """Shift longitudes by multiples of 360° into ``[start, start + 360)``.

    Parameters
    ----------
    longitude : ArrayLike
        Longitudes in degrees, in any convention.
    start : float
        Lower bound of the target range, in degrees.

    Returns
    -------
    np.ndarray
        The same longitudes, each in ``[start, start + 360)``.
    """
    return start + np.mod(np.asarray(longitude, dtype=float) - start, 360.0)


def longitude_difference(lon1: ArrayLike, lon2: ArrayLike) -> np.ndarray:
    """Return ``lon1 - lon2`` reduced to ``[-180, 180)``.

    Two longitudes that denote the same meridian in different conventions
    differ by ``0``.

    Parameters
    ----------
    lon1 : ArrayLike
        Longitudes in degrees.
    lon2 : ArrayLike
        Longitudes in degrees, broadcastable against ``lon1``.

    Returns
    -------
    np.ndarray
        The signed difference in degrees.
    """
    return wrap_longitude(np.asarray(lon1, dtype=float) - np.asarray(lon2), -180.0)


def is_periodic(longitude: ArrayLike, tolerance: float = 1e-4) -> bool:
    """Check whether 1-D grid longitudes go all the way around the globe.

    The grid is periodic when the gap between its last and first longitude,
    across the seam, is no wider than its widest spacing.

    Parameters
    ----------
    longitude : ArrayLike
        1-D grid longitudes in degrees, in any order but within one 360° range.
    tolerance : float, default 1e-4
        Slack in degrees on the comparison.

    Returns
    -------
    bool
        ``True`` if the grid is global in longitude.
    """
    lon = np.sort(np.asarray(longitude, dtype=float))
    if lon.size < 2:
        return False
    seam_gap = lon[0] + 360.0 - lon[-1]
    return bool(seam_gap <= np.diff(lon).max() + tolerance)
