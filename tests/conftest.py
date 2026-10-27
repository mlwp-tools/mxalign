"""Shared fixtures and time coordinates for the ``mxalign`` test suite."""

import numpy as np
import pytest
import xarray as xr

import mxalign  # registers ds.mx accessor  # noqa: F401

# ---------------------------------------------------------------------------
# Shared time coordinates
# ---------------------------------------------------------------------------

T0 = np.datetime64("2020-01-01T00:00", "ns")
H6 = np.timedelta64(6, "h")

REFERENCE_TIMES = np.array([T0, T0 + H6, T0 + 2 * H6])
LEAD_TIMES = np.array([np.timedelta64(h, "h") for h in [0, 6, 12, 18]])

# Forecast value convention: values[i, j] = float(i * 10 + j)
# Row 0 (ref=T0):        0,  1,  2,  3
# Row 1 (ref=T0+6h):    10, 11, 12, 13
# Row 2 (ref=T0+12h):   20, 21, 22, 23
FORECAST_VALUES = np.array([[float(i * 10 + j) for j in range(4)] for i in range(3)])

# Observation covers T0-6h … T0+24h (7 steps)
OBS_TIMES = np.array([T0 + i * H6 for i in range(-1, 5)])  # T0-6h … T0+24h
OBS_VALUES = np.arange(len(OBS_TIMES), dtype=float) * 10.0  # 0, 10, 20, 30, 40, 50

# Ensemble members: CF "realization" coordinate (TODO_ENS.md, convention C2).
# Ensemble value convention: member m adds 100 * m to the deterministic value.
MEMBERS = np.array([0, 1, 2])
MEMBER_ATTRS = {
    "standard_name": "realization",
    "units": "1",
    "long_name": "ensemble member number",
}


def _props(time: str, space: str = "point", uncertainty: str = "deterministic") -> dict:
    """Build the ``mlwp_*`` trait attributes for a dataset.

    Parameters
    ----------
    time : str
        Value of the ``mlwp_time_trait`` attribute, e.g. ``"forecast"``.
    space : str, default "point"
        Value of the ``mlwp_space_trait`` attribute.
    uncertainty : str, default "deterministic"
        Value of the ``mlwp_uncertainty_trait`` attribute.

    Returns
    -------
    dict
        Trait attributes to attach to a dataset's ``attrs``.
    """
    return {
        "mlwp_time_trait": time,
        "mlwp_space_trait": space,
        "mlwp_uncertainty_trait": uncertainty,
    }


def _member_coord(members: np.ndarray = MEMBERS) -> xr.DataArray:
    """Build a ``member`` coordinate with the CF ``realization`` attributes.

    Parameters
    ----------
    members : np.ndarray, default MEMBERS
        Member numbers.

    Returns
    -------
    xr.DataArray
        Integer ``member`` coordinate.
    """
    return xr.DataArray(members, dims="member", attrs=MEMBER_ATTRS)


@pytest.fixture
def ds_fcst() -> xr.Dataset:
    """Provide a point forecast with 3 reference times and 4 lead times.

    Returns
    -------
    xr.Dataset
        Forecast with ``temp[i, j] = i * 10 + j`` over ``reference_time`` and
        ``lead_time``.
    """
    return xr.Dataset(
        {"temp": (["reference_time", "lead_time"], FORECAST_VALUES)},
        coords={"reference_time": REFERENCE_TIMES, "lead_time": LEAD_TIMES},
        attrs=_props("forecast"),
    )


@pytest.fixture
def ds_obs() -> xr.Dataset:
    """Provide a point observation with 6 valid times from T0-6h to T0+24h.

    Returns
    -------
    xr.Dataset
        Observation with ``temp`` values 0, 10, ..., 50 along ``valid_time``.
    """
    return xr.Dataset(
        {"temp": ("valid_time", OBS_VALUES)},
        coords={"valid_time": OBS_TIMES},
        attrs=_props("observation"),
    )


@pytest.fixture
def ds_fcst_ens() -> xr.Dataset:
    """Provide a 3-member point forecast ensemble on the times of ``ds_fcst``.

    Returns
    -------
    xr.Dataset
        Ensemble with ``temp[m, i, j] = 100 * m + i * 10 + j`` over ``member``,
        ``reference_time`` and ``lead_time`` (``member`` leftmost, as CF
        recommends).
    """
    values = 100.0 * MEMBERS[:, np.newaxis, np.newaxis] + FORECAST_VALUES
    return xr.Dataset(
        {"temp": (["member", "reference_time", "lead_time"], values)},
        coords={
            "member": _member_coord(),
            "reference_time": REFERENCE_TIMES,
            "lead_time": LEAD_TIMES,
        },
        attrs=_props("forecast", uncertainty="ensemble"),
    )


@pytest.fixture
def ds_grid_ens() -> xr.Dataset:
    """Provide a 3-member forecast ensemble on a stacked 3×3 lat/lon grid.

    Returns
    -------
    xr.Dataset
        Ensemble with ``temp[m, g] = 100 * m + lat[g] + lon[g]`` over
        ``member`` and ``grid_index``, for latitudes and longitudes 0°, 1°,
        2°. The field is linear in space, so both interpolators reproduce it
        exactly.
    """
    lat, lon = np.meshgrid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], indexing="ij")
    lat, lon = lat.ravel(), lon.ravel()
    values = 100.0 * MEMBERS[:, np.newaxis] + (lat + lon)
    return xr.Dataset(
        {"temp": (["member", "grid_index"], values)},
        coords={
            "member": _member_coord(),
            "latitude": ("grid_index", lat),
            "longitude": ("grid_index", lon),
        },
        attrs=_props("forecast", space="grid", uncertainty="ensemble"),
    )
