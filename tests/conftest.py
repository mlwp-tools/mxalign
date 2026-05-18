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


def _props(time: str) -> dict:
    return {
        "mlwp_time_trait": time,
        "mlwp_space_trait": "point",
        "mlwp_uncertainty_trait": "deterministic",
    }


@pytest.fixture
def ds_fcst() -> xr.Dataset:
    return xr.Dataset(
        {"temp": (["reference_time", "lead_time"], FORECAST_VALUES)},
        coords={"reference_time": REFERENCE_TIMES, "lead_time": LEAD_TIMES},
        attrs=_props("forecast"),
    )


@pytest.fixture
def ds_obs() -> xr.Dataset:
    return xr.Dataset(
        {"temp": ("valid_time", OBS_VALUES)},
        coords={"valid_time": OBS_TIMES},
        attrs=_props("observation"),
    )
