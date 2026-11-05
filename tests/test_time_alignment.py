"""Unit tests for the start of the forecast-observation alignment.

Runs standalone (``python tests/test_time_alignment.py``), like test_spread_skill.py.
A run is scored only if all its leads are at or after the first observation, so every
lead has the same runs.
"""
import sys
import traceback

import numpy as np
import pandas as pd
import xarray as xr

from mxalign.accessors.time import _align_observation_forecast
from mxalign.properties.properties import Space, Time, Uncertainty, Properties
from mxalign.properties.utils import properties_to_attrs


def _attrs(time):
    props = Properties(space=Space.POINT, time=time, uncertainty=Uncertainty.DETERMINISTIC)
    return {"properties": properties_to_attrs(props)}


def _forecast(inits, leads=(6, 12)):
    r, l = pd.to_datetime(inits), pd.to_timedelta(leads, "h")
    return xr.Dataset({"t": (("reference_time", "lead_time", "point_index"), np.ones((len(r), len(l), 1)))},
                      coords={"reference_time": r, "lead_time": l, "point_index": [0]}, attrs=_attrs(Time.FORECAST))


def _obs(start, end):
    v = pd.date_range(start, end, freq="6h")
    return xr.Dataset({"t": (("valid_time", "point_index"), np.ones((len(v), 1)))},
                      coords={"valid_time": v, "point_index": [0]}, attrs=_attrs(Time.OBSERVATION))


def _scored_runs_per_lead(obs, fct, only_common):
    aligned, _ = _align_observation_forecast(obs, fct, only_common=only_common)
    return aligned.t.isel(point_index=0).notnull().sum("reference_time").values.tolist()


def _kept_runs(obs, fct):
    aligned, _ = _align_observation_forecast(obs, fct, only_common=True)
    return [pd.Timestamp(r).strftime("%dT%H") for r in aligned.reference_time.values]


def test_first_run_is_kept_when_its_shortest_lead_has_obs():
    fct = _forecast(["2026-07-02T00", "2026-07-02T06", "2026-07-02T12"])
    obs = _obs("2026-07-02T06", "2026-07-03T00")
    assert _kept_runs(obs, fct) == ["02T00", "02T06", "02T12"]
    for only_common in (True, False):
        assert _scored_runs_per_lead(obs, fct, only_common) == [3, 3]


def test_run_partly_before_obs_is_dropped_so_leads_have_equal_runs():
    fct = _forecast(["2026-07-01T18", "2026-07-02T00", "2026-07-02T06"])
    obs = _obs("2026-07-02T06", "2026-07-02T18")
    assert _kept_runs(obs, fct) == ["02T00", "02T06"]
    for only_common in (True, False):
        assert _scored_runs_per_lead(obs, fct, only_common) == [2, 2]


def test_with_lead_zero_a_run_before_the_obs_is_dropped():
    fct = _forecast(["2026-07-02T00", "2026-07-02T06"], leads=(0, 6))
    obs = _obs("2026-07-02T06", "2026-07-02T18")
    assert _kept_runs(obs, fct) == ["02T06"]


def test_run_past_the_last_obs_is_dropped():
    fct = _forecast(["2026-07-02T00", "2026-07-02T06"])
    obs = _obs("2026-07-02T06", "2026-07-02T12")
    assert _kept_runs(obs, fct) == ["02T00"]
    assert _scored_runs_per_lead(obs, fct, True) == [1, 1]


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
