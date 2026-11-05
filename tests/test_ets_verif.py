"""Unit tests for mxalign.metrics.ets_verif.

Runs standalone (``python tests/test_ets_verif.py``), like test_spread_skill.py. Expected
values follow verif.metric.Ets: a hit, b false alarm, c miss, d correct rejection.
"""
import sys
import traceback

import numpy as np
import xarray as xr

from mxalign.metrics import ets, ets_verif


def _ds(values, var="10si"):
    values = np.asarray(values, dtype="float64")
    return xr.Dataset({var: ("point_index", values)}, coords={"point_index": np.arange(values.size)})


def _score(obs, fct, thresholds, **kwargs):
    return ets_verif(_ds(obs), _ds(fct), thresholds, **kwargs)["10si"]


def test_hand_computed_value():
    # a=2 b=1 c=1 d=6: ar = 3 * 3 / 10 = 0.9, ETS = 1.1 / 3.1
    obs = [5, 5, 5, 0, 0, 0, 0, 0, 0, 0]
    fct = [5, 5, 0, 5, 0, 0, 0, 0, 0, 0]
    assert np.isclose(float(_score(obs, fct, 1.0).squeeze()), 1.1 / 3.1, rtol=1e-14)


def test_event_is_strictly_above_threshold():
    # Equal to the threshold is not an event: obs has no event, fct has one false alarm at 3.5 only.
    result = _score([2.0, 2.0, 0.0], [3.5, 2.0, 0.0], 2.0).squeeze()
    assert float(result) == 0.0


def test_perfect_forecast_scores_one():
    obs = [0, 1, 2, 3, 4, 5]
    assert float(_score(obs, obs, 2.5).squeeze()) == 1.0


def test_no_event_at_all_is_nan():
    assert np.isnan(float(_score([0, 1], [0, 1], 10.0).squeeze()))


def test_no_forecast_event_scores_zero():
    assert float(_score([5, 0], [0, 0], 1.0).squeeze()) == 0.0


def test_no_minimum_number_of_events():
    # One event only, which ets() masks with its default min_events=10.
    assert float(_score([5, 0, 0, 0], [5, 0, 0, 0], 1.0).squeeze()) == 1.0
    old = ets(_ds([5, 0, 0, 0]), _ds([5, 0, 0, 0]), 1.0, 2.0, 1.0)["10si"]
    assert np.isnan(float(old.squeeze()))


def test_only_valid_pairs_are_counted():
    obs = [5, 5, 0, 0, np.nan, 0]
    fct = [5, 0, 5, 0, 5, np.nan]
    clean = _score([5, 5, 0, 0], [5, 0, 5, 0], 1.0)
    assert float(_score(obs, fct, 1.0).squeeze()) == float(clean.squeeze())


def test_threshold_dimension_and_kept_dims():
    obs = xr.Dataset({"10si": (("lead_time", "point_index"), [[5.0, 0.0], [0.0, 0.0]])},
                     coords={"lead_time": [6, 12], "point_index": [0, 1]})
    result = ets_verif(obs, obs, [1.0, 2.0])["10si"]
    assert set(result.dims) == {"threshold", "lead_time"}
    assert list(result.threshold.values) == [1.0, 2.0]
    assert float(result.sel(lead_time=6, threshold=1.0)) == 1.0
    assert np.isnan(float(result.sel(lead_time=12, threshold=1.0)))


def test_pools_over_several_dims_and_selects_variables():
    obs = xr.Dataset({"10si": (("t", "p"), [[5.0, 0.0], [0.0, 5.0]]), "2t": (("t", "p"), [[1.0, 1.0], [1.0, 1.0]])})
    result = ets_verif(obs, obs, 1.5, dim=["t", "p"], variables=["10si"])
    assert list(result.data_vars) == ["10si"]
    assert float(result["10si"].squeeze()) == 1.0


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
