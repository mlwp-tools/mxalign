"""Unit tests for mxalign.metrics.n_cases.

Runs standalone (``python tests/test_n_cases.py``), like test_spread_skill.py.
"""
import sys
import traceback

import numpy as np
import xarray as xr

from mxalign.metrics import n_cases

DIMS = ["reference_time", "point_index"]


def _make_ds(n_m=5, n_t=3, n_lead=2, n_p=4, variables=("2t",)):
    """Aligned (observations, forecasts) datasets without any NaN."""
    obs = xr.Dataset({v: (("reference_time", "lead_time", "point_index"),
                          np.zeros((n_t, n_lead, n_p), dtype="float32")) for v in variables})
    fct = xr.Dataset({v: (("member", "reference_time", "lead_time", "point_index"),
                          np.zeros((n_m, n_t, n_lead, n_p), dtype="float32")) for v in variables})
    return obs, fct


def test_all_finite():
    obs, fct = _make_ds()
    out = n_cases(fct, obs, DIMS)
    assert out["2t"].dims == ("lead_time",)
    assert out["2t"].dtype == "float64"
    assert out["2t"].values.tolist() == [12, 12]


def test_one_missing_member_drops_the_case():
    obs, fct = _make_ds()
    fct["2t"][2, 0, 1, 3] = np.nan
    assert n_cases(fct, obs, DIMS)["2t"].values.tolist() == [12, 11]


def test_missing_observation_drops_the_case():
    obs, fct = _make_ds()
    obs["2t"][1, 0, 0] = np.nan
    assert n_cases(fct, obs, DIMS)["2t"].values.tolist() == [11, 12]


def test_same_case_missing_in_both_counts_once():
    obs, fct = _make_ds()
    obs["2t"][1, 0, 0] = np.nan
    fct["2t"][0, 1, 0, 0] = np.nan
    assert n_cases(fct, obs, DIMS)["2t"].values.tolist() == [11, 12]


def test_whole_lead_missing_gives_zero():
    obs, fct = _make_ds()
    fct["2t"][4, :, 1, :] = np.nan
    assert n_cases(fct, obs, DIMS)["2t"].values.tolist() == [12, 0]


def test_variables_are_independent_and_common_only():
    obs, fct = _make_ds(variables=("2t", "msl"))
    fct["10u"] = fct["2t"]
    obs["msl"][0, 0, 0] = np.nan
    out = n_cases(fct, obs, DIMS)
    assert set(out.data_vars) == {"2t", "msl"}
    assert out["2t"].values.tolist() == [12, 12]
    assert out["msl"].values.tolist() == [11, 12]


def test_dask_gives_the_same():
    obs, fct = _make_ds()
    fct["2t"][2, 0, 1, 3] = np.nan
    out = n_cases(fct.chunk({"lead_time": 1}), obs.chunk({"lead_time": 1}), DIMS)
    assert out["2t"].values.tolist() == [12, 11]


def _main():
    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
        else:
            print(f"ok    {fn.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
