"""Unit tests for mxalign.metrics.fair_crps.

Runs standalone (``python tests/test_fair_crps.py``), like test_spread_skill.py.
"""
import sys
import traceback

import numpy as np
import xarray as xr
from scores.probability import crps_for_ensemble

from mxalign.metrics import bias, fair_crps, n_cases

DIMS = ["reference_time", "point_index"]


def _make_ds(n_m=5, n_t=3, n_lead=2, n_p=40, dtype="float64"):
    rng = np.random.default_rng(1)
    obs = xr.Dataset({"2t": (("reference_time", "lead_time", "point_index"),
                             rng.normal(280, 5, (n_t, n_lead, n_p)).astype(dtype))})
    fct = xr.Dataset({"2t": (("member", "reference_time", "lead_time", "point_index"),
                             rng.normal(280, 5, (n_m, n_t, n_lead, n_p)).astype(dtype))})
    return obs, fct


def _numpy_fair_crps(fct, obs):
    """Independent computation, complete cases only; one value per lead."""
    x, y = fct["2t"].values, obs["2t"].values
    m = x.shape[0]
    out = []
    for lead in range(y.shape[1]):
        xl, yl = x[:, :, lead].reshape(m, -1), y[:, lead].ravel()
        ok = np.isfinite(yl) & np.isfinite(xl).all(axis=0)
        xl, yl = xl[:, ok], yl[ok]
        skill = np.abs(xl - yl).mean(axis=0)
        pairs = np.abs(xl[:, None] - xl[None]).sum(axis=(0, 1))
        out.append((skill - pairs / (2 * m * (m - 1))).mean() if ok.any() else np.nan)
    return np.array(out)


def _scores(fct, obs):
    return crps_for_ensemble(fct, obs, ensemble_member_dim="member", method="fair", reduce_dims=DIMS)


def test_no_nan_identical_to_scores():
    for dtype in ("float32", "float64"):
        obs, fct = _make_ds(dtype=dtype)
        np.testing.assert_array_equal(fair_crps(fct, obs, DIMS)["2t"], _scores(fct, obs)["2t"])


def test_matches_numpy_with_nans():
    for m in (5, 10):
        obs, fct = _make_ds(n_m=m)
        fct["2t"][2, 0, 1, 3] = np.nan
        fct["2t"][:2, 1, 0, 7] = np.nan
        obs["2t"][2, 1, 5:9] = np.nan
        np.testing.assert_allclose(fair_crps(fct, obs, DIMS)["2t"], _numpy_fair_crps(fct, obs), rtol=1e-12)


def test_missing_member_drops_the_case():
    """The case must not be scored with the remaining members, as scores does."""
    obs, fct = _make_ds(n_lead=1)
    fct["2t"][4, 0, 0, 0] = np.nan
    # same result as a truth missing at that case
    obs2, fct2 = _make_ds(n_lead=1)
    obs2["2t"][0, 0, 0] = np.nan
    got = fair_crps(fct, obs, DIMS)["2t"].values
    np.testing.assert_array_equal(got, fair_crps(fct2, obs2, DIMS)["2t"].values)
    assert abs(got[0] - _scores(fct, obs)["2t"].values[0]) > 1e-6
    assert n_cases(fct, obs, DIMS)["2t"].values.tolist() == [3 * 40 - 1]


def test_whole_lead_missing_is_nan():
    obs, fct = _make_ds()
    fct["2t"][1, :, 1, :] = np.nan
    out = fair_crps(fct, obs, DIMS)["2t"].values
    assert np.isfinite(out[0]) and np.isnan(out[1])


def test_dask_gives_the_same():
    obs, fct = _make_ds()
    fct["2t"][2, 0, 1, 3] = np.nan
    lazy = fair_crps(fct.chunk({"member": -1, "lead_time": 1}), obs.chunk({"lead_time": 1}), DIMS)
    np.testing.assert_allclose(lazy["2t"].values, fair_crps(fct, obs, DIMS)["2t"].values, rtol=1e-14)


def test_bias_on_complete_cases():
    obs, fct = _make_ds(dtype="float32")
    fct["2t"][2, 0, 1, 3] = np.nan
    obs["2t"][1, 0, 5:9] = np.nan
    x, y = fct["2t"].values.astype("float64"), obs["2t"].values.astype("float64")
    ok = np.isfinite(y) & np.isfinite(x).all(axis=0)
    want = [np.where(ok, x.mean(axis=0) - y, np.nan)[:, k][ok[:, k]].mean() for k in range(x.shape[2])]
    got = bias(fct, obs, DIMS)["2t"]
    assert got.dtype == "float64"
    np.testing.assert_allclose(got.values, want, rtol=1e-12)


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
