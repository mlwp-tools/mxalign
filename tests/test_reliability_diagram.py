"""Unit tests for mxalign.metrics.reliability_diagram.

Runs standalone (``python tests/test_reliability_diagram.py``), like
test_spread_skill.py.
"""
import sys
import traceback

import numpy as np
import xarray as xr

from mxalign.metrics import n_cases, reliability_diagram

DIMS = ["reference_time", "point_index"]


def _make_ds(obs, fct, var="10si"):
    """obs (time, lead, point), fct (member, time, lead, point)."""
    dims = ("reference_time", "lead_time", "point_index")
    return xr.Dataset({var: (dims, obs)}), xr.Dataset({var: (("member", *dims), fct)})


def _random(n_m=5, n_t=3, n_lead=2, n_p=200, seed=2):
    rng = np.random.default_rng(seed)
    base = rng.gamma(2, 3, (n_t, n_lead, n_p))
    obs = base + rng.normal(0, 1, base.shape)
    return obs.astype("float32"), (base + rng.normal(0, 1.5, (n_m, *base.shape))).astype("float32")


def _numpy(x, y, threshold):
    """x (member, case), y (case): n_forecasts and n_events per bin, complete cases."""
    ok = np.isfinite(y) & np.isfinite(x).all(axis=0)
    x, y = x[:, ok].astype("float64"), y[ok].astype("float64")
    above, event = (x > threshold).sum(axis=0), y > threshold
    bins = range(x.shape[0] + 1)
    return np.array([(above == k).sum() for k in bins]), np.array([event[above == k].sum() for k in bins])


def test_matches_numpy_with_nans():
    for m in (5, 10):
        o, f = _random(n_m=m)
        o[1, 0, 3:6] = np.nan
        f[2, 0, 1, 7] = np.nan
        out = reliability_diagram(*reversed(_make_ds(o, f)), {"10si": 8.0}, DIMS)["10si"]
        np.testing.assert_allclose(out["probability"], np.arange(m + 1) / m)
        for lead in range(2):
            n_f, n_e = _numpy(f[:, :, lead].reshape(m, -1), o[:, lead].ravel(), 8.0)
            got = out.isel(lead_time=lead)
            np.testing.assert_array_equal(got.sel(curve="n_forecasts"), n_f)
            np.testing.assert_array_equal(got.sel(curve="n_events"), n_e)
            np.testing.assert_allclose(got.sel(curve="forecast_frequency"), n_f / n_f.sum())
            with np.errstate(invalid="ignore"):
                np.testing.assert_allclose(got.sel(curve="reliability"), n_e / np.where(n_f > 0, n_f, np.nan))


def test_counts_add_up_to_n_cases():
    o, f = _random()
    f[0, 0, 0, :5] = np.nan
    obs, fct = _make_ds(o, f)
    out = reliability_diagram(fct, obs, {"10si": 8.0}, DIMS)["10si"]
    np.testing.assert_array_equal(
        out.sel(curve="n_forecasts").sum("probability"), n_cases(fct, obs, DIMS)["10si"])


def test_strict_and_float64_comparison():
    """A value equal to the threshold is not an event; 10.1 in float32 is above 10.1 in float64."""
    for value, threshold, is_event in ((10.0, 10.0, 0), (10.1, 10.1, 1)):
        x = np.full((2, 1, 1, 4), value, dtype="float32")
        out = reliability_diagram(*reversed(_make_ds(x[0], x)), {"10si": threshold}, DIMS)["10si"]
        k = 2 * is_event
        assert float(out.sel(curve="n_forecasts").isel(probability=k).squeeze()) == 4
        assert float(out.sel(curve="n_events").isel(probability=k).squeeze()) == 4 * is_event


def test_empty_bin_is_nan_and_empty_lead_is_nan():
    o, f = _random()
    f[:, :, 1] = np.nan
    f[:, :, 0] = 100.0
    out = reliability_diagram(*reversed(_make_ds(o, f)), {"10si": 8.0}, DIMS)["10si"]
    rel, freq = out.sel(curve="reliability"), out.sel(curve="forecast_frequency")
    assert np.isnan(rel.isel(lead_time=0, probability=0)) and np.isfinite(rel.isel(lead_time=0, probability=5))
    assert float(freq.isel(lead_time=0, probability=5)) == 1.0
    assert np.isnan(freq.isel(lead_time=1)).all()


def test_dask_gives_the_same():
    obs, fct = _make_ds(*_random())
    lazy = reliability_diagram(fct.chunk({"lead_time": 1}), obs.chunk({"lead_time": 1}), {"10si": 8.0}, DIMS)
    xr.testing.assert_identical(lazy.compute(), reliability_diagram(fct, obs, {"10si": 8.0}, DIMS))


def test_unknown_variable_raises():
    obs, fct = _make_ds(*_random())
    try:
        reliability_diagram(fct, obs, {"2t": 280.0}, DIMS)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


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
