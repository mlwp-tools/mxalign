"""Unit tests for mxalign.metrics.unbiased_spread_skill.

Runs standalone (``python tests/test_unbiased_spread_skill.py``), like
test_spread_skill.py.
"""
import sys
import traceback

import numpy as np
import xarray as xr

from mxalign.metrics import spread_skill, unbiased_spread_skill

DIM = ["point_index", "reference_time"]
CURVES = ["spread", "rmse", "unbiased_rmse", "unbiased_spread_skill_ratio"]


def _make_ds(obs, fct):
    """obs (time, lead, point), fct (member, time, lead, point)."""
    dims = ("reference_time", "lead_time", "point_index")
    return xr.Dataset({"2t": (dims, obs)}), xr.Dataset({"2t": (("member", *dims), fct)})


def _random(n_m=5, n_t=4, n_lead=2, n_p=50, seed=5):
    rng = np.random.default_rng(seed)
    return rng.normal(280, 5, (n_t, n_lead, n_p)), rng.normal(280, 3, (n_m, n_t, n_lead, n_p))


def _get(out, curve):
    return out["2t"].sel(curve=curve).values


def test_matches_numpy_with_nans():
    for m in (5, 10):
        o, f = _random(n_m=m)
        o[1, 0, 3:6] = np.nan
        f[2, 0, 1, 7] = np.nan
        out = unbiased_spread_skill(*_make_ds(o, f), DIM)
        assert list(out["curve"].values) == CURVES
        for lead in range(2):
            x, y = f[:, :, lead].reshape(m, -1), o[:, lead].ravel()
            ok = np.isfinite(y) & np.isfinite(x).all(axis=0)
            x, y = x[:, ok], y[ok]
            var, mse = x.var(axis=0, ddof=1).mean(), ((x.mean(axis=0) - y) ** 2).mean()
            want = [var**0.5, mse**0.5, (mse - var / m) ** 0.5, (var / (mse - var / m)) ** 0.5]
            got = [_get(out, c)[lead] for c in CURVES]
            np.testing.assert_allclose(got, want, rtol=1e-12)


def test_spread_and_rmse_are_those_of_spread_skill():
    obs, fct = _make_ds(*_random())
    raw, out = spread_skill(obs, fct, DIM, complete_cases=True), unbiased_spread_skill(obs, fct, DIM)
    for curve in ("spread", "rmse"):
        np.testing.assert_array_equal(_get(out, curve), _get(raw, curve))


def test_calibrated_ensemble_gives_one_for_any_size():
    """Truth and members drawn from the same distribution, whose centre varies."""
    rng = np.random.default_rng(0)
    for m in (2, 5, 10):
        centre = rng.normal(0, 10, (20, 1, 5000))
        o, f = rng.normal(centre, 2.0), rng.normal(centre, 2.0, (m, *centre.shape))
        out = unbiased_spread_skill(*_make_ds(o, f), DIM)
        assert abs(_get(out, "unbiased_spread_skill_ratio")[0] - 1) < 0.02, m
        raw_ratio = _get(out, "spread")[0] / _get(out, "rmse")[0]
        assert abs(raw_ratio - np.sqrt(m / (m + 1))) < 0.02, m  # what the correction removes


def test_rmse_too_small_is_nan():
    o, f = _random(n_lead=1)
    o = f.mean(axis=0)  # the ensemble mean is the truth: rmse ~ 0 < spread^2 / M
    out = unbiased_spread_skill(*_make_ds(o, f), DIM)
    assert np.isnan(_get(out, "unbiased_rmse")[0])
    assert np.isnan(_get(out, "unbiased_spread_skill_ratio")[0])
    assert np.isfinite(_get(out, "spread")[0])


def test_one_member_raises():
    try:
        unbiased_spread_skill(*_make_ds(*_random(n_m=1)), DIM)
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
