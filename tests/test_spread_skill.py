"""Unit tests for mxalign.metrics.spread_skill.

Runs standalone (``python tests/test_spread_skill.py``) because the mxalign
venv has no pytest installed; the ``test_*`` function layout means
``pytest tests/`` will also collect it unchanged once pytest is available.

Covers the 2026-08 rework that added ``ddof`` and ``spread_method``:
  - ``spread_method="mean_std", ddof=0`` reproduces the historical formula
  - the default ``spread_method="rms", ddof=1`` is the RMSE-consistent,
    ensemble-size-unbiased quantity
  - the finite-M spread/rmse relationship for a calibrated ensemble
  - rmse is untouched by the rework
"""
import sys
import traceback

import numpy as np
import xarray as xr

from mxalign.metrics import spread_skill


def _make_ds(obs_vals, fct_vals, var="2t"):
    """Build aligned (observations, forecasts) datasets.

    obs_vals : (n_time, n_point)
    fct_vals : (n_member, n_time, n_point)
    A length-1 lead_time dim is inserted so the shape matches the real
    pipeline (reference_time, lead_time, point_index[, member]).
    """
    obs = np.asarray(obs_vals, dtype="float64")[:, None, :]
    fct = np.asarray(fct_vals, dtype="float64")[:, :, None, :]
    n_m, n_t, _, n_p = fct.shape
    return (
        xr.Dataset(
            {var: (("reference_time", "lead_time", "point_index"), obs)},
            coords={"reference_time": np.arange(n_t), "lead_time": [0],
                    "point_index": np.arange(n_p)},
        ),
        xr.Dataset(
            {var: (("member", "reference_time", "lead_time", "point_index"), fct)},
            coords={"member": np.arange(n_m), "reference_time": np.arange(n_t),
                    "lead_time": [0], "point_index": np.arange(n_p)},
        ),
    )


def _curve(res, name, var="2t"):
    return res[var].sel(curve=name).squeeze("lead_time", drop=True).values


DIM = ["point_index", "reference_time"]


def test_legacy_combo_reproduces_historical_formula():
    """spread_method='mean_std', ddof=0 == old  fct.std(ddof=0).mean(dim)."""
    rng = np.random.default_rng(0)
    obs = rng.normal(280, 5, size=(4, 60))
    fct = rng.normal(280, 5, size=(9, 4, 60))
    ds_o, ds_f = _make_ds(obs, fct)

    res = spread_skill(ds_o, ds_f, dim=DIM, spread_method="mean_std", ddof=0)

    fct_da = ds_f["2t"]
    want = float(fct_da.std("member", ddof=0).mean(DIM).squeeze())
    assert np.isclose(_curve(res, "spread"), want, rtol=1e-12, atol=1e-12), \
        (_curve(res, "spread"), want)


def test_rms_ge_mean_std_jensen():
    """sqrt(mean(var)) >= mean(std), with equality only if spread is uniform."""
    rng = np.random.default_rng(1)
    # heterogeneous spread: point p has member noise ~ N(0, (1+p)^2)
    n_p = 40
    scale = (1.0 + np.arange(n_p))[None, None, :]
    fct = 280 + rng.normal(size=(12, 5, n_p)) * scale
    obs = 280 + rng.normal(size=(5, n_p))
    ds_o, ds_f = _make_ds(obs, fct)

    rms = _curve(spread_skill(ds_o, ds_f, dim=DIM, spread_method="rms", ddof=1), "spread")
    ms = _curve(spread_skill(ds_o, ds_f, dim=DIM, spread_method="mean_std", ddof=1), "spread")
    assert rms > ms, (rms, ms)

    # uniform spread -> the two agree
    fct_u = 280 + rng.normal(size=(12, 5, n_p)) * 3.0
    ds_o2, ds_f2 = _make_ds(obs, fct_u)
    rms_u = _curve(spread_skill(ds_o2, ds_f2, dim=DIM, spread_method="rms", ddof=1), "spread")
    ms_u = _curve(spread_skill(ds_o2, ds_f2, dim=DIM, spread_method="mean_std", ddof=1), "spread")
    assert np.isclose(rms_u, ms_u, rtol=0.05), (rms_u, ms_u)


def test_ddof_rescales_rms_spread_by_sqrt_M_over_Mm1():
    """For spread_method='rms', spread(ddof=1)/spread(ddof=0) == sqrt(M/(M-1)) exactly."""
    rng = np.random.default_rng(2)
    for M in (3, 5, 31):
        fct = 280 + rng.normal(size=(M, 6, 50)) * (1 + rng.random((1, 1, 50)))
        obs = 280 + rng.normal(size=(6, 50))
        ds_o, ds_f = _make_ds(obs, fct)
        s1 = _curve(spread_skill(ds_o, ds_f, dim=DIM, spread_method="rms", ddof=1), "spread")
        s0 = _curve(spread_skill(ds_o, ds_f, dim=DIM, spread_method="rms", ddof=0), "spread")
        assert np.isclose(s1 / s0, np.sqrt(M / (M - 1)), rtol=1e-10), (M, s1 / s0)


def test_rmse_curve_matches_plain_ensemble_mean_rmse_and_is_unchanged_by_options():
    rng = np.random.default_rng(3)
    obs = rng.normal(280, 5, size=(8, 70))
    fct = rng.normal(280, 5, size=(11, 8, 70))
    ds_o, ds_f = _make_ds(obs, fct)

    want = float(np.sqrt(
        ((ds_f["2t"].mean("member") - ds_o["2t"]) ** 2).mean(DIM)
    ).squeeze())
    for kw in (dict(), dict(ddof=0), dict(spread_method="mean_std"),
               dict(spread_method="mean_std", ddof=0)):
        got = _curve(spread_skill(ds_o, ds_f, dim=DIM, **kw), "rmse")
        assert np.isclose(got, want, rtol=1e-12, atol=1e-12), (kw, got, want)


def test_calibrated_ensemble_matches_finite_M_relationship():
    """Draw truth and members i.i.d. N(0, s^2); the default (rms, ddof=1) spread
    should satisfy  rmse ~ spread * sqrt((M+1)/M)  to sampling tolerance,
    and the legacy (mean_std, ddof=0) combo should visibly UNDER-shoot it."""
    rng = np.random.default_rng(4)
    M, n_t, n_p, s = 5, 400, 400, 2.0
    truth = rng.normal(0, s, size=(n_t, n_p))
    members = rng.normal(0, s, size=(M, n_t, n_p))  # exchangeable with truth
    ds_o, ds_f = _make_ds(truth, members)

    good = spread_skill(ds_o, ds_f, dim=DIM, spread_method="rms", ddof=1)
    sp, rm = float(_curve(good, "spread")), float(_curve(good, "rmse"))
    target = rm * np.sqrt(M / (M + 1))          # what a calibrated M-ens should give
    assert abs(sp - target) / target < 0.03, (sp, target)

    legacy = spread_skill(ds_o, ds_f, dim=DIM, spread_method="mean_std", ddof=0)
    sp_leg = float(_curve(legacy, "spread"))
    # ddof=0 (x sqrt((M-1)/M)) plus mean-of-std Jensen: legacy must read well low
    assert sp_leg < 0.92 * sp, (sp_leg, sp)


def test_nan_points_are_skipped_not_fatal():
    rng = np.random.default_rng(5)
    obs = rng.normal(280, 5, size=(6, 30))
    fct = rng.normal(280, 5, size=(7, 6, 30))
    obs[:, ::4] = np.nan            # some stations never report
    fct[:, :, 1::7] = np.nan
    ds_o, ds_f = _make_ds(obs, fct)
    res = spread_skill(ds_o, ds_f, dim=DIM)   # defaults
    assert np.isfinite(_curve(res, "spread")) and np.isfinite(_curve(res, "rmse"))


def test_valid_range_masks_before_reducing():
    rng = np.random.default_rng(6)
    obs = rng.normal(280, 3, size=(5, 25))
    fct = rng.normal(280, 3, size=(6, 5, 25))
    fct[0, 0, 0] = 1e6             # one absurd member value
    ds_o, ds_f = _make_ds(obs, fct)
    with_vr = spread_skill(ds_o, ds_f, dim=DIM, valid_range={"2t": [150, 360]})
    assert np.isfinite(_curve(with_vr, "spread"))
    # without the range guard the rogue value dominates the spread
    without = spread_skill(ds_o, ds_f, dim=DIM)
    assert _curve(without, "spread") > 100 * _curve(with_vr, "spread")


def test_bad_spread_method_raises():
    ds_o, ds_f = _make_ds(np.zeros((2, 3)), np.zeros((4, 2, 3)))
    try:
        spread_skill(ds_o, ds_f, dim=DIM, spread_method="stdev")
    except ValueError as exc:
        assert "spread_method" in str(exc)
        return
    raise AssertionError("expected ValueError for unknown spread_method")


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
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
