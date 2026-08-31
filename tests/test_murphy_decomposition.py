"""Unit tests for mxalign.metrics.murphy_decomposition.

Runs standalone (``python tests/test_murphy_decomposition.py``) because the
mxalign venv has no pytest installed; the ``test_*`` function layout means
``pytest tests/`` will also collect it unchanged once pytest is available.

The synthetic cases are built so every expected value is exact in floating
point (integer-valued offsets, halving factors), so the assertions use tight
tolerances rather than fuzzy ones.
"""
import sys
import traceback

import numpy as np
import xarray as xr

from mxalign.metrics import murphy_decomposition, murphy_decomposition_domain_mean

RTOL = 1e-12
ATOL = 1e-12


def _make_ds(obs_vals, fct_vals, var="2t"):
    """Build aligned (observations, forecasts) datasets.

    Inputs are (n_time, n_station) arrays.  A length-1 ``lead_time`` dimension
    is added so the tests exercise the same dimensionality as the real
    pipeline (reference_time, lead_time, point_index).
    """
    obs = np.asarray(obs_vals, dtype="float64")[:, None, :]
    fct = np.asarray(fct_vals, dtype="float64")[:, None, :]
    n_t, _, n_s = obs.shape
    dims = ("reference_time", "lead_time", "point_index")
    coords = {
        "reference_time": np.arange(n_t),
        "lead_time": np.array([0]),
        "code": ("point_index", np.arange(100, 100 + n_s)),
        "latitude": ("point_index", np.linspace(50.0, 60.0, n_s)),
        "longitude": ("point_index", np.linspace(0.0, 10.0, n_s)),
        "altitude": ("point_index", np.linspace(0.0, 500.0, n_s)),
    }
    return (
        xr.Dataset({var: (dims, obs)}, coords=coords),
        xr.Dataset({var: (dims, fct)}, coords=coords),
    )


def _stat(res, name, var="2t"):
    """Per-station values of one stat, squeezed to a 1-D station vector."""
    return res[var].sel(stat=name).squeeze("lead_time", drop=True).values


def test_perfect_forecast_is_all_zero():
    """f == o exactly -> every decomposition term is 0, r == 1, ratio == 1."""
    rng = np.random.default_rng(0)
    obs = rng.normal(280.0, 5.0, size=(12, 3))
    ds_o, ds_f = _make_ds(obs, obs.copy())

    res = murphy_decomposition(ds_o, ds_f, dim="reference_time")

    for term in ("bias", "spread", "corr", "mse"):
        np.testing.assert_allclose(_stat(res, term), 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(_stat(res, "r"), 1.0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(_stat(res, "ratio"), 1.0, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(_stat(res, "N"), np.full(3, 12.0))


def test_constant_offset_is_pure_bias():
    """f == o + c -> bias == c^2, spread == corr == 0, MSE == c^2."""
    rng = np.random.default_rng(1)
    obs = rng.normal(280.0, 5.0, size=(12, 3))
    offsets = np.array([2.0, -3.0, 0.5])
    ds_o, ds_f = _make_ds(obs, obs + offsets[None, :])

    res = murphy_decomposition(ds_o, ds_f, dim="reference_time")

    np.testing.assert_allclose(_stat(res, "bias"), offsets ** 2, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(_stat(res, "spread"), 0.0, rtol=RTOL, atol=1e-10)
    np.testing.assert_allclose(_stat(res, "corr"), 0.0, rtol=RTOL, atol=1e-10)
    np.testing.assert_allclose(_stat(res, "mse"), offsets ** 2, rtol=1e-10, atol=1e-10)
    # A pure offset does not disturb variance or phase.
    np.testing.assert_allclose(_stat(res, "r"), 1.0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(_stat(res, "ratio"), 1.0, rtol=1e-10, atol=1e-10)


def test_damped_variance_is_the_smoothing_signature():
    """The over-smoothing case: f = ō + 0.5*(o - ō).

    Shrinking *toward the station mean* (rather than a plain ``f = 0.5*o``)
    keeps f̄ == ō, so bias stays exactly 0 and the damped variance is isolated
    in ``spread`` alone.  A plain 0.5*o would also halve the mean and leak a
    large spurious bias term, which would not test what it claims to.
    """
    rng = np.random.default_rng(2)
    obs = rng.normal(280.0, 5.0, size=(20, 3))
    o_mean = obs.mean(axis=0, keepdims=True)
    fct = o_mean + 0.5 * (obs - o_mean)
    ds_o, ds_f = _make_ds(obs, fct)

    res = murphy_decomposition(ds_o, ds_f, dim="reference_time")

    sigma_o = obs.std(axis=0)  # ddof=0, matching the metric
    # Perfectly correlated (monotone linear map) but variance halved:
    np.testing.assert_allclose(_stat(res, "r"), 1.0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(_stat(res, "ratio"), 0.5, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(_stat(res, "bias"), 0.0, rtol=RTOL, atol=1e-10)
    np.testing.assert_allclose(_stat(res, "corr"), 0.0, rtol=RTOL, atol=1e-10)
    # spread = (0.5*sigma_o - sigma_o)^2 = 0.25 * sigma_o^2  >> 0
    np.testing.assert_allclose(_stat(res, "spread"), 0.25 * sigma_o ** 2,
                               rtol=1e-10, atol=1e-10)
    assert np.all(_stat(res, "spread") > 0.0)


def test_identity_holds_including_degenerate_constant_station():
    """bias+spread+corr == MSE everywhere, even where sigma == 0 and r is NaN."""
    rng = np.random.default_rng(3)
    obs = rng.normal(280.0, 5.0, size=(10, 3))
    fct = obs + rng.normal(0.0, 2.0, size=(10, 3))
    # Station 2: both series constant -> sigma_f == sigma_o == 0, r undefined.
    obs[:, 2] = 275.0
    fct[:, 2] = 277.0
    ds_o, ds_f = _make_ds(obs, fct)

    res = murphy_decomposition(ds_o, ds_f, dim="reference_time")

    recomposed = _stat(res, "bias") + _stat(res, "spread") + _stat(res, "corr")
    np.testing.assert_allclose(recomposed, _stat(res, "mse"), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(_stat(res, "identity_residual"), 0.0, rtol=RTOL, atol=1e-9)
    # Degenerate station: r is NaN but the decomposition is still exact.
    assert np.isnan(_stat(res, "r")[2])
    np.testing.assert_allclose(_stat(res, "bias")[2], 4.0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(_stat(res, "mse")[2], 4.0, rtol=1e-10, atol=1e-10)


def test_joint_nan_mask_and_pair_counts():
    """N counts jointly-valid pairs only; NaNs on either side are excluded."""
    obs = np.arange(24, dtype="float64").reshape(8, 3)
    fct = obs + 1.0
    obs[0, 0] = np.nan          # obs-side gap
    fct[1, 0] = np.nan          # forecast-side gap (different time)
    obs[:6, 1] = np.nan         # station 1 mostly missing
    ds_o, ds_f = _make_ds(obs, fct)

    res = murphy_decomposition(ds_o, ds_f, dim="reference_time")

    np.testing.assert_array_equal(_stat(res, "N"), np.array([6.0, 2.0, 8.0]))
    # Offset is +1 everywhere it is observed, so bias == 1 regardless of gaps.
    np.testing.assert_allclose(_stat(res, "bias"), 1.0, rtol=1e-10, atol=1e-10)


def test_unweighted_per_station_differs_from_naive_pair_pooling():
    """The core requirement: imbalanced N must not tilt the aggregate.

    Station 0 has 10 pairs with a +1 offset; station 1 has 2 pairs with a +5
    offset.  Unweighted per-station averaging gives (1 + 25)/2 = 13.  Pooling
    all 12 pairs first collapses toward the densely-observed station and gives
    a completely different (and for this purpose wrong) answer.
    """
    rng = np.random.default_rng(4)
    obs = rng.normal(280.0, 5.0, size=(10, 2))
    fct = obs.copy()
    fct[:, 0] += 1.0
    fct[:, 1] += 5.0
    obs[2:, 1] = np.nan          # station 1 keeps only 2 valid pairs
    ds_o, ds_f = _make_ds(obs, fct)

    res = murphy_decomposition(ds_o, ds_f, dim="reference_time")

    np.testing.assert_array_equal(_stat(res, "N"), np.array([10.0, 2.0]))
    np.testing.assert_allclose(_stat(res, "bias"), np.array([1.0, 25.0]),
                               rtol=1e-10, atol=1e-10)

    # (1) unweighted per-station mean -- what aggregate_murphy_results.py does
    unweighted = np.nanmean(_stat(res, "bias"))
    np.testing.assert_allclose(unweighted, 13.0, rtol=1e-10, atol=1e-10)

    # (2) naive per-pair pooling -- explicitly rejected by the methodology
    diff = (fct - obs)[np.isfinite(obs) & np.isfinite(fct)]
    pooled = diff.mean() ** 2
    np.testing.assert_allclose(pooled, (20.0 / 12.0) ** 2, rtol=1e-10, atol=1e-10)

    # They must genuinely disagree, otherwise the test proves nothing.
    assert abs(unweighted - pooled) > 10.0


def test_station_metadata_is_preserved():
    """code/lat/lon/altitude survive for later stratification by terrain."""
    rng = np.random.default_rng(5)
    obs = rng.normal(280.0, 5.0, size=(6, 4))
    ds_o, ds_f = _make_ds(obs, obs + 1.0)

    res = murphy_decomposition(ds_o, ds_f, dim="reference_time")

    for coord in ("code", "latitude", "longitude", "altitude"):
        assert coord in res.coords, f"{coord} lost from result"
        assert res[coord].dims == ("point_index",)
    np.testing.assert_array_equal(res["code"].values, np.arange(100, 104))


def test_rejects_ensemble_input():
    """An ensemble forecast is an error, not silently averaged."""
    rng = np.random.default_rng(6)
    obs = rng.normal(280.0, 5.0, size=(6, 2))
    ds_o, ds_f = _make_ds(obs, obs.copy())
    ds_f = ds_f.expand_dims(member=[0, 1])

    try:
        murphy_decomposition(ds_o, ds_f, dim="reference_time")
    except ValueError as exc:
        assert "member" in str(exc)
        return
    raise AssertionError("expected ValueError for ensemble forecast input")


def test_rejects_spatial_dim_in_reduce_dims():
    """Reducing over point_index would defeat the per-station design."""
    rng = np.random.default_rng(7)
    obs = rng.normal(280.0, 5.0, size=(6, 2))
    ds_o, ds_f = _make_ds(obs, obs.copy())

    try:
        murphy_decomposition(ds_o, ds_f, dim=["reference_time", "point_index"])
    except ValueError as exc:
        assert "point_index" in str(exc)
        return
    raise AssertionError("expected ValueError for spatial dim in reduce dims")


def test_ddof0_is_required_for_the_identity():
    """Document why ddof=0: sample moments would break the identity.

    Recomputes the decomposition by hand with ddof=1 and shows it misses MSE
    by the expected N/(N-1) factor -- the reason the metric uses population
    moments and offers no ddof switch.
    """
    rng = np.random.default_rng(8)
    n = 20
    obs = rng.normal(280.0, 5.0, size=(n, 1))
    fct = obs + rng.normal(0.0, 2.0, size=(n, 1))
    ds_o, ds_f = _make_ds(obs, fct)

    res = murphy_decomposition(ds_o, ds_f, dim="reference_time")
    np.testing.assert_allclose(_stat(res, "identity_residual"), 0.0, rtol=RTOL, atol=1e-9)

    o, f = obs[:, 0], fct[:, 0]
    s_o1, s_f1 = o.std(ddof=1), f.std(ddof=1)
    cov1 = np.cov(f, o, ddof=1)[0, 1]
    ddof1_sum = (f.mean() - o.mean()) ** 2 + (s_f1 - s_o1) ** 2 + 2 * (s_f1 * s_o1 - cov1)
    mse = float(_stat(res, "mse")[0])

    # ddof=1 inflates the variance terms by exactly n/(n-1).
    centred = mse - (f.mean() - o.mean()) ** 2
    np.testing.assert_allclose(ddof1_sum - (f.mean() - o.mean()) ** 2,
                               centred * n / (n - 1), rtol=1e-10, atol=1e-10)
    assert abs(ddof1_sum - mse) > 1e-3, "ddof=1 should visibly break the identity"


def _make_grid_ds(obs_vals, fct_vals, var="2t"):
    """Build aligned (observations, forecasts) datasets with a grid_index dim.

    Mirrors ``_make_ds`` but uses ``grid_index`` (the dimension name the
    HA-analysis-reference alignment actually produces) instead of
    ``point_index``, and skips the SYNOP station-metadata coords, which have
    no equivalent on a full model grid.
    """
    obs = np.asarray(obs_vals, dtype="float64")[:, None, :]
    fct = np.asarray(fct_vals, dtype="float64")[:, None, :]
    n_t, _, n_g = obs.shape
    dims = ("reference_time", "lead_time", "grid_index")
    coords = {
        "reference_time": np.arange(n_t),
        "lead_time": np.array([0]),
    }
    return (
        xr.Dataset({var: (dims, obs)}, coords=coords),
        xr.Dataset({var: (dims, fct)}, coords=coords),
    )


def _stat_grid(res, name, var="2t"):
    """Value of one stat with lead_time squeezed away (no spatial dim left)."""
    return res[var].sel(stat=name).squeeze("lead_time", drop=True).values


def test_domain_mean_identity_holds_on_grid_pooling():
    """bias+spread+corr == MSE when pooling BOTH reference_time and grid_index.

    Uses the function's default ``dim`` (not passed explicitly) so this also
    exercises that the default really is ``("reference_time", "grid_index")``
    -- the shape the *_murphy_vs_ha_analysis.sh scripts rely on.
    """
    rng = np.random.default_rng(9)
    obs = rng.normal(280.0, 5.0, size=(15, 50))  # 15 reference_times x 50 cells
    fct = obs + rng.normal(0.0, 2.0, size=(15, 50))
    ds_o, ds_f = _make_grid_ds(obs, fct)

    res = murphy_decomposition_domain_mean(ds_o, ds_f)

    recomposed = _stat_grid(res, "bias") + _stat_grid(res, "spread") + _stat_grid(res, "corr")
    np.testing.assert_allclose(recomposed, _stat_grid(res, "mse"), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(_stat_grid(res, "identity_residual"), 0.0, rtol=RTOL, atol=1e-9)
    # Pooled over both dims jointly -> N == n_reference_time * n_grid_index
    np.testing.assert_allclose(_stat_grid(res, "N"), 15.0 * 50.0)
    # No spatial dimension survives pooling (unlike murphy_decomposition).
    assert "grid_index" not in res.dims
    assert "point_index" not in res.dims


def test_domain_mean_differs_from_unweighted_percell_mean_on_imbalanced_case():
    """Confirms this is real pooling, not a relabeled per-cell unweighted mean.

    This is the whole point of Bastien's domain-mean design choice: for the
    equally-weighted HA-analysis grid, pooling every (reference_time,
    grid_index) pair together is legitimate (see the function's docstring),
    unlike for uneven SYNOP stations -- but the two routes must actually give
    different numbers on an imbalanced case, otherwise this metric would just
    be silently doing what ``murphy_decomposition`` already does under a new
    name. Same imbalanced construction as
    ``test_unweighted_per_station_differs_from_naive_pair_pooling`` above,
    just relabeled onto a grid_index dimension.
    """
    rng = np.random.default_rng(10)
    obs = rng.normal(280.0, 5.0, size=(10, 2))
    fct = obs.copy()
    fct[:, 0] += 1.0
    fct[:, 1] += 5.0
    obs[2:, 1] = np.nan  # cell 1 keeps only 2 valid pairs
    ds_o, ds_f = _make_grid_ds(obs, fct)

    # Route 1: per-cell decomposition (dim=reference_time only, grid_index
    # survives), then an unweighted mean across cells -- same shape as what
    # murphy_decomposition + aggregate_murphy_results.py would produce if
    # naively pointed at a grid.
    res_percell = murphy_decomposition(ds_o, ds_f, dim="reference_time")
    unweighted_mean_bias = np.nanmean(_stat_grid(res_percell, "bias"))
    np.testing.assert_allclose(unweighted_mean_bias, 13.0, rtol=1e-10, atol=1e-10)

    # Route 2: this function's domain-mean pooling.
    res_pooled = murphy_decomposition_domain_mean(ds_o, ds_f, dim=("reference_time", "grid_index"))
    pooled_bias = float(_stat_grid(res_pooled, "bias"))

    diff = (fct - obs)[np.isfinite(obs) & np.isfinite(fct)]
    expected_pooled = diff.mean() ** 2
    np.testing.assert_allclose(pooled_bias, expected_pooled, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(pooled_bias, (20.0 / 12.0) ** 2, rtol=1e-10, atol=1e-10)

    # The two routes must genuinely disagree, otherwise this test (and the
    # whole rationale for a separate domain-mean function) proves nothing.
    assert abs(pooled_bias - unweighted_mean_bias) > 10.0


def test_domain_mean_rejects_ensemble_input():
    """Same deterministic-only restriction as murphy_decomposition."""
    rng = np.random.default_rng(11)
    obs = rng.normal(280.0, 5.0, size=(6, 4))
    ds_o, ds_f = _make_grid_ds(obs, obs.copy())
    ds_f = ds_f.expand_dims(member=[0, 1])

    try:
        murphy_decomposition_domain_mean(ds_o, ds_f, dim=("reference_time", "grid_index"))
    except ValueError as exc:
        assert "member" in str(exc)
        return
    raise AssertionError("expected ValueError for ensemble forecast input")


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
