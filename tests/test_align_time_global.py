"""Tests for module-level ``align_time`` — multi-dataset global alignment.

The pairwise ``ds.mx.align_time_with`` cases are covered by
``test_align_time.py``. This file exercises the *global* regime triggered
when the reference is an observation and one or more forecasts are present:

  1. R* = intersection of reference_time across forecasts
  2. L* = union of lead_time across forecasts (never pruned)
  3. R* is pruned to ref_times r where every (r + l ∈ L*) lies inside
     [ref.valid_time.min(), ref.valid_time.max()]
  4. Forecasts reindex(reference_time=R*, lead_time=L*); refresh valid_time
  5. Observations reshape onto (R*, L*); trait flips to forecast
"""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.typing import ArrayLike

from mxalign.alignment.time import align_time

T0 = np.datetime64("2020-01-01T00:00", "ns")
H6 = np.timedelta64(6, "h")


def _props(time: str) -> dict:
    """Build the ``mlwp_*`` trait attributes for a deterministic point dataset.

    Parameters
    ----------
    time : str
        Value of the ``mlwp_time_trait`` attribute, e.g. ``"forecast"``.

    Returns
    -------
    dict
        Trait attributes to attach to a dataset's ``attrs``.
    """
    return {
        "mlwp_time_trait": time,
        "mlwp_space_trait": "point",
        "mlwp_uncertainty_trait": "deterministic",
    }


def fcst(reference_times: ArrayLike, lead_times: ArrayLike) -> xr.Dataset:
    """Build a point forecast with ``temp`` counting up from 0 in row-major order.

    Parameters
    ----------
    reference_times : ArrayLike
        Reference times, parseable by ``pd.to_datetime``.
    lead_times : ArrayLike
        Lead times, parseable by ``pd.to_timedelta``.

    Returns
    -------
    xr.Dataset
        Forecast dataset over ``reference_time`` and ``lead_time``.
    """
    rt = pd.to_datetime(np.asarray(reference_times))
    lt = pd.to_timedelta(np.asarray(lead_times))
    values = np.arange(len(rt) * len(lt), dtype=float).reshape(len(rt), len(lt))
    return xr.Dataset(
        {"temp": (["reference_time", "lead_time"], values)},
        coords={"reference_time": rt, "lead_time": lt},
        attrs=_props("forecast"),
    )


def obs(valid_times: ArrayLike, offset: float = 0.0) -> xr.Dataset:
    """Build a point observation with ``temp`` counting up from ``offset``.

    Parameters
    ----------
    valid_times : ArrayLike
        Valid times, parseable by ``pd.to_datetime``.
    offset : float, default 0.0
        Value of the first observation.

    Returns
    -------
    xr.Dataset
        Observation dataset along ``valid_time``.
    """
    vt = pd.to_datetime(np.asarray(valid_times))
    return xr.Dataset(
        {"temp": (["valid_time"], np.arange(len(vt), dtype=float) + offset)},
        coords={"valid_time": vt},
        attrs=_props("observation"),
    )


# ---------------------------------------------------------------------------
# Global regime: 2 forecasts + 1 observation
# ---------------------------------------------------------------------------


class TestGlobalForecastsObservation:
    """Group tests of the global regime with two forecasts and an observation."""

    @pytest.fixture
    def datasets(self) -> dict[str, xr.Dataset]:
        """Provide two partly overlapping forecasts and a reference observation.

        Returns
        -------
        dict[str, xr.Dataset]
            Datasets ``f1``, ``f2`` and the observation ``ref``.
        """
        f1 = fcst(
            ["2020-01-01", "2020-01-02", "2020-01-03"],
            ["0h", "6h", "12h"],
        )
        f2 = fcst(
            ["2020-01-02", "2020-01-03", "2020-01-04"],
            ["6h", "12h", "18h"],
        )
        ref = obs(pd.date_range("2020-01-01", "2020-01-05", freq="3h"), offset=100.0)
        return {"f1": f1, "f2": f2, "ref": ref}

    def test_r_star_is_intersection_of_forecast_ref_times(
        self, datasets: dict[str, xr.Dataset]
    ) -> None:
        """Check that R* is the intersection of the forecasts' reference times.

        Parameters
        ----------
        datasets : dict[str, xr.Dataset]
            Two forecasts and a reference observation.
        """
        out = align_time(datasets, reference="ref")
        np.testing.assert_array_equal(
            out["f1"].reference_time.values.astype("datetime64[D]"),
            np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
        )
        assert out["f2"].sizes["reference_time"] == 2

    def test_l_star_is_union_of_forecast_lead_times(
        self, datasets: dict[str, xr.Dataset]
    ) -> None:
        """Check that L* is the union of the forecasts' lead times.

        Parameters
        ----------
        datasets : dict[str, xr.Dataset]
            Two forecasts and a reference observation.
        """
        out = align_time(datasets, reference="ref")
        expected = np.array([0, 6, 12, 18], dtype="int64")
        np.testing.assert_array_equal(
            out["f1"].lead_time.values.astype("timedelta64[h]").astype("int64"),
            expected,
        )

    def test_all_outputs_share_grid(self, datasets: dict[str, xr.Dataset]) -> None:
        """Check that all aligned datasets end up with the same shape.

        Parameters
        ----------
        datasets : dict[str, xr.Dataset]
            Two forecasts and a reference observation.
        """
        out = align_time(datasets, reference="ref")
        shapes = {k: out[k].temp.shape for k in out}
        assert len(set(shapes.values())) == 1, f"shapes differ: {shapes}"

    def test_reference_observation_becomes_forecast_shaped(
        self, datasets: dict[str, xr.Dataset]
    ) -> None:
        """Check that the reference observation is reshaped to forecast dims.

        Parameters
        ----------
        datasets : dict[str, xr.Dataset]
            Two forecasts and a reference observation.
        """
        out = align_time(datasets, reference="ref")
        assert out["ref"].mx.is_forecast()
        assert out["ref"].temp.dims == ("reference_time", "lead_time")
        assert out["ref"].valid_time.dims == ("reference_time", "lead_time")

    def test_forecast_nan_filled_for_missing_lead_times(
        self, datasets: dict[str, xr.Dataset]
    ) -> None:
        """Check that forecasts are NaN at lead times they do not have.

        Parameters
        ----------
        datasets : dict[str, xr.Dataset]
            Two forecasts and a reference observation.
        """
        out = align_time(datasets, reference="ref")
        # f1 has no 18h lead → its 18h column is all-NaN across R*
        assert np.isnan(out["f1"].temp.sel(lead_time="18h").values).all()
        # f2 has no 0h lead → its 0h column is all-NaN
        assert np.isnan(out["f2"].temp.sel(lead_time="0h").values).all()
        # f1's 0h column has real data
        assert not np.isnan(out["f1"].temp.sel(lead_time="0h").values).any()


# ---------------------------------------------------------------------------
# Overlapping runs: several (reference_time, lead_time) share a valid_time
# ---------------------------------------------------------------------------


class TestOverlappingValidTimes:
    """Group tests of forecast runs whose valid times overlap."""

    def test_observation_broadcast_when_runs_overlap(self) -> None:
        """Check that one observation fills every cell sharing its valid time."""
        # Runs 6h apart with leads up to 12h → each valid_time is hit up to 3 times
        f1 = fcst(
            ["2020-01-01 00:00", "2020-01-01 06:00", "2020-01-01 12:00"],
            ["0h", "6h", "12h"],
        )
        ref = obs(pd.date_range("2020-01-01", "2020-01-02", freq="6h"))

        out = align_time({"f1": f1, "ref": ref}, reference="ref")

        # obs values are 0..4 at 00, 06, 12, 18, 24 UTC; cell (r, l) holds obs at r + l
        np.testing.assert_array_equal(
            out["ref"].temp.values,
            [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
        )


# ---------------------------------------------------------------------------
# R* pruning by valid_time range
# ---------------------------------------------------------------------------


class TestReferenceTimePruning:
    """Group tests of pruning R* to the reference valid-time window."""

    def test_prunes_ref_times_before_reference_window(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Check that reference times before the reference window are pruned.

        Parameters
        ----------
        caplog : pytest.LogCaptureFixture
            Captures the pruning warning.
        """
        f1 = fcst(["2020-01-01", "2020-01-02", "2020-01-03"], ["0h", "6h", "12h"])
        f2 = fcst(["2020-01-01", "2020-01-02", "2020-01-03"], ["0h", "6h", "12h"])
        ref = obs(pd.date_range("2020-01-03", "2020-01-05", freq="3h"))

        with caplog.at_level(logging.WARNING, logger="mxalign.alignment.time"):
            out = align_time({"f1": f1, "f2": f2, "ref": ref}, reference="ref")

        np.testing.assert_array_equal(
            out["f1"].reference_time.values.astype("datetime64[D]"),
            np.array(["2020-01-03"], dtype="datetime64[D]"),
        )
        # The reference observation must end up gap-free after pruning
        assert not np.isnan(out["ref"].temp.values).any()
        assert any("pruned" in r.message for r in caplog.records)

    def test_keeps_long_lead_time_when_reference_covers_it(self) -> None:
        """Check that L* keeps a long lead time that only one forecast has.

        Forecast F2 has a long lead_time; F1 doesn't. L* must keep it.
        """
        f1 = fcst(["2020-01-02", "2020-01-03"], ["0h", "6h", "12h"])
        f2 = fcst(["2020-01-02", "2020-01-03"], ["0h", "6h", "12h", "240h"])
        ref = obs(pd.date_range("2020-01-01", "2020-01-31", freq="3h"))

        out = align_time({"f1": f1, "f2": f2, "ref": ref}, reference="ref")

        leads = out["f1"].lead_time.values.astype("timedelta64[h]").astype("int64")
        assert 240 in leads, "L* must keep F2's long lead_time"
        # F1 doesn't have 240h → that column is NaN, but F1's own short leads survive
        assert np.isnan(out["f1"].temp.sel(lead_time="240h").values).all()
        assert not np.isnan(out["f1"].temp.sel(lead_time="0h").values).any()

    def test_raises_when_no_ref_time_fits_all_lead_times(self) -> None:
        """Check that an empty R* after pruning raises ``ValueError``.

        Long lead_time + short reference → R* empties → ValueError.
        """
        f1 = fcst(["2020-01-02", "2020-01-03"], ["0h"])
        f2 = fcst(["2020-01-02", "2020-01-03"], ["0h", "240h"])
        ref = obs(pd.date_range("2020-01-01", "2020-01-04", freq="3h"))

        with pytest.raises(ValueError, match="no reference_time"):
            align_time({"f1": f1, "f2": f2, "ref": ref}, reference="ref")


# ---------------------------------------------------------------------------
# Multiple observations
# ---------------------------------------------------------------------------


class TestMultipleObservations:
    """Group tests of global alignment with several observations."""

    def test_both_observations_reshape_to_forecast_grid(self) -> None:
        """Check that every observation is reshaped onto the forecast grid."""
        f1 = fcst(["2020-01-02", "2020-01-03"], ["0h", "6h", "12h"])
        f2 = fcst(["2020-01-02", "2020-01-03"], ["6h", "12h", "18h"])
        o1 = obs(pd.date_range("2020-01-01", "2020-01-05", freq="3h"), offset=100.0)
        o2 = obs(
            pd.date_range("2020-01-02 06", "2020-01-03 18", freq="6h"), offset=200.0
        )

        out = align_time({"f1": f1, "f2": f2, "o1": o1, "o2": o2}, reference="o1")

        assert out["o1"].mx.is_forecast()
        assert out["o2"].mx.is_forecast()
        assert out["o1"].temp.shape == out["o2"].temp.shape == out["f1"].temp.shape
        # Reference observation has full coverage; o2 has partial coverage → some NaNs
        assert not np.isnan(out["o1"].temp.values).any()
        assert np.isnan(out["o2"].temp.values).any()


# ---------------------------------------------------------------------------
# Pairwise fallback
# ---------------------------------------------------------------------------


class TestPairwiseFallback:
    """Group tests of falling back to pairwise alignment."""

    def test_forecast_reference_uses_pairwise(self) -> None:
        """Check that a forecast reference triggers pairwise alignment.

        When the reference is a forecast, no global pruning happens —
        each non-ref dataset is aligned pairwise to the reference.
        """
        f1 = fcst(["2020-01-01", "2020-01-02", "2020-01-03"], ["0h", "6h", "12h"])
        f_ref = fcst(["2020-01-02", "2020-01-03"], ["6h", "12h"])

        out = align_time({"f1": f1, "ref": f_ref}, reference="ref")

        # f1 reindexed onto f_ref's axes; ref untouched
        np.testing.assert_array_equal(
            out["f1"].reference_time.values, f_ref.reference_time.values
        )
        np.testing.assert_array_equal(
            out["f1"].lead_time.values, f_ref.lead_time.values
        )
        assert out["ref"] is f_ref

    def test_all_observations_uses_pairwise(self) -> None:
        """Check that observations without forecasts are aligned pairwise.

        Reference is obs and no forecasts → pairwise reindex.
        """
        o1 = obs(pd.date_range("2020-01-01", "2020-01-05", freq="6h"))
        o2 = obs(pd.date_range("2020-01-02", "2020-01-04", freq="6h"))

        out = align_time({"o1": o1, "o2": o2}, reference="o2")

        np.testing.assert_array_equal(out["o1"].valid_time.values, o2.valid_time.values)
        assert out["o1"].mx.is_observation()


# ---------------------------------------------------------------------------
# lead_time option
# ---------------------------------------------------------------------------


class TestLeadTimeOption:
    """Group tests of the ``lead_time`` option of ``align_time``."""

    def test_forecast_reference_forwards_lead_time(self) -> None:
        """Check that ``lead_time`` is forwarded to pairwise forecast alignment."""
        f1 = fcst(["2020-01-02", "2020-01-03"], ["0h", "6h"])
        f_ref = fcst(["2020-01-02", "2020-01-03"], ["6h", "12h"])

        out = align_time({"f1": f1, "ref": f_ref}, reference="ref", lead_time="union")

        leads = out["f1"].lead_time.values.astype("timedelta64[h]").astype("int64")
        np.testing.assert_array_equal(leads, [0, 6, 12])

    def test_forecast_reference_rejects_unknown_lead_time(self) -> None:
        """Check that an invalid forecast-to-forecast ``lead_time`` raises."""
        f1 = fcst(["2020-01-02"], ["0h"])
        f_ref = fcst(["2020-01-02"], ["0h"])

        with pytest.raises(ValueError, match="Unknown lead_time"):
            align_time({"f1": f1, "ref": f_ref}, reference="ref", lead_time="shortest")

    @pytest.mark.parametrize("with_forecast", [True, False])
    def test_observation_reference_rejects_lead_time(self, with_forecast: bool) -> None:
        """Check that ``lead_time`` is rejected when the reference is an observation.

        Parameters
        ----------
        with_forecast : bool
            Whether to include a forecast alongside the reference observation.
        """
        datasets = {"ref": obs(pd.date_range("2020-01-01", "2020-01-05", freq="6h"))}
        if with_forecast:
            datasets["f1"] = fcst(["2020-01-02"], ["0h", "6h"])

        with pytest.raises(ValueError, match="only supported when the reference"):
            align_time(datasets, reference="ref", lead_time="union")


# ---------------------------------------------------------------------------
# Ensembles in the global regime
# ---------------------------------------------------------------------------


class TestEnsemble:
    """Group tests of global alignment with an ensemble among the forecasts."""

    def test_ensemble_keeps_members_on_common_grid(
        self, ds_fcst: xr.Dataset, ds_fcst_ens: xr.Dataset, ds_obs: xr.Dataset
    ) -> None:
        """Check that an ensemble is put on the common grid member by member.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Point forecast fixture, equal to member 0 of the ensemble.
        ds_fcst_ens : xr.Dataset
            Point forecast ensemble fixture.
        ds_obs : xr.Dataset
            Point observation fixture, used as the reference.
        """
        out = align_time(
            {"det": ds_fcst, "ens": ds_fcst_ens, "obs": ds_obs}, reference="obs"
        )

        ens = out["ens"]
        assert ens.mx.is_ensemble()
        assert ens["member"].attrs["standard_name"] == "realization"
        for member in ens["member"].values:
            xr.testing.assert_identical(
                ens["temp"].sel(member=member, drop=True) - 100.0 * member,
                out["det"]["temp"],
            )
        # The observation is reshaped onto the time grid only
        assert set(out["obs"]["temp"].dims) == {"reference_time", "lead_time"}
