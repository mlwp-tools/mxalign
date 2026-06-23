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

from mxalign.align.time import align_time

T0 = np.datetime64("2020-01-01T00:00", "ns")
H6 = np.timedelta64(6, "h")


def _props(time: str) -> dict:
    return {
        "mlwp_time_trait": time,
        "mlwp_space_trait": "point",
        "mlwp_uncertainty_trait": "deterministic",
    }


def fcst(reference_times, lead_times):
    rt = pd.to_datetime(np.asarray(reference_times))
    lt = pd.to_timedelta(np.asarray(lead_times))
    values = np.arange(len(rt) * len(lt), dtype=float).reshape(len(rt), len(lt))
    return xr.Dataset(
        {"temp": (["reference_time", "lead_time"], values)},
        coords={"reference_time": rt, "lead_time": lt},
        attrs=_props("forecast"),
    )


def obs(valid_times, offset=0.0):
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
    @pytest.fixture
    def datasets(self):
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

    def test_r_star_is_intersection_of_forecast_ref_times(self, datasets):
        out = align_time(datasets, reference="ref")
        np.testing.assert_array_equal(
            out["f1"].reference_time.values.astype("datetime64[D]"),
            np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
        )
        assert out["f2"].sizes["reference_time"] == 2

    def test_l_star_is_union_of_forecast_lead_times(self, datasets):
        out = align_time(datasets, reference="ref")
        expected = np.array([0, 6, 12, 18], dtype="int64")
        np.testing.assert_array_equal(
            out["f1"].lead_time.values.astype("timedelta64[h]").astype("int64"),
            expected,
        )

    def test_all_outputs_share_grid(self, datasets):
        out = align_time(datasets, reference="ref")
        shapes = {k: out[k].temp.shape for k in out}
        assert len(set(shapes.values())) == 1, f"shapes differ: {shapes}"

    def test_reference_observation_becomes_forecast_shaped(self, datasets):
        out = align_time(datasets, reference="ref")
        assert out["ref"].mx.is_forecast()
        assert out["ref"].temp.dims == ("reference_time", "lead_time")
        assert out["ref"].valid_time.dims == ("reference_time", "lead_time")

    def test_forecast_nan_filled_for_missing_lead_times(self, datasets):
        out = align_time(datasets, reference="ref")
        # f1 has no 18h lead → its 18h column is all-NaN across R*
        assert np.isnan(out["f1"].temp.sel(lead_time="18h").values).all()
        # f2 has no 0h lead → its 0h column is all-NaN
        assert np.isnan(out["f2"].temp.sel(lead_time="0h").values).all()
        # f1's 0h column has real data
        assert not np.isnan(out["f1"].temp.sel(lead_time="0h").values).any()


# ---------------------------------------------------------------------------
# R* pruning by valid_time range
# ---------------------------------------------------------------------------


class TestReferenceTimePruning:
    def test_prunes_ref_times_before_reference_window(self, caplog):
        f1 = fcst(["2020-01-01", "2020-01-02", "2020-01-03"], ["0h", "6h", "12h"])
        f2 = fcst(["2020-01-01", "2020-01-02", "2020-01-03"], ["0h", "6h", "12h"])
        ref = obs(pd.date_range("2020-01-03", "2020-01-05", freq="3h"))

        with caplog.at_level(logging.WARNING, logger="mxalign.align.time"):
            out = align_time({"f1": f1, "f2": f2, "ref": ref}, reference="ref")

        np.testing.assert_array_equal(
            out["f1"].reference_time.values.astype("datetime64[D]"),
            np.array(["2020-01-03"], dtype="datetime64[D]"),
        )
        # The reference observation must end up gap-free after pruning
        assert not np.isnan(out["ref"].temp.values).any()
        assert any("pruned" in r.message for r in caplog.records)

    def test_keeps_long_lead_time_when_reference_covers_it(self):
        """Forecast F2 has a long lead_time; F1 doesn't. L* must keep it."""
        f1 = fcst(["2020-01-02", "2020-01-03"], ["0h", "6h", "12h"])
        f2 = fcst(["2020-01-02", "2020-01-03"], ["0h", "6h", "12h", "240h"])
        ref = obs(pd.date_range("2020-01-01", "2020-01-31", freq="3h"))

        out = align_time({"f1": f1, "f2": f2, "ref": ref}, reference="ref")

        leads = out["f1"].lead_time.values.astype("timedelta64[h]").astype("int64")
        assert 240 in leads, "L* must keep F2's long lead_time"
        # F1 doesn't have 240h → that column is NaN, but F1's own short leads survive
        assert np.isnan(out["f1"].temp.sel(lead_time="240h").values).all()
        assert not np.isnan(out["f1"].temp.sel(lead_time="0h").values).any()

    def test_raises_when_no_ref_time_fits_all_lead_times(self):
        """Long lead_time + short reference → R* empties → ValueError."""
        f1 = fcst(["2020-01-02", "2020-01-03"], ["0h"])
        f2 = fcst(["2020-01-02", "2020-01-03"], ["0h", "240h"])
        ref = obs(pd.date_range("2020-01-01", "2020-01-04", freq="3h"))

        with pytest.raises(ValueError, match="no reference_time"):
            align_time({"f1": f1, "f2": f2, "ref": ref}, reference="ref")


# ---------------------------------------------------------------------------
# Multiple observations
# ---------------------------------------------------------------------------


class TestMultipleObservations:
    def test_both_observations_reshape_to_forecast_grid(self):
        f1 = fcst(["2020-01-02", "2020-01-03"], ["0h", "6h", "12h"])
        f2 = fcst(["2020-01-02", "2020-01-03"], ["6h", "12h", "18h"])
        o1 = obs(pd.date_range("2020-01-01", "2020-01-05", freq="3h"), offset=100.0)
        o2 = obs(pd.date_range("2020-01-02 06", "2020-01-03 18", freq="6h"), offset=200.0)

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
    def test_forecast_reference_uses_pairwise(self):
        """When the reference is a forecast, no global pruning happens —
        each non-ref dataset is aligned pairwise to the reference."""
        f1 = fcst(["2020-01-01", "2020-01-02", "2020-01-03"], ["0h", "6h", "12h"])
        f_ref = fcst(["2020-01-02", "2020-01-03"], ["6h", "12h"])

        out = align_time({"f1": f1, "ref": f_ref}, reference="ref")

        # f1 reindexed onto f_ref's axes; ref untouched
        np.testing.assert_array_equal(
            out["f1"].reference_time.values, f_ref.reference_time.values
        )
        np.testing.assert_array_equal(out["f1"].lead_time.values, f_ref.lead_time.values)
        assert out["ref"] is f_ref

    def test_all_observations_uses_pairwise(self):
        """Reference is obs and no forecasts → pairwise reindex."""
        o1 = obs(pd.date_range("2020-01-01", "2020-01-05", freq="6h"))
        o2 = obs(pd.date_range("2020-01-02", "2020-01-04", freq="6h"))

        out = align_time({"o1": o1, "o2": o2}, reference="o2")

        np.testing.assert_array_equal(
            out["o1"].valid_time.values, o2.valid_time.values
        )
        assert out["o1"].mx.is_observation()
