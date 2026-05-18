"""Tests for ds.mx.align_time_with() covering all four alignment cases.

Fixtures (defined in conftest.py):

  ds_fcst  — 3 reference_times × 4 lead_times, values[i,j] = float(i*10 + j)
              ref:  T0,  T0+6h, T0+12h
              lead: 0h,  6h,   12h,   18h

  ds_obs   — 6 valid_times from T0-6h to T0+24h (step 6h)
              values: 0, 10, 20, 30, 40, 50  (each 10 apart for easy reading)

Valid-time coverage from ds_fcst:
  T0     → only (ref=T0,    lead=0h)  = 0
  T0+6h  → (T0,6h)=1 or (T0+6h,0h)=10
  T0+12h → (T0,12h)=2 or (T0+6h,6h)=11 or (T0+12h,0h)=20
  T0+18h → (T0,18h)=3 or (T0+6h,12h)=12 or (T0+12h,6h)=21
  T0+24h → (T0+6h,18h)=13 or (T0+12h,12h)=22
  T0+30h → only (T0+12h,18h)=23  [not in obs]
"""

import numpy as np
import pytest
import xarray as xr

T0 = np.datetime64("2020-01-01T00:00", "ns")
H6 = np.timedelta64(6, "h")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _props(time: str) -> dict:
    return {
        "mlwp_time_trait": time,
        "mlwp_space_trait": "point",
        "mlwp_uncertainty_trait": "deterministic",
    }


def obs(valid_times, values):
    return xr.Dataset(
        {"temp": ("valid_time", np.asarray(values, dtype=float))},
        coords={"valid_time": np.asarray(valid_times)},
        attrs=_props("observation"),
    )


def fcst(reference_times, lead_times, values):
    return xr.Dataset(
        {"temp": (["reference_time", "lead_time"], np.asarray(values, dtype=float))},
        coords={
            "reference_time": np.asarray(reference_times),
            "lead_time": np.asarray(lead_times),
        },
        attrs=_props("forecast"),
    )


# ---------------------------------------------------------------------------
# Case 1: Forecast → Observation
# ---------------------------------------------------------------------------


class TestForecastToObservation:
    def test_shortest_lead_time(self, ds_fcst, ds_obs):
        result = ds_fcst.mx.align_time_with(ds_obs, lead_time="shortest")

        assert result.mx.is_observation()
        assert list(result.valid_time.values) == list(ds_obs.valid_time.values)

        # T0-6h has no forecast coverage → NaN
        assert np.isnan(result["temp"].sel(valid_time=T0 - H6).item())

        # For each covered time, shortest lead_time wins
        assert result["temp"].sel(valid_time=T0).item() == 0.0  # only (T0, 0h)
        assert (
            result["temp"].sel(valid_time=T0 + H6).item() == 10.0
        )  # (T0+6h, 0h) beats (T0, 6h)
        assert (
            result["temp"].sel(valid_time=T0 + 2 * H6).item() == 20.0
        )  # (T0+12h, 0h) is shortest
        assert (
            result["temp"].sel(valid_time=T0 + 3 * H6).item() == 21.0
        )  # (T0+12h, 6h) is shortest
        assert (
            result["temp"].sel(valid_time=T0 + 4 * H6).item() == 22.0
        )  # (T0+12h, 12h) is shortest

    def test_longest_lead_time(self, ds_fcst, ds_obs):
        result = ds_fcst.mx.align_time_with(ds_obs, lead_time="longest")

        assert result.mx.is_observation()
        assert np.isnan(result["temp"].sel(valid_time=T0 - H6).item())

        assert result["temp"].sel(valid_time=T0).item() == 0.0  # only one entry
        assert (
            result["temp"].sel(valid_time=T0 + H6).item() == 1.0
        )  # (T0, 6h) beats (T0+6h, 0h)
        assert (
            result["temp"].sel(valid_time=T0 + 2 * H6).item() == 2.0
        )  # (T0, 12h) is longest
        assert (
            result["temp"].sel(valid_time=T0 + 3 * H6).item() == 3.0
        )  # (T0, 18h) is longest
        assert (
            result["temp"].sel(valid_time=T0 + 4 * H6).item() == 13.0
        )  # (T0+6h, 18h) beats (T0+12h, 12h)

    def test_specific_lead_time(self, ds_fcst, ds_obs):
        lt = np.timedelta64(6, "h")
        result = ds_fcst.mx.align_time_with(ds_obs, lead_time=lt)

        assert result.mx.is_observation()
        # Only T0+6h, T0+12h, T0+18h are produced by lead_time=6h
        assert np.isnan(result["temp"].sel(valid_time=T0 - H6).item())
        assert np.isnan(result["temp"].sel(valid_time=T0).item())
        assert result["temp"].sel(valid_time=T0 + H6).item() == 1.0  # (T0, 6h)
        assert result["temp"].sel(valid_time=T0 + 2 * H6).item() == 11.0  # (T0+6h, 6h)
        assert result["temp"].sel(valid_time=T0 + 3 * H6).item() == 21.0  # (T0+12h, 6h)
        assert np.isnan(result["temp"].sel(valid_time=T0 + 4 * H6).item())

    def test_nan_filled_for_times_not_in_forecast(self, ds_fcst, ds_obs):
        result = ds_fcst.mx.align_time_with(ds_obs, lead_time="shortest")
        # T0-6h is in obs but never produced by any (ref_time, lead_time) pair
        assert np.isnan(result["temp"].sel(valid_time=T0 - H6).item())

    def test_result_has_observation_property(self, ds_fcst, ds_obs):
        result = ds_fcst.mx.align_time_with(ds_obs, lead_time="shortest")
        assert result.mx.is_observation()
        assert not result.mx.is_forecast()


# ---------------------------------------------------------------------------
# Case 2: Observation → Forecast
# ---------------------------------------------------------------------------


class TestObservationToForecast:
    def test_values_placed_at_correct_positions(self, ds_obs, ds_fcst):
        result = ds_obs.mx.align_time_with(ds_fcst)

        assert result.mx.is_forecast()
        assert set(result.dims) == {"reference_time", "lead_time"}

        # obs values: T0→10, T0+6h→20, T0+12h→30, T0+18h→40, T0+24h→50
        assert (
            result["temp"]
            .sel(reference_time=T0, lead_time=np.timedelta64(0, "h"))
            .item()
            == 10.0
        )
        assert (
            result["temp"]
            .sel(reference_time=T0, lead_time=np.timedelta64(6, "h"))
            .item()
            == 20.0
        )
        assert (
            result["temp"]
            .sel(reference_time=T0 + H6, lead_time=np.timedelta64(6, "h"))
            .item()
            == 30.0
        )
        assert (
            result["temp"]
            .sel(reference_time=T0 + H6, lead_time=np.timedelta64(18, "h"))
            .item()
            == 50.0
        )

    def test_nan_where_obs_missing(self, ds_obs, ds_fcst):
        result = ds_obs.mx.align_time_with(ds_fcst)

        # T0+30h is not in obs; it appears at (T0+12h, lead=18h)
        assert np.isnan(
            result["temp"]
            .sel(
                reference_time=T0 + 2 * H6,
                lead_time=np.timedelta64(18, "h"),
            )
            .item()
        )

    def test_obs_value_repeated_for_shared_valid_times(self, ds_obs, ds_fcst):
        result = ds_obs.mx.align_time_with(ds_fcst)

        # T0+12h appears at (T0,12h), (T0+6h,6h), (T0+12h,0h) — all should equal 30.0
        assert (
            result["temp"]
            .sel(reference_time=T0, lead_time=np.timedelta64(12, "h"))
            .item()
            == 30.0
        )
        assert (
            result["temp"]
            .sel(reference_time=T0 + H6, lead_time=np.timedelta64(6, "h"))
            .item()
            == 30.0
        )
        assert (
            result["temp"]
            .sel(reference_time=T0 + 2 * H6, lead_time=np.timedelta64(0, "h"))
            .item()
            == 30.0
        )

    def test_result_has_forecast_property(self, ds_obs, ds_fcst):
        result = ds_obs.mx.align_time_with(ds_fcst)
        assert result.mx.is_forecast()
        assert not result.mx.is_observation()


# ---------------------------------------------------------------------------
# Case 3: Observation → Observation
# ---------------------------------------------------------------------------


class TestObservationToObservation:
    @pytest.fixture
    def ds_obs1(self):
        times = np.array([T0, T0 + H6, T0 + 2 * H6, T0 + 3 * H6])
        return obs(times, [0.0, 1.0, 2.0, 3.0])

    @pytest.fixture
    def ds_obs2(self):
        times = np.array([T0 + H6, T0 + 2 * H6, T0 + 3 * H6, T0 + 4 * H6])
        return obs(times, [10.0, 20.0, 30.0, 40.0])

    def test_reindexes_to_ds2_valid_times(self, ds_obs1, ds_obs2):
        result = ds_obs1.mx.align_time_with(ds_obs2)

        assert list(result.valid_time.values) == list(ds_obs2.valid_time.values)
        # T0+24h is in ds2 but not ds1 → NaN
        assert np.isnan(result["temp"].sel(valid_time=T0 + 4 * H6).item())
        # Overlapping times retain ds1 values
        assert result["temp"].sel(valid_time=T0 + H6).item() == 1.0
        assert result["temp"].sel(valid_time=T0 + 2 * H6).item() == 2.0
        assert result["temp"].sel(valid_time=T0 + 3 * H6).item() == 3.0

    def test_ds1_only_times_are_dropped(self, ds_obs1, ds_obs2):
        result = ds_obs1.mx.align_time_with(ds_obs2)
        assert T0 not in result.valid_time.values  # only in ds1

    def test_result_stays_observation(self, ds_obs1, ds_obs2):
        result = ds_obs1.mx.align_time_with(ds_obs2)
        assert result.mx.is_observation()


# ---------------------------------------------------------------------------
# Case 4: Forecast → Forecast
# ---------------------------------------------------------------------------


class TestForecastToForecast:
    @pytest.fixture
    def ds_fcst2(self):
        ref = np.array([T0 + H6, T0 + 2 * H6, T0 + 3 * H6])
        lead = np.array([np.timedelta64(6, "h"), np.timedelta64(12, "h")])
        values = np.zeros((3, 2))
        return fcst(ref, lead, values)

    def test_reindexes_to_ds2_reference_times(self, ds_fcst, ds_fcst2):
        result = ds_fcst.mx.align_time_with(ds_fcst2, lead_time="reference")

        np.testing.assert_array_equal(
            result.reference_time.values, ds_fcst2.reference_time.values
        )

    def test_lead_time_reference_drops_ds1_only_leads(self, ds_fcst, ds_fcst2):
        result = ds_fcst.mx.align_time_with(ds_fcst2, lead_time="reference")

        np.testing.assert_array_equal(
            result.lead_time.values, ds_fcst2.lead_time.values
        )
        assert np.timedelta64(0, "h") not in result.lead_time.values
        assert np.timedelta64(18, "h") not in result.lead_time.values

    def test_nan_for_ref_times_not_in_ds1(self, ds_fcst, ds_fcst2):
        result = ds_fcst.mx.align_time_with(ds_fcst2, lead_time="reference")

        # T0+18h is in ds_fcst2 but not ds_fcst → all NaN
        assert np.isnan(result["temp"].sel(reference_time=T0 + 3 * H6).values).all()

    def test_values_preserved_for_common_ref_times(self, ds_fcst, ds_fcst2):
        result = ds_fcst.mx.align_time_with(ds_fcst2, lead_time="reference")

        assert (
            result["temp"]
            .sel(reference_time=T0 + H6, lead_time=np.timedelta64(6, "h"))
            .item()
            == 11.0
        )
        assert (
            result["temp"]
            .sel(reference_time=T0 + H6, lead_time=np.timedelta64(12, "h"))
            .item()
            == 12.0
        )
        assert (
            result["temp"]
            .sel(reference_time=T0 + 2 * H6, lead_time=np.timedelta64(6, "h"))
            .item()
            == 21.0
        )
        assert (
            result["temp"]
            .sel(reference_time=T0 + 2 * H6, lead_time=np.timedelta64(12, "h"))
            .item()
            == 22.0
        )

    def test_lead_time_intersection_keeps_common_leads(self, ds_fcst, ds_fcst2):
        result = ds_fcst.mx.align_time_with(ds_fcst2, lead_time="intersection")

        expected_lead = np.array([np.timedelta64(6, "h"), np.timedelta64(12, "h")])
        np.testing.assert_array_equal(result.lead_time.values, expected_lead)
        assert np.timedelta64(0, "h") not in result.lead_time.values
        assert np.timedelta64(18, "h") not in result.lead_time.values

    def test_lead_time_union_keeps_all_leads(self, ds_fcst, ds_fcst2):
        result = ds_fcst.mx.align_time_with(ds_fcst2, lead_time="union")

        expected_lead = np.array([np.timedelta64(h, "h") for h in [0, 6, 12, 18]])
        np.testing.assert_array_equal(result.lead_time.values, expected_lead)

    def test_result_has_valid_time_coord(self, ds_fcst, ds_fcst2):
        result = ds_fcst.mx.align_time_with(ds_fcst2, lead_time="reference")
        assert "valid_time" in result.coords

    def test_result_stays_forecast(self, ds_fcst, ds_fcst2):
        result = ds_fcst.mx.align_time_with(ds_fcst2, lead_time="reference")
        assert result.mx.is_forecast()
