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
from mlwp_data_specs.specs.traits.uncertainty import Uncertainty
from mlwp_data_specs.specs.traits.uncertainty import (
    validate_dataset as validate_uncertainty,
)
from numpy.typing import ArrayLike

T0 = np.datetime64("2020-01-01T00:00", "ns")
H6 = np.timedelta64(6, "h")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def obs(valid_times: ArrayLike, values: ArrayLike) -> xr.Dataset:
    """Build a point observation dataset.

    Parameters
    ----------
    valid_times : ArrayLike
        Values of the ``valid_time`` coordinate.
    values : ArrayLike
        Values of ``temp`` along ``valid_time``.

    Returns
    -------
    xr.Dataset
        Observation dataset with a ``temp`` variable.
    """
    return xr.Dataset(
        {"temp": ("valid_time", np.asarray(values, dtype=float))},
        coords={"valid_time": np.asarray(valid_times)},
        attrs=_props("observation"),
    )


def fcst(
    reference_times: ArrayLike, lead_times: ArrayLike, values: ArrayLike
) -> xr.Dataset:
    """Build a point forecast dataset.

    Parameters
    ----------
    reference_times : ArrayLike
        Values of the ``reference_time`` coordinate.
    lead_times : ArrayLike
        Values of the ``lead_time`` coordinate.
    values : ArrayLike
        Values of ``temp`` with shape ``(reference_time, lead_time)``.

    Returns
    -------
    xr.Dataset
        Forecast dataset with a ``temp`` variable.
    """
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
    """Group tests of aligning a forecast to observation valid times."""

    def test_shortest_lead_time(self, ds_fcst: xr.Dataset, ds_obs: xr.Dataset) -> None:
        """Check that ``lead_time="shortest"`` picks the shortest lead per valid time.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Point forecast fixture.
        ds_obs : xr.Dataset
            Point observation fixture.
        """
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

    def test_longest_lead_time(self, ds_fcst: xr.Dataset, ds_obs: xr.Dataset) -> None:
        """Check that ``lead_time="longest"`` picks the longest lead per valid time.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Point forecast fixture.
        ds_obs : xr.Dataset
            Point observation fixture.
        """
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

    def test_specific_lead_time(self, ds_fcst: xr.Dataset, ds_obs: xr.Dataset) -> None:
        """Check that a specific ``lead_time`` only fills times it produces.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Point forecast fixture.
        ds_obs : xr.Dataset
            Point observation fixture.
        """
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

    def test_lead_time_as_string(self, ds_fcst: xr.Dataset, ds_obs: xr.Dataset) -> None:
        """Check that a string ``lead_time`` matches the equivalent timedelta.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Point forecast fixture.
        ds_obs : xr.Dataset
            Point observation fixture.
        """
        expected = ds_fcst.mx.align_time_with(ds_obs, lead_time=np.timedelta64(6, "h"))
        result = ds_fcst.mx.align_time_with(ds_obs, lead_time="6h")
        xr.testing.assert_identical(result, expected)

    @pytest.mark.parametrize(
        "lead_time",
        [["6h", "12h"], np.array([6, 12], dtype="timedelta64[h]")],
        ids=["list", "ndarray"],
    )
    def test_multiple_lead_times(
        self,
        ds_fcst: xr.Dataset,
        ds_obs: xr.Dataset,
        lead_time: list[str] | np.ndarray,
    ) -> None:
        """Check that several lead times are combined, earliest reference first.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Point forecast fixture.
        ds_obs : xr.Dataset
            Point observation fixture.
        lead_time : list[str] | np.ndarray
            Lead times to select, as strings or a timedelta array.
        """
        result = ds_fcst.mx.align_time_with(ds_obs, lead_time=lead_time)

        assert np.isnan(result["temp"].sel(valid_time=T0).item())  # only 0h reaches T0
        assert result["temp"].sel(valid_time=T0 + H6).item() == 1.0  # (T0, 6h)
        # (T0, 12h) and (T0+6h, 6h) both match → earliest reference_time wins
        assert result["temp"].sel(valid_time=T0 + 2 * H6).item() == 2.0

    @pytest.mark.parametrize(
        "lead_time, match",
        [
            ("shortst", "Unknown lead_time"),
            ("union", "Unknown lead_time"),
            (6, "needs a unit"),
            ("7h", "not in forecast lead times"),
            (["6h", "24h"], "not in forecast lead times"),
        ],
    )
    def test_invalid_lead_time_raises(
        self,
        ds_fcst: xr.Dataset,
        ds_obs: xr.Dataset,
        lead_time: str | int | list[str],
        match: str,
    ) -> None:
        """Check that an invalid ``lead_time`` raises ``ValueError``.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Point forecast fixture.
        ds_obs : xr.Dataset
            Point observation fixture.
        lead_time : str | int | list[str]
            Invalid lead time option.
        match : str
            Pattern expected in the error message.
        """
        with pytest.raises(ValueError, match=match):
            ds_fcst.mx.align_time_with(ds_obs, lead_time=lead_time)

    def test_nan_filled_for_times_not_in_forecast(
        self, ds_fcst: xr.Dataset, ds_obs: xr.Dataset
    ) -> None:
        """Check that valid times the forecast never reaches are NaN.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Point forecast fixture.
        ds_obs : xr.Dataset
            Point observation fixture.
        """
        result = ds_fcst.mx.align_time_with(ds_obs, lead_time="shortest")
        # T0-6h is in obs but never produced by any (ref_time, lead_time) pair
        assert np.isnan(result["temp"].sel(valid_time=T0 - H6).item())

    def test_result_has_observation_property(
        self, ds_fcst: xr.Dataset, ds_obs: xr.Dataset
    ) -> None:
        """Check that the result carries the observation time trait.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Point forecast fixture.
        ds_obs : xr.Dataset
            Point observation fixture.
        """
        result = ds_fcst.mx.align_time_with(ds_obs, lead_time="shortest")
        assert result.mx.is_observation()
        assert not result.mx.is_forecast()


# ---------------------------------------------------------------------------
# Case 2: Observation → Forecast
# ---------------------------------------------------------------------------


class TestObservationToForecast:
    """Group tests of reshaping an observation onto a forecast grid."""

    def test_values_placed_at_correct_positions(
        self, ds_obs: xr.Dataset, ds_fcst: xr.Dataset
    ) -> None:
        """Check that observations land at the matching (reference, lead) cells.

        Parameters
        ----------
        ds_obs : xr.Dataset
            Point observation fixture.
        ds_fcst : xr.Dataset
            Point forecast fixture.
        """
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

    def test_nan_where_obs_missing(
        self, ds_obs: xr.Dataset, ds_fcst: xr.Dataset
    ) -> None:
        """Check that cells whose valid time has no observation are NaN.

        Parameters
        ----------
        ds_obs : xr.Dataset
            Point observation fixture.
        ds_fcst : xr.Dataset
            Point forecast fixture.
        """
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

    def test_obs_value_repeated_for_shared_valid_times(
        self, ds_obs: xr.Dataset, ds_fcst: xr.Dataset
    ) -> None:
        """Check that one observation fills every cell sharing its valid time.

        Parameters
        ----------
        ds_obs : xr.Dataset
            Point observation fixture.
        ds_fcst : xr.Dataset
            Point forecast fixture.
        """
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

    def test_result_has_forecast_property(
        self, ds_obs: xr.Dataset, ds_fcst: xr.Dataset
    ) -> None:
        """Check that the result carries the forecast time trait.

        Parameters
        ----------
        ds_obs : xr.Dataset
            Point observation fixture.
        ds_fcst : xr.Dataset
            Point forecast fixture.
        """
        result = ds_obs.mx.align_time_with(ds_fcst)
        assert result.mx.is_forecast()
        assert not result.mx.is_observation()


# ---------------------------------------------------------------------------
# Case 3: Observation → Observation
# ---------------------------------------------------------------------------


class TestObservationToObservation:
    """Group tests of aligning an observation to another observation."""

    @pytest.fixture
    def ds_obs1(self) -> xr.Dataset:
        """Provide an observation at T0 .. T0+18h with values 0..3.

        Returns
        -------
        xr.Dataset
            Source observation dataset.
        """
        times = np.array([T0, T0 + H6, T0 + 2 * H6, T0 + 3 * H6])
        return obs(times, [0.0, 1.0, 2.0, 3.0])

    @pytest.fixture
    def ds_obs2(self) -> xr.Dataset:
        """Provide an observation at T0+6h .. T0+24h with values 10..40.

        Returns
        -------
        xr.Dataset
            Target observation dataset.
        """
        times = np.array([T0 + H6, T0 + 2 * H6, T0 + 3 * H6, T0 + 4 * H6])
        return obs(times, [10.0, 20.0, 30.0, 40.0])

    def test_reindexes_to_ds2_valid_times(
        self, ds_obs1: xr.Dataset, ds_obs2: xr.Dataset
    ) -> None:
        """Check that the result is reindexed onto the target valid times.

        Parameters
        ----------
        ds_obs1 : xr.Dataset
            Source observation fixture.
        ds_obs2 : xr.Dataset
            Target observation fixture.
        """
        result = ds_obs1.mx.align_time_with(ds_obs2)

        assert list(result.valid_time.values) == list(ds_obs2.valid_time.values)
        # T0+24h is in ds2 but not ds1 → NaN
        assert np.isnan(result["temp"].sel(valid_time=T0 + 4 * H6).item())
        # Overlapping times retain ds1 values
        assert result["temp"].sel(valid_time=T0 + H6).item() == 1.0
        assert result["temp"].sel(valid_time=T0 + 2 * H6).item() == 2.0
        assert result["temp"].sel(valid_time=T0 + 3 * H6).item() == 3.0

    def test_ds1_only_times_are_dropped(
        self, ds_obs1: xr.Dataset, ds_obs2: xr.Dataset
    ) -> None:
        """Check that valid times only in the source are dropped.

        Parameters
        ----------
        ds_obs1 : xr.Dataset
            Source observation fixture.
        ds_obs2 : xr.Dataset
            Target observation fixture.
        """
        result = ds_obs1.mx.align_time_with(ds_obs2)
        assert T0 not in result.valid_time.values  # only in ds1

    def test_result_stays_observation(
        self, ds_obs1: xr.Dataset, ds_obs2: xr.Dataset
    ) -> None:
        """Check that observation-to-observation alignment keeps the trait.

        Parameters
        ----------
        ds_obs1 : xr.Dataset
            Source observation fixture.
        ds_obs2 : xr.Dataset
            Target observation fixture.
        """
        result = ds_obs1.mx.align_time_with(ds_obs2)
        assert result.mx.is_observation()


# ---------------------------------------------------------------------------
# Case 4: Forecast → Forecast
# ---------------------------------------------------------------------------


class TestForecastToForecast:
    """Group tests of aligning a forecast to another forecast."""

    @pytest.fixture
    def ds_fcst2(self) -> xr.Dataset:
        """Provide a zero-valued forecast at T0+6h .. T0+18h with 6h and 12h leads.

        Returns
        -------
        xr.Dataset
            Target forecast dataset.
        """
        ref = np.array([T0 + H6, T0 + 2 * H6, T0 + 3 * H6])
        lead = np.array([np.timedelta64(6, "h"), np.timedelta64(12, "h")])
        values = np.zeros((3, 2))
        return fcst(ref, lead, values)

    def test_reindexes_to_ds2_reference_times(
        self, ds_fcst: xr.Dataset, ds_fcst2: xr.Dataset
    ) -> None:
        """Check that the result is reindexed onto the target reference times.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Source point forecast fixture.
        ds_fcst2 : xr.Dataset
            Target forecast fixture.
        """
        result = ds_fcst.mx.align_time_with(ds_fcst2, lead_time="reference")

        np.testing.assert_array_equal(
            result.reference_time.values, ds_fcst2.reference_time.values
        )

    def test_lead_time_reference_drops_ds1_only_leads(
        self, ds_fcst: xr.Dataset, ds_fcst2: xr.Dataset
    ) -> None:
        """Check that ``lead_time="reference"`` drops leads only in the source.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Source point forecast fixture.
        ds_fcst2 : xr.Dataset
            Target forecast fixture.
        """
        result = ds_fcst.mx.align_time_with(ds_fcst2, lead_time="reference")

        np.testing.assert_array_equal(
            result.lead_time.values, ds_fcst2.lead_time.values
        )
        assert np.timedelta64(0, "h") not in result.lead_time.values
        assert np.timedelta64(18, "h") not in result.lead_time.values

    def test_nan_for_ref_times_not_in_ds1(
        self, ds_fcst: xr.Dataset, ds_fcst2: xr.Dataset
    ) -> None:
        """Check that reference times missing from the source are all NaN.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Source point forecast fixture.
        ds_fcst2 : xr.Dataset
            Target forecast fixture.
        """
        result = ds_fcst.mx.align_time_with(ds_fcst2, lead_time="reference")

        # T0+18h is in ds_fcst2 but not ds_fcst → all NaN
        assert np.isnan(result["temp"].sel(reference_time=T0 + 3 * H6).values).all()

    def test_values_preserved_for_common_ref_times(
        self, ds_fcst: xr.Dataset, ds_fcst2: xr.Dataset
    ) -> None:
        """Check that source values are kept at shared (reference, lead) cells.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Source point forecast fixture.
        ds_fcst2 : xr.Dataset
            Target forecast fixture.
        """
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

    def test_lead_time_intersection_keeps_common_leads(
        self, ds_fcst: xr.Dataset, ds_fcst2: xr.Dataset
    ) -> None:
        """Check that ``lead_time="intersection"`` keeps only shared leads.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Source point forecast fixture.
        ds_fcst2 : xr.Dataset
            Target forecast fixture.
        """
        result = ds_fcst.mx.align_time_with(ds_fcst2, lead_time="intersection")

        expected_lead = np.array([np.timedelta64(6, "h"), np.timedelta64(12, "h")])
        np.testing.assert_array_equal(result.lead_time.values, expected_lead)
        assert np.timedelta64(0, "h") not in result.lead_time.values
        assert np.timedelta64(18, "h") not in result.lead_time.values

    def test_lead_time_union_keeps_all_leads(
        self, ds_fcst: xr.Dataset, ds_fcst2: xr.Dataset
    ) -> None:
        """Check that ``lead_time="union"`` keeps the leads of both forecasts.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Source point forecast fixture.
        ds_fcst2 : xr.Dataset
            Target forecast fixture.
        """
        result = ds_fcst.mx.align_time_with(ds_fcst2, lead_time="union")

        expected_lead = np.array([np.timedelta64(h, "h") for h in [0, 6, 12, 18]])
        np.testing.assert_array_equal(result.lead_time.values, expected_lead)

    def test_result_has_valid_time_coord(
        self, ds_fcst: xr.Dataset, ds_fcst2: xr.Dataset
    ) -> None:
        """Check that the aligned forecast has a ``valid_time`` coordinate.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Source point forecast fixture.
        ds_fcst2 : xr.Dataset
            Target forecast fixture.
        """
        result = ds_fcst.mx.align_time_with(ds_fcst2, lead_time="reference")
        assert "valid_time" in result.coords

    def test_result_stays_forecast(
        self, ds_fcst: xr.Dataset, ds_fcst2: xr.Dataset
    ) -> None:
        """Check that forecast-to-forecast alignment keeps the forecast trait.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Source point forecast fixture.
        ds_fcst2 : xr.Dataset
            Target forecast fixture.
        """
        result = ds_fcst.mx.align_time_with(ds_fcst2, lead_time="reference")
        assert result.mx.is_forecast()


# ---------------------------------------------------------------------------
# Ensembles: the member dimension rides along and is never matched
# ---------------------------------------------------------------------------


def _assert_ensemble(ds: xr.Dataset, members: ArrayLike) -> None:
    """Assert that ``ds`` is still a valid ensemble with the given members.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to check.
    members : ArrayLike
        Expected values of the ``member`` coordinate.
    """
    assert ds.mx.is_ensemble()
    np.testing.assert_array_equal(ds["member"].values, members)
    assert ds["member"].attrs["standard_name"] == "realization"
    report, _ = validate_uncertainty(ds, trait=Uncertainty.ENSEMBLE)
    assert not report.has_fails()


def _per_member(ds: xr.Dataset, member: int) -> xr.DataArray:
    """Return ``temp`` of one member, minus the 100-per-member offset.

    Parameters
    ----------
    ds : xr.Dataset
        Ensemble built from the ``ds_fcst_ens`` fixture.
    member : int
        Member to select.

    Returns
    -------
    xr.DataArray
        The member's values, comparable to the deterministic ``ds_fcst``.
    """
    return ds["temp"].sel(member=member, drop=True) - 100.0 * member


class TestEnsemble:
    """Group tests of time alignment with a ``member`` dimension."""

    @pytest.mark.parametrize("lead_time", ["shortest", "longest", "6h"])
    def test_forecast_ensemble_to_observation(
        self,
        ds_fcst_ens: xr.Dataset,
        ds_fcst: xr.Dataset,
        ds_obs: xr.Dataset,
        lead_time: str,
    ) -> None:
        """Check that each member is aligned like the deterministic forecast.

        Parameters
        ----------
        ds_fcst_ens : xr.Dataset
            Point forecast ensemble fixture.
        ds_fcst : xr.Dataset
            Point forecast fixture, equal to member 0.
        ds_obs : xr.Dataset
            Point observation fixture.
        lead_time : str
            Lead time strategy for forecast-to-observation alignment.
        """
        result = ds_fcst_ens.mx.align_time_with(ds_obs, lead_time=lead_time)
        expected = ds_fcst.mx.align_time_with(ds_obs, lead_time=lead_time)

        assert result.mx.is_observation()
        _assert_ensemble(result, ds_fcst_ens["member"])
        for member in result["member"].values:
            xr.testing.assert_equal(_per_member(result, member), expected["temp"])

    def test_observation_to_forecast_ensemble_adds_no_member(
        self, ds_obs: xr.Dataset, ds_fcst_ens: xr.Dataset, ds_fcst: xr.Dataset
    ) -> None:
        """Check that observations reshape onto an ensemble's times only.

        Parameters
        ----------
        ds_obs : xr.Dataset
            Point observation fixture.
        ds_fcst_ens : xr.Dataset
            Point forecast ensemble fixture.
        ds_fcst : xr.Dataset
            Point forecast fixture with the same times as the ensemble.
        """
        result = ds_obs.mx.align_time_with(ds_fcst_ens)

        xr.testing.assert_identical(result, ds_obs.mx.align_time_with(ds_fcst))
        assert result.mx.is_deterministic()

    def test_forecast_ensemble_to_forecast(
        self, ds_fcst_ens: xr.Dataset, ds_fcst: xr.Dataset
    ) -> None:
        """Check that each member is aligned like the deterministic forecast.

        Parameters
        ----------
        ds_fcst_ens : xr.Dataset
            Point forecast ensemble fixture.
        ds_fcst : xr.Dataset
            Point forecast fixture, equal to member 0.
        """
        target = ds_fcst.isel(reference_time=[1, 2], lead_time=[0, 1])

        result = ds_fcst_ens.mx.align_time_with(target)
        expected = ds_fcst.mx.align_time_with(target)

        _assert_ensemble(result, ds_fcst_ens["member"])
        for member in result["member"].values:
            xr.testing.assert_equal(_per_member(result, member), expected["temp"])

    def test_forecast_to_forecast_ensemble_adds_no_member(
        self, ds_fcst: xr.Dataset, ds_fcst_ens: xr.Dataset
    ) -> None:
        """Check that a deterministic forecast aligns to an ensemble's times only.

        Parameters
        ----------
        ds_fcst : xr.Dataset
            Point forecast fixture.
        ds_fcst_ens : xr.Dataset
            Point forecast ensemble fixture.
        """
        target = ds_fcst_ens.isel(reference_time=[1, 2], lead_time=[0, 1])

        result = ds_fcst.mx.align_time_with(target)

        xr.testing.assert_identical(
            result, ds_fcst.mx.align_time_with(target.isel(member=0, drop=True))
        )

    def test_members_are_not_matched_between_ensembles(
        self, ds_fcst_ens: xr.Dataset
    ) -> None:
        """Check that aligning to an ensemble with other members keeps all members.

        Member 1 of one ensemble has nothing to do with member 1 of another, so
        the ``member`` dimension must not be reindexed.

        Parameters
        ----------
        ds_fcst_ens : xr.Dataset
            Point forecast ensemble fixture with members 0, 1, 2.
        """
        target = ds_fcst_ens.isel(member=[1, 2]).assign_coords(member=[10, 11])

        result = ds_fcst_ens.mx.align_time_with(target)

        _assert_ensemble(result, ds_fcst_ens["member"])
        xr.testing.assert_equal(
            result["temp"].drop_vars("valid_time"), ds_fcst_ens["temp"]
        )
