"""Tests for ``broadcast_nans``: propagating NaN masks across datasets.

The datasets are point observations over 4 valid times, with
``temp = 0, 1, 2, 3``. Ensembles add a ``member`` dimension, and member ``m``
adds ``100 * m``. NaNs are placed by position.
"""

from collections.abc import Sequence

import dask.array as dda
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.typing import ArrayLike

from mxalign.alignment.nans import broadcast_nans

TIMES = pd.date_range("2020-01-01", periods=4, freq="6h")
MEMBER_ATTRS = {"standard_name": "realization", "units": "1"}


def _traits(uncertainty: str) -> dict:
    """Build the ``mlwp_*`` trait attributes for a point observation.

    Parameters
    ----------
    uncertainty : str
        Value of the ``mlwp_uncertainty_trait`` attribute.

    Returns
    -------
    dict
        Trait attributes to attach to a dataset's ``attrs``.
    """
    return {
        "mlwp_space_trait": "point",
        "mlwp_time_trait": "observation",
        "mlwp_uncertainty_trait": uncertainty,
    }


def _det(nan_at: Sequence[int] = (), times: pd.DatetimeIndex = TIMES) -> xr.Dataset:
    """Build a deterministic dataset with NaNs at the given time positions.

    Parameters
    ----------
    nan_at : sequence of int, default ()
        Positions along ``valid_time`` to set to NaN.
    times : pd.DatetimeIndex, default TIMES
        Values of the ``valid_time`` coordinate.

    Returns
    -------
    xr.Dataset
        Dataset with a ``temp`` variable along ``valid_time``.
    """
    values = np.arange(len(times), dtype=float)
    values[list(nan_at)] = np.nan
    return xr.Dataset(
        {"temp": ("valid_time", values, {"units": "K"})},
        coords={"valid_time": times},
        attrs=_traits("deterministic"),
    )


def _ens(
    nan_at: Sequence[tuple[int, int]] = (), members: ArrayLike = (0, 1, 2)
) -> xr.Dataset:
    """Build an ensemble with NaNs at the given (member, time) positions.

    Parameters
    ----------
    nan_at : sequence of tuple of int, default ()
        ``(member, valid_time)`` positions to set to NaN.
    members : ArrayLike, default (0, 1, 2)
        Values of the ``member`` coordinate.

    Returns
    -------
    xr.Dataset
        Ensemble with a ``temp`` variable over ``member`` and ``valid_time``.
    """
    members = np.asarray(members)
    values = 100.0 * members[:, np.newaxis] + np.arange(len(TIMES))
    for m, t in nan_at:
        values[m, t] = np.nan
    return xr.Dataset(
        {"temp": (["member", "valid_time"], values, {"units": "K"})},
        coords={"member": ("member", members, MEMBER_ATTRS), "valid_time": TIMES},
        attrs=_traits("ensemble"),
    )


def _missing(ds: xr.Dataset) -> list[bool]:
    """Return for each valid time whether ``temp`` is missing.

    Also asserts that an ensemble is missing either all its members or none.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a ``temp`` variable along ``valid_time``.

    Returns
    -------
    list of bool
        Whether ``temp`` is NaN, per valid time.
    """
    missing = ds["temp"].isnull()
    if "member" in missing.dims:
        xr.testing.assert_equal(missing.any("member"), missing.all("member"))
        missing = missing.any("member")
    return missing.values.tolist()


# ---------------------------------------------------------------------------
# Deterministic datasets
# ---------------------------------------------------------------------------


class TestDeterministic:
    """Group tests of NaN propagation between deterministic datasets."""

    def test_nan_propagates_both_ways(self) -> None:
        """Check that a NaN in either dataset ends up in both."""
        a, b = broadcast_nans([_det(nan_at=[1]), _det(nan_at=[2])])

        assert _missing(a) == _missing(b) == [False, True, True, False]

    def test_only_shared_times_are_masked(self) -> None:
        """Check that values outside the shared coordinates are left alone."""
        a = _det(nan_at=[0])
        b = _det(nan_at=[0], times=TIMES[1:])  # NaN at TIMES[1]

        a, b = broadcast_nans([a, b])

        assert _missing(a) == [True, True, False, False]
        assert _missing(b) == [True, False, False]

    def test_inputs_are_not_modified(self) -> None:
        """Check that the input datasets are left unchanged."""
        a, b = _det(nan_at=[1]), _det(nan_at=[2])
        a_before, b_before = a.copy(deep=True), b.copy(deep=True)

        broadcast_nans({"a": a, "b": b})

        xr.testing.assert_identical(a, a_before)
        xr.testing.assert_identical(b, b_before)

    def test_returns_same_container_type(self) -> None:
        """Check that a dict, a list and a single dataset come back as given."""
        ds = _det()

        assert list(broadcast_nans({"x": ds, "y": _det()})) == ["x", "y"]
        assert isinstance(broadcast_nans([ds, _det()]), list)
        assert broadcast_nans(ds) is ds

    def test_variable_in_one_dataset_is_untouched(self) -> None:
        """Check that a variable only one dataset has is not masked."""
        a = _det(nan_at=[1]).assign(rh=("valid_time", np.ones(len(TIMES))))

        a, _ = broadcast_nans([a, _det(nan_at=[2])])

        assert not a["rh"].isnull().any()

    def test_different_dims_raise(self) -> None:
        """Check that a shared variable with other non-uncertainty dims raises."""
        a = _det().expand_dims(level=[500, 850])

        with pytest.raises(ValueError, match="variable 'temp'"):
            broadcast_nans([a, _det()])

    def test_missing_uncertainty_trait_raises(self) -> None:
        """Check that a dataset without an uncertainty trait raises."""
        a = _det()
        del a.attrs["mlwp_uncertainty_trait"]

        with pytest.raises(KeyError, match="mlwp_uncertainty_trait"):
            broadcast_nans([a, _det()])


# ---------------------------------------------------------------------------
# Ensembles: members are reduced, never matched
# ---------------------------------------------------------------------------


class TestEnsemble:
    """Group tests of NaN propagation with ensembles."""

    @pytest.mark.parametrize("ensemble_first", [False, True])
    def test_observation_and_ensemble(self, ensemble_first: bool) -> None:
        """Check that one missing member masks all members and the observation.

        Parameters
        ----------
        ensemble_first : bool
            Whether the ensemble comes before the observation in the input.
        """
        datasets = {"obs": _det(nan_at=[1]), "ens": _ens(nan_at=[(0, 2)])}
        if ensemble_first:
            datasets = dict(reversed(datasets.items()))

        out = broadcast_nans(datasets)

        assert _missing(out["obs"]) == _missing(out["ens"])
        assert _missing(out["obs"]) == [False, True, True, False]

    def test_observation_deterministic_and_ensemble(self) -> None:
        """Check that NaNs propagate across a mix of uncertainty types."""
        out = broadcast_nans(
            [_det(nan_at=[0]), _det(nan_at=[1]), _ens(nan_at=[(2, 3)])]
        )

        assert all(_missing(ds) == [True, True, False, True] for ds in out)

    def test_members_are_not_matched_between_ensembles(self) -> None:
        """Check that ensembles of different sizes keep all their members.

        Member labels of two ensembles are unrelated, so masks are compared
        after reducing over ``member``, not per member.
        """
        small = _ens(nan_at=[(0, 1)])
        large = _ens(nan_at=[(4, 2)], members=[10, 11, 12, 13, 14])

        small, large = broadcast_nans([small, large])

        np.testing.assert_array_equal(small["member"], [0, 1, 2])
        np.testing.assert_array_equal(large["member"], [10, 11, 12, 13, 14])
        assert _missing(small) == _missing(large) == [False, True, True, False]

    def test_dim_order_coords_and_attrs_are_kept(self) -> None:
        """Check that a trailing member dim, coord attrs and var attrs survive."""
        ens = _ens(nan_at=[(1, 0)]).transpose("valid_time", "member")

        _, out = broadcast_nans([_det(), ens])

        assert out["temp"].dims == ("valid_time", "member")
        assert out["temp"].attrs == {"units": "K"}
        assert out["member"].attrs == MEMBER_ATTRS
        assert out.mx.is_ensemble()
        assert _missing(out) == [True, False, False, False]

    def test_dask_inputs_stay_lazy(self) -> None:
        """Check that dask-backed inputs give dask-backed, correct outputs."""
        obs = _det(nan_at=[1]).chunk({"valid_time": 2})
        ens = _ens(nan_at=[(0, 2)]).chunk({"member": 1, "valid_time": 2})

        obs, ens = broadcast_nans([obs, ens])

        assert isinstance(obs["temp"].data, dda.Array)
        assert isinstance(ens["temp"].data, dda.Array)
        assert _missing(obs) == _missing(ens) == [False, True, True, False]
