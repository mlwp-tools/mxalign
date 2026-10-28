"""Tests for ds.mx.align_space_with() covering all spatial alignment cases.

Fixtures:

  ds_grid  — 3×3 lat/lon grid, temp[i, j] = lat[i] + lon[j]
             lat: 0°, 1°, 2°   lon: 0°, 1°, 2°

  ds_point — 3 observation points:
               point 0: (lat=0.0, lon=0.0) — exact grid node  → expected temp=0.0
               point 1: (lat=0.5, lon=0.5) — interior          → expected temp=1.0
               point 2: (lat=1.0, lon=1.0) — exact grid node  → expected temp=2.0

Pure interpolation logic (values, coords) is tested in test_interpolations.py.
These tests focus on trait propagation and accessor dispatch.
"""

import numpy as np
import pytest
import xarray as xr

import mxalign  # registers ds.mx accessor  # noqa: F401

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _props(space: str, time: str = "forecast") -> dict:
    """Build the ``mlwp_*`` trait attributes for a deterministic dataset.

    Parameters
    ----------
    space : str
        Value of the ``mlwp_space_trait`` attribute, e.g. ``"grid"``.
    time : str, default "forecast"
        Value of the ``mlwp_time_trait`` attribute.

    Returns
    -------
    dict
        Trait attributes to attach to a dataset's ``attrs``.
    """
    return {
        "mlwp_space_trait": space,
        "mlwp_time_trait": time,
        "mlwp_uncertainty_trait": "deterministic",
    }


@pytest.fixture
def ds_grid() -> xr.Dataset:
    """Provide a 3×3 lat/lon grid forecast with ``temp[i, j] = lat[i] + lon[j]``.

    Returns
    -------
    xr.Dataset
        Grid forecast on latitudes and longitudes 0, 1, 2.
    """
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([0.0, 1.0, 2.0])
    temp = lats[:, np.newaxis] + lons[np.newaxis, :]
    return xr.Dataset(
        {"temp": (["latitude", "longitude"], temp)},
        coords={"latitude": lats, "longitude": lons},
        attrs=_props("grid"),
    )


@pytest.fixture
def ds_point() -> xr.Dataset:
    """Provide 3 observation points, two on grid nodes and one interior.

    Returns
    -------
    xr.Dataset
        Point observation along ``point_index``.
    """
    return xr.Dataset(
        {"temp": ("point_index", np.array([0.0, 1.0, 2.0]))},
        coords={
            "latitude": ("point_index", np.array([0.0, 0.5, 1.0])),
            "longitude": ("point_index", np.array([0.0, 0.5, 1.0])),
        },
        attrs=_props("point", "observation"),
    )


# ---------------------------------------------------------------------------
# Case 1: Grid → Grid
# ---------------------------------------------------------------------------


class TestGridToGrid:
    """Group tests of aligning a grid dataset to another grid."""

    def test_identical_grids_return_self(self, ds_grid: xr.Dataset) -> None:
        """Check that aligning a grid to itself returns the same object.

        Parameters
        ----------
        ds_grid : xr.Dataset
            Lat/lon grid forecast fixture.
        """
        result = ds_grid.mx.align_space_with(ds_grid)

        assert result is ds_grid

    def test_within_tolerance_treated_as_equal(self, ds_grid: xr.Dataset) -> None:
        """Check that grids differing within tolerance are treated as equal.

        Parameters
        ----------
        ds_grid : xr.Dataset
            Lat/lon grid forecast fixture.
        """
        ds_grid2 = ds_grid.assign_coords(
            latitude=ds_grid.latitude + 1e-5,
            longitude=ds_grid.longitude + 1e-5,
        ).assign_attrs(_props("grid"))
        result = ds_grid.mx.align_space_with(ds_grid2)

        assert result is ds_grid

    def test_different_grids_raise(self, ds_grid: xr.Dataset) -> None:
        """Check that aligning to a different grid raises ``NotImplementedError``.

        Parameters
        ----------
        ds_grid : xr.Dataset
            Lat/lon grid forecast fixture.
        """
        ds_grid2 = ds_grid.assign_coords(
            latitude=ds_grid.latitude + 10.0,
        ).assign_attrs(_props("grid"))

        with pytest.raises(NotImplementedError):
            ds_grid.mx.align_space_with(ds_grid2)

    def test_result_stays_grid(self, ds_grid: xr.Dataset) -> None:
        """Check that grid-to-grid alignment keeps the grid space trait.

        Parameters
        ----------
        ds_grid : xr.Dataset
            Lat/lon grid forecast fixture.
        """
        result = ds_grid.mx.align_space_with(ds_grid)

        assert result.mx.is_grid()
        assert not result.mx.is_point()


# ---------------------------------------------------------------------------
# Case 2: Grid → Point  (xarray interpolator)
# ---------------------------------------------------------------------------


class TestGridToPoint:
    """Group tests of interpolating a grid dataset to point locations."""

    def test_result_has_point_trait(
        self, ds_grid: xr.Dataset, ds_point: xr.Dataset
    ) -> None:
        """Check that grid-to-point alignment sets the point space trait.

        Parameters
        ----------
        ds_grid : xr.Dataset
            Lat/lon grid forecast fixture.
        ds_point : xr.Dataset
            Point observation fixture.
        """
        result = ds_grid.mx.align_space_with(ds_point, method="xarray")

        assert result.mx.is_point()
        assert not result.mx.is_grid()

    def test_result_has_point_index_dim(
        self, ds_grid: xr.Dataset, ds_point: xr.Dataset
    ) -> None:
        """Check that grid-to-point alignment produces a ``point_index`` dim.

        Parameters
        ----------
        ds_grid : xr.Dataset
            Lat/lon grid forecast fixture.
        ds_point : xr.Dataset
            Point observation fixture.
        """
        result = ds_grid.mx.align_space_with(ds_point, method="xarray")

        assert "point_index" in result.dims

    def test_result_has_target_latlon_coords(
        self, ds_grid: xr.Dataset, ds_point: xr.Dataset
    ) -> None:
        """Check that grid-to-point alignment copies the target lat/lon coords.

        Parameters
        ----------
        ds_grid : xr.Dataset
            Lat/lon grid forecast fixture.
        ds_point : xr.Dataset
            Point observation fixture.
        """
        result = ds_grid.mx.align_space_with(ds_point, method="xarray")

        np.testing.assert_array_equal(
            result["latitude"].values, ds_point["latitude"].values
        )
        np.testing.assert_array_equal(
            result["longitude"].values, ds_point["longitude"].values
        )

    def test_interpolated_values_xarray(
        self, ds_grid: xr.Dataset, ds_point: xr.Dataset
    ) -> None:
        """Check that the xarray method interpolates the expected point values.

        Parameters
        ----------
        ds_grid : xr.Dataset
            Lat/lon grid forecast fixture.
        ds_point : xr.Dataset
            Point observation fixture.
        """
        result = ds_grid.mx.align_space_with(ds_point, method="xarray")

        assert result["temp"].isel(point_index=0).item() == pytest.approx(0.0)
        assert result["temp"].isel(point_index=1).item() == pytest.approx(1.0)
        assert result["temp"].isel(point_index=2).item() == pytest.approx(2.0)

    def test_interpolated_values_delaunay(
        self, ds_grid: xr.Dataset, ds_point: xr.Dataset
    ) -> None:
        """Check that the Delaunay method interpolates the expected point values.

        Parameters
        ----------
        ds_grid : xr.Dataset
            Lat/lon grid forecast fixture.
        ds_point : xr.Dataset
            Point observation fixture.
        """
        ds_stacked = ds_grid.stack(grid_index=["latitude", "longitude"]).reset_index(
            "grid_index"
        )
        ds_stacked.attrs.update(_props("grid"))
        result = ds_stacked.mx.align_space_with(ds_point, method="delaunay")

        assert result["temp"].isel(point_index=0).item() == pytest.approx(0.0)
        assert result["temp"].isel(point_index=1).item() == pytest.approx(1.0)
        assert result["temp"].isel(point_index=2).item() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Case 3: Point → * (not implemented)
# ---------------------------------------------------------------------------


class TestPointAlignmentNotImplemented:
    """Group tests that aligning from point datasets is not implemented."""

    def test_point_to_grid_raises(
        self, ds_point: xr.Dataset, ds_grid: xr.Dataset
    ) -> None:
        """Check that aligning a point dataset to a grid raises.

        Parameters
        ----------
        ds_point : xr.Dataset
            Point observation fixture.
        ds_grid : xr.Dataset
            Lat/lon grid forecast fixture.
        """
        with pytest.raises(NotImplementedError):
            ds_point.mx.align_space_with(ds_grid)

    def test_point_to_point_raises(self, ds_point: xr.Dataset) -> None:
        """Check that aligning a point dataset to points raises.

        Parameters
        ----------
        ds_point : xr.Dataset
            Point observation fixture.
        """
        with pytest.raises(NotImplementedError):
            ds_point.mx.align_space_with(ds_point)
