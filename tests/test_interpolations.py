"""Tests for XarrayInterpolator and DelaunayInterpolator core logic.

Both interpolators take a stacked or lat/lon-dim source GRID dataset and
produce a POINT dataset aligned to target point locations.

Fixtures use temp = lat + lon, a linear function, so both bilinear (xarray)
and linear barycentric (Delaunay) interpolation reproduce it exactly.

Trait / accessor-level tests live in test_align_space.py.
"""

import numpy as np
import pytest
import xarray as xr

from mxalign.interpolations.delaunay import DelaunayInterpolator
from mxalign.interpolations.xarray import XarrayInterpolator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grid_latlon():
    """3×3 regular lat/lon grid. temp[i, j] = lat[i] + lon[j]."""
    lats = np.array([0.0, 1.0, 2.0])
    lons = np.array([0.0, 1.0, 2.0])
    temp = lats[:, np.newaxis] + lons[np.newaxis, :]
    return xr.Dataset(
        {"temp": (["latitude", "longitude"], temp)},
        coords={"latitude": lats, "longitude": lons},
    )


@pytest.fixture
def grid_stacked(grid_latlon):
    """grid_latlon stacked to grid_index (required by DelaunayInterpolator)."""
    return grid_latlon.stack(grid_index=["latitude", "longitude"]).reset_index("grid_index")


@pytest.fixture
def target_points():
    """3 target points: two at grid nodes, one at an interior location."""
    return xr.Dataset(
        coords={
            "latitude": ("point_index", np.array([0.0, 0.5, 1.0])),
            "longitude": ("point_index", np.array([0.0, 0.5, 1.0])),
        },
    )


# ---------------------------------------------------------------------------
# XarrayInterpolator
# ---------------------------------------------------------------------------


class TestXarrayInterpolator:
    def test_values_at_grid_nodes(self, grid_latlon, target_points):
        result = XarrayInterpolator(target_points)._interpolate(grid_latlon)

        # lat=0, lon=0 → 0+0=0; lat=1, lon=1 → 1+1=2
        assert result["temp"].isel(point_index=0).item() == pytest.approx(0.0)
        assert result["temp"].isel(point_index=2).item() == pytest.approx(2.0)

    def test_value_at_interior_point(self, grid_latlon, target_points):
        result = XarrayInterpolator(target_points)._interpolate(grid_latlon)

        # lat=0.5, lon=0.5 → 0.5+0.5=1.0 (exact for bilinear on linear function)
        assert result["temp"].isel(point_index=1).item() == pytest.approx(1.0)

    def test_output_has_point_index_dim(self, grid_latlon, target_points):
        result = XarrayInterpolator(target_points)._interpolate(grid_latlon)

        assert "point_index" in result.dims

    def test_output_has_latlon_coords_from_target(self, grid_latlon, target_points):
        result = XarrayInterpolator(target_points)._interpolate(grid_latlon)

        np.testing.assert_array_equal(result["latitude"].values, target_points["latitude"].values)
        np.testing.assert_array_equal(result["longitude"].values, target_points["longitude"].values)


# ---------------------------------------------------------------------------
# DelaunayInterpolator
# ---------------------------------------------------------------------------


class TestDelaunayInterpolator:
    def test_values_at_grid_nodes(self, grid_stacked, target_points):
        result = DelaunayInterpolator(target_points)._interpolate(grid_stacked)

        assert result["temp"].isel(point_index=0).item() == pytest.approx(0.0)
        assert result["temp"].isel(point_index=2).item() == pytest.approx(2.0)

    def test_value_at_interior_point(self, grid_stacked, target_points):
        result = DelaunayInterpolator(target_points)._interpolate(grid_stacked)

        # Barycentric interpolation is exact for any linear function
        assert result["temp"].isel(point_index=1).item() == pytest.approx(1.0)

    def test_output_has_point_index_dim(self, grid_stacked, target_points):
        result = DelaunayInterpolator(target_points)._interpolate(grid_stacked)

        assert "point_index" in result.dims

    def test_output_has_latlon_coords_from_target(self, grid_stacked, target_points):
        result = DelaunayInterpolator(target_points)._interpolate(grid_stacked)

        np.testing.assert_array_equal(result["latitude"].values, target_points["latitude"].values)
        np.testing.assert_array_equal(result["longitude"].values, target_points["longitude"].values)

    def test_weight_matrix_is_cached(self, grid_stacked, target_points):
        interp = DelaunayInterpolator(target_points)
        interp._interpolate(grid_stacked)
        interp._interpolate(grid_stacked)

        assert len(interp._W_cache) == 1

    def test_outside_convex_hull_is_nan(self, grid_stacked):
        far_point = xr.Dataset(
            coords={
                "latitude": ("point_index", np.array([10.0])),
                "longitude": ("point_index", np.array([10.0])),
            },
        )
        result = DelaunayInterpolator(far_point)._interpolate(grid_stacked)

        assert np.isnan(result["temp"].isel(point_index=0).item())
