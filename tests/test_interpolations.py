"""Tests for XarrayInterpolator and DelaunayInterpolator core logic.

Both interpolators take a stacked or lat/lon-dim source GRID dataset and
produce a POINT dataset aligned to target point locations.

Fixtures use temp = lat + lon, a linear function, so both bilinear (xarray)
and linear barycentric (Delaunay) interpolation reproduce it exactly.

Trait / accessor-level tests live in test_align_space.py.
"""

from pathlib import Path

import cartopy.crs as ccrs
import dask
import numpy as np
import pytest
import xarray as xr
from numpy.typing import ArrayLike

from mxalign.alignment.interpolations import delaunay
from mxalign.alignment.interpolations.delaunay import DelaunayInterpolator
from mxalign.alignment.interpolations.interpolate import interpolate
from mxalign.alignment.interpolations.xarray import XarrayInterpolator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _latlon_grid(lats: ArrayLike, lons: ArrayLike) -> xr.Dataset:
    """Build a regular lat/lon grid with ``temp[i, j] = lat[i] + lon[j]``.

    Parameters
    ----------
    lats : ArrayLike
        Latitudes of the grid.
    lons : ArrayLike
        Longitudes of the grid.

    Returns
    -------
    xr.Dataset
        Grid dataset with ``latitude`` and ``longitude`` dims.
    """
    lats, lons = np.asarray(lats), np.asarray(lons)
    temp = lats[:, np.newaxis] + lons[np.newaxis, :]
    return xr.Dataset(
        {"temp": (["latitude", "longitude"], temp)},
        coords={"latitude": lats, "longitude": lons},
    )


def _stacked_grid(lats: ArrayLike, lons: ArrayLike) -> xr.Dataset:
    """Build a regular lat/lon grid stacked to a ``grid_index`` dim.

    Parameters
    ----------
    lats : ArrayLike
        Latitudes of the grid.
    lons : ArrayLike
        Longitudes of the grid.

    Returns
    -------
    xr.Dataset
        Grid dataset with ``temp = lat + lon`` along ``grid_index``.
    """
    return (
        _latlon_grid(lats, lons)
        .stack(grid_index=["latitude", "longitude"])
        .reset_index("grid_index")
    )


@pytest.fixture
def grid_latlon() -> xr.Dataset:
    """Provide a 3×3 regular lat/lon grid with ``temp[i, j] = lat[i] + lon[j]``.

    Returns
    -------
    xr.Dataset
        Grid dataset on latitudes and longitudes 0, 1, 2.
    """
    return _latlon_grid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])


@pytest.fixture
def grid_stacked(grid_latlon: xr.Dataset) -> xr.Dataset:
    """Provide ``grid_latlon`` stacked to ``grid_index``.

    Stacking is required by ``DelaunayInterpolator``.

    Parameters
    ----------
    grid_latlon : xr.Dataset
        Regular lat/lon grid fixture.

    Returns
    -------
    xr.Dataset
        Grid dataset along ``grid_index``.
    """
    return grid_latlon.stack(grid_index=["latitude", "longitude"]).reset_index(
        "grid_index"
    )


@pytest.fixture
def target_points() -> xr.Dataset:
    """Provide 3 target points: two at grid nodes, one at an interior location.

    Returns
    -------
    xr.Dataset
        Coordinate-only dataset with lat/lon along ``point_index``.
    """
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
    """Group tests of ``XarrayInterpolator``."""

    def test_values_at_grid_nodes(
        self, grid_latlon: xr.Dataset, target_points: xr.Dataset
    ) -> None:
        """Check that values at grid nodes are reproduced exactly.

        Parameters
        ----------
        grid_latlon : xr.Dataset
            Regular lat/lon grid fixture.
        target_points : xr.Dataset
            Target point locations fixture.
        """
        result = XarrayInterpolator(target_points)._interpolate(grid_latlon)

        # lat=0, lon=0 → 0+0=0; lat=1, lon=1 → 1+1=2
        assert result["temp"].isel(point_index=0).item() == pytest.approx(0.0)
        assert result["temp"].isel(point_index=2).item() == pytest.approx(2.0)

    def test_value_at_interior_point(
        self, grid_latlon: xr.Dataset, target_points: xr.Dataset
    ) -> None:
        """Check that an interior point is interpolated bilinearly.

        Parameters
        ----------
        grid_latlon : xr.Dataset
            Regular lat/lon grid fixture.
        target_points : xr.Dataset
            Target point locations fixture.
        """
        result = XarrayInterpolator(target_points)._interpolate(grid_latlon)

        # lat=0.5, lon=0.5 → 0.5+0.5=1.0 (exact for bilinear on linear function)
        assert result["temp"].isel(point_index=1).item() == pytest.approx(1.0)

    def test_output_has_point_index_dim(
        self, grid_latlon: xr.Dataset, target_points: xr.Dataset
    ) -> None:
        """Check that the output has a ``point_index`` dim.

        Parameters
        ----------
        grid_latlon : xr.Dataset
            Regular lat/lon grid fixture.
        target_points : xr.Dataset
            Target point locations fixture.
        """
        result = XarrayInterpolator(target_points)._interpolate(grid_latlon)

        assert "point_index" in result.dims

    def test_output_has_latlon_coords_from_target(
        self, grid_latlon: xr.Dataset, target_points: xr.Dataset
    ) -> None:
        """Check that the output carries the target lat/lon coords.

        Parameters
        ----------
        grid_latlon : xr.Dataset
            Regular lat/lon grid fixture.
        target_points : xr.Dataset
            Target point locations fixture.
        """
        result = XarrayInterpolator(target_points)._interpolate(grid_latlon)

        np.testing.assert_array_equal(
            result["latitude"].values, target_points["latitude"].values
        )
        np.testing.assert_array_equal(
            result["longitude"].values, target_points["longitude"].values
        )

    def test_outside_domain_is_nan(self, grid_latlon: xr.Dataset) -> None:
        """Check that a point outside the grid domain is NaN.

        Parameters
        ----------
        grid_latlon : xr.Dataset
            Regular lat/lon grid fixture.
        """
        far_point = xr.Dataset(
            coords={
                "latitude": ("point_index", np.array([10.0])),
                "longitude": ("point_index", np.array([10.0])),
            },
        )
        result = XarrayInterpolator(far_point)._interpolate(grid_latlon)

        assert np.isnan(result["temp"].isel(point_index=0).item())

    def test_projected_source_keeps_target_point_coords(
        self, target_points: xr.Dataset
    ) -> None:
        """Check that a projected source keeps the target's point coordinates.

        Parameters
        ----------
        target_points : xr.Dataset
            Target point locations fixture.
        """
        # xc/yc source with an identity projection, so xc == lon and yc == lat
        xc = yc = np.array([0.0, 1.0, 2.0])
        source = xr.Dataset(
            {"temp": (["yc", "xc"], yc[:, np.newaxis] + xc[np.newaxis, :])},
            coords={"xc": xc, "yc": yc},
            attrs={
                "crs": ccrs.PlateCarree(),
                "mlwp_space_trait": "grid",
                "mlwp_time_trait": "forecast",
            },
        )
        target = target_points.assign_coords(
            point_index=[101, 102, 103], station=("point_index", ["a", "b", "c"])
        )

        result = XarrayInterpolator(target)._interpolate(source)

        np.testing.assert_array_equal(result["point_index"].values, [101, 102, 103])
        np.testing.assert_array_equal(result["station"].values, ["a", "b", "c"])
        np.testing.assert_allclose(result["temp"].values, [0.0, 1.0, 2.0])


# ---------------------------------------------------------------------------
# DelaunayInterpolator
# ---------------------------------------------------------------------------


class TestDelaunayInterpolator:
    """Group tests of ``DelaunayInterpolator``."""

    def test_values_at_grid_nodes(
        self, grid_stacked: xr.Dataset, target_points: xr.Dataset
    ) -> None:
        """Check that values at grid nodes are reproduced exactly.

        Parameters
        ----------
        grid_stacked : xr.Dataset
            Stacked grid fixture.
        target_points : xr.Dataset
            Target point locations fixture.
        """
        result = DelaunayInterpolator(target_points)._interpolate(grid_stacked)

        assert result["temp"].isel(point_index=0).item() == pytest.approx(0.0)
        assert result["temp"].isel(point_index=2).item() == pytest.approx(2.0)

    def test_value_at_interior_point(
        self, grid_stacked: xr.Dataset, target_points: xr.Dataset
    ) -> None:
        """Check that an interior point is interpolated barycentrically.

        Parameters
        ----------
        grid_stacked : xr.Dataset
            Stacked grid fixture.
        target_points : xr.Dataset
            Target point locations fixture.
        """
        result = DelaunayInterpolator(target_points)._interpolate(grid_stacked)

        # Barycentric interpolation is exact for any linear function
        assert result["temp"].isel(point_index=1).item() == pytest.approx(1.0)

    def test_output_has_point_index_dim(
        self, grid_stacked: xr.Dataset, target_points: xr.Dataset
    ) -> None:
        """Check that the output has a ``point_index`` dim.

        Parameters
        ----------
        grid_stacked : xr.Dataset
            Stacked grid fixture.
        target_points : xr.Dataset
            Target point locations fixture.
        """
        result = DelaunayInterpolator(target_points)._interpolate(grid_stacked)

        assert "point_index" in result.dims

    def test_output_has_latlon_coords_from_target(
        self, grid_stacked: xr.Dataset, target_points: xr.Dataset
    ) -> None:
        """Check that the output carries the target lat/lon coords.

        Parameters
        ----------
        grid_stacked : xr.Dataset
            Stacked grid fixture.
        target_points : xr.Dataset
            Target point locations fixture.
        """
        result = DelaunayInterpolator(target_points)._interpolate(grid_stacked)

        np.testing.assert_array_equal(
            result["latitude"].values, target_points["latitude"].values
        )
        np.testing.assert_array_equal(
            result["longitude"].values, target_points["longitude"].values
        )

    def test_weights_shared_across_instances(
        self,
        grid_stacked: xr.Dataset,
        target_points: xr.Dataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Check that interpolation weights are cached across interpolator instances.

        Parameters
        ----------
        grid_stacked : xr.Dataset
            Stacked grid fixture.
        target_points : xr.Dataset
            Target point locations fixture.
        monkeypatch : pytest.MonkeyPatch
            Used to reset the weights cache and count weight builds.
        """
        monkeypatch.setattr(delaunay, "_WEIGHTS_CACHE", {})
        calls = []
        build = delaunay._build_weight_matrix
        monkeypatch.setattr(
            delaunay,
            "_build_weight_matrix",
            lambda *args: calls.append(1) or build(*args),
        )

        # The Runner builds one interpolator per dataset
        DelaunayInterpolator(target_points)._interpolate(grid_stacked)
        DelaunayInterpolator(target_points)._interpolate(grid_stacked.copy())

        assert len(calls) == 1

    def test_cache_distinguishes_grids_with_same_extent(
        self,
        grid_stacked: xr.Dataset,
        target_points: xr.Dataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Check that the weights cache tells apart grids with the same extent.

        Parameters
        ----------
        grid_stacked : xr.Dataset
            Stacked grid fixture.
        target_points : xr.Dataset
            Target point locations fixture.
        monkeypatch : pytest.MonkeyPatch
            Used to reset the weights cache.
        """
        monkeypatch.setattr(delaunay, "_WEIGHTS_CACHE", {})
        # Same shape, first latitude and last longitude, different interior
        other = _stacked_grid(lats=[0.0, 1.0, 2.0], lons=[0.0, 1.5, 2.0])

        DelaunayInterpolator(target_points)._interpolate(grid_stacked)
        result = DelaunayInterpolator(target_points)._interpolate(other)

        assert result["temp"].isel(point_index=1).item() == pytest.approx(1.0)

    def test_rechunk_keeps_block_size_bounded(self, target_points: xr.Dataset) -> None:
        """Check that rechunking ``grid_index`` keeps blocks within the chunk budget.

        Parameters
        ----------
        target_points : xr.Dataset
            Target point locations fixture.
        """
        grid = _stacked_grid(lats=np.linspace(0, 2, 20), lons=np.linspace(0, 2, 50))
        n_steps, n_source = 64, grid.sizes["grid_index"]
        source = (
            grid.expand_dims(step=n_steps)
            .copy()
            .chunk({"step": n_steps, "grid_index": 100})
        )

        with dask.config.set({"array.chunk-size": "64KiB"}):
            result = DelaunayInterpolator(target_points)._interpolate(source)

        # Merging grid_index into one chunk must not grow blocks past the budget
        rows_per_block = max(result["temp"].chunks[0])
        assert rows_per_block * n_source * 8 <= 64 * 1024
        np.testing.assert_allclose(result["temp"].isel(step=0).values, [0.0, 1.0, 2.0])

    def test_outside_convex_hull_is_nan(self, grid_stacked: xr.Dataset) -> None:
        """Check that a point outside the convex hull is NaN.

        Parameters
        ----------
        grid_stacked : xr.Dataset
            Stacked grid fixture.
        """
        far_point = xr.Dataset(
            coords={
                "latitude": ("point_index", np.array([10.0])),
                "longitude": ("point_index", np.array([10.0])),
            },
        )
        result = DelaunayInterpolator(far_point)._interpolate(grid_stacked)

        assert np.isnan(result["temp"].isel(point_index=0).item())


# ---------------------------------------------------------------------------
# Shared interpolator behaviour
# ---------------------------------------------------------------------------

GRID_ATTRS = {
    "crs": ccrs.PlateCarree(),
    "grid_mapping": {"nx": 3, "ny": 3},
    "mlwp_space_trait": "grid",
    "mlwp_time_trait": "forecast",
    "mlwp_uncertainty_trait": "deterministic",
}


@pytest.mark.parametrize(
    "interpolator, source",
    [(XarrayInterpolator, "grid_latlon"), (DelaunayInterpolator, "grid_stacked")],
)
def test_point_output_drops_grid_attrs(
    interpolator: type[XarrayInterpolator] | type[DelaunayInterpolator],
    source: str,
    target_points: xr.Dataset,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """Check that point output drops grid-only attrs and can be saved to netCDF.

    Parameters
    ----------
    interpolator : type[XarrayInterpolator] | type[DelaunayInterpolator]
        Interpolator class under test.
    source : str
        Name of the source grid fixture.
    target_points : xr.Dataset
        Target point locations fixture.
    tmp_path : Path
        Temporary directory to write the output to.
    request : pytest.FixtureRequest
        Used to look up the source grid fixture by name.
    """
    source = request.getfixturevalue(source).assign_attrs(GRID_ATTRS)

    result = interpolator(target_points).interpolate(source)

    assert "crs" not in result.attrs and "grid_mapping" not in result.attrs
    assert result.attrs["mlwp_time_trait"] == "forecast"
    result.to_netcdf(tmp_path / "points.nc")  # a cartopy CRS attr would fail here


def test_interpolate_returns_same_container_type(
    grid_latlon: xr.Dataset, target_points: xr.Dataset
) -> None:
    """Check that ``interpolate`` returns the same container type it is given.

    Parameters
    ----------
    grid_latlon : xr.Dataset
        Regular lat/lon grid fixture.
    target_points : xr.Dataset
        Target point locations fixture.
    """
    out = interpolate(
        {"a": grid_latlon.assign_attrs(GRID_ATTRS)}, target_points, "xarray"
    )
    assert list(out) == ["a"]
