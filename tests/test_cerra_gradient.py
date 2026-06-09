"""Unit tests for the CERRA gradient transformations.

These tests run without the real CERRA dataset by constructing a small
synthetic grid (still 1069x1069 to match the registered defaults) with
analytically-known gradients.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from mxalign.transformations import cerra  # noqa: F401 — registers transforms
from mxalign.transformations.registry import get_transformation
from mxalign.utils.projections import BUILTIN


GRID = BUILTIN["cerra"]["kws_grid"]
NY = int(GRID["ny"])
NX = int(GRID["nx"])
DX = float(GRID["dx"])
DY = float(GRID["dy"])
N = NY * NX


def _flat_field_from_image(image_2d: np.ndarray) -> np.ndarray:
    """Encode a 2D image (row 0 = North) as the 1D anemoi flatten order
    (row 0 of the raw reshape = South).
    """
    return image_2d[::-1, :].reshape(-1)


def _make_dataset(image_2d: np.ndarray, name: str = "f") -> xr.Dataset:
    flat = _flat_field_from_image(image_2d)
    da = xr.DataArray(flat[None, :], dims=("valid_time", "grid_index"))
    return xr.Dataset({name: da})


def test_cerra_gradient_x_linear_field():
    """f(col) = col * DX  =>  df/dx == 1 everywhere (also at the edges)."""
    cols = np.arange(NX, dtype=np.float64)
    image = np.broadcast_to(cols * DX, (NY, NX)).copy()
    ds = _make_dataset(image, name="f")

    fn = get_transformation("cerra_gradient_x")
    ds_out = fn(ds.copy(), variables=["f"], outputs=["fx"])

    gx_flat = ds_out["fx"].values[0]
    gx_image = gx_flat.reshape(NY, NX)[::-1, :]
    np.testing.assert_allclose(gx_image, np.ones_like(gx_image), atol=1e-9)


def test_cerra_gradient_y_linear_field_north_positive():
    """f(row_image) = (NY-1-row_image) * DY  -> values grow going North,
    so df/dy == 1 everywhere (also at the edges).
    """
    rows_image = np.arange(NY, dtype=np.float64)
    f_per_row = (NY - 1 - rows_image) * DY  # increases northward
    image = np.broadcast_to(f_per_row[:, None], (NY, NX)).copy()
    ds = _make_dataset(image, name="f")

    fn = get_transformation("cerra_gradient_y")
    ds_out = fn(ds.copy(), variables=["f"], outputs=["fy"])

    gy_flat = ds_out["fy"].values[0]
    gy_image = gy_flat.reshape(NY, NX)[::-1, :]
    np.testing.assert_allclose(gy_image, np.ones_like(gy_image), atol=1e-9)


def test_cerra_gradient_y_constant_in_x():
    """A field constant in the x-direction has zero x-gradient."""
    rows_image = np.arange(NY, dtype=np.float64)
    image = np.broadcast_to(rows_image[:, None] * DY, (NY, NX)).copy()
    ds = _make_dataset(image, name="f")

    fn = get_transformation("cerra_gradient_x")
    ds_out = fn(ds.copy(), variables=["f"], outputs=["fx"])
    assert np.max(np.abs(ds_out["fx"].values)) < 1e-9


def test_cerra_gradient_preserves_dims_and_coords():
    rng = np.random.default_rng(0)
    image = rng.standard_normal((NY, NX)).astype(np.float32)
    flat = _flat_field_from_image(image)
    times = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    da = xr.DataArray(
        np.stack([flat, flat]),
        dims=("valid_time", "grid_index"),
        coords={"valid_time": times},
        name="f",
    )
    ds = xr.Dataset({"f": da})

    fn = get_transformation("cerra_gradient_x")
    ds_out = fn(ds.copy(), variables=["f"], outputs=["fx"])

    assert ds_out["fx"].dims == ("valid_time", "grid_index")
    assert ds_out["fx"].shape == (2, N)
    np.testing.assert_array_equal(
        ds_out["fx"].coords["valid_time"].values, times
    )


def test_cerra_gradient_backend_parity(monkeypatch):
    """Numpy and torch-CUDA paths must produce identical numerical output.

    Skipped if torch+CUDA is not available.
    """
    torch_mod = pytest.importorskip("torch")
    if not torch_mod.cuda.is_available():
        pytest.skip("CUDA not available")

    rng = np.random.default_rng(1)
    image = rng.standard_normal((NY, NX)).astype(np.float32)
    ds = _make_dataset(image, name="f")

    fn = get_transformation("cerra_gradient_x")

    # GPU path (default since torch+CUDA is present).
    ds_gpu = fn(ds.copy(), variables=["f"], outputs=["fx"])

    # Force numpy path.
    monkeypatch.setattr(cerra, "_torch_cuda", lambda: None)
    ds_cpu = fn(ds.copy(), variables=["f"], outputs=["fx"])

    np.testing.assert_allclose(
        ds_gpu["fx"].values, ds_cpu["fx"].values, rtol=0, atol=1e-6
    )
