"""CERRA-grid spatial-gradient transformations.

Registers ``cerra_gradient_x`` and ``cerra_gradient_y``: discrete first-order
gradients along the projection x and y axes of the CERRA LCC grid, using
central differences in the interior and one-sided differences at the borders.
Units are [variable] / m (projection-plane meters; no map-scale-factor
correction).

The transformation operates on a flat spatial dim (default ``grid_index``)
of length ``ny * nx``. It reshapes to 2D for the stencil and flattens back,
so the resulting variable has the same dims/coords as the input.

Backend selection (per call):

* If PyTorch + CUDA is available, the gradient runs on GPU.
* Otherwise it runs as a vectorised NumPy stencil.

Both backends use the exact same slicing expression and produce identical
numerical output (to floating-point round-off).
"""

from __future__ import annotations

import xarray as xr

from .registry import register_transformation
from ..utils.projections import BUILTIN

# Defaults sourced from the canonical CERRA grid description.
_GRID = BUILTIN["cerra"]["kws_grid"]
_NY_DEFAULT: int = int(_GRID["ny"])
_NX_DEFAULT: int = int(_GRID["nx"])
_DX_DEFAULT: float = float(_GRID["dx"])
_DY_DEFAULT: float = float(_GRID["dy"])


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


def _torch_cuda():
    """Return the ``torch`` module if CUDA is available, else ``None``.

    Import is lazy so the module remains importable without torch installed.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch


def _grad_axis(arr2d, axis_xy: str, dx: float, dy: float, empty_like):
    """Discrete gradient on a 2D array, central interior + one-sided edges.

    ``arr2d`` has shape ``(..., ny, nx)`` in image orientation (row 0 = North,
    col 0 = West). ``axis_xy`` is ``"x"`` (along columns / west-east) or
    ``"y"`` (along rows / south-north). ``empty_like`` is the array
    library's ``empty_like`` constructor; the same slicing expression works
    for NumPy and PyTorch tensors.
    """
    out = empty_like(arr2d)
    if axis_xy == "x":
        out[..., :, 1:-1] = (arr2d[..., :, 2:] - arr2d[..., :, :-2]) / (2.0 * dx)
        out[..., :, 0] = (arr2d[..., :, 1] - arr2d[..., :, 0]) / dx
        out[..., :, -1] = (arr2d[..., :, -1] - arr2d[..., :, -2]) / dx
    elif axis_xy == "y":
        # row 0 = North, so +y (north) corresponds to decreasing row index.
        out[..., 1:-1, :] = (arr2d[..., :-2, :] - arr2d[..., 2:, :]) / (2.0 * dy)
        out[..., 0, :] = (arr2d[..., 0, :] - arr2d[..., 1, :]) / dy
        out[..., -1, :] = (arr2d[..., -2, :] - arr2d[..., -1, :]) / dy
    else:
        raise ValueError(f"axis_xy must be 'x' or 'y', got {axis_xy!r}")
    return out


def _compute_gradient(arr_flat, axis_xy: str, ny: int, nx: int,
                      dx: float, dy: float):
    """Compute the gradient of a NumPy array shaped ``(..., ny*nx)``.

    Reshapes to 2D image orientation, dispatches to the GPU backend if
    available, and returns a NumPy array of the original shape.
    """
    import numpy as np

    arr = np.ascontiguousarray(arr_flat)
    lead_shape = arr.shape[:-1]
    arr2d_image = arr.reshape(*lead_shape, ny, nx)[..., ::-1, :]

    torch = _torch_cuda()
    if torch is not None:
        device = torch.device("cuda")
        t = torch.from_numpy(np.ascontiguousarray(arr2d_image)).to(
            device, non_blocking=True
        )
        out_t = _grad_axis(t, axis_xy, dx, dy, torch.empty_like)
        out2d_image = out_t.detach().cpu().numpy()
    else:
        out2d_image = _grad_axis(arr2d_image, axis_xy, dx, dy, np.empty_like)

    # Reverse the image flip and flatten back.
    out2d = out2d_image[..., ::-1, :]
    return np.ascontiguousarray(out2d).reshape(*lead_shape, ny * nx)


# ---------------------------------------------------------------------------
# DataArray wrapper
# ---------------------------------------------------------------------------


def _cerra_gradient_axis(da: xr.DataArray, axis_xy: str, *, grid_dim: str,
                         ny: int, nx: int, dx: float, dy: float) -> xr.DataArray:
    """Apply the gradient to a single ``DataArray`` and return a new one."""
    if grid_dim not in da.dims:
        raise ValueError(
            f"DataArray has no dim '{grid_dim}'. dims={da.dims}"
        )
    n_expected = ny * nx
    if da.sizes[grid_dim] != n_expected:
        raise ValueError(
            f"DataArray dim '{grid_dim}' has size {da.sizes[grid_dim]}, "
            f"expected ny*nx = {n_expected} (ny={ny}, nx={nx})."
        )

    da_t = da.transpose(..., grid_dim)
    result = _compute_gradient(da_t.values, axis_xy, ny, nx, dx, dy)
    out = xr.DataArray(
        result,
        dims=da_t.dims,
        coords={k: v for k, v in da_t.coords.items() if set(v.dims).issubset(da_t.dims)},
        name=da.name,
        attrs=dict(da.attrs),
    )
    return out.transpose(*da.dims)


def _apply(ds: xr.Dataset, variables, outputs, axis_xy: str, *,
           grid_dim: str = "grid_index",
           ny: int = _NY_DEFAULT, nx: int = _NX_DEFAULT,
           dx: float = _DX_DEFAULT, dy: float = _DY_DEFAULT) -> xr.Dataset:
    vs = [variables] if isinstance(variables, str) else list(variables)
    os_ = [outputs] if isinstance(outputs, str) else list(outputs)
    if len(vs) != len(os_):
        raise ValueError(
            f"variables and outputs must have the same length, "
            f"got {len(vs)} vs {len(os_)}."
        )
    for in_name, out_name in zip(vs, os_):
        ds[out_name] = _cerra_gradient_axis(
            ds[in_name], axis_xy,
            grid_dim=grid_dim, ny=ny, nx=nx, dx=dx, dy=dy,
        )
    return ds


# ---------------------------------------------------------------------------
# Registry entry points
# ---------------------------------------------------------------------------


def _sig_cerra_grad(variables, outputs, **_):
    v = [variables] if isinstance(variables, str) else list(variables)
    o = [outputs] if isinstance(outputs, str) else list(outputs)
    return v, o


@register_transformation("cerra_gradient_x", signature=_sig_cerra_grad)
def transform_cerra_gradient_x(ds, variables, outputs, **grid_kwargs):
    return _apply(ds, variables, outputs, axis_xy="x", **grid_kwargs)


@register_transformation("cerra_gradient_y", signature=_sig_cerra_grad)
def transform_cerra_gradient_y(ds, variables, outputs, **grid_kwargs):
    return _apply(ds, variables, outputs, axis_xy="y", **grid_kwargs)
