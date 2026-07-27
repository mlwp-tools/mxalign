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

import fnmatch
import xarray as xr

from .registry import register_transformation, register_expander
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

    # Use apply_ufunc so the computation is lazy when the input DataArray is
    # backed by dask (e.g. when transform_datasets runs on the full loaded
    # dataset before alignment).  The grid_dim is a core dimension passed as
    # the last axis to _compute_gradient; allow_rechunk ensures the spatial
    # axis is never split across chunks (required for the 2-D reshape).
    result = xr.apply_ufunc(
        _compute_gradient,
        da_t,
        kwargs=dict(axis_xy=axis_xy, ny=ny, nx=nx, dx=dx, dy=dy),
        input_core_dims=[[grid_dim]],
        output_core_dims=[[grid_dim]],
        dask="parallelized",
        output_dtypes=[da_t.dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    result.name = da.name
    result.attrs.update(da.attrs)
    return result.transpose(*da.dims)


def _expand_vars(patterns, ds_vars):
    """Expand a list of variable name patterns (may contain ``*`` / ``?`` globs)
    against the concrete variable names in ``ds_vars``.

    Literal names that contain no wildcards are kept as-is (and will cause an
    error later if they are absent from the dataset, which is the intended
    behaviour).  Glob patterns that match nothing are silently dropped.
    """
    result = []
    for p in (patterns if not isinstance(patterns, str) else [patterns]):
        if any(c in p for c in ("*", "?", "[")):
            result.extend(sorted(v for v in ds_vars if fnmatch.fnmatch(v, p)))
        else:
            result.append(p)
    return result


def _apply(ds: xr.Dataset, variables, outputs=None, axis_xy: str = "x", *,
           grid_dim: str = "grid_index",
           ny: int = _NY_DEFAULT, nx: int = _NX_DEFAULT,
           dx: float = _DX_DEFAULT, dy: float = _DY_DEFAULT) -> xr.Dataset:
    vs = _expand_vars(
        [variables] if isinstance(variables, str) else list(variables),
        list(ds.data_vars),
    )
    if outputs is None:
        suffix = f"_grad_{axis_xy}"
        os_ = [f"{v}{suffix}" for v in vs]
    else:
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


def _sig_cerra_grad(variables, outputs=None, **_):
    v = [variables] if isinstance(variables, str) else list(variables)
    if outputs is None:
        # Outputs can't be derived without the dataset when globs are present;
        # the expander will have resolved them to concrete names before this
        # is called by _derive_source_vars, so outputs will not be None there.
        raise ValueError(
            "cerra_gradient signature called with outputs=None; ensure the "
            "transformation expander ran before recording kwargs."
        )
    o = [outputs] if isinstance(outputs, str) else list(outputs)
    return v, o


def _make_expander(axis_xy: str):
    suffix = f"_grad_{axis_xy}"

    def expander(ds, kwargs: dict) -> dict:
        kw = dict(kwargs)
        vars_raw = kw.get("variables", [])
        expanded = _expand_vars(
            [vars_raw] if isinstance(vars_raw, str) else list(vars_raw),
            list(ds.data_vars),
        )
        kw["variables"] = expanded
        if kw.get("outputs") is None:
            kw["outputs"] = [f"{v}{suffix}" for v in expanded]
        return kw

    return expander


register_expander("cerra_gradient_x")(_make_expander("x"))
register_expander("cerra_gradient_y")(_make_expander("y"))


@register_transformation("cerra_gradient_x", signature=_sig_cerra_grad)
def transform_cerra_gradient_x(ds, variables, outputs=None, **grid_kwargs):
    return _apply(ds, variables, outputs, axis_xy="x", **grid_kwargs)


@register_transformation("cerra_gradient_y", signature=_sig_cerra_grad)
def transform_cerra_gradient_y(ds, variables, outputs=None, **grid_kwargs):
    return _apply(ds, variables, outputs, axis_xy="y", **grid_kwargs)
