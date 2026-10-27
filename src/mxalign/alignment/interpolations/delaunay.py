"""Grid-to-point linear interpolator based on a Delaunay triangulation."""

import hashlib
import logging
from functools import partial

import numpy as np
import dask.array as dda
import xarray as xr

from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix

from .base import BaseInterpolator
from .registry import register_interpolator

from mlwp_data_specs.specs.traits.spatial_coordinate import Space

logger = logging.getLogger(__name__)

# Weight matrices keyed by a hash of the full source and target coordinates.
# Module-level so they are shared across interpolator instances (the Runner
# builds one per dataset); bounded so long sessions don't accumulate them.
_WEIGHTS_CACHE: dict[tuple[str, str], csr_matrix] = {}
_WEIGHTS_CACHE_SIZE = 8


@register_interpolator
class DelaunayInterpolator(BaseInterpolator):
    """Linear grid-to-point interpolation via Delaunay triangulation.

    The source grid points are triangulated in (latitude, longitude) and each
    target point gets barycentric weights from its enclosing triangle. The
    sparse weight matrix is cached per source/target coordinate pair. Target
    points outside the convex hull of the source grid get NaN.

    Parameters
    ----------
    target_dataset : xr.Dataset
        Point dataset with ``longitude`` and ``latitude`` coordinates.
    **options : object
        Interpolator options; only ``method="linear"`` (the default) is
        supported.

    Raises
    ------
    ValueError
        If ``method`` is anything other than ``"linear"``.
    """

    name = "delaunay"
    source_space = Space.GRID
    target_space = Space.POINT

    def __init__(self, target_dataset: xr.Dataset, **options: object) -> None:
        super().__init__(target_dataset, **options)
        method = self.options.get("method", "linear")
        if method != "linear":
            raise ValueError(
                f"Method: {method}. Delaunay interpolation only supports linear interpolation"
            )

    def _interpolate(self, source_dataset: xr.Dataset) -> xr.Dataset:
        """Interpolate a stacked grid dataset to the target points.

        Variables without a ``grid_index`` dimension are skipped. In the
        output, ``point_index`` takes the place of ``grid_index`` and the other
        dimensions keep their order.

        Parameters
        ----------
        source_dataset : xr.Dataset
            Grid dataset with a ``grid_index`` dimension and ``latitude``/
            ``longitude`` coordinates.

        Returns
        -------
        xr.Dataset
            The interpolated variables along ``point_index``, with the target
            ``latitude``/``longitude`` coordinates and the source attributes.

        Raises
        ------
        NotImplementedError
            If the source dataset has no ``grid_index`` dimension.
        """
        if "grid_index" not in source_dataset.dims:
            raise NotImplementedError(
                "Delaunay interpolation currently only supports stacked grids"
            )

        if "latitude" in source_dataset.dims:
            lon_grid, lat_grid = np.meshgrid(
                source_dataset["longitude"].values, source_dataset["latitude"].values
            )
            source_points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
        else:
            source_points = np.column_stack(
                (source_dataset["latitude"].values, source_dataset["longitude"].values)
            )

        target_points = np.column_stack(
            (
                self.target_dataset["latitude"].values,
                self.target_dataset["longitude"].values,
            )
        )

        # Compute triangulation and sparse weight matrix ONCE, shared across all variables
        W = _get_weights(source_points, target_points)

        arrays_out = {}
        for var in source_dataset.data_vars:
            da = source_dataset[var]
            if "grid_index" not in da.dims:
                logger.warning(
                    "Delaunay interpolation: skipping variable '%s' without a "
                    "grid_index dimension.",
                    var,
                )
                continue
            # The weight matmul needs grid_index last; put point_index back in
            # its place afterwards so e.g. a trailing member dim stays trailing
            out_dims = [("point_index" if d == "grid_index" else d) for d in da.dims]
            da = da.transpose(..., "grid_index")
            arrays_out[var] = interpolate_da(da, W, target_points).transpose(*out_dims)

        ds_out = xr.Dataset(arrays_out).assign_coords(
            latitude=self.target_dataset["latitude"],
            longitude=self.target_dataset["longitude"],
        )
        ds_out.attrs.update(source_dataset.attrs)
        return ds_out


def _fingerprint(points: np.ndarray) -> str:
    """Hash a full coordinate array (~20 ms for a 1M-point grid).

    Parameters
    ----------
    points : np.ndarray
        Coordinate array to hash.

    Returns
    -------
    str
        Hex digest of the array contents.
    """
    return hashlib.blake2b(np.ascontiguousarray(points), digest_size=16).hexdigest()


def _get_weights(source_points: np.ndarray, target_points: np.ndarray) -> csr_matrix:
    """Return the cached weight matrix for these points, building it on a miss.

    Parameters
    ----------
    source_points : np.ndarray
        Source coordinates, shape ``(n_source, 2)`` as (latitude, longitude).
    target_points : np.ndarray
        Target coordinates, shape ``(n_target, 2)`` as (latitude, longitude).

    Returns
    -------
    csr_matrix
        Sparse ``(n_target, n_source)`` barycentric weight matrix.
    """
    key = (_fingerprint(source_points), _fingerprint(target_points))
    if key not in _WEIGHTS_CACHE:
        if len(_WEIGHTS_CACHE) >= _WEIGHTS_CACHE_SIZE:
            _WEIGHTS_CACHE.pop(next(iter(_WEIGHTS_CACHE)))  # drop the oldest entry
        _WEIGHTS_CACHE[key] = _build_weight_matrix(
            Delaunay(source_points), source_points, target_points
        )
    return _WEIGHTS_CACHE[key]


def _build_weight_matrix(
    triangulation: Delaunay,
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> csr_matrix:
    """Build a sparse ``(n_target, n_source)`` barycentric weight matrix.

    Target points outside the convex hull of the source points get NaN
    weights.

    Parameters
    ----------
    triangulation : Delaunay
        Delaunay triangulation of ``source_points``.
    source_points : np.ndarray
        Source coordinates, shape ``(n_source, ndim)``.
    target_points : np.ndarray
        Target coordinates, shape ``(n_target, ndim)``.

    Returns
    -------
    csr_matrix
        Sparse ``(n_target, n_source)`` weight matrix.
    """
    print("Calculating interpolation-weight matrix")

    n_target = len(target_points)
    n_source = len(source_points)
    ndim = source_points.shape[1]  # 2 for lat/lon

    # Find which simplex each target point falls in; -1 means outside convex hull
    simplex_indices = triangulation.find_simplex(target_points)  # (n_target,)

    # Map outside points to simplex 0 temporarily to avoid index errors —
    # their weights will be NaN'd out below
    safe_indices = np.where(simplex_indices >= 0, simplex_indices, 0)

    # Vertices of each target point's simplex: (n_target, ndim+1)
    simplex_vertices = triangulation.simplices[safe_indices]

    # Recover barycentric coordinates using the affine transforms stored in
    # triangulation.transform: shape (nsimplex, ndim+1, ndim)
    #   transform[s, :ndim, :] — inverse of the edge matrix for simplex s
    #   transform[s,  ndim, :] — the ndim-th vertex (origin) of simplex s
    Tinv = triangulation.transform[safe_indices, :ndim, :]  # (n_target, ndim, ndim)
    origin = triangulation.transform[safe_indices, ndim, :]  # (n_target, ndim)

    r = target_points - origin  # (n_target, ndim)
    bary_partial = np.einsum("nij,nj->ni", Tinv, r)  # (n_target, ndim)
    last = 1.0 - bary_partial.sum(axis=1, keepdims=True)
    bary = np.concatenate([bary_partial, last], axis=1)  # (n_target, ndim+1)

    # Flatten into coordinate format (COO) for sparse matrix construction
    rows = np.repeat(np.arange(n_target), ndim + 1)
    cols = simplex_vertices.ravel()
    vals = bary.ravel()

    # NaN out weights for points outside the convex hull
    outside = simplex_indices == -1
    vals[np.repeat(outside, ndim + 1)] = np.nan

    W = csr_matrix((vals, (rows, cols)), shape=(n_target, n_source))

    print("Done")

    return W


def interpolate_da(
    da: xr.DataArray, W: csr_matrix, target_points: np.ndarray
) -> xr.DataArray:
    """Apply weight matrix ``W`` to ``da`` over its ``grid_index`` dimension.

    The computation is lazy (``map_blocks``); dask-backed inputs whose
    ``grid_index`` is split over several chunks are rechunked to one chunk.

    Parameters
    ----------
    da : xr.DataArray
        Source data whose last dimension is ``grid_index``.
    W : csr_matrix
        Sparse ``(n_target, n_source)`` weight matrix.
    target_points : np.ndarray
        Target coordinates, shape ``(n_target, 2)``.

    Returns
    -------
    xr.DataArray
        Interpolated data with ``grid_index`` replaced by ``point_index``.
    """
    n_target = len(target_points)
    leading_dims = da.dims[:-1]

    # Delaunay needs the full source axis per block (every target point pulls
    # from every source point via the sparse weight matrix). If grid_index is
    # split across multiple chunks, rechunk to a single chunk before the matmul,
    # and let dask re-chunk the leading dims ("auto") so blocks stay near
    # ``array.chunk-size`` instead of growing by the number of grid chunks.
    # ``unify_chunks`` then reconciles any leftover mismatch between the data
    # array and its coords (e.g. when ``chunks="auto"`` produced different
    # chunkings for reference_time on data vs. coord).
    if isinstance(da.data, dda.Array):
        grid_chunks = dict(zip(da.dims, da.chunks)).get("grid_index")
        if grid_chunks is not None and len(grid_chunks) > 1:
            logger.warning(
                "Delaunay interpolation: rechunking grid_index from %d chunks to 1 "
                "(required for the sparse weight matmul).",
                len(grid_chunks),
            )
            da = da.chunk({"grid_index": -1, **{d: "auto" for d in leading_dims}})
        da = da.unify_chunks()

    # Build the template
    # Get chunking info for leading dims
    shape_tmp = tuple(da.sizes[d] for d in leading_dims) + (n_target,)

    if isinstance(da.data, dda.Array):
        dim_to_chunks = dict(zip(da.dims, da.chunks))
    else:
        dim_to_chunks = {dim: (da.sizes[dim],) for dim in da.dims}

    chunks_tmp = tuple(
        dim_to_chunks[dim] if dim in dim_to_chunks else (da.sizes[dim],)
        for dim in leading_dims
    ) + ((n_target,),)

    # Create a dask array template matching the chunking pattern
    tmp = dda.empty(shape=shape_tmp, chunks=chunks_tmp, dtype=da.dtype)
    tmp = xr.DataArray(
        tmp,
        dims=leading_dims + ("point_index",),
        coords={d: da.coords[d].load() for d in leading_dims},
    )

    # Drop coords tied to grid_index to avoid dimension mismatch in map_blocks
    spatial_coords = [c for c in da.coords if "grid_index" in da[c].dims]
    da_clean = da.drop_vars(spatial_coords)

    da_interp = da_clean.map_blocks(
        partial(interpolate_block, W=W, target_points=target_points), template=tmp
    )

    return da_interp


def interpolate_block(
    block: xr.DataArray,
    W: csr_matrix,
    target_points: np.ndarray,
) -> xr.DataArray:
    """Apply the sparse weight matrix to a single block.

    Used by ``map_blocks`` inside ``interpolate_da``.

    Parameters
    ----------
    block : xr.DataArray
        Block of source data whose last dimension is ``grid_index``.
    W : csr_matrix
        Sparse ``(n_target, n_source)`` weight matrix.
    target_points : np.ndarray
        Target coordinates, shape ``(n_target, 2)``.

    Returns
    -------
    xr.DataArray
        Interpolated block with the last dimension replaced by
        ``point_index``.
    """
    data = block.values  # shape = (.., npoints)
    original_shape = data.shape[:-1]
    data_flat = data.reshape(
        -1, data.shape[-1]
    )  # shape = (ndim1 * ndim2 * ... , npoints)

    # Identify NaN source points
    nan_mask = np.isnan(data_flat)  # (nleading, n_source)

    if nan_mask.any():
        print(f"Warning, interpolating NaNs for variable {block.name}")

    # Single sparse matrix multiply replaces the per-row interpolator loop:
    # (nleading, n_source) @ (n_source, n_target) -> (nleading, n_target)
    interpolated_flat = data_flat @ W.T
    interpolated = interpolated_flat.reshape(*original_shape, target_points.shape[0])

    new_dims = block.dims[:-1] + ("point_index",)
    new_coords = {dim: block.coords[dim] for dim in block.dims[:-1]}
    return xr.DataArray(interpolated, dims=new_dims, coords=new_coords)
