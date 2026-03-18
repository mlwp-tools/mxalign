from functools import partial

import numpy as np
import dask.array as dda
import xarray as xr

from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix

from .base import BaseInterpolator
from .registry import register_interpolator
from ..properties.properties import Space


@register_interpolator
class DelaunayInterpolator(BaseInterpolator):
    name = "delaunay"
    source_space = Space.GRID
    target_space = Space.POINT

    def __init__(self, target_dataset, **options):
        super().__init__(target_dataset, **options)
        method = self.options.get("method", "linear")
        self._W_cache = {}  # keyed by source grid hash
        if method != "linear":
            raise ValueError(
                f"Method: {method}. Delaunay interpolation only supports linear interpolation"
            )

    def _get_weights(self, source_points, target_points):
        key = (
            source_points.shape,
            source_points[0, 0],
            source_points[-1, 1],
        )  # cheap fingerprint
        if key not in self._W_cache:
            triangulation = Delaunay(source_points)
            self._W_cache[key] = _build_weight_matrix(
                triangulation, source_points, target_points
            )
        return self._W_cache[key]

    def _interpolate(self, source_dataset):
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
        W = self._get_weights(source_points, target_points)

        arrays_out = {}
        for var in source_dataset.data_vars:
            da = source_dataset[var]
            if da.dims[-1] != "grid_index":
                print(
                    f"Skipping variable '{var}' - doesn't end with spatial dimension grid_index"
                )
                continue
            else:
                arrays_out[var] = interpolate_da(da, W, target_points)

        ds_out = xr.Dataset(arrays_out).assign_coords(
            latitude=self.target_dataset["latitude"],
            longitude=self.target_dataset["longitude"],
        )
        ds_out.attrs["properties"] = source_dataset.attrs["properties"]
        return ds_out


def _build_weight_matrix(
    triangulation: Delaunay,
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> csr_matrix:
    """
    Precompute a sparse (n_target, n_source) weight matrix from the triangulation.

    Applying W to a (n_source,) value vector gives (n_target,) interpolated values
    via a simple sparse matrix multiply. Target points outside the convex hull
    receive NaN weights.
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
    n_target = len(target_points)
    leading_dims = da.dims[:-1]

    # Validate that grid_index is not chunked
    if isinstance(da.data, dda.Array):
        grid_chunks = dict(zip(da.dims, da.chunks)).get("grid_index")
        if grid_chunks is not None and len(grid_chunks) > 1:
            raise ValueError(
                f"grid_index must not be chunked for Delaunay interpolation "
                f"(found {len(grid_chunks)} chunks). Rechunk with da.chunk({{'grid_index': -1}}) "
                f"or enforce this on the loading side."
            )

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
