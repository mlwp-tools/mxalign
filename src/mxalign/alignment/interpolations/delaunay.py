"""Grid-to-point linear interpolator based on a Delaunay triangulation."""

import hashlib
import logging
from functools import partial

import numpy as np
import dask.array as dda
import xarray as xr

from scipy.spatial import ConvexHull, cKDTree
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

# A target counts as inside a triangle when its most negative coordinate is
# within this fraction of the coordinates' total (points on an edge).
_INSIDE_TOLERANCE = 1e-12
# A hull facet is part of the triangulation of the sphere when the origin is
# at least this far inside its plane. A triangle on the sphere is only closer
# when its circumcircle is nearly a great circle.
_PLANE_OFFSET_TOLERANCE = 1e-6
# Walks from the triangle with the nearest centre mostly take a few steps, but
# can cross a long fan of thin triangles (qhull's split of a cocircular row of
# grid points). This only bounds a walk that doesn't converge, which then
# falls back to a full search.
_MAX_WALK_STEPS = 4096


@register_interpolator
class DelaunayInterpolator(BaseInterpolator):
    """Linear grid-to-point interpolation via Delaunay triangulation.

    The source grid points are triangulated on the sphere and each target
    point gets barycentric weights from its enclosing triangle, so the source
    and target may use different longitude conventions (0–360 or
    [-180, 180)). The sparse weight matrix is cached per source/target
    coordinate pair. Target points outside the spherical convex hull of the
    source grid get NaN.

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

        Variables whose last dimension is not ``grid_index`` are skipped.

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
        _WEIGHTS_CACHE[key] = _build_weight_matrix(source_points, target_points)
    return _WEIGHTS_CACHE[key]


def _to_unit_vectors(points: np.ndarray) -> np.ndarray:
    """Convert (latitude, longitude) in degrees to 3-D unit vectors.

    Parameters
    ----------
    points : np.ndarray
        Coordinates, shape ``(n, 2)`` as (latitude, longitude) in degrees.

    Returns
    -------
    np.ndarray
        Unit vectors, shape ``(n, 3)``.
    """
    lat, lon = np.radians(points[:, 0]), np.radians(points[:, 1])
    return np.column_stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))
    )


def _cone_coordinates(corners: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Solve ``targets = w @ corners`` for each target and its triangle.

    All three coordinates are non-negative exactly when the target direction
    lies in the cone that the origin and the triangle span. Divided by their
    sum, they are the barycentric coordinates of the point where that
    direction crosses the triangle.

    Parameters
    ----------
    corners : np.ndarray
        Triangle corners, shape ``(n, 3, 3)``: ``corners[i, k]`` is corner
        ``k`` of target ``i``'s triangle.
    targets : np.ndarray
        Target unit vectors, shape ``(n, 3)``.

    Returns
    -------
    np.ndarray
        Coordinates, shape ``(n, 3)``; NaN for a degenerate triangle.
    """
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    w = np.column_stack(
        (
            np.einsum("ij,ij->i", targets, np.cross(b, c)),
            np.einsum("ij,ij->i", a, np.cross(targets, c)),
            np.einsum("ij,ij->i", a, np.cross(b, targets)),
        )
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return w / det[:, np.newaxis]


def _build_weight_matrix(
    source_points: np.ndarray, target_points: np.ndarray
) -> csr_matrix:
    """Build a sparse ``(n_target, n_source)`` barycentric weight matrix.

    The source points are triangulated on the unit sphere: the convex hull of
    their 3-D unit vectors is their spherical Delaunay triangulation. A
    target's weights are the barycentric coordinates of the point where its
    direction crosses the triangle it falls in. This doesn't depend on the
    longitude convention of either dataset and has no seam; the poles need no
    special case.

    Each target's triangle is found by a walk over neighbouring triangles,
    starting at the triangle with the nearest centre. Targets outside the
    spherical convex hull of the source points get NaN weights. A source that
    doesn't fit in a hemisphere covers the whole sphere, so a hole in it (e.g.
    a missing polar cap) is bridged rather than left NaN.

    Parameters
    ----------
    source_points : np.ndarray
        Source coordinates, shape ``(n_source, 2)`` as (latitude, longitude)
        in degrees.
    target_points : np.ndarray
        Target coordinates, shape ``(n_target, 2)`` as (latitude, longitude)
        in degrees.

    Returns
    -------
    csr_matrix
        Sparse ``(n_target, n_source)`` weight matrix.
    """
    logger.info("Calculating interpolation-weight matrix")

    n_target = len(target_points)
    n_source = len(source_points)
    source = _to_unit_vectors(source_points)
    target = _to_unit_vectors(target_points)

    hull = ConvexHull(source)
    triangles = hull.simplices  # (n_triangle, 3) source indices
    # Hull facets with the origin on or beyond their plane close the hull of a
    # regional grid from below; they're not part of the triangulation of the
    # sphere. ``equations`` holds (normal, offset), with the inside negative.
    # The margin catches lid facets along a meridian, whose plane holds the
    # origin up to rounding.
    on_sphere = hull.equations[:, 3] < -_PLANE_OFFSET_TOLERANCE

    found = np.full(n_target, -1, dtype=np.intp)
    coords = np.full((n_target, 3), np.nan)
    active = np.arange(n_target)

    # The walk's edge tests only hold for targets less than 90° from the
    # triangles. For a regional source, drop targets outside a cap around it:
    # they are outside the triangulation, and the rest are close enough. This
    # also spares the nearest-centre search below its slow far-away queries.
    if not on_sphere.all():
        centre = source.mean(axis=0)
        centre /= np.linalg.norm(centre)
        cos_radius = (source @ centre).min()
        if cos_radius > 0:
            active = active[target @ centre >= cos_radius - _INSIDE_TOLERANCE]

    # Start each walk at the triangle on the sphere with the nearest centre
    sphere_triangles = np.flatnonzero(on_sphere)
    centres = source[triangles[sphere_triangles]].mean(axis=1)
    current = sphere_triangles[cKDTree(centres).query(target[active])[1]]

    for _ in range(_MAX_WALK_STEPS):
        w = _cone_coordinates(source[triangles[current]], target[active])
        k = np.argmin(np.where(np.isnan(w), np.inf, w), axis=1)
        inside = w[np.arange(len(active)), k] >= -_INSIDE_TOLERANCE * np.abs(w).sum(
            axis=1
        )
        found[active[inside]] = current[inside]
        coords[active[inside]] = w[inside]
        # Step across the edge opposite the most negative coordinate. Leaving
        # the triangulation of the sphere means the target is outside it.
        step = hull.neighbors[current[~inside], k[~inside]]
        keep = on_sphere[step]
        active, current = active[~inside][keep], step[keep]
        if active.size == 0:
            break
    else:
        logger.warning(
            "Delaunay interpolation: %d target points not located by walking, "
            "searching all triangles.",
            active.size,
        )
        corners = source[triangles[sphere_triangles]]
        for i in active:
            w = _cone_coordinates(
                corners, np.broadcast_to(target[i], (len(sphere_triangles), 3))
            )
            hit = np.flatnonzero(
                w.min(axis=1) >= -_INSIDE_TOLERANCE * np.abs(w).sum(axis=1)
            )
            if hit.size:
                found[i] = sphere_triangles[hit[0]]
                coords[i] = w[hit[0]]

    located = found >= 0

    bary = coords / coords.sum(axis=1, keepdims=True)
    bary[~located] = np.nan
    cols = triangles[np.where(located, found, 0)]

    # Flatten into coordinate format (COO) for sparse matrix construction
    rows = np.repeat(np.arange(n_target), 3)
    W = csr_matrix((bary.ravel(), (rows, cols.ravel())), shape=(n_target, n_source))

    logger.info("Done")

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
