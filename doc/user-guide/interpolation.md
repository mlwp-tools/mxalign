# Interpolation methods

How the two built-in interpolators differ and when to use each: `"xarray"`
(wraps `xr.Dataset.interp`, needs a regular or projected grid) and
`"delaunay"` (barycentric weights on a Delaunay triangulation, works on any
flattened grid, and caches its weights). It will also cover
`mx.interpolate` for interpolating many datasets at once, and writing your
own interpolator with `register_interpolator`.

!!! note "Coming soon"
    This page is still being written. For now, see the
    [API reference](../reference/interpolations.md).
