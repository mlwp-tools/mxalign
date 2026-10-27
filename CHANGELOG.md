# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Breaking:** the pipeline config is validated against an explicit pydantic schema (`mxalign.pipeline.config.PipelineConfig`, loaded with `load_config`) instead of being read as a free-form dict. Unknown sections or keys, and missing required keys, raise a `ValidationError` when the `Runner` is created; `Runner.config` is now the validated model. Per-dataset `dates` entries override individual fields of the top-level `dates`. @leifdenby
- **Breaking:** file path patterns (`datasets.<name>.files` with a `dates:` section, and `alignment.save.path`) now use Python `str.format` syntax instead of earthkit-data patterns: `{reference_time:%Y%m%d%H}` instead of `{reference_time:strftime(%Y%m%d%H)}`, `{year:02d}` instead of `{year:int(%02d)}`. `{lead_time}` (integer hours) and `{valid_time}` are now supported in `files` patterns. `earthkit-data` is no longer a dependency. @michielv
- **Breaking:** temporal alignment always runs in the pipeline; `alignment.time` is optional and only accepts `lead_time` (`reference` | `intersection` | `union`, forecast reference only). Unknown keys such as `return_as` or `method` now raise an error. `align_time` takes `reference` and an explicit `lead_time` instead of `**kwargs`. @michielv
- Remove dataset loading and dataset validation and outsource it to [mlwp-data-loaders](https://github.com/mlwp-tools/mlwp-data-loaders) and [mlwp-data-specs](https://github.com/mlwp-tools/mlwp-data-specs) @michielv
- Store dataset properties as netCDF-safe individual attributes while keeping read compatibility with legacy `attrs["properties"]` dict/JSON data. [\#21](https://github.com/mlwp-tools/mxalign/pull/21) @observingClouds
- Added CI test workflow with first unit tests. [\#21](https://github.com/mlwp-tools/mxalign/pull/21) @observingClouds
- Added optional `ifs` dependency group with `cfgrib`, `eccodes`, and `eccodeslib`. [\#21](https://github.com/mlwp-tools/mxalign/pull/21) @observingClouds
- Added CI action for package build and upload to pypi.org on releases. [\#28](https://github.com/mlwp-tools/mxalign/pull/28) @leifdenby
- `transform()` and `interpolate()` return the same container type they are given: a one-item list now stays a list instead of being unwrapped to a bare dataset, and a dict with a single dataset no longer raises `KeyError`. @michielv
- Delaunay interpolation weights are shared across datasets on the same grid, so aligning several models to the same points triangulates the grid once. When a multi-chunk `grid_index` is merged into a single chunk, the leading dimensions are re-chunked automatically so blocks stay near dask's `array.chunk-size`. @michielv

### Fixed
- `ds.mx.add_xy(crs=...)` and `ds.mx.unstack(crs=...)` no longer replace the dataset held by the cached `ds.mx` accessor, so later `ds.mx` calls work on `ds` itself instead of a copy with the CRS attached. @michielv
- Interpolated point datasets no longer inherit the source grid's `crs` and `grid_mapping` attributes, so they can be saved with `to_netcdf`. @michielv
- Saving aligned datasets with `alignment.save.datasets: merge` no longer fails with `UnboundLocalError`; the merged output fills `{name}` in the path with `merged`. @michielv
- A pipeline config without a `transformations:` section no longer crashes. @michielv
- `ds.mx.stack()` now stacks regular grids with `latitude`/`longitude` dimensions (it looked for `lat`/`lon`). @michielv
- Delaunay interpolation could silently reuse the weights of a different grid with the same shape and corner coordinates, giving wrong values; weights are now cached by a hash of the full source and target coordinates. @michielv

## [0.1.0](https://github.com/mlwp-tools/mxalign/releases/tag/v0.1.0)

First release of `mxalign`, an xarray-based package for alignment of meteorological datasets, with the following functionality and configuration:

- Execution of the verification tooling pipeline.
- Core alignment capabilities (handling of space, time, and NaNs).
- Base data loaders, including support for Anemoi datasets and inference.
- Validation functionality for datasets.
- Implementations for interpolations (e.g., Delaunay, xarray).
- Accessors (space, time), transformations, and properties management tools.
- Introductory Jupyter notebook (`examples/introduction.ipynb`) demonstrating interactive usage.
- Integration of `uv.lock` dependency locking to ensure specific versions are used in testing, allowing for safe and reliable releases.
- Dependabot configuration with a strategy where PRs for flexible requirements will only be merged once tests confirm that `mxalign` works correctly with the newer upstream packages.
