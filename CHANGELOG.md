# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Remove dataset loading and dataset validation and outsource it to [mlwp-data-loaders](https://github.com/mlwp-tools/mlwp-data-loaders) and [mlwp-data-specs](https://github.com/mlwp-tools/mlwp-data-specs) @michielv
- Store dataset properties as netCDF-safe individual attributes while keeping read compatibility with legacy `attrs["properties"]` dict/JSON data. [\#21](https://github.com/mlwp-tools/mxalign/pull/21) @observingClouds
- Added CI test workflow with first unit tests. [\#21](https://github.com/mlwp-tools/mxalign/pull/21) @observingClouds
- Added optional `ifs` dependency group with `cfgrib`, `eccodes`, and `eccodeslib`. [\#21](https://github.com/mlwp-tools/mxalign/pull/21) @observingClouds
- Added CI action for package build and upload to pypi.org on releases. [\#28](https://github.com/mlwp-tools/mxalign/pull/28) @leifdenby

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
