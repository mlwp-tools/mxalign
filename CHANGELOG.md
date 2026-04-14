# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-14

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
