# API reference

This reference is generated from the docstrings in the source code.

## Top-level functions

Everything below is importable from the `mxalign` package itself
(`import mxalign as mx`):

| Function | Purpose | Page |
|---|---|---|
| `mx.align_time(datasets, reference, lead_time=None)` | Align many datasets in time to a reference | [Alignment](align.md#mxalign.align.time.align_time) |
| `mx.align_space(datasets, reference, **kwargs)` | Align many datasets in space to a reference | [Alignment](align.md#mxalign.align.space.align_space) |
| `mx.interpolate(source_datasets, target_dataset, method)` | Interpolate one or more grids to a target's points | [Interpolations](interpolations.md#mxalign.interpolations.interpolate.interpolate) |
| `mx.available_interpolations()` | Names of registered interpolators | [Interpolations](interpolations.md#mxalign.interpolations.registry.available_interpolations) |
| `mx.register_interpolator` | Class decorator that registers an interpolator | [Interpolations](interpolations.md#mxalign.interpolations.registry.register_interpolator) |
| `mx.transform(name, datasets, ...)` | Apply a registered transformation | [Transformations](transformations.md#mxalign.transformations.transform.transform) |
| `mx.available_transformations()` | Names of registered transformations | [Transformations](transformations.md#mxalign.transformations.registry.available_transformations) |
| `mx.register_transformation(name)` | Decorator that registers a transformation | [Transformations](transformations.md#mxalign.transformations.registry.register_transformation) |

## Elsewhere

| Object | Import from | Page |
|---|---|---|
| `ds.mx` accessor | registered on `import mxalign` | [ds.mx accessor](accessor.md) |
| `broadcast_nans` | `mxalign.align.nans` | [Alignment](align.md#mxalign.align.nans.broadcast_nans) |
| `Runner` | `mxalign.runner` | [Runner](runner.md) |
| `update_space_trait`, `update_time_trait`, `update_uncertainty_trait` | `mxalign.utils.traits` | [Alignment](align.md#trait-helpers) |
