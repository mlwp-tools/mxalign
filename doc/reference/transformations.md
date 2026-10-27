# Transformations

::: mxalign.transformations.transform.transform

## Registry

::: mxalign.transformations.registry.available_transformations

::: mxalign.transformations.registry.register_transformation

::: mxalign.transformations.registry.get_transformation

## Built-in transformations

Pass the name in quotes as the first argument to `mx.transform`. The
documented signature is that of the registered function; `mx.transform`
supplies `ds` for you.

### `"rename"`

::: mxalign.transformations.base.transform_rename
    options:
      show_root_heading: false

### `"kelvin_to_celcius"`

::: mxalign.transformations.base.transform_kelvin_to_celcius
    options:
      show_root_heading: false

### `"uv_to_speed"`

::: mxalign.transformations.base.transform
    options:
      show_root_heading: false

### `"external"`

::: mxalign.transformations.external.transform
    options:
      show_root_heading: false
