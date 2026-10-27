# Traits

An `xr.Dataset` on its own doesn't say whether it holds a forecast or
observations, a grid or stations, one realisation or fifty. **Traits** pin
that down. They are three dataset attributes, one per axis:

| Axis | Attribute | Values |
|---|---|---|
| **Space** | `mlwp_space_trait` | `grid` · `point` |
| **Time** | `mlwp_time_trait` | `forecast` · `observation` |
| **Uncertainty** | `mlwp_uncertainty_trait` | `deterministic` · `ensemble` · `quantile` |

```python
ds.attrs
# {'mlwp_space_trait': 'grid',
#  'mlwp_time_trait': 'forecast',
#  'mlwp_uncertainty_trait': 'deterministic'}
```

Traits are defined by
[`mlwp-data-specs`](https://github.com/mlwp-tools/mlwp-data-specs) and are
usually set by a [loader](ecosystem.md#bring-your-own-loader). mxalign reads
the space and time traits to decide how to align two datasets. See
[How alignment works](alignment.md).

## Each trait is a contract

A trait does more than label a dataset. It also requires certain dimensions,
coordinates and coordinate metadata. A dataset that claims a trait but breaks
its contract fails validation.

### Space

| Trait | Spatial dimension(s) | Required coordinates |
|---|---|---|
| `grid` | one of: `grid_index` (flattened), `(latitude, longitude)` (regular lat/lon), `(yc, xc)` (projected) | `latitude`, `longitude` |
| `point` | `point_index` | `latitude`, `longitude` on `point_index` |

`latitude` and `longitude` need CF metadata: `standard_name` of `latitude` /
`longitude` and `units` of `degrees_north` / `degrees_east`. Projected `xc` /
`yc` coordinates, when present, need projection-coordinate metadata.

### Time

| Trait | Time dimension(s) | Required coordinates and metadata |
|---|---|---|
| `forecast` | `(reference_time, lead_time)` | `reference_time` with `standard_name` `forecast_reference_time` or `time`. `lead_time` with `standard_name` `forecast_period` and `units` in `s`, `seconds`, `h` or `hours`. An optional `valid_time` coordinate must have `standard_name` `time`. |
| `observation` | `valid_time` | `valid_time` with `standard_name` `time` |

A forecast's valid time is `reference_time + lead_time`. Use
`ds.mx.add_valid_time()` to add it as a 2-D `valid_time` coordinate.

### Uncertainty

| Trait | Extra dimension | Required coordinate |
|---|---|---|
| `deterministic` | none | none |
| `ensemble` | `member` | `member` with `standard_name` `realization` |
| `quantile` | `quantile` | `quantile` with `standard_name` `quantile`, `units` `1`, values in `[0, 1]` |

## Typical layouts

Putting the axes together, these are the shapes you will meet most often:

| Dataset | Traits | Dimensions |
|---|---|---|
| NWP / ML forecast | grid · forecast · deterministic | `(reference_time, lead_time, grid_index)` |
| Ensemble forecast | grid · forecast · ensemble | `(reference_time, lead_time, member, grid_index)` |
| Analysis / reanalysis | grid · observation · deterministic | `(valid_time, grid_index)` |
| Station observations | point · observation · deterministic | `(valid_time, point_index)` |
| Forecast at stations, after spatial alignment | point · forecast · deterministic | `(reference_time, lead_time, point_index)` |

An analysis or reanalysis has the `observation` time trait. In time it behaves
like observations, a single value per valid time, even though it is a grid in
space.

## Traits change as you align

Alignment returns datasets with updated traits. Interpolating a grid to
stations turns `grid` into `point`. Aligning observations to a forecast gives
them the forecast's `(reference_time, lead_time)` layout and the `forecast`
time trait. The output is therefore always a valid input for the next step.

## What mxalign uses today

- **Space** and **time** traits drive alignment.
- The **uncertainty** trait is carried through unchanged: ensemble `member`
  and `quantile` dimensions travel along like any other dimension. Alignment
  *along* the uncertainty axis, such as ensemble against deterministic, is
  planned but not implemented.

## Setting traits yourself

The usual way to get traits is a loader. When you build or modify a dataset
yourself, you can set the attributes directly:

```python
ds.attrs.update(
    mlwp_space_trait="point",
    mlwp_time_trait="observation",
    mlwp_uncertainty_trait="deterministic",
)
```

and check the result against the contract:

```python
from mlwp_data_specs.api import validate_dataset

report = validate_dataset(ds)
report.has_fails()        # True if any requirement is broken
report.console_print()    # per-check details
```

`mxalign.utils.traits` also provides `update_space_trait`,
`update_time_trait` and `update_uncertainty_trait`. They set the attribute
from the `mlwp-data-specs` enums (`Space.POINT`, `Time.FORECAST`, …), which
avoids typos in the trait values. They do not reject an invalid dataset, so
call `validate_dataset` when you need that check.
