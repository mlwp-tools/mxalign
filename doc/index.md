# mxalign

**Align meteorological datasets in space and time, with plain `xarray`.**

Before you can compare a forecast with observations, an analysis or another
model, you have to bring them onto the same space and time: interpolate a grid
to station locations, line up forecast lead times with observation times, match
units. Every verification study repeats this step, usually with throwaway code.

`mxalign` does that step for you. Each dataset carries three small attributes
called [**traits**](concepts/traits.md): grid or points, forecast or
observation, deterministic or ensemble. From those `mxalign` works out how two
datasets should be aligned, so you don't have to spell it out:

```python
import mxalign  # registers the ds.mx accessor

# ds_fcst: gridded forecast, ds_obs: station observations
fcst_at_stations = ds_fcst.mx.align_space_with(ds_obs, method="delaunay")
obs_per_forecast = ds_obs.mx.align_time_with(fcst_at_stations)

error = fcst_at_stations["2t"] - obs_per_forecast["2t"]  # same dims, ready to score
```

Inputs and outputs are ordinary `xr.Dataset`s, and Dask-backed data stays lazy
until you call `.compute()`.

## What mxalign does

- **Spatial alignment**: interpolate gridded data to points (`xarray` or
  Delaunay-based interpolation), and check that two grids match.
- **Temporal alignment**: match forecasts with observations or with other
  forecasts, with control over which lead times are used.
- **Consistent missing data**: propagate NaNs across datasets so every
  dataset is scored on the same samples.
- **Transformations**: unit conversion, renaming, derived variables, or any
  function you register.
- **Scaling up**: a YAML-driven pipeline that loads, transforms and aligns
  data, then hands off to your metrics. It runs on a local Dask cluster or on
  Slurm.

## What mxalign is not

mxalign is **not a verification package**. It doesn't define scores. Once the
data is aligned you compute metrics with whatever you like: plain `xarray`,
[`xskillscore`](https://xskillscore.readthedocs.io), or your own functions.

It doesn't read file formats either. Loading is the job of
[`mlwp-data-loaders`](https://github.com/mlwp-tools/mlwp-data-loaders), and
when no loader exists for your data you write a short one yourself. See
[the mlwp-tools ecosystem](concepts/ecosystem.md).

## Where next

<div class="grid cards" markdown>

-   :material-download: **[Installation](getting-started/installation.md)**

    Install mxalign and its optional extras.

-   :material-rocket-launch: **[Quickstart](getting-started/quickstart.md)**

    Score a public ML forecast against stations in about 30 lines.

-   :material-lightbulb-on: **[Concepts](concepts/traits.md)**

    Traits, the ecosystem, and how alignment is worked out from them.

-   :material-api: **[API reference](reference/index.md)**

    Every public function and the `ds.mx` accessor.

</div>
