# Quickstart

This page scores a machine-learning weather forecast against weather stations,
from loading to an RMSE per lead time. Every step runs on public data, so you
can paste the code blocks into a notebook in order. The full run takes well
under a minute.

You will:

1. load a gridded forecast with traits attached,
2. describe a handful of stations as a *point observation* dataset,
3. interpolate the forecast to the stations (**space**),
4. line the observations up with each forecast (**time**),
5. convert units and compute a score.

## 1. Load a forecast

The sample data are two runs (2020-02-01 and 2020-02-02 at 00 UTC) of an
[anemoi](https://anemoi.readthedocs.io) limited-area model over Europe. They
are stored on ECMWF's public object store. `mlwp-data-loaders` ships a loader
for anemoi-inference output, which reads the files and attaches the
[traits](../concepts/traits.md):

```python
from mlwp_data_loaders import load_and_validate_dataset

STORAGE_OPTS = {
    "endpoint_url": "https://object-store.os-api.cci2.ecmwf.int",
    "anon": True,
}
BASE = "s3://mlwp-sample-datasets/anemoi-inference/unknown-revision"

ds_fcst = load_and_validate_dataset(
    [
        f"{BASE}/anemoi-inference-lam_2020020100.nc",
        f"{BASE}/anemoi-inference-lam_2020020200.nc",
    ],
    loader="mlwp_data_loaders.loaders.anemoi.anemoi_inference",
    storage_options=STORAGE_OPTS,
)
ds_fcst
```

```text
<xarray.Dataset> Size: 932MB
Dimensions:         (reference_time: 2, lead_time: 25, grid_index: 1142761)
Coordinates:
  * reference_time  (reference_time) datetime64[ns] 2020-02-01 2020-02-02
  * lead_time       (lead_time) timedelta64[ns] 0 days 00:00:00 ... 6 days
    latitude        (grid_index) float64 ...
    longitude       (grid_index) float64 ...
Data variables:
    10u             (reference_time, lead_time, grid_index) float64 dask.array<...>
    2t              (reference_time, lead_time, grid_index) float64 dask.array<...>
Attributes:
    mlwp_time_trait:         Time.FORECAST
    mlwp_space_trait:        Space.GRID
    mlwp_uncertainty_trait:  Uncertainty.DETERMINISTIC
```

The data are opened lazily with Dask; nothing is downloaded yet apart from
the coordinates. The three `mlwp_*_trait` attributes say what this dataset
*is*: a deterministic forecast on a grid.

!!! tip "Quieter validation"
    `mlwp-data-specs` logs every check it runs at `INFO` level. To silence
    it, run `from loguru import logger; logger.disable("mlwp_data_specs")`.

## 2. Describe the stations

Normally your observations come from a loader as well, either one shipped
with `mlwp-data-loaders` or [one you write yourself](../concepts/ecosystem.md#bring-your-own-loader).
To keep this page self-contained, we build a small point-observation dataset
by hand: six stations, 6-hourly 2 m temperature in °C.

!!! note
    The temperatures below are **random numbers**. The point is to show the
    mechanics, so the scores at the end mean nothing.

```python
import numpy as np
import pandas as pd
import xarray as xr

stations = {
    # name: (latitude, longitude)
    "Uccle": (50.80, 4.36),
    "De Bilt": (52.10, 5.18),
    "Paris": (48.82, 2.34),
    "Frankfurt": (50.03, 8.57),
    "Zurich": (47.38, 8.57),
    "Vienna": (48.25, 16.36),
}
valid_time = pd.date_range("2020-02-01", "2020-02-08", freq="6h")

rng = np.random.default_rng(0)
t2m = 5 + rng.normal(0, 3, size=(len(valid_time), len(stations)))  # °C, made up

ds_obs = xr.Dataset(
    {"2t": (("valid_time", "point_index"), t2m)},
    coords={
        "valid_time": ("valid_time", valid_time, {"standard_name": "time"}),
        "point_index": list(stations),
        "latitude": (
            "point_index",
            [lat for lat, _ in stations.values()],
            {"standard_name": "latitude", "units": "degrees_north"},
        ),
        "longitude": (
            "point_index",
            [lon for _, lon in stations.values()],
            {"standard_name": "longitude", "units": "degrees_east"},
        ),
    },
    attrs={
        "mlwp_space_trait": "point",
        "mlwp_time_trait": "observation",
        "mlwp_uncertainty_trait": "deterministic",
    },
)
```

That is all a dataset needs to take part in alignment: the dimension names
and coordinate metadata required by its traits, and the three trait
attributes. You can check it against the contract:

```python
from mlwp_data_specs.api import validate_dataset

validate_dataset(ds_obs).has_fails()  # False
```

## 3. Align in space

Importing `mxalign` registers the [`ds.mx` accessor](../user-guide/accessor.md)
on every `xr.Dataset`. It reads the traits:

```python
import mxalign as mx

ds_fcst.mx.is_grid(), ds_fcst.mx.is_forecast()      # (True, True)
ds_obs.mx.is_point(), ds_obs.mx.is_observation()    # (True, True)
```

Because the forecast is a grid and the observations are points, aligning the
forecast to the observations means interpolating it to the station
locations:

```python
fcst_pts = ds_fcst.mx.align_space_with(ds_obs, method="delaunay")
dict(fcst_pts.sizes)
# {'reference_time': 2, 'lead_time': 25, 'point_index': 6}
fcst_pts.attrs["mlwp_space_trait"]
# 'point'
```

The forecast now has the stations' `point_index` and its space trait has
become `point`. The interpolation weights are computed now, but the
interpolated values stay lazy until you compute them.
`method="delaunay"` works directly on this flattened (`grid_index`) grid; the
default `"xarray"` method needs the grid's projection first. See
[Interpolation methods](../user-guide/interpolation.md).

## 4. Align in time

The forecast is indexed by `(reference_time, lead_time)`, the observations by
`valid_time`. Aligning the observations *to* the forecast picks, for every
forecast run and lead time, the observation valid at
`reference_time + lead_time`:

```python
obs_fc = ds_obs.mx.align_time_with(fcst_pts)
dict(obs_fc.sizes)
# {'reference_time': 2, 'lead_time': 25, 'point_index': 6}
```

Both datasets now have identical dimensions. Alignment always works in one
direction: the dataset you call `.mx` on is reshaped to match the argument,
which is never modified. See
[How alignment works](../concepts/alignment.md#reference-semantics).

## 5. Transform and score

The forecast's `2t` is in Kelvin and the observations are in °C. Fix that
with a built-in [transformation](../user-guide/transformations.md), then
compute the RMSE per lead time using plain `xarray`:

```python
fcst_pts = mx.transform("kelvin_to_celcius", fcst_pts, variables="2t")

err = fcst_pts["2t"] - obs_fc["2t"]
rmse = np.sqrt((err**2).mean(["reference_time", "point_index"]))
rmse.compute()
```

`.compute()` is the moment data is actually read and interpolated.

Any package that works on aligned `xarray` objects works here too. For
example, [xskillscore](https://xskillscore.readthedocs.io) (the `verification`
extra) gives the same numbers. It needs each dimension it reduces over in a
single Dask chunk:

```python
import xskillscore as xs

xs.rmse(
    fcst_pts["2t"].chunk(reference_time=-1),
    obs_fc["2t"],
    dim=["reference_time", "point_index"],
).compute()
```

## Next steps

- [Traits](../concepts/traits.md) and [How alignment works](../concepts/alignment.md)
  explain what happened in steps 3 and 4.
- [The mlwp-tools ecosystem](../concepts/ecosystem.md) shows how to write a
  loader for your own data instead of building datasets by hand.
- The [introduction notebook](https://github.com/mlwp-tools/mxalign/blob/main/examples/introduction.ipynb)
  is a longer tour: projections, unstacking to the native grid, comparing
  interpolators, and `xskillscore`.
