# Installation

mxalign needs **Python 3.12 or newer**.

!!! warning "Install from GitHub, not PyPI"
    The `mxalign` 0.1.0 release on PyPI predates the split into
    `mlwp-data-specs` / `mlwp-data-loaders` and does not match this
    documentation. Also, `mlwp-data-loaders` is not on PyPI yet. Until the
    next release, install all three packages from GitHub as shown below.

=== "uv"

    ```bash
    uv add \
      "mxalign @ git+https://github.com/mlwp-tools/mxalign" \
      "mlwp-data-loaders @ git+https://github.com/mlwp-tools/mlwp-data-loaders" \
      "mlwp-data-specs @ git+https://github.com/mlwp-tools/mlwp-data-specs@059f382"
    ```

=== "pip"

    ```bash
    pip install \
      "mxalign @ git+https://github.com/mlwp-tools/mxalign" \
      "mlwp-data-loaders @ git+https://github.com/mlwp-tools/mlwp-data-loaders" \
      "mlwp-data-specs @ git+https://github.com/mlwp-tools/mlwp-data-specs@059f382"
    ```

The two sibling packages come from the
[mlwp-tools ecosystem](../concepts/ecosystem.md):

- [`mlwp-data-specs`](https://github.com/mlwp-tools/mlwp-data-specs) defines
  and validates the dataset [traits](../concepts/traits.md). It is pinned to
  the commit mxalign is tested against.
- [`mlwp-data-loaders`](https://github.com/mlwp-tools/mlwp-data-loaders) loads
  datasets and attaches those traits.

## Optional extras

Request extras on the `mxalign` requirement, for example
`"mxalign[verification,jobqueue] @ git+https://github.com/mlwp-tools/mxalign"`.

| Extra | Adds | Needed for |
|---|---|---|
| `verification` | `xskillscore` | Metrics such as `xskillscore.rmse` in the [config-driven pipeline](../user-guide/pipeline.md) |
| `jobqueue` | `dask-jobqueue` | `mxalign slurm …`: running the pipeline on a Slurm cluster |
| `earthkit` | `earthkit-meteo` | Meteorological functions used as [transformations](../user-guide/transformations.md) |
| `ifs` | `cfgrib`, `eccodes`, `eccodeslib` | Reading IFS GRIB forecasts with the `mlwp-data-loaders` IFS loader |

## Development install

Clone the repository and sync the environment with [uv](https://docs.astral.sh/uv/).
Inside the repository, uv resolves the sibling packages from GitHub
automatically:

```bash
git clone https://github.com/mlwp-tools/mxalign
cd mxalign
uv sync --all-extras          # library, extras and the dev group (pytest, ipykernel)
uv run pytest                 # run the test suite
```

To build this documentation locally:

```bash
uv sync --group docs
uv run mkdocs serve           # live preview at http://localhost:8000
```

## Check the install

```python
import mxalign as mx

mx.available_interpolations()   # ['xarray', 'delaunay']
mx.available_transformations()  # ['rename', 'kelvin_to_celcius', 'uv_to_speed', 'external']
```
