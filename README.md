# Meteo-xAlign

**An xarray based package for alignment of meteorological datasets**

## What is this?

`mxalign` is an `xarray`-based package designed for the alignment and verification of meteorological datasets. It standardizes operations across datasets by attaching properties along three main axes:
- **Space:** Grid or point-based data
- **Time:** Forecasts, observations, or climatology
- **Uncertainty:** Deterministic, ensemble, or quantile forecasts

Currently, `mxalign` also acts as a full execution engine. It can load datasets (e.g., Anemoi inference outputs, observation datasets), apply transformations, align datasets in both space and time to match a reference, safely broadcast NaNs, and execute verification metrics on scaled Dask clusters (Local or Slurm).

> ⚠️ **Roadmap & Future Architecture Changes (planned for v0.2.0):**
> Currently, `mxalign` handles both alignment and the execution of the verification tooling pipeline, including loading and validation. In the upcoming `v0.2.0` release, this architecture will be refactored:
> - **Loading** will be split out into [`mlwp-data-loaders`](https://github.com/mlwp-tools/mlwp-data-loaders).
> - **Validation** of loaded `xr.Dataset`s will be moved to [`mlwp-data-specs`](https://github.com/mlwp-tools/mlwp-data-specs) (which will contain the requirements for each of the dataset traits and the validation logic).
> - **Execution** of the full verification pipeline (loading, transformations, alignment, and verification) from configuration files may be moved to a separate package in future releases.
> - **Tests** will be added to `mxalign` (building on test datasets already integrated into `mlwp-data-loaders`) that ensure that all alignment operations work correctly (Testing notebook execution inside `mxalign` is explicitly excluded from the current roadmap).

## Python API

`mxalign` provides building blocks for manual alignment, transformations, and interpolations of `xarray` datasets. This is ideal for interactive use in Jupyter notebooks or custom Python scripts.

```python
import xarray as xr
from mxalign import load, align_space, align_time, transform

# Load datasets (using registered loaders)
ds_obs = load(name="observations_loader", files=["obs.nc"])
ds_fcst = load(name="anemoi_inference", files=["forecast.nc"])

# Align the forecast spatially to match the observation reference
ds_fcst_aligned_space = align_space(ds_fcst, reference=ds_obs, method="interpolation")

# Align datasets temporally
datasets = {"obs": ds_obs, "fcst": ds_fcst_aligned_space}
aligned_datasets = align_time(datasets, method="intersection")
```

For a more comprehensive interactive example, check out the [introductory notebook](./examples/introduction.ipynb).

## Executing via a Configuration

For full verification pipeline execution, `mxalign` uses a YAML configuration file. This allows you to declaratively define how datasets are loaded, transformed, aligned, and verified.

### Configuration Contents

The configuration file is divided into several main sections:

```yaml
datasets:
  # Define datasets to load, specifying the loader, files, and variables
  obs_data:
    loader: observations_loader
    files: ["obs.nc"]
  fcst_data:
    loader: anemoi_inference
    files: ["forecast.nc"]

transformations:
  # Apply transformations to loaded datasets

alignment:
  # Define reference dataset and alignment methods (space, time, NaN broadcasting)
  reference: obs_data
  time:
    method: intersection

verification:
  # Specify the reference dataset and the metrics to calculate
  reference: obs_data
  metrics:
    # define metrics here
```

### Running from the Command Line

The CLI uses Dask to distribute the workload and supports both local execution and execution on Slurm-managed HPC clusters.

**Local Execution**
Run the pipeline on a local Dask cluster:
```bash
mxalign local path/to/config.yaml --n_workers 4 --threads_per_worker 1
```

**Slurm Execution**
Run the pipeline on a Slurm cluster:
```bash
mxalign slurm path/to/config.yaml --account your_account --queue your_queue --cores 8 --memory 64GB
```

### Running from Python

You can also execute the entire configuration-driven pipeline directly from Python using the `Runner` class.

```python
from mxalign.runner import Runner

# Initialize the runner with a YAML config file or a dictionary
runner = Runner("path/to/config.yaml")

# Execute the pipeline: loads, transforms, aligns, and verifies the datasets
runner.run()

# The resulting aligned datasets and computed metrics are accessible via:
aligned_datasets = runner.datasets
metrics = runner.metrics
```
