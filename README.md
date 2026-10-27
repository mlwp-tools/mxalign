# Meteo-xAlign

**An xarray based package for alignment of meteorological datasets**

## What is this?

`mxalign` is an `xarray`-based package for aligning meteorological datasets. It operates on datasets that carry **traits** — metadata attributes that describe the nature of a dataset along three axes:

`mxalign` is an `xarray`-based package for aligning meteorological datasets. It operates on datasets that carry **traits** — metadata attributes that describe the nature of a dataset along three axes:
- **Space:** `grid` or `point`
- **Time:** `forecast`, `observation`, or `climatology`
- **Uncertainty:** `deterministic`, `ensemble`, or `quantile`

These traits are defined and validated by [`mlwp-data-specs`](https://github.com/mlwp-tools/mlwp-data-specs) and attached to datasets by [`mlwp-data-loaders`](https://github.com/mlwp-tools/mlwp-data-loaders). `mxalign` reads them to infer how datasets should be aligned, without needing to know how they were loaded.

`mxalign` currently supports alignment in **space** and **time**. Alignment along the **uncertainty** axis (e.g. ensemble to deterministic) is planned for a future release.

## Python API

`mxalign` provides building blocks for spatial and temporal alignment of `xarray` datasets. This is ideal for interactive use in Jupyter notebooks or custom Python scripts.

```python
import mlwp_data_loaders as dl
import mxalign as mx

# Load datasets — traits are attached by the loader
ds_obs = dl.load("observations_loader", files=["obs.nc"])
ds_fcst = dl.load("anemoi_inference", files=["forecast.nc"])

# Align the forecast spatially to match the observation reference
ds_fcst_aligned = mx.align_space(ds_fcst, reference=ds_obs, method="interpolation")

# Align datasets temporally
datasets = {"obs": ds_obs, "fcst": ds_fcst_aligned}
aligned_datasets = mx.align_time(datasets, method="intersection")
```

For a more comprehensive interactive example, check out the [introductory notebook](./examples/introduction.ipynb).

## Executing via a Configuration

`mxalign` can drive a full verification pipeline from a YAML configuration file, orchestrating dataset loading (via `mlwp-data-loaders`), transformations, alignment, and verification.

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
