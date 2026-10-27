# Config-driven pipeline

Running load → transform → align → verify from a single YAML file, either
from Python with `mxalign.runner.Runner` or from the command line with
`mxalign local` (a local Dask cluster) and `mxalign slurm` (a Slurm cluster
through `dask-jobqueue`). It will include a reference for every section of
the configuration: `dates`, `datasets`, `transformations`, `alignment` and
`verification`.

!!! note "Coming soon"
    This page is still being written. Until then, the
    [README](https://github.com/mlwp-tools/mxalign#executing-via-a-configuration)
    and [`examples/config.yaml`](https://github.com/mlwp-tools/mxalign/blob/main/examples/config.yaml)
    describe the configuration.
