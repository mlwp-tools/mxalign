import os
import time
import xarray as xr

from .utils.config import Config
from .loaders.loader import load  # noqa: F401  (kept for external API back-compat)
from .loaders.registry import get_loader
from .transformations.transform import transform
from .align.time import align_time
from .align.space import align_space
from .align.nans import broadcast_nans
from .utils.save import save_dataset, save_metrics
from .verification import Metric
from ._progress import (
    ProgressTicker,
    count_tasks,
    log_dashboard,
    log_phase_done,
    log_phase_start,
)


class Runner:
    def __init__(self, config: str | dict):
        self.config = Config(config)
        self.datasets = {}
        # Bookkeeping required by the fused verification engine.
        # Populated by load_datasets / transform_datasets and ignored by
        # the legacy xarray engine.
        self.loaders: dict[str, object] = {}
        self._transforms_by_ds: dict[str, list[tuple[str, dict]]] = {}

    def run(self):
        # 1. Load the datasets
        self.load_datasets()

        # 2. Transform the datasets
        self.transform_datasets()
        self.align()
        self.verify()

    def load_datasets(self):
        config = self.config["datasets"]
        if config is None:
            return ValueError("No datasets section in the config.")
        for name, config_ds in config.items():
            config_ds = config_ds.copy()
            # Check if all the files exist
            loader_name = config_ds.pop("loader")
            variables = config_ds.pop("variables", None)
            grid_mapping = config_ds.pop("grid_mapping", None)
            files = []
            # Check if all the files exist
            for file in config_ds.pop("files"):
                if os.path.exists(file):
                    files.append(file)
                else:
                    print(f"File: {file} is missing, skipping.")
            loader_cls = get_loader(loader_name)
            loader_inst = loader_cls(
                files,
                variables=variables,
                grid_mapping=grid_mapping,
                **config_ds,
            )
            self.datasets[name] = loader_inst.load()
            self.loaders[name] = loader_inst

    def transform_datasets(self):
        config = self.config["transformations"]
        if config is None:
            pass
        for transformation, config_trans in config.items():
            config_trans = config_trans.copy()
            # if no datasets specified, apply to all datasets
            names_ds = config_trans.pop("datasets", self.datasets.keys())
            for name in names_ds:
                ds = self.datasets[name]
                self.datasets[name] = transform(
                    name=transformation, datasets=ds, **config_trans
                )
                # Record (transform_name, kwargs) in application order for
                # the fused engine to replay on per-rt slices.
                self._transforms_by_ds.setdefault(name, []).append(
                    (transformation, dict(config_trans))
                )

    def align(self):
        config = self.config["alignment"]
        reference = config.pop("reference")
        brdcst_nans = config.pop("broadcast_nans", True)
        config_align_time = config.get("time", None)
        config_align_space = config.get("space", None)
        config_align_save = config.get("save", None)

        # align in time
        if config_align_time:
            self.align_time(config_align_time)
        else:
            print("Skipping temporal alignment")

        # align in space
        if config_align_space:
            self.align_space(reference=reference, config=config_align_space)
        else:
            print("Skipping spatial alignment")

        # broadcast NaNs
        if brdcst_nans:
            self.datasets = broadcast_nans(self.datasets)

        # Save aligned datasets
        if config_align_save:
            config = config_align_save.copy()
            method = config.pop("method")
            datasets = config.pop("datasets", "all")
            if datasets == "all":
                for name, ds in self.datasets.items():
                    save_dataset(method, name, ds, **config)
            elif datasets == "merge":
                ds = xr.concat(
                    self.datasets.values(),
                    dim=xr.Variable("model", list(self.datasets.keys())),
                )
                save_dataset(method, name, ds, **config)
            else:
                raise ValueError("Unknown option for dataset saving.")

    def verify(self):
        config = self.config["verification"]
        reference = self.datasets[config["reference"]]
        config_metrics = config.get("metrics", None)
        config_save_metrics = config.get("save", None)

        common_vars = set(reference.data_vars)
        for ds in self.datasets.values():
            common_vars.intersection_update(set(ds.data_vars))
        common_vars = list(common_vars)

        rechunk_lead_time = config.get("rechunk_lead_time", True)
        engine = config.get("engine", "xarray")

        if config_metrics and engine == "fused":
            from .verification_fused import compute_metrics_fused
            log_phase_start(
                "verify-build",
                engine="fused",
                n_models=len(self.datasets) - 1,
                n_metrics=len(config_metrics),
                n_vars=len(common_vars),
                n_rt=int(reference.sizes.get("reference_time", -1)),
                n_lt=int(reference.sizes.get("lead_time", -1)),
            )
            t_build = time.perf_counter()
            self.metrics = compute_metrics_fused(
                datasets=self.datasets,
                loaders=self.loaders,
                transforms_by_ds=self._transforms_by_ds,
                reference_name=config["reference"],
                common_vars=common_vars,
                metrics_cfg=config["metrics"],
                engine_cfg=config,
            )
            log_phase_done(
                "verify-build+exec",
                time.perf_counter() - t_build,
                engine="fused",
            )
        elif config_metrics:
            log_phase_start(
                "verify-build",
                n_models=len(self.datasets) - 1,
                n_metrics=len(config_metrics),
                n_vars=len(common_vars),
                n_rt=int(reference.sizes.get("reference_time", -1)),
                n_lt=int(reference.sizes.get("lead_time", -1)),
                rechunk_lead_time=rechunk_lead_time,
            )
            t_build = time.perf_counter()
            ds_ref_for_metric = (
                _rechunk_for_metric(reference[common_vars])
                if rechunk_lead_time
                else reference[common_vars]
            )
            metrics = {}
            for metric_name, config_metric in config["metrics"].items():
                config_metric = config_metric.copy()
                func_path = config_metric.pop("function")
                inputs = config_metric.pop("inputs")

                metric = Metric(
                    name=metric_name,
                    func_path=func_path,
                    ds_ref=ds_ref_for_metric,
                    inputs=inputs,
                    **config_metric,
                )
                models = {}
                for ds_name, ds in self.datasets.items():
                    if ds_name != config["reference"]:
                        ds_slice = (
                            _rechunk_for_metric(ds[common_vars])
                            if rechunk_lead_time
                            else ds[common_vars]
                        )
                        models[ds_name] = metric.compute(ds_slice)
                models = xr.concat(
                    models.values(), dim=xr.Variable("model", list(models.keys()))
                )
                metrics[metric.name] = models
            metrics = xr.concat(
                metrics.values(), dim=xr.Variable("metric", list(metrics.keys()))
            )
            metrics_lazy = metrics.transpose("model", "metric", ...)
            n_tasks = count_tasks(metrics_lazy)
            log_phase_done(
                "verify-build",
                time.perf_counter() - t_build,
                n_tasks=n_tasks,
            )
            log_dashboard()
            log_phase_start("verify-exec")
            t_exec = time.perf_counter()
            with ProgressTicker("verify-exec"):
                self.metrics = metrics_lazy.compute()
            log_phase_done("verify-exec", time.perf_counter() - t_exec)

        if config_save_metrics:
            config = config_save_metrics.copy()
            method = config.pop("method")
            save_metrics(method, self.metrics, **config)

    def align_time(self, config):
        self.datasets = align_time(self.datasets, **config)

    def align_space(self, reference, config):
        ds_ref = self.datasets[reference]
        for name, ds in self.datasets.items():
            if name != reference:
                options = config.get(get_spatial_alignment(ds, ds_ref), {})
                self.datasets[name] = align_space(ds, ds_ref, **options)


def get_spatial_alignment(ds, reference):
    if reference.space.is_point() and ds.space.is_grid():
        return "interpolation"
    if reference.space.is_grid() and ds.space.is_grid():
        return "regrid"
    return "null"


def _rechunk_for_metric(ds: xr.Dataset) -> xr.Dataset:
    """Rechunk to (reference_time=1, lead_time=-1, ...) before metric graph build.

    This aligns the ERA5 observation chunks (typically (1,1,40320) after time
    alignment) with the forecast chunks (1, n_lt, n_grid) produced by the
    anemoi-inference loader.  Without this, xarray/dask fans out 144 tasks per
    (reference_time, variable) cell when it tries to broadcast mismatched
    lead_time chunks, turning an O(N_rt) graph into an O(N_rt * N_lt) one.

    Only rechunks dims that are present; leaves grid_index at its natural size.
    """
    chunks: dict[str, int] = {}
    if "reference_time" in ds.dims:
        chunks["reference_time"] = 1
    if "lead_time" in ds.dims:
        chunks["lead_time"] = -1
    if not chunks:
        return ds
    return ds.chunk(chunks)
