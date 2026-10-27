import os
import xarray as xr

from mlwp_data_loaders import load_and_validate_dataset

from .config import (
    PipelineConfig,
    SpaceAlignmentConfig,
    TimeAlignmentConfig,
    load_config,
)
from .transformations.transform import transform
from ..alignment.time import align_time
from ..alignment.nans import broadcast_nans
from .save import save_dataset, save_metrics
from .verification import Metric


class Runner:
    def __init__(self, config: str | dict):
        self.config: PipelineConfig = load_config(config)
        self.datasets = {}

    def run(self):
        # 1. Load the datasets
        self.load_datasets()

        # 2. Transform the datasets
        self.transform_datasets()
        self.align()
        self.verify()

    def load_datasets(self):
        for name, config_ds in self.config.datasets.items():
            variables = config_ds.variables
            grid_mapping = config_ds.grid_mapping

            raw_files = config_ds.files
            if isinstance(raw_files, str):
                raw_files = [raw_files]
            files = []
            for file in raw_files:
                if os.path.exists(file):
                    files.append(file)
                else:
                    print(f"File: {file} is missing, skipping.")
            dataset_path = files[0] if len(files) == 1 else files

            ds = load_and_validate_dataset(
                dataset_path,
                loader=config_ds.loader,
                **config_ds.kwargs,
            )

            if variables is not None:
                ds = ds[[variables] if isinstance(variables, str) else list(variables)]
            if grid_mapping is not None:
                ds = ds.mx.add_crs(grid_mapping)
                ds = ds.mx.add_grid_mapping(grid_mapping)

            self.datasets[name] = ds

    def transform_datasets(self):
        if self.config.transformations is None:
            return
        for transformation, config_trans in self.config.transformations.items():
            # if no datasets specified, apply to all datasets
            names_ds = config_trans.datasets
            if names_ds is None:
                names_ds = self.datasets.keys()
            for name in names_ds:
                ds = self.datasets[name]
                self.datasets[name] = transform(
                    name=transformation, datasets=ds, **config_trans.kwargs
                )

    def align(self):
        config = self.config.alignment
        reference = config.reference

        # align in time (always; the time section only holds options)
        self.align_time(config.time, reference_name=reference)

        # align in space
        if config.space is not None:
            self.align_space(reference=reference, config=config.space)
        else:
            print("Skipping spatial alignment")

        # broadcast NaNs
        if config.broadcast_nans:
            self.datasets = broadcast_nans(self.datasets)

        # Save aligned datasets
        if config.save is not None:
            save = config.save
            kwargs = {"path": save.path, **save.kwargs}
            if save.datasets == "all":
                for name, ds in self.datasets.items():
                    save_dataset(save.method, name, ds, **kwargs)
            elif save.datasets == "merge":
                ds = xr.concat(
                    self.datasets.values(),
                    dim=xr.Variable("model", list(self.datasets.keys())),
                )
                save_dataset(save.method, "merged", ds, **kwargs)

    def verify(self):
        config = self.config.verification
        if config is None:
            return
        reference = self.datasets[config.reference]

        common_vars = set(reference.data_vars)
        for ds in self.datasets.values():
            common_vars.intersection_update(set(ds.data_vars))
        common_vars = list(common_vars)

        if config.metrics:
            metrics = {}
            for metric_name, config_metric in config.metrics.items():
                metric = Metric(
                    name=metric_name,
                    func_path=config_metric.function,
                    ds_ref=reference[common_vars],
                    inputs=config_metric.inputs,
                    **config_metric.kwargs,
                )
                models = {}
                for ds_name, ds in self.datasets.items():
                    if ds_name != config.reference:
                        models[ds_name] = metric.compute(ds[common_vars])
                models = xr.concat(
                    models.values(), dim=xr.Variable("model", list(models.keys()))
                )
                metrics[metric.name] = models
            metrics = xr.concat(
                metrics.values(), dim=xr.Variable("metric", list(metrics.keys()))
            )
            self.metrics = metrics.transpose("model", "metric", ...).compute()

        if config.save is not None:
            save = config.save
            save_metrics(save.method, self.metrics, path=save.path, **save.kwargs)

    def align_time(
        self, config: TimeAlignmentConfig | None, reference_name: str | None = None
    ):
        lead_time = config.lead_time if config is not None else None
        self.datasets = align_time(
            self.datasets, reference=reference_name, lead_time=lead_time
        )

    def align_space(self, reference: str, config: SpaceAlignmentConfig):
        ds_ref = self.datasets[reference]
        for name, ds in self.datasets.items():
            if name != reference:
                kind = get_spatial_alignment(ds, ds_ref)
                options = getattr(config, kind, None) or {}
                self.datasets[name] = ds.mx.align_space_with(ds_ref, **options)


def get_spatial_alignment(ds, reference):
    if reference.mx.is_point() and ds.mx.is_grid():
        return "interpolation"
    if reference.mx.is_grid() and ds.mx.is_grid():
        return "regrid"
    return "null"
