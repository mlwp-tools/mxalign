import os
import numpy as np
import xarray as xr

from .utils.config import Config
from .utils.dates import Dates
from .loaders.loader import load
from .transformations.transform import transform
from .align.time import align_time
from .align.space import align_space
from .align.nans import broadcast_nans
from .utils.save import save_dataset, save_metrics
from .verification import Metric



class Runner():
    def __init__(self, config: str | dict):
        self.config = Config(config)
        self.datasets = {}
    
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

        dates_config = self.config.dates
        dates = Dates(**dates_config) if dates_config else None

        for name, config_ds in config.items():
            config_ds = config_ds.copy()
            # Check if all the files exist
            loader = config_ds.pop("loader")
            variables = config_ds.pop("variables", None)
            grid_mapping = config_ds.pop("grid_mapping", None)
            ens_size = config_ds.pop("ens_size", 1)
            files = []
            # Check if all the files exist
            for file in config_ds.pop("files"):
                if os.path.exists(file):
                    files.append(file)
                else:
                    print(f"File: {file} is missing, skipping.")
            ds = load(
                name=loader,
                files=files,
                variables=variables,
                grid_mapping=grid_mapping,
                ens_size=ens_size,
                **config_ds
            )
            if dates is not None:
                ds = _filter_by_dates(ds, dates)
            self.datasets[name] = ds
    
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
                    name=transformation,
                    datasets=ds,
                    **config_trans
                )

    def align(self):
        config = self.config["alignment"]
        reference = config.pop("reference")
        brdcst_nans = config.pop("broadcast_nans", True)
        config_align_time = config.get("time",None)
        config_align_space = config.get("space", None)
        config_align_save = config.get("save", None)

        def _debug_reftimes(label):
            for name, ds in self.datasets.items():
                if "reference_time" in ds.dims:
                    hours = set(ds["reference_time"].dt.hour.values.tolist())
                    print(f"  [{label}] {name}: {ds.sizes['reference_time']} ref_times, hours={sorted(hours)}")

        _debug_reftimes("after load")

        # align in time
        if config_align_time:
            self.align_time(config_align_time)
        else:
            print("Skipping temporal alignment")

        _debug_reftimes("after time align")

        # align in space
        if config_align_space:
            self.align_space(reference=reference, config=config_align_space)
        else:
            print("Skipping spatial alignment")

        _debug_reftimes("after space align")

        # broadcast NaNs
        if brdcst_nans:
            self.datasets = broadcast_nans(self.datasets)

        _debug_reftimes("after broadcast_nans")

        # Save aligned datasets
        if config_align_save:
            config = config_align_save.copy()
            method = config.pop("method")
            datasets = config.pop("datasets","all")
            if datasets == "all":
                for name, ds in self.datasets.items():
                    save_dataset(method, name, ds, **config)
            elif datasets == "merge":
                ds = xr.concat(
                    self.datasets.values(),
                    dim = xr.Varialbe("model", list(self.datasets.keys()))
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

        if config_metrics:
            metrics = {}
            for metric_name, config_metric in config["metrics"].items():
                config_metric = config_metric.copy()
                func_path = config_metric.pop("function")
                inputs = config_metric.pop("inputs")
                
                metric = Metric(
                    name=metric_name,
                    func_path=func_path,
                    ds_ref=reference[common_vars],
                    inputs=inputs,
                    **config_metric
                )
                models = {}
                for ds_name, ds in self.datasets.items():
                    if ds_name != config["reference"]:
                        models[ds_name] = metric.compute(ds[common_vars])
                models = xr.concat(
                    models.values(),
                    dim = xr.Variable("model", list(models.keys()))
                )
                metrics[metric.name] = models
            metrics = xr.concat(
                metrics.values(),
                dim = xr.Variable("metric", list(metrics.keys()))
            )
            self.metrics = metrics.transpose("model", "metric", ...).compute()
        
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
                self.datasets[name], ds_ref = align_space(ds, ds_ref, **options)
        self.datasets[reference] = ds_ref
        
    

def get_spatial_alignment(ds, reference):
    if reference.space.is_point() and ds.space.is_grid():
        return "interpolation"
    if reference.space.is_grid() and ds.space.is_grid():
        return "regrid"
    return "null"


def _filter_by_dates(ds, dates):
    """Slice a dataset to keep only the times defined by a Dates object."""
    if "reference_time" in ds.dims:
        ref_times = np.array(sorted(dates.reference_times), dtype="datetime64[ns]")
        ref_times = ref_times[np.isin(ref_times, ds["reference_time"].values)]
        ds = ds.sel(reference_time=ref_times)
        if "lead_time" in ds.dims:
            lead_times = np.array(
                [np.timedelta64(lt, "s") for lt in dates.lead_times], dtype="timedelta64[ns]"
            )
            lead_times = lead_times[np.isin(lead_times, ds["lead_time"].values)]
            ds = ds.sel(lead_time=lead_times)
    elif "valid_time" in ds.dims:
        valid_times = np.array(sorted(dates.valid_times), dtype="datetime64[ns]")
        valid_times = valid_times[np.isin(valid_times, ds["valid_time"].values)]
        ds = ds.sel(valid_time=valid_times)
    return ds

            