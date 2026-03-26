import xarray as xr
import numpy as np

from ..properties.properties import Properties, Time, Space, Uncertainty
from ..properties.utils import properties_to_attrs

def align_time(reference: xr.Dataset, datasets: list[xr.Dataset] | dict[str, xr.Dataset], return_as: str = "forecast"):
    if isinstance(datasets, (xr.Dataset, xr.DataArray)):
        datasets = [datasets]
    if isinstance(datasets, dict):
        keys = datasets.keys()
        datasets = datasets.values()
    else:
        keys = None

    if return_as != "forecast":
        NotImplementedError("Currently only temporal alignment return forecast structure is supported.")
    print("Aligning datasets in time...")
    print("Datasets before alignment:")
    for i, ds in enumerate(datasets):
        print(f"  Dataset {i}: {ds}")
    # Get the first forecast to start building the valid times
    valid_times_fcst = None
    valid_times_obs = None
    first_fcst = True
    first_obs = True
    for ds in datasets:
        if ds.time.is_forecast():
            if first_fcst:
                print("First forecast dataset found, initializing valid times with this dataset.")
                valid_times_fcst = ds.time.add_valid_time()["valid_time"].to_dataset(name="valid_times")
                valid_times_fcst = valid_times_fcst.assign_attrs(ds.attrs)
                print("Initial valid times from first forecast dataset:", valid_times_fcst)
                first_fcst = False
            else:
                _ds = ds.time.add_valid_time()["valid_time"].to_dataset(name="valid_times")
                _ds = _ds.assign_attrs(ds.attrs)
                _, valid_times_fcst = _ds.time.align_with(valid_times_fcst)
        elif ds.time.is_observation():
            if first_obs:
                valid_times_obs = ds["valid_time"].to_dataset(name="valid_times")
                valid_times_obs = valid_times_obs.assign_attrs(ds.attrs)
                fist_obs = False
            else:
                _ds = ds["valid_time"].to_dataset(name="valid_times")
                _ds = _ds.assign_attrs(ds.attrs)
                _, valid_times_obs = _ds.time.align_with(valid_times_obs)

    if (valid_times_obs is None) and (valid_times_fcst is None):
        raise ValueError("No observations or forecasts found")
    elif valid_times_fcst is None:
        valid_times = valid_times_obs
    elif valid_times_obs is None:
        valid_times = valid_times_fcst
    else:
        _, valid_times = valid_times_obs.time.align_with(valid_times_fcst)
    
    datasets = [
        ds.time.align_with(valid_times)[0]
        for ds in datasets
        ]
    print("Valid times after alignment:", valid_times)
    print("Datasets after alignment:")
    for i, ds in enumerate(datasets):
        print(f"  Dataset {i}: {ds}")

    if keys is None:
        return datasets
    else:
        return {key: value for (key, value) in zip(keys, datasets)}
    
