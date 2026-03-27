import xarray as xr
import numpy as np

from ..properties.properties import Time
from ..properties.utils import properties_from_attrs, update_time_property
@xr.register_dataset_accessor("time")
class TimeAccessor:
    def __init__(self, ds):
        self._time = properties_from_attrs(ds).time
        self._ds = ds

    def is_forecast(self):
        return self._time == Time.FORECAST
    
    def is_observation(self):
        return self._time == Time.OBSERVATION

    def add_valid_time(self):
        if self.is_forecast():          
            valid_time = self._ds["reference_time"].values[:,np.newaxis] + self._ds["lead_time"].values
            ds_out = self._ds.assign_coords(
                {
                    "valid_time": (["reference_time", "lead_time"], valid_time)
                }
            )
        else:
            ds_out = self._ds
        return ds_out
    
    def align_with(self, ds, **kwargs):
        if self.is_forecast():
            if ds.time.is_forecast():
                return _align_forecast_forecast(self._ds, ds, **kwargs)
            elif ds.time.is_observation():
                return _align_forecast_observation(self._ds, ds, **kwargs)
        elif self.is_observation():
            if ds.time.is_observation():
                return _align_observation_observation(self._ds, ds, **kwargs)
            elif ds.time.is_forecast():
                return _align_observation_forecast(self._ds, ds, **kwargs)
        else:
            raise ValueError("Datasets do not have compatible temporal properties")

def _align_forecast_forecast(ds1, ds2, only_common=False):
    # Align the reference times
    common_reference_times = ds1.indexes["reference_time"].intersection(ds2.indexes["reference_time"])
    ds1_aligned = ds1.sel(reference_time=common_reference_times)
    ds2_aligned = ds2.sel(reference_time=common_reference_times)
    
    # Align the lead times
    if only_common:
        common_lead_times = ds1_aligned.indexes["lead_time"].intersection(ds2_aligned.indexes["lead_time"])
        ds1_aligned = ds1_aligned.sel(lead_time=common_lead_times)
        ds2_aligned = ds2_aligned.sel(lead_time=common_lead_times)
    else:
        non_aligning_dims = (set(ds1.dims) | set(ds2.dims)) - set(["lead_time"])
        ds1_aligned, ds2_aligned = xr.align(ds1_aligned, ds2_aligned, join="outer", exclude=non_aligning_dims)
        ds1_aligned = ds1_aligned.time.add_valid_time()
        ds2_aligned = ds2_aligned.time.add_valid_time()
    return ds1_aligned, ds2_aligned

def _align_forecast_observation(ds_forecast, ds_observation, only_common=False,lead_time="start-min"):
    ds_forecast = ds_forecast.time.add_valid_time()

    # Check if reference_times are continuous
    reference_time_diff = ds_forecast.reference_time.diff("reference_time").values
    if not (reference_time_diff[0] == reference_time_diff).all():
        raise NotImplementedError("Aligning a forecast with non-continuous reference times with an observation is not implemented.")
    if lead_time == "start-min":
        min_diff = reference_time_diff[0]
        ds_forecast_reduced = ds_forecast.where(ds_forecast.lead_time < min_diff, drop=True)
    elif lead_time == "start-max":    
        max_diff = ds_forecast.lead_time.max().values
        reference_times = np.arange(ds_forecast.reference_time.min().values, ds_forecast.reference_time.max().values, max_diff, dtype="datetime64[ns]")
        ds_forecast_reduced = ds_forecast.sel(reference_time=reference_times)
    else:
        raise ValueError("Invalid value for lead_time. Expected 'start-min' or 'start-max'.")
    
    ds_forecast_stacked = ds_forecast_reduced.\
        stack(time=["reference_time", "lead_time"]).\
        reset_index("time").\
        swap_dims({"time": "valid_time"}).transpose("valid_time",...)
    if only_common:
        ds_forecast_aligned, ds_observation_aligned = xr.align(
            ds_forecast_stacked, 
            ds_observation, 
            join="inner", 
            exclude=set(ds_forecast_stacked.coords) | set(ds_observation.coords) - set(["valid_time"]))
    else:
        ds_forecast_aligned, ds_observation_aligned = xr.align(
            ds_forecast_stacked, 
            ds_observation, 
            join="outer", 
            exclude=set(ds_forecast_stacked.coords) | set(ds_observation.coords) - set(["valid_time"]))
    ds_forecast_aligned = update_time_property(ds_forecast_aligned, Time.OBSERVATION)
    return ds_forecast_aligned, ds_observation_aligned

def _align_observation_observation(ds1, ds2, only_common=False):
    exclude = (set(ds1.dims) | set(ds2.dims)) - set(["valid_time"])
    if only_common:
        ds1_aligned, ds2_aligned = xr.align(ds1, ds2, join="inner", exclude=exclude)
    else:
        ds1_aligned, ds2_aligned = xr.align(ds1, ds2, join="outer", exclude=exclude)
    return ds1_aligned, ds2_aligned

def _align_observation_forecast(ds_observation, ds_forecast, only_common=False):
    ds_forecast_cut = ds_forecast.time.add_valid_time()
    if ds_forecast_cut.reference_time.min().values < ds_observation.valid_time.min().values:
        ds_forecast_cut = ds_forecast_cut.sel(
            reference_time=slice(
                ds_observation.valid_time.min().values, None)
        )
    if ds_forecast_cut.valid_time.max().values > ds_observation.valid_time.max().values:
        # The forecast time-step/lead times might not always align with the maximum observation time
        valid_diff = (ds_forecast_cut["valid_time"]-(ds_observation["valid_time"].max())).isel(lead_time=-1)
        last_valid_index = int(np.abs(valid_diff.where(valid_diff<=0, drop=True)).argmin(dim="reference_time").values)
        max_reference_time = ds_forecast_cut.isel(reference_time=last_valid_index)["reference_time"].values


        #max_reference_time = ds_observation.valid_time.max().values - (ds_forecast_cut.lead_time.max().values - shift)
        ds_forecast_cut = ds_forecast_cut.sel(reference_time=slice(None, max_reference_time))

    # Reindex obs to cover all forecast valid_times (NaN for missing obs),
    # then do the 2D sel — avoids KeyError when obs has gaps.
    flat_valid_times = np.unique(ds_forecast_cut.valid_time.values.ravel())
    ds_observation = ds_observation.reindex(valid_time=flat_valid_times)
    ds_observation_aligned = ds_observation.sel(valid_time=ds_forecast_cut.valid_time)
    ds_observation_aligned = ds_observation_aligned.transpose("reference_time", "lead_time", ...)
    ds_observation_aligned = update_time_property(ds_observation_aligned, Time.FORECAST)
    if only_common:
        return ds_observation_aligned, ds_forecast_cut
    else:
        ds_observation_aligned, ds_forecast_aligned = xr.align(
            ds_observation_aligned, 
            ds_forecast.time.add_valid_time(), 
            join="outer", 
            exclude=(set(ds_observation_aligned.coords) | set(ds_forecast_cut.coords)) - set(["reference_time", "lead_time"]))
        ds_observation_aligned["valid_time"] = ds_forecast_aligned["valid_time"]
        return ds_observation_aligned, ds_forecast_aligned


