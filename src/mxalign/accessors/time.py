import numpy as np
import pandas as pd
import xarray as xr

from mlwp_data_specs.api import TIME_TRAIT_ATTR
from mlwp_data_specs.specs.traits.time_coordinate import Time


def _add_valid_time(ds_fcst):
    valid_time = (
        ds_fcst["reference_time"].values[:, np.newaxis] + ds_fcst["lead_time"].values
    )
    return ds_fcst.assign_coords(
        {"valid_time": (["reference_time", "lead_time"], valid_time)}
    )


def align_forecast_to_observation(ds_fcst, ds_obs, lead_time="shortest"):
    ds_with_vt = _add_valid_time(ds_fcst)
    ds_stacked = ds_with_vt.stack(time=["reference_time", "lead_time"]).reset_index("time")

    vt_vals = ds_stacked.valid_time.values
    lt_vals = ds_stacked.lead_time.values

    if lead_time in ("shortest", "longest"):
        df = pd.DataFrame({"vt": vt_vals, "lt": lt_vals})
        agg = "min" if lead_time == "shortest" else "max"
        is_extreme = (df.groupby("vt")["lt"].transform(agg) == df["lt"]).values
        # Among entries that match the extreme lead_time, keep first per valid_time
        # (handles ties: same vt + same extreme lt appearing via different ref_times)
        extreme_positions = np.where(is_extreme)[0]
        _, first_in_group = np.unique(vt_vals[extreme_positions], return_index=True)
        positions = extreme_positions[first_in_group]
    elif isinstance(lead_time, (list, np.ndarray)):
        lt_set = set(np.asarray(lead_time).tolist())
        seen_vt = set()
        positions = []
        for i, (vt, lt) in enumerate(zip(vt_vals, lt_vals)):
            if lt in lt_set and vt not in seen_vt:
                positions.append(i)
                seen_vt.add(vt)
        positions = np.array(positions)
    else:
        # single lead_time value — filter directly
        positions = np.where(lt_vals == lead_time)[0]

    ds_1d = ds_stacked.isel(time=positions)
    ds_1d = ds_1d.swap_dims({"time": "valid_time"})
    ds_1d = ds_1d.drop_vars(
        [v for v in ["reference_time", "lead_time", "time"] if v in ds_1d.coords]
    )
    ds_1d = ds_1d.transpose("valid_time", ...)

    ds_1d = ds_1d.reindex(valid_time=ds_obs.valid_time)
    ds_1d.attrs[TIME_TRAIT_ATTR] = Time.OBSERVATION.value
    return ds_1d


def align_observation_to_forecast(ds_obs, ds_fcst):
    ds_fcst_with_vt = _add_valid_time(ds_fcst)
    valid_time_2d = ds_fcst_with_vt["valid_time"]  # shape (reference_time, lead_time)

    # Reindex obs onto all unique fcst valid_times (NaN-fills fcst valid_times not in obs)
    fcst_vt_flat = np.unique(valid_time_2d.values.ravel())
    obs_reindexed = ds_obs.reindex(valid_time=fcst_vt_flat)

    # sel with a 2D DataArray indexer broadcasts 1D obs → (reference_time, lead_time)
    ds_out = obs_reindexed.sel(valid_time=valid_time_2d)

    ds_out.attrs[TIME_TRAIT_ATTR] = Time.FORECAST.value
    return ds_out


def align_observation_to_observation(ds1, ds2):
    return ds1.reindex(valid_time=ds2.valid_time)


def align_forecast_to_forecast(ds1, ds2, lead_time="reference"):
    ds_out = ds1.reindex(reference_time=ds2.reference_time)

    if lead_time == "reference":
        ds_out = ds_out.reindex(lead_time=ds2.lead_time)
    elif lead_time == "intersection":
        common_lt = np.intersect1d(ds_out.lead_time.values, ds2.lead_time.values)
        ds_out = ds_out.sel(lead_time=common_lt)
    elif lead_time == "union":
        all_lt = np.union1d(ds_out.lead_time.values, ds2.lead_time.values)
        ds_out = ds_out.reindex(lead_time=all_lt)
    else:
        raise ValueError(f"Unknown lead_time option for F→F alignment: {lead_time!r}")

    # Refresh valid_time
    if "valid_time" in ds_out.coords:
        ds_out = ds_out.drop_vars("valid_time")
    return _add_valid_time(ds_out)
