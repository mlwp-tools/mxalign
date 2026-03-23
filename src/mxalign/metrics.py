import numpy as np
import xarray as xr
from scores.processing import binary_discretise


def _build_thresholds(threshold_min, threshold_max, threshold_by):
    """Build a sorted, deduplicated threshold array from range spec(s).

    Each of threshold_min / threshold_max / threshold_by can be either
    a scalar or a list (for multiple segments with different step sizes).

    Examples
    --------
    Single segment:  min=0, max=30, by=1  → np.arange(0, 30, 1)
    Multi-segment:   min=[0, 5], max=[5, 30], by=[0.5, 2]
                     → fine steps 0–5, coarse steps 5–30
    """
    if not isinstance(threshold_min, (list, tuple)):
        threshold_min = [threshold_min]
        threshold_max = [threshold_max]
        threshold_by  = [threshold_by]

    segments = [
        np.arange(mn, mx, by)
        for mn, mx, by in zip(threshold_min, threshold_max, threshold_by)
    ]
    thresholds = np.unique(np.concatenate(segments)).astype(float)
    return thresholds


def ets(observations, forecasts, threshold_min, threshold_max, threshold_by,
        dim="point_index"):
    """Equitable Threat Score over a range of thresholds.

    Parameters
    ----------
    observations, forecasts : xr.Dataset
        Aligned observation and forecast datasets.
    threshold_min, threshold_max, threshold_by : scalar or list
        Define threshold range(s) via np.arange(min, max, by).
        Use lists for multiple segments, e.g.
            min=[0, 5], max=[5, 30], by=[0.5, 2]
    dim : str
        Dimension to reduce over (typically 'point_index').

    Returns
    -------
    xr.Dataset with an extra 'threshold' dimension.
    """
    thresholds = _build_thresholds(threshold_min, threshold_max, threshold_by)

    n            = observations.count(dim)  # total valid points (before binarizing)
    obs_bin = binary_discretise(observations, thresholds=thresholds, mode=">=") == 1
    fct_bin = binary_discretise(forecasts,    thresholds=thresholds, mode=">=") == 1

    hits         = (fct_bin  &  obs_bin).sum(dim)
    misses       = (~fct_bin &  obs_bin).sum(dim)
    false_alarms = (fct_bin  & ~obs_bin).sum(dim)

    hits_random = (hits + misses) * (hits + false_alarms) / n
    denom = hits + misses + false_alarms - hits_random
    return (hits - hits_random) / denom.where(denom > 0)


def mse_by_domain(observations, forecasts, dim, lsm_zarr_path,
                  lsm_variable="lsm", skipna=True):
    """MSE computed separately for all domain, land only, and ocean only.

    Loads a static land-sea mask from ``lsm_zarr_path`` and applies it as a
    spatial mask before computing MSE.  Returns a Dataset with an extra
    ``domain`` dimension with values ['all', 'land', 'ocean'].

    Parameters
    ----------
    observations, forecasts : xr.Dataset
        Aligned datasets (grid_index dimension).
    dim : str or list of str
        Spatial dimension(s) to reduce over (typically 'grid_index').
    lsm_zarr_path : str
        Path to the zarr file containing the land-sea mask variable.
    lsm_variable : str
        Name of the lsm variable in the zarr file (default: 'lsm').
    skipna : bool
        Whether to skip NaNs in the MSE computation.
    """
    import xskillscore as xs

    ds_raw = xr.open_zarr(lsm_zarr_path, consolidated=False)
    variables = list(ds_raw.attrs["variables"])
    lsm = (
        ds_raw["data"]
        .assign_coords(variable=variables)
        .sel(variable=lsm_variable)
        .isel(ensemble=0, time=0)
        .rename({"cell": "grid_index"})
        .load()
    )

    masks = {
        "land":  lsm > 0.5,
        "ocean": lsm <= 0.5,
    }

    # Rechunk so that the reduction dim (grid_index) is one chunk per time step,
    # matching what Metric._rechunk() does for xskillscore functions.
    dim_list = [dim] if isinstance(dim, str) else list(dim)
    dim_other = [d for d in observations.dims if d not in dim_list]
    chunks = {d: -1 for d in dim_list}
    chunks.update({d: 1 for d in dim_other})
    observations = observations.chunk(chunks)
    forecasts    = forecasts.chunk(chunks)

    results = {}
    for domain, mask in masks.items():
        obs_m = observations.where(mask) if mask is not None else observations
        fct_m = forecasts.where(mask)    if mask is not None else forecasts
        results[domain] = xs.mse(fct_m, obs_m, dim=dim, skipna=skipna)

    return xr.concat(
        list(results.values()),
        dim=xr.Variable("domain", list(results.keys()))
    )


def qq_plot_data(observations, forecasts, quantiles_min, quantiles_max, quantiles_by, dim):
    """Compute empirical quantiles of observations and forecasts for a QQ plot.

    Pools all values across `dim` (e.g. point_index, reference_time, lead_time),
    then computes empirical quantiles at probability levels built from
    np.arange(quantiles_min, quantiles_max, quantiles_by).  Each argument can be
    a scalar or a list for multi-segment grids (same convention as ``ets``), e.g.:
        quantiles_min: [0.0, 0.9]
        quantiles_max: [0.9, 1.0]
        quantiles_by:  [0.1, 0.01]

    Works with all mxalign loaders (anemoi-datasets, anemoi-inference, zarr).
    After alignment all datasets share the same dimension names; ``dim`` just
    needs to match those aligned dimensions.

    Parameters
    ----------
    observations, forecasts : xr.Dataset
        Aligned datasets.
    quantiles_min, quantiles_max, quantiles_by : scalar or list
        Define probability level range(s) via np.arange(min, max, by).
    dim : str or list of str
        Dimension(s) to pool over. Remaining dims are kept in the output
        (e.g. omit ``lead_time`` to get one QQ curve per lead time).

    Returns
    -------
    xr.Dataset with variables ``obs_q`` and ``fct_q``, both indexed by a
    ``quantile`` coordinate.
    """
    q = _build_thresholds(quantiles_min, quantiles_max, quantiles_by)
    q = np.clip(q, 0.0, 1.0)
    obs_q = observations.quantile(q, dim=dim).rename({v: f"{v}_obs" for v in observations.data_vars})
    fct_q = forecasts.quantile(q, dim=dim).rename({v: f"{v}_fct" for v in forecasts.data_vars})
    return xr.merge([obs_q, fct_q])
