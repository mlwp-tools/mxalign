import numpy as np
import xarray as xr
from scipy.fft import dctn
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
        dim="point_index", variables=None, min_events=10):
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
    variables : list of str, optional
        Subset of variables to compute ETS for.  If None, all variables are used.
    min_events : int
        Minimum number of observed events (hits + misses) required to compute
        ETS. Thresholds with fewer events are masked as NaN to avoid
        small-sample artifacts (e.g. ETS≈1 from a single lucky hit).

    Returns
    -------
    xr.Dataset with an extra 'threshold' dimension.
    """
    if variables is not None:
        observations = observations[variables]
        forecasts    = forecasts[variables]

    thresholds = _build_thresholds(threshold_min, threshold_max, threshold_by)

    n            = observations.count(dim)  # total valid points (before binarizing)
    obs_bin = binary_discretise(observations, thresholds=thresholds, mode=">=") == 1
    fct_bin = binary_discretise(forecasts,    thresholds=thresholds, mode=">=") == 1

    hits         = (fct_bin  &  obs_bin).sum(dim)
    misses       = (~fct_bin &  obs_bin).sum(dim)
    false_alarms = (fct_bin  & ~obs_bin).sum(dim)

    n_obs_events = hits + misses
    hits_random  = n_obs_events * (hits + false_alarms) / n
    denom        = n_obs_events + false_alarms - hits_random
    valid        = (denom > 0) & (n_obs_events >= min_events)
    return (hits - hits_random) / denom.where(valid)


def mse_by_domain(observations, forecasts, dim, lsm_zarr_path,
                  lsm_variable="lsm", skipna=True, valid_range=None):
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
    valid_range : dict, optional
        Per-variable physical bounds, e.g. {"10si": [0, 150], "2t": [150, 360]}.
        Values outside [min, max] are replaced with NaN before MSE computation.
        Applied on top of the generic fill-value mask (abs > 1e10).
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
        "all":   xr.ones_like(lsm, dtype=bool),
        "land":  lsm > 0.5,
        "ocean": lsm <= 0.5,
    }

    dim_list = [dim] if isinstance(dim, str) else list(dim)

    # Rechunk to consolidate the many tiny chunks produced by temporal alignment.
    # All spatial points in one chunk so each task covers a full time-slice;
    # "auto" on time dims lets dask choose a batch size that matches n_workers.
    chunk_spec = {d: -1 for d in dim_list}
    for d in observations.dims:
        if d not in chunk_spec:
            chunk_spec[d] = "auto"
    observations = observations.chunk(chunk_spec)
    forecasts = forecasts.chunk(chunk_spec)

    # Process one variable at a time to bound memory usage, but build both
    # domain tasks lazily and compute them together so dask can parallelise
    # across domains within each compute() call.
    per_var = {}
    for var in list(observations.data_vars):
        obs_v = observations[var]
        fct_v = forecasts[var]

        # Mask unphysical fill values (e.g. 9.96921e+36 from zarr/netCDF)
        obs_v = obs_v.where(np.abs(obs_v) < 1e10)
        fct_v = fct_v.where(np.abs(fct_v) < 1e10)

        # Per-variable physical bounds
        if valid_range and var in valid_range:
            vmin, vmax = valid_range[var]
            obs_v = obs_v.where((obs_v >= vmin) & (obs_v <= vmax))
            fct_v = fct_v.where((fct_v >= vmin) & (fct_v <= vmax))

        domain_results = {}
        for domain, mask in masks.items():
            obs_m = obs_v.where(mask)
            fct_m = fct_v.where(mask)
            domain_results[domain] = xs.mse(
                fct_m, obs_m, dim=dim, skipna=skipna
            )  # keep lazy — compute both domains together below

        per_var[var] = xr.concat(
            list(domain_results.values()),
            dim=xr.Variable("domain", list(domain_results.keys()))
        ).compute()   # one compute() per variable covers both domains in parallel

    return xr.Dataset(per_var)


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


def spread_skill(observations, forecasts, dim, member_dim="member", skipna=True, valid_range=None):
    """Ensemble spread and RMSE of the ensemble mean vs observations.

    Both curves are averaged over ``dim`` (typically ``reference_time`` and the
    spatial dimension), leaving ``lead_time`` on the x-axis.  A well-calibrated
    ensemble satisfies spread ≈ RMSE.

    Parameters
    ----------
    observations : xr.Dataset
        Reference dataset, dims (..., lead_time, point_index/grid_index).
    forecasts : xr.Dataset
        Ensemble forecast, dims (..., member, lead_time, point_index/grid_index).
    dim : str or list of str
        Dimensions to average over (e.g. ["point_index", "reference_time"]).
    member_dim : str
        Name of the ensemble member dimension (default: "member").
    skipna : bool
        Whether to skip NaNs.
    valid_range : dict, optional
        Per-variable physical bounds, e.g. {"2t": [150, 360]}.

    Returns
    -------
    xr.Dataset
        For each variable a DataArray with a ``curve`` dimension
        (values: ``["spread", "rmse"]``) and remaining dims (typically
        ``lead_time``).
    """
    dim_list = [dim] if isinstance(dim, str) else list(dim)

    # Mirror the chunking strategy used by _rechunk() for xskillscore metrics:
    # reduction dims and member get chunk=-1, all other dims (e.g. lead_time)
    # get chunk=1 so that each dask task covers only one lead_time slice.
    # This matches how CRPS is processed and keeps peak memory bounded.
    obs_chunk = {d: -1 for d in dim_list}
    for d in observations.dims:
        if d not in obs_chunk:
            obs_chunk[d] = 1
    observations = observations.chunk(obs_chunk)

    fct_chunk = {d: -1 for d in dim_list + [member_dim]}
    for d in forecasts.dims:
        if d not in fct_chunk:
            fct_chunk[d] = 1
    forecasts = forecasts.chunk(fct_chunk)

    per_var = {}
    for var in list(observations.data_vars):
        obs_v = observations[var]
        fct_v = forecasts[var]

        # Mask unphysical fill values (e.g. 9.96921e+36 from zarr/netCDF)
        obs_v = obs_v.where(np.abs(obs_v) < 1e10)
        fct_v = fct_v.where(np.abs(fct_v) < 1e10)

        if valid_range and var in valid_range:
            vmin, vmax = valid_range[var]
            obs_v = obs_v.where((obs_v >= vmin) & (obs_v <= vmax))
            fct_v = fct_v.where((fct_v >= vmin) & (fct_v <= vmax))

        ens_mean = fct_v.mean(dim=member_dim, skipna=skipna)
        spread = fct_v.std(dim=member_dim, skipna=skipna).mean(dim=dim_list, skipna=skipna)
        rmse = np.sqrt(((ens_mean - obs_v) ** 2).mean(dim=dim_list, skipna=skipna))

        per_var[var] = xr.concat(
            [spread, rmse],
            dim=xr.Variable("curve", ["spread", "rmse"])
        ).compute()

    return xr.Dataset(per_var)


def brier_score(fcst, obs, threshold_min, threshold_max, threshold_by,
                reduce_dims, ensemble_member_dim="member", variables=None,
                fair_correction=False):
    """Brier Score for ensemble forecasts, with optional per-call variable filtering.

    Thin wrapper around ``scores.probability.brier_score_for_ensemble`` that
    adds a ``variables`` parameter so that different thresholds can be used for
    different variables by defining multiple metrics in the config.
    Threshold ranges follow the same convention as ``ets``.

    Parameters
    ----------
    observations, forecasts : xr.Dataset
        Aligned datasets.
    threshold_min, threshold_max, threshold_by : scalar or list
        Define threshold range(s) via np.arange(min, max, by).
        Use lists for multiple segments, e.g.
            min=[0, 5], max=[5, 30], by=[0.5, 2]
    reduce_dims : list of str
        Dimensions to reduce over (e.g. ["grid_index", "reference_time"]).
    ensemble_member_dim : str
        Name of the ensemble member dimension (default: "member").
    variables : list of str, optional
        Subset of variables to compute BS for.  All common variables if None.
    fair_correction : bool
        Apply the fair Brier score correction (default: False).

    Returns
    -------
    xr.Dataset
        Brier Score for each variable with an extra ``threshold`` dimension.
    """
    from scores.probability import brier_score_for_ensemble

    if variables is not None:
        obs  = obs[variables]
        fcst = fcst[variables]

    thresholds = _build_thresholds(threshold_min, threshold_max, threshold_by).tolist()

    return brier_score_for_ensemble(
        fcst, obs,
        ensemble_member_dim=ensemble_member_dim,
        event_thresholds=thresholds,
        reduce_dims=reduce_dims,
        fair_correction=fair_correction,
    )


def brier_score_chunked(fcst, obs, threshold_min, threshold_max, threshold_by,
                        reduce_dims, ensemble_member_dim="member", variables=None,
                        fair_correction=False):
    """Brier Score computed with explicit chunking to avoid Dask shuffle operations.

    Replaces the ``scores`` library call in ``brier_score`` with a purely
    element-wise implementation:

        BS(θ) = mean( (P̂(X > θ) - I(Y > θ))² )

    where P̂(X > θ) = fraction of ensemble members exceeding θ.  This is
    computed locally per grid-point/time-step — no sorting, no data
    redistribution, no Dask shuffles.

    Chunking strategy (same as spread_skill): reduction dims get chunk=-1 so
    each task covers a full spatial/temporal slice; all other dims (e.g.
    lead_time) get chunk=1 so tasks are independent.

    Parameters
    ----------
    fcst, obs : xr.Dataset
        Aligned forecast (with member dim) and observation datasets.
    threshold_min, threshold_max, threshold_by : scalar or list
        Define threshold range(s) via np.arange(min, max, by).
    reduce_dims : list of str
        Dimensions to average over (e.g. ["grid_index", "reference_time"]).
    ensemble_member_dim : str
        Name of the ensemble member dimension (default: "member").
    variables : list of str, optional
        Subset of variables to compute BS for.
    fair_correction : bool
        Apply the fair (unbiased) Brier score correction: multiply the
        spread term by m/(m-1) where m is the ensemble size (default: False).

    Returns
    -------
    xr.Dataset
        Brier Score for each variable with an extra ``threshold`` dimension.
    """
    if variables is not None:
        obs  = obs[variables]
        fcst = fcst[variables]

    thresholds = _build_thresholds(threshold_min, threshold_max, threshold_by)
    dim_list   = [reduce_dims] if isinstance(reduce_dims, str) else list(reduce_dims)

    # Chunk: reduction dims fully in-memory per task, lead_time one slice at a time
    obs_chunk = {d: -1 for d in dim_list}
    for d in obs.dims:
        if d not in obs_chunk:
            obs_chunk[d] = 1
    obs = obs.chunk(obs_chunk)

    fct_chunk = {d: -1 for d in dim_list + [ensemble_member_dim]}
    for d in fcst.dims:
        if d not in fct_chunk:
            fct_chunk[d] = 1
    fcst = fcst.chunk(fct_chunk)

    m = fcst.sizes[ensemble_member_dim]  # ensemble size

    per_var = {}
    for var in list(obs.data_vars):
        obs_v = obs[var].where(np.abs(obs[var]) < 1e10)
        fct_v = fcst[var].where(np.abs(fcst[var]) < 1e10)

        bs_per_thresh = []
        for thresh in thresholds:
            # Exceedance probability: fraction of members > threshold (local op)
            p_fcst = (fct_v > thresh).sum(dim=ensemble_member_dim) / m
            o_bin  = (obs_v > thresh).astype(float)

            if fair_correction:
                # Fair BS: BS_fair = BS - p*(1-p)/(m-1)
                bs = ((p_fcst - o_bin) ** 2 - p_fcst * (1 - p_fcst) / (m - 1)).mean(
                    dim=dim_list, skipna=True
                )
            else:
                bs = ((p_fcst - o_bin) ** 2).mean(dim=dim_list, skipna=True)

            bs_per_thresh.append(bs)

        per_var[var] = xr.concat(
            bs_per_thresh,
            dim=xr.Variable("threshold", thresholds)
        ).compute()

    return xr.Dataset(per_var)


def power_spectrum(observations, forecasts, dim_x, dim_y, res,
                   variables=None, n_bins=None, compute_obs=True, compute_fct=True, ref_time_chunk=50):
    """2D isotropic power spectrum, one curve per (reference_time, lead_time).

    Follows the DCTPowerSpectrum approach: physical wavenumbers in rad/m,
    uniform linear binning via np.digitize, per-bin mean of DCT-II power.
    Ensemble members (``member`` dim) are averaged over if present.

    Parameters
    ----------
    observations, forecasts : xr.Dataset
        Aligned datasets with dimensions ``(reference_time, lead_time, grid_index)``
        (plus optional ``member`` dim for ensemble forecasts).
        ``len(grid_index)`` must equal ``dim_x * dim_y``.
    dim_x, dim_y : int
        Shape of the 2-D grid (rows × columns).
    res : float
        Grid spacing in **metres** (uniform, dx = dy = res).
    variables : list of str, optional
        Subset of variables to process.  All data_vars of ``forecasts`` are used if None.
    n_bins : int, optional
        Number of wavenumber bins.  Defaults to ``min(dim_x, dim_y) // 2``.
    compute_obs : bool
        If False, skip the observations spectrum (``{var}_obs``).
    compute_fct : bool
        If False, skip the forecast spectrum (``{var}_fct``).
        Use ``compute_obs=True, compute_fct=False`` when the dataset of interest
        is the reference (e.g. computing the reanalysis spectrum standalone).
    ref_time_chunk : int
        Number of reference times loaded per ``.load()`` call (default: 50).

    Returns
    -------
    xr.Dataset
        ``{var}_fct`` for each variable (plus ``{var}_obs`` if ``compute_obs=True``),
        with dimensions ``(reference_time, lead_time, k)``.
        ``k`` is the bin-centre wavenumber in rad/m.
    """
    if variables is not None:
        if compute_obs:
            observations = observations[variables]
        if compute_fct:
            forecasts = forecasts[variables]

    # ── Wavenumber grid (rad/m) ───────────────────────────────────────────────
    dx = dy = res
    kx = np.pi * np.arange(dim_x) / (dim_x * dx)
    ky = np.pi * np.arange(dim_y) / (dim_y * dy)
    KX, KY    = np.meshgrid(kx, ky, indexing='ij')
    k         = np.sqrt(KX ** 2 + KY ** 2)
    k_max     = k.max()
    _n_bins   = n_bins if n_bins is not None else min(dim_x, dim_y) // 2
    k_edges   = np.linspace(0.0, k_max, _n_bins + 1)
    k_centers = 0.5 * (k_edges[1:] + k_edges[:-1])
    digitized = np.digitize(k.flatten(), k_edges)

    def _field_spectrum(field_1d):
        """Isotropic power spectrum of a single 1-D field (grid_index order)."""
        field_2d = field_1d.reshape(dim_x, dim_y)
        if np.any(np.isnan(field_2d)):
            return None
        P = np.abs(dctn(field_2d, type=2, norm='ortho')) ** 2
        return np.array(
            [P.flatten()[digitized == i].mean() if np.any(digitized == i) else 0.0
             for i in range(1, _n_bins + 1)],
            dtype=np.float32,
        )

    def _spectrum(da):
        lead_times = da.lead_time.values
        ref_times  = da.reference_time.values
        n_rt       = len(ref_times)
        n_lt       = len(lead_times)
        has_member = "member" in da.dims
        members    = da.member.values if has_member else None
        n_mbr      = len(members) if has_member else 1

        # Identify the spatial dim (everything that is not a time or member dim)
        grid_dim = next(d for d in da.dims if d not in ("reference_time", "lead_time", "member"))

        # Normalise dim order so numpy indexing is predictable
        if has_member:
            da = da.transpose("reference_time", "lead_time", grid_dim, "member")
            out = np.full((n_rt, n_lt, n_mbr, _n_bins), np.nan, dtype=np.float32)
        else:
            da = da.transpose("reference_time", "lead_time", grid_dim)
            out = np.full((n_rt, n_lt, _n_bins), np.nan, dtype=np.float32)

        for lt_idx in range(n_lt):
            da_lt = da.isel(lead_time=lt_idx)
            for t_start in range(0, n_rt, ref_time_chunk):
                t_end = min(t_start + ref_time_chunk, n_rt)
                chunk = da_lt.isel(reference_time=slice(t_start, t_end)).load().values
                # chunk shape: (t_chunk, grid) or (t_chunk, grid, member)
                if chunk.ndim == 1:
                    chunk = chunk[np.newaxis, :]
                for t in range(chunk.shape[0]):
                    if has_member:
                        for m_idx in range(n_mbr):
                            s = _field_spectrum(chunk[t, :, m_idx])
                            if s is not None:
                                out[t_start + t, lt_idx, m_idx] = s
                    else:
                        s = _field_spectrum(chunk[t])
                        if s is not None:
                            out[t_start + t, lt_idx] = s

        if has_member:
            return xr.DataArray(
                out,
                dims=["reference_time", "lead_time", "member", "k"],
                coords={"reference_time": ref_times, "lead_time": lead_times,
                        "member": members, "k": k_centers},
            )
        return xr.DataArray(
            out,
            dims=["reference_time", "lead_time", "k"],
            coords={"reference_time": ref_times, "lead_time": lead_times, "k": k_centers},
        )

    iter_vars = list(observations.data_vars) if not compute_fct else list(forecasts.data_vars)

    per_var = {}
    for var in iter_vars:
        if compute_obs:
            print(f"  power_spectrum: {var} obs ...", flush=True)
            per_var[f"{var}_obs"] = _spectrum(observations[var])
        if compute_fct:
            print(f"  power_spectrum: {var} fct ...", flush=True)
            per_var[f"{var}_fct"] = _spectrum(forecasts[var])

    return xr.Dataset(per_var)
