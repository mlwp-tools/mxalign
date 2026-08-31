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


def spread_skill(observations, forecasts, dim, member_dim="member", skipna=True,
                 valid_range=None, ddof=1, spread_method="rms"):
    """Ensemble spread and RMSE of the ensemble mean vs observations.

    Both curves are reduced over ``dim`` (typically ``reference_time`` and the
    spatial dimension), leaving ``lead_time`` on the x-axis.

    ``rmse`` is ``sqrt(< (ens_mean - obs)^2 >)`` where ``< . >`` is the mean
    over ``dim`` — i.e. the RMSE aggregates by pooling squared errors then
    taking one square root (same convention as ``scores.continuous.rmse``).

    ``spread`` is, by default (``spread_method="rms"``), the matching
    root-**mean**-square quantity::

        spread = sqrt( < ensemble variance > )

    so spread and rmse are on the same footing: the mean of a squared
    quantity, square-rooted once.

    Interpreting spread vs rmse
    ---------------------------
    A reliable ensemble does NOT have ``spread == rmse`` at finite ensemble
    size ``M``.  If truth and members are exchangeable draws from the same
    distribution (variance ``sigma^2``)::

        E[rmse^2]   = sigma^2 * (M + 1) / M      (sigma^2 / M sampling error
                                                  of the ensemble mean, plus
                                                  sigma^2 error of the truth)
        E[spread^2] = sigma^2
        =>  spread / rmse  ->  sqrt(M / (M + 1))     ( 0.91 at M=5,
                                                       0.98 at M=31 )

    This function returns the **raw** spread and rmse; it does not apply the
    finite-``M`` inflation.  For a single "1.0 == calibrated" line that is
    comparable across ensemble sizes, multiply spread by ``sqrt((M+1)/M)``
    (or divide rmse by it) in the plotting/interpretation layer.

    Parameters
    ----------
    observations : xr.Dataset
        Reference dataset, dims (..., lead_time, point_index/grid_index).
    forecasts : xr.Dataset
        Ensemble forecast, dims (..., member, lead_time, point_index/grid_index).
    dim : str or list of str
        Dimensions to reduce over (e.g. ["point_index", "reference_time"]).
    member_dim : str
        Name of the ensemble member dimension (default: "member").
    skipna : bool
        Whether to skip NaNs.
    valid_range : dict, optional
        Per-variable physical bounds, e.g. {"2t": [150, 360]}.
    ddof : int
        Delta-degrees-of-freedom for the ensemble variance (default: 1, the
        unbiased estimator ``1/(M-1) * sum (x_i - xbar)^2``).  ``ddof=0``
        under-estimates ``sigma`` by ``sqrt((M-1)/M)`` — 11.8 % at M=5 vs
        1.7 % at M=31 — which biases cross-ensemble-size comparisons.
    spread_method : {"rms", "mean_std"}
        How the per-point spread is aggregated over ``dim``:

        - ``"rms"`` (default): ``sqrt( mean( ensemble variance ) )`` — the
          quantity that pairs with rmse (see above).
        - ``"mean_std"``: ``mean( ensemble std )``.  Jensen's inequality
          makes this <= the ``"rms"`` value, by a gap that grows with how
          heterogeneous the spread is across ``dim`` (~20 % for 2t on a
          regional domain) and that is itself mildly ensemble-size
          dependent.  Kept only for reproducing results produced before
          this parameter existed: ``spread_method="mean_std", ddof=0``
          reproduces the historical output exactly.

    Returns
    -------
    xr.Dataset
        For each variable a DataArray with a ``curve`` dimension
        (values: ``["spread", "rmse"]``) and remaining dims (typically
        ``lead_time``).
    """
    if spread_method not in ("rms", "mean_std"):
        raise ValueError(
            f"spread_method must be 'rms' or 'mean_std', got {spread_method!r}"
        )

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
        ens_var = fct_v.var(dim=member_dim, ddof=ddof, skipna=skipna)
        if spread_method == "rms":
            spread = np.sqrt(ens_var.mean(dim=dim_list, skipna=skipna))
        else:  # "mean_std"
            spread = np.sqrt(ens_var).mean(dim=dim_list, skipna=skipna)
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


def spread_skill_spectrum(observations, forecasts, dim_x, dim_y, res,
                          variables=None, n_bins=None, ref_time_chunk=50,
                          member_dim="member"):
    """Ensemble spread–skill as a function of spatial scale (spectral space).

    Adapted from SpectralGridded (bris). Order of operations:
      1. DCT-II each ensemble member and the analysis independently
      2. spread_variance(kx,ky) = sample variance across members of DCT coefficients
         error_variance(kx,ky)  = (ensemble-mean DCT − analysis DCT)²
      3. Isotropic-average over k-angle bins → 1-D spectra per (time, leadtime)

    Step 4 (time-average + sqrt) is deferred to the caller:
        spread(k) = sqrt(mean(spread_variance, axis="reference_time"))
        rmse(k)   = sqrt(mean(error_variance,  axis="reference_time"))

    Parameters
    ----------
    observations : xr.Dataset
        Analysis/reference dataset, dims (reference_time, lead_time, grid_index).
    forecasts : xr.Dataset
        Ensemble forecasts, dims (reference_time, lead_time, grid_index, member_dim).
    dim_x, dim_y : int
        2-D grid shape; len(grid_index) must equal dim_x * dim_y.
    res : float
        Grid spacing in metres (uniform, dx = dy = res).
    variables : list of str, optional
        Subset of variables. All data_vars of forecasts are used if None.
    n_bins : int, optional
        Number of wavenumber bins. Defaults to min(dim_x, dim_y) // 2.
    ref_time_chunk : int
        Reference times loaded per .load() call (default: 50).
    member_dim : str
        Name of the ensemble member dimension (default: "member").

    Returns
    -------
    xr.Dataset
        For each variable, four DataArrays (all dims reference_time, lead_time, k
        except spectrum_fct which has an extra member dim):
          {var}_spread_var    : sample variance of DCT coefficients across members
          {var}_error_var     : squared error of ensemble-mean DCT vs analysis DCT
          {var}_spectrum_fct  : power spectrum per ensemble member
          {var}_spectrum_obs  : power spectrum of the analysis
        k is the bin-centre wavenumber in rad/m.
    """
    if variables is not None:
        observations = observations[variables]
        forecasts    = forecasts[variables]

    # Wavenumber grid, same convention as power_spectrum / DCTPowerSpectrum
    dx = dy = res
    kx = np.pi * np.arange(dim_x) / (dim_x * dx)
    ky = np.pi * np.arange(dim_y) / (dim_y * dy)
    KX, KY    = np.meshgrid(kx, ky, indexing='ij')
    k         = np.sqrt(KX**2 + KY**2)
    k_max     = k.max()
    _n_bins   = n_bins if n_bins is not None else min(dim_x, dim_y) // 2
    k_edges   = np.linspace(0.0, k_max, _n_bins + 1)
    k_centers = 0.5 * (k_edges[1:] + k_edges[:-1])
    digitized = np.digitize(k.flatten(), k_edges)

    def _bin_average_2d(field_2d):
        """Angle-average a (dim_x, dim_y) spectral field → 1-D array of length n_bins."""
        flat = field_2d.flatten()
        return np.array(
            [flat[digitized == j].mean() if np.any(digitized == j) else np.nan
             for j in range(1, _n_bins + 1)],
            dtype=np.float32,
        )

    def _bin_average_2d_per_member(field_2d_ens):
        """Angle-average a (dim_x, dim_y, N) array → (n_bins, N)."""
        flat = field_2d_ens.reshape(-1, field_2d_ens.shape[-1])  # (grid, N)
        return np.array(
            [flat[digitized == j].mean(axis=0) if np.any(digitized == j) else np.full(flat.shape[-1], np.nan)
             for j in range(1, _n_bins + 1)],
            dtype=np.float32,
        )

    def _process_variable(fct_da, obs_da):
        ref_times  = fct_da.reference_time.values
        lead_times = fct_da.lead_time.values
        n_rt       = len(ref_times)
        n_lt       = len(lead_times)
        members    = fct_da[member_dim].values
        N          = len(members)

        grid_dim = next(d for d in fct_da.dims if d not in ("reference_time", "lead_time", member_dim))

        fct_da = fct_da.transpose("reference_time", "lead_time", grid_dim, member_dim)
        obs_da = obs_da.transpose("reference_time", "lead_time", grid_dim)

        spread_var   = np.full((n_rt, n_lt, _n_bins), np.nan, dtype=np.float32)
        error_var    = np.full((n_rt, n_lt, _n_bins), np.nan, dtype=np.float32)
        spectrum_fct = np.full((n_rt, n_lt, _n_bins, N), np.nan, dtype=np.float32)
        spectrum_obs = np.full((n_rt, n_lt, _n_bins), np.nan, dtype=np.float32)

        for t_start in range(0, n_rt, ref_time_chunk):
            t_end      = min(t_start + ref_time_chunk, n_rt)
            fct_chunk  = fct_da.isel(reference_time=slice(t_start, t_end)).load().values
            obs_chunk  = obs_da.isel(reference_time=slice(t_start, t_end)).load().values
            # fct_chunk: (t_chunk, lt, grid, member) ; obs_chunk: (t_chunk, lt, grid)
            t_chunk = fct_chunk.shape[0]

            # Reshape spatial dimension: (t, lt, grid, member) → (t, lt, nx, ny, member)
            obs_2d = obs_chunk.reshape(t_chunk, n_lt, dim_x, dim_y)
            fct_2d = fct_chunk.reshape(t_chunk, n_lt, dim_x, dim_y, N)

            # Step 1: DCT over spatial axes (2, 3) for the whole chunk at once
            obs_k  = dctn(obs_2d, axes=(2, 3), type=2, norm='ortho')   # (t, lt, kx, ky)
            pred_k = dctn(fct_2d, axes=(2, 3), type=2, norm='ortho')   # (t, lt, kx, ky, N)

            # Step 2: spread and error variance in spectral space
            ens_mean        = pred_k.mean(axis=4)                         # (t, lt, kx, ky)
            spread_var_k    = ((pred_k - ens_mean[..., None]) ** 2).sum(axis=4) / (N - 1)
            error_var_k     = (ens_mean - obs_k) ** 2                    # (t, lt, kx, ky)
            P_obs           = obs_k ** 2                                  # (t, lt, kx, ky)
            P_pred          = pred_k ** 2                                 # (t, lt, kx, ky, N)

            # Step 3: angle-average over k, per (t, lt)
            for t in range(t_chunk):
                if np.any(np.isnan(fct_chunk[t])) or np.any(np.isnan(obs_chunk[t])):
                    continue
                for lt in range(n_lt):
                    spread_var[t_start + t, lt]    = _bin_average_2d(spread_var_k[t, lt])
                    error_var[t_start + t, lt]     = _bin_average_2d(error_var_k[t, lt])
                    spectrum_fct[t_start + t, lt]  = _bin_average_2d_per_member(P_pred[t, lt])
                    spectrum_obs[t_start + t, lt]  = _bin_average_2d(P_obs[t, lt])

        coords_3d = {"reference_time": ref_times, "lead_time": lead_times, "k": k_centers}
        return {
            "spread_var":   xr.DataArray(spread_var,   dims=["reference_time", "lead_time", "k"], coords=coords_3d),
            "error_var":    xr.DataArray(error_var,    dims=["reference_time", "lead_time", "k"], coords=coords_3d),
            "spectrum_fct": xr.DataArray(spectrum_fct, dims=["reference_time", "lead_time", "k", member_dim],
                                         coords={**coords_3d, member_dim: members}),
            "spectrum_obs": xr.DataArray(spectrum_obs, dims=["reference_time", "lead_time", "k"], coords=coords_3d),
        }

    per_var = {}
    for var in list(forecasts.data_vars):
        if var not in observations.data_vars:
            print(f"  spread_skill_spectrum: '{var}' not in observations — skipping.", flush=True)
            continue
        print(f"  spread_skill_spectrum: {var} ...", flush=True)
        result = _process_variable(forecasts[var], observations[var])
        for suffix, da in result.items():
            per_var[f"{var}_{suffix}"] = da

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


_MURPHY_STATS = [
    "N", "f_mean", "o_mean", "sigma_f", "sigma_o", "r",
    "bias", "spread", "corr", "ratio", "mse", "identity_residual",
]

#: Station-metadata coordinates carried through from the observation dataset.
_STATION_COORDS = ("code", "latitude", "longitude", "altitude")


def _drop_nondim_coords(da):
    """Strip non-dimension coordinates so concat along a new dim can't conflict.

    Forecast and observation datasets carry different point_index coordinates
    (obs has ``code``/``altitude``, the delaunay-interpolated forecast only has
    ``latitude``/``longitude``), so statistics derived from each side disagree
    on coords.  They are stripped here and reattached once, from the
    observations, in ``murphy_decomposition``.
    """
    return da.drop_vars([c for c in da.coords if c not in da.dims], errors="ignore")


def murphy_decomposition(observations, forecasts, dim="reference_time",
                         valid_range=None, member_dim="member",
                         check_identity=True, identity_rtol=1e-6,
                         identity_atol=1e-8):
    """Per-station Murphy decomposition of MSE into bias / spread / correlation.

    For each station ``j`` independently, using only that station's valid
    (forecast, observation) pairs::

        bias_j   = (f̄_j - ō_j)²
        spread_j = (σ_f,j - σ_o,j)²
        corr_j   = 2·σ_f,j·σ_o,j·(1 - r_j)
        MSE_j    = bias_j + spread_j + corr_j        (exact identity)

    The decomposition separates *systematic offset* (bias), *variance
    mismatch* (spread) and *phase/timing error* (corr).  Its purpose here is
    to diagnose whether a model scores well because of genuine skill or
    because it is over-smoothed: a damped-variance model shows
    ``ratio = σ_f/σ_o < 1`` and a large ``spread`` term while its ``corr``
    term stays low.

    **Deterministic forecasts only.**  A forecast carrying ``member_dim`` is
    rejected rather than silently collapsed — pick the ensemble mean or a
    single member explicitly in the config instead.

    Aggregation across stations is deliberately NOT done here — see
    ``code/aggregate_murphy_results.py``.  Keeping this function purely
    per-station means the (subjective) ``min_obs_per_station`` cutoff can be
    re-tuned without re-running the expensive alignment/verification step.

    Parameters
    ----------
    observations : xr.Dataset
        Reference dataset, dims (..., ``dim``, point_index).
    forecasts : xr.Dataset
        Deterministic forecast, aligned onto the same point_index.
    dim : str or list of str
        Dimension(s) to reduce over when forming each station's sample —
        typically ``"reference_time"``.  The spatial dimension must NOT be
        listed here; that is what makes the result per-station.
    valid_range : dict, optional
        Per-variable physical bounds, e.g. ``{"2t": [150, 360]}``.  Applied on
        top of the generic fill-value mask (``abs > 1e10``), same convention as
        ``mse_by_domain`` / ``spread_skill``.
    member_dim : str
        Ensemble dimension name whose presence is an error (default "member").
    check_identity : bool
        Verify ``bias + spread + corr == MSE`` to floating-point tolerance and
        raise if violated (default True).
    identity_rtol, identity_atol : float
        Tolerances for that check.

    Returns
    -------
    xr.Dataset
        One variable per input variable, each with an extra ``stat`` dimension
        taking the values ``N, f_mean, o_mean, sigma_f, sigma_o, r, bias,
        spread, corr, ratio, mse, identity_residual`` and retaining
        ``lead_time`` and ``point_index``.  Station metadata (``code``,
        ``latitude``, ``longitude``, ``altitude``) is preserved as point_index
        coordinates for later stratification by elevation/region.

    Notes
    -----
    **Population moments (ddof=0) are used deliberately, not sample moments.**
    The identity ``MSE = bias + spread + corr`` holds exactly only when σ_f,
    σ_o and the covariance behind ``r`` share the same ``÷N`` normalisation.
    With ddof=1 the three terms miss MSE by a systematic factor ~N/(N-1) —
    about 5% at N=20 — which would defeat the identity check above.  Sample
    standard deviations are therefore intentionally not offered here.

    ``corr`` is evaluated as ``2·(σ_f·σ_o - cov)``, which is algebraically
    identical to ``2·σ_f·σ_o·(1 - r)`` but stays finite when a station's
    series is constant (σ = 0, ``r`` undefined).  The decomposition therefore
    remains exact for degenerate stations while ``r`` itself is reported NaN.

    ``N`` is stored as a float (the ``stat`` dimension is homogeneous);
    it is an exact integer count of jointly-valid pairs.
    """
    dim_list = [dim] if isinstance(dim, str) else list(dim)

    if member_dim in forecasts.dims:
        raise ValueError(
            f"murphy_decomposition is a deterministic metric but the forecast "
            f"has an ensemble dimension '{member_dim}' of size "
            f"{forecasts.sizes[member_dim]}. Reduce it explicitly first (e.g. "
            f"ensemble mean, or select one member) rather than relying on this "
            f"metric to choose for you."
        )

    # The whole point is a per-station sample, so the spatial dim must survive.
    spatial_dims = [d for d in ("point_index", "grid_index") if d in observations.dims]
    for d in dim_list:
        if d in spatial_dims:
            raise ValueError(
                f"dim={dim!r} includes the spatial dimension '{d}', which would "
                f"pool stations together and defeat the per-station design. "
                f"Use dim=['reference_time'] and aggregate afterwards with "
                f"code/aggregate_murphy_results.py."
            )
    if not spatial_dims:
        print("  murphy_decomposition: no point_index/grid_index dimension found — "
              "results will not be per-station.", flush=True)

    per_var = {}
    for var in list(observations.data_vars):
        if var not in forecasts.data_vars:
            print(f"  murphy_decomposition: '{var}' not in forecasts — skipping.", flush=True)
            continue

        obs_v = observations[var]
        fct_v = forecasts[var]

        # Mask unphysical fill values (e.g. 9.96921e+36 from zarr/netCDF)
        obs_v = obs_v.where(np.abs(obs_v) < 1e10)
        fct_v = fct_v.where(np.abs(fct_v) < 1e10)

        if valid_range and var in valid_range:
            vmin, vmax = valid_range[var]
            obs_v = obs_v.where((obs_v >= vmin) & (obs_v <= vmax))
            fct_v = fct_v.where((fct_v >= vmin) & (fct_v <= vmax))

        # Joint mask: a pair counts only if BOTH sides are valid, so every
        # moment below is computed over the identical sample.  Obs coverage is
        # highly variable here (tp reports at 00/12Z only for most stations),
        # so this is what makes N_j meaningful.
        valid = obs_v.notnull() & fct_v.notnull()
        obs_v = obs_v.where(valid)
        fct_v = fct_v.where(valid)

        # Promote to float64 before accumulating moments. The decomposition
        # differences large, similar numbers (f̄ ≈ ō ≈ 280 K for 2t), so in
        # float32 the identity residual lands around 1e-5 -- larger than MSE
        # itself times the check tolerance, i.e. the identity check would fail
        # on precision alone. Cheap here: point obs are ~2131 stations, not a
        # 3M-cell grid.
        obs_v = obs_v.astype("float64")
        fct_v = fct_v.astype("float64")

        n = valid.sum(dim=dim_list)

        f_mean = fct_v.mean(dim=dim_list, skipna=True)
        o_mean = obs_v.mean(dim=dim_list, skipna=True)

        fa = fct_v - f_mean
        oa = obs_v - o_mean

        # skipna means each of these divides by the same count n -> ddof=0
        var_f = (fa ** 2).mean(dim=dim_list, skipna=True)
        var_o = (oa ** 2).mean(dim=dim_list, skipna=True)
        cov   = (fa * oa).mean(dim=dim_list, skipna=True)

        sigma_f = np.sqrt(var_f)
        sigma_o = np.sqrt(var_o)

        mse    = ((fct_v - obs_v) ** 2).mean(dim=dim_list, skipna=True)
        bias   = (f_mean - o_mean) ** 2
        spread = (sigma_f - sigma_o) ** 2
        # == 2*sigma_f*sigma_o*(1-r), but finite when either sigma is 0
        corr   = 2.0 * (sigma_f * sigma_o - cov)

        denom = sigma_f * sigma_o
        r     = cov / denom.where(denom > 0)
        ratio = sigma_f / sigma_o.where(sigma_o > 0)

        residual = bias + spread + corr - mse

        stats = {
            "N": n.astype("float64"), "f_mean": f_mean, "o_mean": o_mean,
            "sigma_f": sigma_f, "sigma_o": sigma_o, "r": r,
            "bias": bias, "spread": spread, "corr": corr,
            "ratio": ratio, "mse": mse, "identity_residual": residual,
        }
        assert list(stats) == _MURPHY_STATS, "stat ordering drifted from _MURPHY_STATS"

        per_var[var] = xr.concat(
            [_drop_nondim_coords(stats[s]) for s in _MURPHY_STATS],
            dim=xr.Variable("stat", _MURPHY_STATS),
        ).compute()

    result = xr.Dataset(per_var)

    # Reattach station metadata (dropped above so the concat can't conflict).
    station_coords = {
        c: observations.coords[c]
        for c in _STATION_COORDS
        if c in observations.coords and set(observations.coords[c].dims) <= set(result.dims)
    }
    if station_coords:
        result = result.assign_coords(station_coords)

    if check_identity:
        _check_murphy_identity(result, rtol=identity_rtol, atol=identity_atol)

    return result


def _check_murphy_identity(result, rtol=1e-6, atol=1e-8):
    """Raise if bias + spread + corr departs from MSE beyond tolerance."""
    for var in result.data_vars:
        da  = result[var]
        res = da.sel(stat="identity_residual").values
        mse = da.sel(stat="mse").values
        finite = np.isfinite(res) & np.isfinite(mse)
        if not finite.any():
            continue
        tol  = atol + rtol * np.abs(mse[finite])
        bad  = np.abs(res[finite]) > tol
        if bad.any():
            worst = np.abs(res[finite])[bad].max()
            raise ValueError(
                f"murphy_decomposition: MSE identity violated for '{var}' at "
                f"{int(bad.sum())}/{int(finite.sum())} (station, lead_time) points; "
                f"max |bias+spread+corr-MSE| = {worst:.3e}. This indicates a bug in "
                f"the decomposition, not a data problem."
            )


def murphy_decomposition_domain_mean(observations, forecasts,
                                     dim=("reference_time", "grid_index"),
                                     valid_range=None, member_dim="member",
                                     check_identity=True, identity_rtol=1e-6,
                                     identity_atol=1e-8):
    """Domain-mean Murphy decomposition of MSE into bias / spread / correlation.

    Same identity and math as ``murphy_decomposition``::

        bias   = (f̄ - ō)²
        spread = (σ_f - σ_o)²
        corr   = 2·(σ_f·σ_o - cov)
        MSE    = bias + spread + corr        (exact identity)
        ratio  = σ_f / σ_o                   <-- the smoothing diagnostic

    but pooled into **one number per lead_time**: by default BOTH
    ``reference_time`` AND the spatial dimension (``grid_index``) are pooled
    into a single sample, instead of keeping a per-station/per-cell result.

    Why pooling the spatial dimension here is safe, unlike ``murphy_decomposition``
    ---------------------------------------------------------------------------
    ``murphy_decomposition`` explicitly *refuses* to let ``dim`` include the
    spatial dimension (``point_index``/``grid_index``) because pooling SYNOP
    stations would bias the result: station density correlates with
    terrain/country, so pooling pairs would let the aggregate be dominated by
    wherever the observation network happens to be dense, silently turning a
    model score into a partial measure of network geometry rather than model
    skill (see that function's docstring, and design decision 5 in
    ``murphy-decomposition.md``).

    This function targets a structurally different reference: HA-AROME's own
    analysis on **its own regular model grid** (3,071,581 cells). Every grid
    cell there is an equally-weighted, regularly-gridded reference value by
    construction — there is no equivalent of "station density" bias to guard
    against. Pooling every cell (and every reference date) into one
    domain-mean number per lead_time is therefore a legitimate, unbiased
    spatial average for this reference type, not a shortcut around the same
    problem ``murphy_decomposition`` exists to prevent. That is also why this
    is a new, separate function rather than a flag on the existing one: the
    two functions make *opposite* default assumptions about whether the
    spatial dimension may appear in ``dim`` because the underlying safety
    argument for each reference type is genuinely different, not a
    superset/subset of the other. Naively pointing the per-station function
    at the full 3.07M-cell grid instead would also produce a per-cell output
    of roughly 12 stats × up to 25 leads × 3.07M cells × 5 vars — tens of GB
    per run — which this domain-mean reduction avoids by design, not just by
    convention.

    Mirrors the ``reduce_dims: ["grid_index", "reference_time"]`` convention
    already used by the ``scores.continuous.rmse``-based
    ``*_RMSE_vs_ha_analysis.sh`` scripts, so results from this function are
    directly comparable in shape (``stat`` x ``lead_time`` per variable) to
    those RMSE outputs.

    Parameters
    ----------
    observations : xr.Dataset
        HA-AROME analysis (mbr000), dims (reference_time, lead_time, grid_index).
    forecasts : xr.Dataset
        Deterministic forecast, aligned onto the same grid_index (no
        interpolation needed when both share the HA-AROME grid).
    dim : str or list/tuple of str
        Dimension(s) to pool over. Defaults to BOTH ``reference_time`` and
        ``grid_index``, so the output keeps only ``lead_time`` (plus
        ``stat``). Unlike ``murphy_decomposition``, including the spatial
        dimension here is the intended default usage, not an error.
    valid_range : dict, optional
        Per-variable physical bounds, e.g. ``{"2t": [150, 360]}``. Same
        convention as ``murphy_decomposition``.
    member_dim : str
        Ensemble dimension name whose presence is an error (default
        "member") — same deterministic-only restriction as
        ``murphy_decomposition``; ensemble-mean-vs-single-member is the
        caller's decision, not the metric's.
    check_identity : bool
        Verify ``bias + spread + corr == MSE`` to floating-point tolerance
        and raise if violated (default True).
    identity_rtol, identity_atol : float
        Tolerances for that check.

    Returns
    -------
    xr.Dataset
        One variable per input variable, each with a ``stat`` dimension
        (same order/values as ``murphy_decomposition``: ``N, f_mean, o_mean,
        sigma_f, sigma_o, r, bias, spread, corr, ratio, mse,
        identity_residual``) and whatever dims survive pooling (typically
        just ``lead_time``). No ``point_index``/``grid_index`` dimension
        survives, so — unlike ``murphy_decomposition`` — there is no
        per-cell/per-station coordinate to reattach afterwards.

    Notes
    -----
    Same float64 promotion / ddof=0 population-moments / joint valid-pair
    mask / ``corr = 2·(σ_f·σ_o - cov)`` design as ``murphy_decomposition`` —
    see that function's docstring for the detailed rationale (identity
    exactness, degenerate-cell safety), all of which applies unchanged here.
    The math is identical between the two functions; only which dimensions
    get pooled, and therefore what "one sample" means, differs.
    """
    dim_list = [dim] if isinstance(dim, str) else list(dim)

    if member_dim in forecasts.dims:
        raise ValueError(
            f"murphy_decomposition_domain_mean is a deterministic metric but "
            f"the forecast has an ensemble dimension '{member_dim}' of size "
            f"{forecasts.sizes[member_dim]}. Reduce it explicitly first (e.g. "
            f"ensemble mean, or select one member) rather than relying on "
            f"this metric to choose for you."
        )

    per_var = {}
    for var in list(observations.data_vars):
        if var not in forecasts.data_vars:
            print(f"  murphy_decomposition_domain_mean: '{var}' not in forecasts — skipping.", flush=True)
            continue

        obs_v = observations[var]
        fct_v = forecasts[var]

        # Mask unphysical fill values (e.g. 9.96921e+36 from zarr/netCDF)
        obs_v = obs_v.where(np.abs(obs_v) < 1e10)
        fct_v = fct_v.where(np.abs(fct_v) < 1e10)

        if valid_range and var in valid_range:
            vmin, vmax = valid_range[var]
            obs_v = obs_v.where((obs_v >= vmin) & (obs_v <= vmax))
            fct_v = fct_v.where((fct_v >= vmin) & (fct_v <= vmax))

        # Joint mask: a pair counts only if BOTH sides are valid, so every
        # moment below is computed over the identical (pooled) sample.
        valid = obs_v.notnull() & fct_v.notnull()
        obs_v = obs_v.where(valid)
        fct_v = fct_v.where(valid)

        # Promote to float64 before accumulating moments -- same rationale as
        # murphy_decomposition (large, similar-magnitude numbers, e.g.
        # f̄≈ō≈280K for 2t): in float32 the identity residual would land
        # around 1e-5, larger than the check tolerance would allow.
        obs_v = obs_v.astype("float64")
        fct_v = fct_v.astype("float64")

        n = valid.sum(dim=dim_list)

        f_mean = fct_v.mean(dim=dim_list, skipna=True)
        o_mean = obs_v.mean(dim=dim_list, skipna=True)

        fa = fct_v - f_mean
        oa = obs_v - o_mean

        # skipna means each of these divides by the same count n -> ddof=0
        var_f = (fa ** 2).mean(dim=dim_list, skipna=True)
        var_o = (oa ** 2).mean(dim=dim_list, skipna=True)
        cov   = (fa * oa).mean(dim=dim_list, skipna=True)

        sigma_f = np.sqrt(var_f)
        sigma_o = np.sqrt(var_o)

        mse    = ((fct_v - obs_v) ** 2).mean(dim=dim_list, skipna=True)
        bias   = (f_mean - o_mean) ** 2
        spread = (sigma_f - sigma_o) ** 2
        # == 2*sigma_f*sigma_o*(1-r), but finite when either sigma is 0
        corr   = 2.0 * (sigma_f * sigma_o - cov)

        denom = sigma_f * sigma_o
        r     = cov / denom.where(denom > 0)
        ratio = sigma_f / sigma_o.where(sigma_o > 0)

        residual = bias + spread + corr - mse

        stats = {
            "N": n.astype("float64"), "f_mean": f_mean, "o_mean": o_mean,
            "sigma_f": sigma_f, "sigma_o": sigma_o, "r": r,
            "bias": bias, "spread": spread, "corr": corr,
            "ratio": ratio, "mse": mse, "identity_residual": residual,
        }
        assert list(stats) == _MURPHY_STATS, "stat ordering drifted from _MURPHY_STATS"

        per_var[var] = xr.concat(
            [_drop_nondim_coords(stats[s]) for s in _MURPHY_STATS],
            dim=xr.Variable("stat", _MURPHY_STATS),
        ).compute()

    result = xr.Dataset(per_var)

    # No spatial dimension survives pooling, so unlike murphy_decomposition
    # there is no per-cell/per-station coordinate to reattach here.

    if check_identity:
        _check_murphy_identity(result, rtol=identity_rtol, atol=identity_atol)

    return result


def taylor_diagram(observations, forecasts, dim, member_dim="member", skipna=True, valid_range=None):
    """Taylor diagram statistics per ensemble member and ensemble mean.

    For each (reference_time, lead_time) pair and each forecast field
    (individual members + ensemble mean):
      1. Spatial anomalies are formed by subtracting the domain mean.
      2. Spatial std, Pearson correlation, and RMSE of anomalies are computed
         over ``dim``.
    Results are then averaged over reference_time (variance-preserving RMS for
    std, arithmetic mean for correlation), yielding one triplet per
    (lead_time, member).

    The RMSE here is the RMSE of the anomaly fields — equivalent to the
    centered RMSE (CRMSE) used in Taylor diagrams, i.e. the distance to the
    reference point on the diagram: RMSE² = σ_f² + σ_o² − 2·R·σ_f·σ_o.

    Parameters
    ----------
    observations : xr.Dataset
    forecasts : xr.Dataset
        Ensemble forecast with a ``member_dim`` dimension.
    dim : str or list of str
        Spatial dimension(s) to compute statistics over (e.g. ``"grid_index"``).
        Do NOT include ``lead_time`` or ``reference_time`` here.
    member_dim : str
        Name of the ensemble member dimension (default: ``"member"``).
    skipna : bool
    valid_range : dict, optional
        Per-variable physical bounds, e.g. ``{"2t": [150, 360]}``.

    Returns
    -------
    xr.Dataset
        For each variable, a DataArray with dimensions ``(stat, lead_time, member)``
        where ``stat ∈ ["std_fct", "std_obs", "corr", "crmse"]`` and ``member``
        contains all individual member labels plus ``"ens_mean"``.
        ``std_obs`` is the same for all members (the reference std).
    """
    dim_list = [dim] if isinstance(dim, str) else list(dim)

    # All members in memory at once (needed for ensemble mean); spatial dims
    # also fully in-memory. lead_time and reference_time each get chunk=1.
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

        obs_v = obs_v.where(np.abs(obs_v) < 1e10)
        fct_v = fct_v.where(np.abs(fct_v) < 1e10)

        if valid_range and var in valid_range:
            vmin, vmax = valid_range[var]
            obs_v = obs_v.where((obs_v >= vmin) & (obs_v <= vmax))
            fct_v = fct_v.where((fct_v >= vmin) & (fct_v <= vmax))

        # Observation anomaly and its variance (same for all members)
        obs_anom = obs_v - obs_v.mean(dim=dim_list, skipna=skipna)
        var_obs   = (obs_anom ** 2).mean(dim=dim_list, skipna=skipna)

        # Determine which dims to average over (everything except lead_time)
        time_dims = [d for d in var_obs.dims if d != "lead_time"]

        # RMS-average of obs variance over time → std_obs per lead_time
        std_obs = np.sqrt(var_obs.mean(dim=time_dims, skipna=skipna))

        # Build forecast slices: one per member + ensemble mean
        members = fct_v[member_dim].values
        fct_slices = (
            [(str(m), fct_v.sel({member_dim: m})) for m in members]
            + [("ens_mean", fct_v.mean(dim=member_dim, skipna=skipna))]
        )

        member_results = []
        for _label, fct_da in fct_slices:
            # Drop scalar member coordinate left behind by .sel() so that stats
            # computed from fct_da and obs_anom share the same coordinates.
            fct_da = fct_da.drop_vars(member_dim, errors="ignore")
            fct_anom = fct_da - fct_da.mean(dim=dim_list, skipna=skipna)
            var_fct  = (fct_anom ** 2).mean(dim=dim_list, skipna=skipna)
            cov      = (fct_anom * obs_anom).mean(dim=dim_list, skipna=skipna)

            std_fct_t = np.sqrt(var_fct)
            denom     = std_fct_t * np.sqrt(var_obs)
            corr_t    = cov / denom.where(denom > 0)

            std_fct = np.sqrt(var_fct.mean(dim=time_dims, skipna=skipna))
            corr    = corr_t.mean(dim=time_dims, skipna=skipna)
            rmse    = np.sqrt(
                (std_fct ** 2 + std_obs ** 2 - 2 * corr * std_fct * std_obs).clip(0)
            )

            member_results.append(xr.concat(
                [std_fct, std_obs, corr, rmse],
                dim=xr.Variable("stat", ["std_fct", "std_obs", "corr", "crmse"])
            ))

        member_labels = [str(m) for m in members] + ["ens_mean"]
        per_var[var] = xr.concat(
            member_results,
            dim=xr.Variable(member_dim, member_labels)
        ).compute()

    return xr.Dataset(per_var)


def power_spectrum_patches(observations, forecasts, dim_x, dim_y, res,
                            patch_size_x, patch_size_y,
                            patch_stride_x=None, patch_stride_y=None,
                            variables=None, n_bins=None,
                            compute_obs=True, compute_fct=True,
                            ref_time_chunk=50):
    """2D isotropic power spectrum computed independently over spatial patches.

    Tiles the (dim_x, dim_y) domain into patches and computes a separate
    spectrum per patch, allowing diagnosis of spatial non-stationarity
    (e.g. flat terrain vs mountains).  Patch layout matches the SpectralCRPS
    patch loss used in training (patch_size, patch_stride).

    Boundary patches that are smaller than patch_size are zero-padded so that
    all patches share the same wavenumber grid.

    Parameters
    ----------
    observations, forecasts : xr.Dataset
        Same as power_spectrum.
    dim_x, dim_y : int
        Full 2-D grid shape (rows × columns).
    res : float
        Grid spacing in metres.
    patch_size_x, patch_size_y : int
        Patch size in rows and columns.  E.g. 403, 478 for a 4×4 tiling of
        the 1609×1909 UWC-W domain.
    patch_stride_x, patch_stride_y : int, optional
        Stride between patches.  Defaults to patch_size (non-overlapping).
    variables, n_bins, compute_obs, compute_fct, ref_time_chunk :
        Same as power_spectrum.

    Returns
    -------
    xr.Dataset
        Same variables as power_spectrum ({var}_fct, {var}_obs) but with an
        extra leading 'patch' dimension whose values are "p{row}_{col}".
    """
    _psx   = patch_size_x
    _psy   = patch_size_y
    _pstx  = patch_stride_x if patch_stride_x is not None else _psx
    _psty  = patch_stride_y if patch_stride_y is not None else _psy

    # Enumerate patches: (row_idx, col_idx, row0, col0, actual_rows, actual_cols)
    patches = []
    ri, row0 = 0, 0
    while row0 < dim_x:
        ci, col0 = 0, 0
        while col0 < dim_y:
            pr = min(_psx, dim_x - row0)
            pc = min(_psy, dim_y - col0)
            patches.append((ri, ci, row0, col0, pr, pc))
            col0 += _psty
            ci   += 1
        row0 += _pstx
        ri   += 1

    patch_names = [f"p{ri}_{ci}" for ri, ci, *_ in patches]

    # Wavenumber grid — fixed to nominal patch size so all patches share k
    dx = dy = res
    kx        = np.pi * np.arange(_psx) / (_psx * dx)
    ky        = np.pi * np.arange(_psy) / (_psy * dy)
    KX, KY    = np.meshgrid(kx, ky, indexing='ij')
    k         = np.sqrt(KX**2 + KY**2)
    k_max     = k.max()
    _n_bins   = n_bins if n_bins is not None else min(_psx, _psy) // 2
    k_edges   = np.linspace(0.0, k_max, _n_bins + 1)
    k_centers = 0.5 * (k_edges[1:] + k_edges[:-1])
    digitized = np.digitize(k.flatten(), k_edges)

    def _field_spectrum_patch(field_1d, row0, col0, pr, pc):
        """Spectrum of one patch; zero-pads to (_psx, _psy) if boundary patch."""
        field_2d = field_1d.reshape(dim_x, dim_y)[row0:row0 + pr, col0:col0 + pc]
        if np.any(np.isnan(field_2d)):
            return None
        if pr < _psx or pc < _psy:
            padded = np.zeros((_psx, _psy), dtype=field_2d.dtype)
            padded[:pr, :pc] = field_2d
            field_2d = padded
        P = np.abs(dctn(field_2d, type=2, norm='ortho')) ** 2
        return np.array(
            [P.flatten()[digitized == i].mean() if np.any(digitized == i) else 0.0
             for i in range(1, _n_bins + 1)],
            dtype=np.float32,
        )

    def _spectrum_patches(da):
        lead_times = da.lead_time.values
        ref_times  = da.reference_time.values
        n_rt       = len(ref_times)
        n_lt       = len(lead_times)
        n_patches  = len(patches)
        has_member = "member" in da.dims
        members    = da.member.values if has_member else None
        n_mbr      = len(members) if has_member else 1

        grid_dim = next(d for d in da.dims if d not in ("reference_time", "lead_time", "member"))

        if has_member:
            da  = da.transpose("reference_time", "lead_time", grid_dim, "member")
            out = np.full((n_patches, n_rt, n_lt, n_mbr, _n_bins), np.nan, dtype=np.float32)
        else:
            da  = da.transpose("reference_time", "lead_time", grid_dim)
            out = np.full((n_patches, n_rt, n_lt, _n_bins), np.nan, dtype=np.float32)

        for lt_idx in range(n_lt):
            da_lt = da.isel(lead_time=lt_idx)
            for t_start in range(0, n_rt, ref_time_chunk):
                t_end = min(t_start + ref_time_chunk, n_rt)
                chunk = da_lt.isel(reference_time=slice(t_start, t_end)).load().values
                if chunk.ndim == 1:
                    chunk = chunk[np.newaxis, :]
                for t in range(chunk.shape[0]):
                    for pi, (_, _, row0, col0, pr, pc) in enumerate(patches):
                        if has_member:
                            for m_idx in range(n_mbr):
                                s = _field_spectrum_patch(chunk[t, :, m_idx], row0, col0, pr, pc)
                                if s is not None:
                                    out[pi, t_start + t, lt_idx, m_idx] = s
                        else:
                            s = _field_spectrum_patch(chunk[t], row0, col0, pr, pc)
                            if s is not None:
                                out[pi, t_start + t, lt_idx] = s

        patch_coord = xr.Variable("patch", patch_names)
        if has_member:
            return xr.DataArray(
                out,
                dims=["patch", "reference_time", "lead_time", "member", "k"],
                coords={"patch": patch_coord, "reference_time": ref_times,
                        "lead_time": lead_times, "member": members, "k": k_centers},
            )
        return xr.DataArray(
            out,
            dims=["patch", "reference_time", "lead_time", "k"],
            coords={"patch": patch_coord, "reference_time": ref_times,
                    "lead_time": lead_times, "k": k_centers},
        )

    if variables is not None:
        if compute_obs:
            observations = observations[variables]
        if compute_fct:
            forecasts    = forecasts[variables]

    n_row = ri
    n_col = patches[-1][1] + 1
    print(f"  power_spectrum_patches: {n_row}×{n_col} = {len(patches)} patches "
          f"of nominal size {_psx}×{_psy}", flush=True)

    iter_vars = list(observations.data_vars) if not compute_fct else list(forecasts.data_vars)

    per_var = {}
    for var in iter_vars:
        if compute_obs:
            print(f"  power_spectrum_patches: {var} obs ...", flush=True)
            per_var[f"{var}_obs"] = _spectrum_patches(observations[var])
        if compute_fct:
            print(f"  power_spectrum_patches: {var} fct ...", flush=True)
            per_var[f"{var}_fct"] = _spectrum_patches(forecasts[var])

    return xr.Dataset(per_var)
