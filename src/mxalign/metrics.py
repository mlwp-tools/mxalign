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


def power_spectrum(observations, forecasts, dim_x, dim_y, res,
                   variables=None, ref_time_chunk=50):
    """2D power spectrum averaged over reference_time, one curve per lead_time.

    Applies a 2-D rfft2 to every (reference_time, lead_time) slice of each
    variable, then averages spectra across reference_time.  Both forecast and
    reference (observation) spectra are returned so that a single saved file
    contains all curves needed for comparison plots.

    Grid layout assumption: ``grid_index`` is ordered with the y-axis varying
    slowest (row-major), i.e. ``field.reshape(dim_x, dim_y)`` gives the 2-D
    field.  A 90° rotation is applied before the FFT (same convention as the
    standalone ``compute_power_spectra.py``).

    Parameters
    ----------
    observations, forecasts : xr.Dataset
        Aligned datasets, both with dimensions
        ``(reference_time, lead_time, grid_index)``.
        ``len(grid_index)`` must equal ``dim_x * dim_y``.
    dim_x, dim_y : int
        Shape of the 2-D grid (rows × columns after reshape + rot90).
    res : float
        Grid spacing in **metres** — used to convert FFT bin indices to
        physical wavelengths.
    variables : list of str, optional
        Subset of variables to process.  All common data_vars are used if None.
    ref_time_chunk : int
        Number of reference times loaded per ``.load()`` call.  Decrease if
        memory is tight; increase to reduce I/O round-trips (default: 50).

    Returns
    -------
    xr.Dataset
        Variables ``{var}_obs`` and ``{var}_fct`` for each processed variable,
        with dimensions ``(reference_time, lead_time, wavelength)``.
        ``wavelength`` is in metres.  Fields with any NaN are stored as NaN
        in the output so that averaging over reference_time can be done in
        post-processing.
    """
    if variables is not None:
        observations = observations[variables]
        forecasts    = forecasts[variables]

    # ── Isotropic radial wavenumber binning ─────────────────────────────────
    # DCT-II on (dim_x, dim_y) gives a real output of the same shape.
    # Cosine mode (kx, ky) has spatial frequency kx/(2*dim_x*res) in x and
    # ky/(2*dim_y*res) in y — the factor 2 arises from the DCT's implicit
    # even-symmetric mirroring of the domain.
    # We bin coefficients by radial wavenumber k = sqrt(kx²+ky²) (in grid
    # units), using integer bins so each bin has roughly the same annular width.
    kx_idx = np.arange(dim_x, dtype=float)               # shape (dim_x,)
    ky_idx = np.arange(dim_y, dtype=float)                # shape (dim_y,)
    KX, KY = np.meshgrid(kx_idx, ky_idx, indexing='ij')  # (dim_x, dim_y)

    K_norm  = np.sqrt((KX / dim_x) ** 2 + (KY / dim_y) ** 2)
    k_bins  = np.round(K_norm * min(dim_x, dim_y)).astype(int)  # integer bin index

    n_freq     = min(dim_x, dim_y) // 2
    valid_bins = np.arange(1, n_freq + 1)

    # Representative physical wavelength per bin (m): 1 / mean(K_phys) in bin
    # Factor 2 in denominator: DCT mirrors the domain, doubling effective length
    K_phys = np.sqrt((KX / (2 * dim_x * res)) ** 2 + (KY / (2 * dim_y * res)) ** 2)
    wavelengths = np.zeros(n_freq, dtype=np.float64)
    for _b in valid_bins:
        _m = k_bins == _b
        if _m.any():
            wavelengths[_b - 1] = 1.0 / K_phys[_m].mean()

    k_bins_flat = k_bins.ravel()

    def _spectrum(da):
        """One isotropic power spectrum per (reference_time, lead_time) slice."""
        lead_times = da.lead_time.values
        ref_times  = da.reference_time.values
        n_rt       = len(ref_times)
        n_lt       = len(lead_times)

        out = np.full((n_rt, n_lt, n_freq), np.nan, dtype=np.float64)

        for lt_idx in range(n_lt):
            da_lt = da.isel(lead_time=lt_idx)
            for t_start in range(0, n_rt, ref_time_chunk):
                t_end = min(t_start + ref_time_chunk, n_rt)
                chunk = da_lt.isel(reference_time=slice(t_start, t_end)).load().values
                if chunk.ndim == 1:
                    chunk = chunk[np.newaxis, :]
                for t in range(chunk.shape[0]):
                    field = chunk[t]
                    if np.any(np.isnan(field)):
                        continue
                    field_2d = field.reshape(dim_x, dim_y)
                    field_2d = field_2d - np.mean(field_2d)
                    sp        = dctn(field_2d, type=2, norm='ortho')  # real output
                    power_2d  = sp ** 2                               # no rfft_w needed
                    binned    = np.bincount(k_bins_flat, weights=power_2d.ravel(),
                                            minlength=n_freq + 1)
                    out[t_start + t, lt_idx] = binned[1: n_freq + 1]

        return xr.DataArray(
            out,
            dims=["reference_time", "lead_time", "wavelength"],
            coords={
                "reference_time": ref_times,
                "lead_time":      lead_times,
                "wavelength":     wavelengths,
            },
        )

    per_var = {}
    for var in list(observations.data_vars):
        print(f"  power_spectrum: {var} obs ...", flush=True)
        per_var[f"{var}_obs"] = _spectrum(observations[var])
        print(f"  power_spectrum: {var} fct ...", flush=True)
        per_var[f"{var}_fct"] = _spectrum(forecasts[var])

    return xr.Dataset(per_var)
