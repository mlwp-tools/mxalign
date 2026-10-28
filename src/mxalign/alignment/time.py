"""Temporal alignment of forecast and observation datasets."""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from mlwp_data_specs.api import TIME_TRAIT_ATTR
from mlwp_data_specs.specs.traits.time_coordinate import Time

logger = logging.getLogger(__name__)


def align_time(
    datasets: xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset],
    reference: str | xr.Dataset,
    lead_time: str | None = None,
) -> xr.Dataset | list[xr.Dataset] | dict[str, xr.Dataset]:
    """Align all datasets temporally to a reference dataset.

    Two regimes:

    * **Global** — when the reference is an observation and the inputs contain
      one or more forecasts. Computes a single common ``(reference_time,
      lead_time)`` grid across all forecasts and aligns every dataset onto it:

      1. ``R*`` = intersection of ``reference_time`` across all forecasts.
      2. ``L*`` = union of ``lead_time`` across all forecasts.
      3. ``R*`` is pruned to ref_times ``r`` for which **every** ``l ∈ L*``
         yields a ``valid_time`` inside the reference observation's range
         ``[ref.valid_time.min(), ref.valid_time.max()]``. ``L*`` is never
         pruned — keeping a long lead_time only costs an extra column in the
         output (NaN where the reference doesn't reach) and avoids dropping
         lead_times from forecasts that did fit. Raises if no ref_time
         survives.
      4. Each forecast → ``reindex(reference_time=R*, lead_time=L*)``,
         NaN-padding lead_times the forecast doesn't have, and a refreshed
         ``valid_time`` coordinate.
      5. Each observation → subset to ``V* = R* + L*`` then broadcast onto
         ``(R*, L*)``. Output carries the forecast time trait.

    * **Pairwise** — for every other case (reference is a forecast, or all
      datasets are observations), each non-reference dataset is aligned via
      ``ds.mx.align_time_with(ref_ds, lead_time=lead_time)``. The reference is
      fixed.

    Parameters
    ----------
    datasets : xr.Dataset or list of xr.Dataset or dict of str to xr.Dataset
        Single dataset, list, or dict of datasets to align.
    reference : str or xr.Dataset
        Key into *datasets* dict, or an xr.Dataset to align to.
    lead_time : str, optional
        Only allowed when the reference is a forecast: how the lead times of
        the other forecasts are matched to the reference's — ``"reference"``
        (default), ``"intersection"`` or ``"union"``.

    Returns
    -------
    xr.Dataset or list of xr.Dataset or dict of str to xr.Dataset
        The aligned datasets: a dict with the same keys if ``datasets`` is a
        dict, otherwise a single dataset if there is only one, else a list.

    Raises
    ------
    ValueError
        If ``lead_time`` is given but the reference is not a forecast, if no
        ``reference_time`` survives global alignment, or if a pairwise
        alignment fails (e.g. an unknown ``lead_time`` option).
    KeyError
        If ``reference`` is not a key of ``datasets``, or a dataset has no
        time trait attribute.
    """
    if isinstance(datasets, dict):
        keys = list(datasets.keys())
        ds_list = list(datasets.values())
        ref_ds = datasets[reference] if isinstance(reference, str) else reference
    else:
        ds_list = [datasets] if isinstance(datasets, xr.Dataset) else list(datasets)
        keys = None
        ref_ds = reference

    if lead_time is not None and not ref_ds.mx.is_forecast():
        raise ValueError(
            f"lead_time={lead_time!r} is only supported when the reference is a forecast."
        )

    forecasts = [ds for ds in ds_list if ds.mx.is_forecast()]
    use_global = ref_ds.mx.is_observation() and len(forecasts) > 0

    if use_global:
        aligned = _align_time_global(ds_list, forecasts, ref_ds)
    else:
        aligned = [
            ds if ds is ref_ds else ds.mx.align_time_with(ref_ds, lead_time=lead_time)
            for ds in ds_list
        ]

    if keys is not None:
        return dict(zip(keys, aligned))
    return aligned[0] if len(aligned) == 1 else aligned


def _align_time_global(
    ds_list: list[xr.Dataset],
    forecasts: list[xr.Dataset],
    ref_ds: xr.Dataset,
) -> list[xr.Dataset]:
    """Align all datasets onto a common ``(reference_time, lead_time)`` grid.

    See the "Global" regime in :func:`align_time` for the algorithm.

    Parameters
    ----------
    ds_list : list of xr.Dataset
        All datasets to align, forecasts and observations.
    forecasts : list of xr.Dataset
        The forecast datasets among ``ds_list``.
    ref_ds : xr.Dataset
        Reference observation dataset bounding the valid times.

    Returns
    -------
    list of xr.Dataset
        The aligned datasets, in the order of ``ds_list``; all carry the
        forecast time trait.

    Raises
    ------
    ValueError
        If no ``reference_time`` keeps all lead times within the reference
        observation's ``valid_time`` range.
    """
    # R* = intersection of reference_time, L* = union of lead_time.
    # Pull the aligned *coord* (= the merged index) rather than the array's
    # data — outer-aligning a coord-DataArray NaN-fills the data where the
    # index extends, but the coord itself carries the correct union.

    r_star = xr.align(*[f.reference_time for f in forecasts], join="inner")[
        0
    ].reference_time
    l_star = xr.align(*[f.lead_time for f in forecasts], join="outer")[0].lead_time

    n_r = r_star.size
    r_star = _constrain_to_reference(r_star, l_star, ref_ds.valid_time)
    if r_star.size < n_r:
        logger.warning(
            "Global time alignment: pruned %d reference_time(s) (%d → %d) "
            "whose (r + l) falls outside the reference observation's range.",
            n_r - r_star.size,
            n_r,
            r_star.size,
        )

    # xarray broadcasts r_star (reference_time) + l_star (lead_time) → 2D
    valid_time_2d = r_star + l_star
    # Coordinate-only template of the common grid, used to reshape observations
    grid = xr.Dataset(coords={"reference_time": r_star, "lead_time": l_star})

    aligned = []
    for ds in ds_list:
        if ds.mx.is_forecast():
            out = ds.reindex(reference_time=r_star, lead_time=l_star)
            out = out.assign_coords(valid_time=valid_time_2d)
        else:
            out = align_observation_to_forecast(ds, grid)
        aligned.append(out)
    return aligned


def _constrain_to_reference(
    r_star: xr.DataArray,
    l_star: xr.DataArray,
    ref_vt: xr.DataArray,
) -> xr.DataArray:
    """Drop ref_times whose ``(r + l)`` falls outside the reference range.

    A ref_time ``r`` is kept only when every ``l ∈ L*`` yields a valid_time
    in ``[ref_vt.min(), ref_vt.max()]``. ``L*`` is never pruned.

    Parameters
    ----------
    r_star : xr.DataArray
        Candidate reference times ``R*`` along ``reference_time``.
    l_star : xr.DataArray
        Lead times ``L*`` along ``lead_time``.
    ref_vt : xr.DataArray
        Valid times of the reference observation.

    Returns
    -------
    xr.DataArray
        The subset of ``r_star`` whose valid times all fall within range.

    Raises
    ------
    ValueError
        If no reference time satisfies the constraint.
    """
    vt = r_star + l_star
    in_range = (vt >= ref_vt.min()) & (vt <= ref_vt.max())
    rows_ok = in_range.all(dim="lead_time")
    if not bool(rows_ok.any()):
        raise ValueError(
            "Global time alignment: no reference_time keeps all lead_times "
            "within the reference observation's valid_time range."
        )
    return r_star.where(rows_ok, drop=True)


# --- Pairwise alignment (used by ``ds.mx.align_time_with``) ---


def _add_valid_time(ds_fcst: xr.Dataset) -> xr.Dataset:
    """Add a 2-D ``valid_time`` coordinate (``reference_time + lead_time``) to a forecast.

    Parameters
    ----------
    ds_fcst : xr.Dataset
        Forecast dataset with ``reference_time`` and ``lead_time`` coordinates.

    Returns
    -------
    xr.Dataset
        ``ds_fcst`` with a ``valid_time`` coordinate on
        ``(reference_time, lead_time)``.
    """
    valid_time = (
        ds_fcst["reference_time"].values[:, np.newaxis] + ds_fcst["lead_time"].values
    )
    return ds_fcst.assign_coords(
        {"valid_time": (["reference_time", "lead_time"], valid_time)}
    )


def align_forecast_to_observation(
    ds_fcst: xr.Dataset,
    ds_obs: xr.Dataset,
    lead_time: str | list[str | np.timedelta64] | np.timedelta64 = "shortest",
) -> xr.Dataset:
    """Turn a forecast into an observation-like dataset on ``ds_obs.valid_time``.

    ``ds_fcst`` is stacked to valid time, deduplicated according to the
    ``lead_time`` strategy, and reindexed to ``ds_obs.valid_time``.

    Parameters
    ----------
    ds_fcst : xr.Dataset
        Forecast dataset with ``reference_time`` and ``lead_time``.
    ds_obs : xr.Dataset
        Observation dataset with a ``valid_time`` coordinate.
    lead_time : str or np.timedelta64 or list, default "shortest"
        ``"shortest"`` or ``"longest"`` keeps, per valid time, the entry with
        the shortest/longest lead time. Otherwise one or more lead times
        (timedelta-like or strings such as ``"6h"``) to keep. Ties for the
        same valid time are resolved by keeping the first entry.

    Returns
    -------
    xr.Dataset
        Dataset with a ``valid_time`` dimension matching ``ds_obs`` (NaN where
        the forecast has no data) and the observation time trait.

    Raises
    ------
    ValueError
        If ``lead_time`` is not a valid option, lacks a unit, or contains lead
        times not present in the forecast.
    """
    ds_with_vt = _add_valid_time(ds_fcst)
    ds_stacked = ds_with_vt.stack(time=["reference_time", "lead_time"]).reset_index(
        "time"
    )

    vt_vals = ds_stacked.valid_time.values
    lt_vals = ds_stacked.lead_time.values

    if isinstance(lead_time, str) and lead_time in ("shortest", "longest"):
        df = pd.DataFrame({"vt": vt_vals, "lt": lt_vals})
        agg = "min" if lead_time == "shortest" else "max"
        is_candidate = (df.groupby("vt")["lt"].transform(agg) == df["lt"]).values
    else:
        requested = _parse_lead_times(lead_time)
        missing = np.setdiff1d(requested, lt_vals)
        if missing.size:
            raise ValueError(
                f"lead_time(s) {pd.to_timedelta(missing).tolist()} not in forecast "
                f"lead times {pd.to_timedelta(np.unique(lt_vals)).tolist()}"
            )
        is_candidate = np.isin(lt_vals, requested)

    # Among candidate entries keep the first per valid_time (handles ties: the
    # same valid_time reached via several reference_times)
    candidate_positions = np.where(is_candidate)[0]
    _, first_in_group = np.unique(vt_vals[candidate_positions], return_index=True)
    positions = candidate_positions[first_in_group]

    ds_1d = ds_stacked.isel(time=positions)
    ds_1d = ds_1d.swap_dims({"time": "valid_time"})
    ds_1d = ds_1d.drop_vars(
        [v for v in ["reference_time", "lead_time", "time"] if v in ds_1d.coords]
    )
    ds_1d = ds_1d.transpose("valid_time", ...)

    ds_1d = ds_1d.reindex(valid_time=ds_obs.valid_time)
    ds_1d.attrs[TIME_TRAIT_ATTR] = Time.OBSERVATION.value
    return ds_1d


def _parse_lead_times(
    lead_time: str | list[str | np.timedelta64] | np.timedelta64,
) -> np.ndarray:
    """Convert one or more lead times to a ``timedelta64[ns]`` array.

    Parameters
    ----------
    lead_time : str or np.timedelta64 or list
        One or more lead times, as timedelta-like values or strings such as
        ``"6h"``.

    Returns
    -------
    np.ndarray
        1-D array of lead times with dtype ``timedelta64[ns]``.

    Raises
    ------
    ValueError
        If a lead time is a bare number without unit, or cannot be parsed.
    """
    values = np.atleast_1d(np.asarray(lead_time, dtype=object))
    # np.timedelta64 subclasses np.number, so exclude it explicitly
    if any(
        isinstance(v, (int, float, np.number)) and not isinstance(v, np.timedelta64)
        for v in values
    ):
        raise ValueError(
            f"lead_time {lead_time!r} needs a unit, e.g. '6h' or np.timedelta64(6, 'h')."
        )
    try:
        return pd.to_timedelta(values).to_numpy().astype("timedelta64[ns]")
    except ValueError:
        raise ValueError(
            f"Unknown lead_time option for F→O alignment: {lead_time!r}. Use "
            "'shortest', 'longest', or one or more lead times such as '6h'."
        ) from None


def align_observation_to_forecast(
    ds_obs: xr.Dataset, ds_fcst: xr.Dataset
) -> xr.Dataset:
    """Broadcast 1-D observations onto the ``(reference_time, lead_time)`` forecast grid.

    Observations are selected at the forecast valid times
    (``reference_time + lead_time``), NaN-filled where missing.

    Parameters
    ----------
    ds_obs : xr.Dataset
        Observation dataset with a ``valid_time`` dimension.
    ds_fcst : xr.Dataset
        Forecast dataset (or coordinate template) with ``reference_time`` and
        ``lead_time``.

    Returns
    -------
    xr.Dataset
        Observations on ``(reference_time, lead_time)``, with the forecast
        time trait.
    """
    ds_fcst_with_vt = _add_valid_time(ds_fcst)
    valid_time_2d = ds_fcst_with_vt["valid_time"]  # shape (reference_time, lead_time)

    # Reindex obs onto all unique fcst valid_times (NaN-fills fcst valid_times not in obs)
    fcst_vt_flat = np.unique(valid_time_2d.values.ravel())
    obs_reindexed = ds_obs.reindex(valid_time=fcst_vt_flat)

    # sel with a 2D DataArray indexer broadcasts 1D obs → (reference_time, lead_time)
    ds_out = obs_reindexed.sel(valid_time=valid_time_2d)

    ds_out.attrs[TIME_TRAIT_ATTR] = Time.FORECAST.value
    return ds_out


def align_observation_to_observation(ds1: xr.Dataset, ds2: xr.Dataset) -> xr.Dataset:
    """Reindex ``ds1`` to ``ds2.valid_time`` with NaN-fill for missing times.

    Parameters
    ----------
    ds1 : xr.Dataset
        Observation dataset to align.
    ds2 : xr.Dataset
        Reference observation dataset.

    Returns
    -------
    xr.Dataset
        ``ds1`` reindexed to the valid times of ``ds2``.
    """
    return ds1.reindex(valid_time=ds2.valid_time)


def align_forecast_to_forecast(
    ds1: xr.Dataset, ds2: xr.Dataset, lead_time: str = "reference"
) -> xr.Dataset:
    """Reindex ``ds1`` to ``ds2.reference_time`` and align lead times by a strategy.

    Parameters
    ----------
    ds1 : xr.Dataset
        Forecast dataset to align.
    ds2 : xr.Dataset
        Reference forecast dataset.
    lead_time : str, default "reference"
        ``"reference"`` reindexes to the lead times of ``ds2``,
        ``"intersection"`` keeps only lead times shared by both, and
        ``"union"`` reindexes to the lead times of either.

    Returns
    -------
    xr.Dataset
        ``ds1`` aligned to ``ds2`` with a refreshed ``valid_time`` coordinate.

    Raises
    ------
    ValueError
        If ``lead_time`` is not one of the supported options.
    """
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
