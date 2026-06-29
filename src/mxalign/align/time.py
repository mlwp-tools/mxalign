import logging

import xarray as xr

from mlwp_data_specs.api import TIME_TRAIT_ATTR
from mlwp_data_specs.specs.traits.time_coordinate import Time

logger = logging.getLogger(__name__)


def align_time(
    datasets: list[xr.Dataset] | dict[str, xr.Dataset],
    reference: str | xr.Dataset,
    **kwargs,
):
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
      ``ds.mx.align_time_with(ref_ds, **kwargs)``. The reference is fixed.

    Parameters
    ----------
    datasets : list or dict of xr.Dataset
    reference : str or xr.Dataset
        Key into *datasets* dict, or an xr.Dataset to align to.
    """
    if isinstance(datasets, dict):
        keys = list(datasets.keys())
        ds_list = list(datasets.values())
        ref_ds = datasets[reference] if isinstance(reference, str) else reference
    else:
        ds_list = [datasets] if isinstance(datasets, xr.Dataset) else list(datasets)
        keys = None
        ref_ds = reference

    forecasts = [ds for ds in ds_list if ds.mx.is_forecast()]
    use_global = ref_ds.mx.is_observation() and len(forecasts) > 0

    if use_global:
        aligned = _align_time_global(ds_list, forecasts, ref_ds)
    else:
        aligned = [
            ds if ds is ref_ds else ds.mx.align_time_with(ref_ds, **kwargs)
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

    aligned = []
    for ds in ds_list:
        if ds.mx.is_forecast():
            out = ds.reindex(reference_time=r_star, lead_time=l_star)
            out = out.assign_coords(valid_time=valid_time_2d)
        else:
            out = ds.reindex(valid_time=valid_time_2d.values.ravel()).sel(
                valid_time=valid_time_2d
            )
            out.attrs[TIME_TRAIT_ATTR] = Time.FORECAST.value
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
