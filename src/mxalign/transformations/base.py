import numpy as np
from .registry import register_transformation

@register_transformation("rename")
def transform(ds, rename_dict):
    new_dict = {}
    for new_name, old_names in rename_dict.items():
        for name in ds.keys():
            if name in old_names:
                new_dict[name]= new_name
            else:
                pass
    return ds.rename(new_dict)

@register_transformation("kelvin_to_celcius")
def transform(ds, variables , inverse=False):
    T_C2K = 273.15
    if isinstance(variables, str):
        variables = [variables]
    if inverse:
        t = T_C2K
    else:
        t = -T_C2K

    for var in variables:
        ds[var] = ds[var] + t

    return(ds)

@register_transformation("pa_to_hpa")
def transform(ds, variables, inverse=False):
    if isinstance(variables, str):
        variables = [variables]
    factor = 100.0 if inverse else 0.01
    for var in variables:
        ds[var] = ds[var] * factor
    return ds

@ register_transformation("uv_to_speed")
def transform(ds, u, v, speed):
    import numpy as np
    result = np.sqrt(ds[u]**2 + ds[v]**2)
    ds[speed] = result
    return ds

@register_transformation("scale")
def transform(ds, variables, factor):
    if isinstance(variables, str):
        variables = [variables]
    for var in variables:
        if var in ds:
            ds[var] = ds[var] * factor
    return ds

@register_transformation("valid_range")
def transform(ds, variables, min=None, max=None, datasets=None):
    """Mask values outside [min, max] with NaN.

    Applied *before* spatial/temporal alignment so that fill values and bad
    sensor readings propagate as NaN rather than inflating error metrics.
    This is distinct from the valid_range check inside mse_by_domain, which
    acts as a safety net on the forecast side after alignment.

    Typical use case: SYNOP AccPcp6h can carry fill values (e.g. 62622 mm)
    that pass the station QC but are physically impossible. Setting max=800
    discards them before they corrupt the MSE average.
    """
    if isinstance(variables, str):
        variables = [variables]
    for var in variables:
        if var in ds:
            mask = True
            if min is not None:
                mask = mask & (ds[var] >= min)
            if max is not None:
                mask = mask & (ds[var] <= max)
            ds[var] = ds[var].where(mask)
    return ds

@register_transformation("clip")
def transform(ds, variables, min=None, max=None):
    """Censor values to [min, max] (replace, don't mask; NaNs stay NaN).

    Typical use case: de-accumulated precipitation carries small negative
    round-off values (HARMONIE tp, ~ -0.005 mm/6h). Masking them with
    valid_range drops individual ensemble members at that point (always the
    dry ones), which biases ensemble mean and spread; clipping at min=0 is
    the standard treatment (cf. MET censor_thresh/censor_val).
    """
    if isinstance(variables, str):
        variables = [variables]
    for var in variables:
        if var in ds:
            ds[var] = ds[var].clip(min=min, max=max)
    return ds
@register_transformation("select_points")
def transform(ds, points_file, tolerance_deg=1e-3):
    """Keep only the grid points whose (latitude, longitude) are in `points_file`.

    `points_file` is a .npz with arrays `latitudes` and `longitudes` (e.g. the grid of a
    trimmed-domain forecast). Each point of `ds` is matched to its nearest target point;
    points farther than `tolerance_deg` are dropped. Original grid order is kept, so two
    datasets on the same parent grid end up with identical grid_index order.

    Typical use case: verify full-domain and trim_edge forecasts on one common (trimmed)
    domain, since grid-grid alignment requires identical grids.
    """
    from scipy.spatial import cKDTree

    pts = np.load(points_file)
    target = np.column_stack([pts["latitudes"], ((pts["longitudes"] + 180) % 360) - 180])
    lat = ds["latitude"].values.astype(np.float64)
    lon = ((ds["longitude"].values.astype(np.float64) + 180) % 360) - 180
    dist, _ = cKDTree(target).query(np.column_stack([lat, lon]), distance_upper_bound=tolerance_deg)
    keep = np.flatnonzero(np.isfinite(dist))
    if len(keep) != len(target):
        raise ValueError(f"select_points: matched {len(keep)} points, expected {len(target)}")
    return ds.isel(grid_index=keep)
