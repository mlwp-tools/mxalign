import cartopy.crs as ccrs


def create_cartopy_crs(
    projection: str,
    kws_projection: dict,
    kws_globe: dict | None = None,
) -> ccrs.Projection:
    """Build a Cartopy CRS from a projection name and keyword dicts.

    Parameters
    ----------
    projection:
        Key into :data:`PROJECTIONS` (e.g. ``"lcc"``, ``"latlon"``).
    kws_projection:
        Keyword arguments forwarded to the projection constructor.
    kws_globe:
        Optional keyword arguments forwarded to :class:`cartopy.crs.Globe`.

    Raises
    ------
    ValueError
        If ``projection`` is not a key in :data:`PROJECTIONS`.
    """

    # - Get the cartopy projection class
    try:
        proj_cls = PROJECTIONS[projection]
    except KeyError:
        raise ValueError(f"Unsupported projection: {projection}")

    # copy projection kws to avoid mutating caller's dict
    kws_projection = kws_projection.copy()

    # - Build globe if keywords provided
    globe = None
    if kws_globe:
        globe = ccrs.Globe(**kws_globe)

    crs = proj_cls(globe=globe, **kws_projection)
    return crs


PROJECTIONS = dict(
    lcc=ccrs.LambertConformal,
    latlon=ccrs.PlateCarree,
    PlateCarree=ccrs.PlateCarree,
    Mercator=ccrs.Mercator,
    Orthographic=ccrs.Orthographic,
)

BUILTIN = dict(
    cerra=dict(
        projection="lcc",
        kws_globe=dict(
            semimajor_axis=6371229.0,
            semiminor_axis=6371229.0,
        ),
        kws_projection=dict(
            central_longitude=8.0,
            central_latitude=50.0,
            standard_parallels=[50.0, 50.0],
        ),
        kws_grid=dict(
            lon_ll=-17.4859,
            lat_ll=20.2923,
            lon_ur=74.1051,
            lat_ur=63.7695,
            dx=5500.0,
            dy=5500.0,
            nx=1069,
            ny=1069,
        ),
    ),
    uwcw=dict(
        projection="lcc",
        kws_globe=dict(
            semimajor_axis=6371229.0,
            semiminor_axis=6371229.0,
        ),
        kws_projection=dict(
            central_longitude=-1.96590281,
            central_latitude=55.5164337,
            standard_parallels=[55.499996, 55.499996],
        ),
        kws_grid=dict(
            lon_ll=-25.4470005,
            lat_ll=39.6389999,
            lon_ur=40.1508102,
            lat_ur=62.6713715,
            dx=2000.0,
            dy=2000.0,
            nx=1909,
            ny=1609,
        ),
    ),
)
