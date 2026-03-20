
import cartopy.crs as ccrs

def create_cartopy_crs(projection, kws_projection, kws_globe = None) -> ccrs.Projection:
    """Create a Cartopy coordinate reference system (CRS) based on the specified projection.

    This function creates a Cartopy projection object using the provided projection name
    and associated keyword arguments.
    
    Parameters
    ----------
    projection : str
        Name of the projection to create. Must be one of the supported projections
        defined in PROJECTIONS.
    projection_kws : dict[str, str]
        Dictionary of keyword arguments to pass to the projection constructor.
    globe_kws: dict[str, str], optional
        Optional globe parameters which will be used to create a ccrs.Globe object.
    
    Returns
    -------
    ccrs.Projection
        The created Cartopy projection object.
    Raises
    ------
    AssertionError
        If the specified projection is not supported (not in PROJECTIONS).
    
    Examples
    --------
    >>> projection_kws = {'central_longitude': 0,}
    >>> globe_kws = {'ellipse': 'WGS84'}}
    >>> crs = create_cartopy_crs('latlon', projection_kws, globe_kws)
    """

    
    
    # - Get the cartopy projection (crs)
    try:
        projection = PROJECTIONS[projection]
    except KeyError:
        raise ValueError(f"Unsupported projection: {projection}")
    kwargs = kws_projection.copy()

    # - Move globe keywords to different dictionary
    if kws_globe:
        globe = ccrs.Globe(**kws_globe)
    
    crs = projection(globe=globe, **kws_projection)
    return crs

PROJECTIONS=dict(
    lcc=ccrs.LambertConformal,
    latlon=ccrs.PlateCarree,
    PlateCarree=ccrs.PlateCarree,
    Mercator=ccrs.Mercator,
    Orthographic=ccrs.Orthographic
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
            ny=1069
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
            ny=1609
        ),
    ),
)
