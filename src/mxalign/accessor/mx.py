"""The ``ds.mx`` xarray Dataset accessor for space/time predicates and alignment."""

import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs

from mlwp_data_specs.api import SPACE_TRAIT_ATTR, TIME_TRAIT_ATTR
from mlwp_data_specs.specs.traits.spatial_coordinate import Space
from mlwp_data_specs.specs.traits.time_coordinate import Time

from ..utils.projections import create_cartopy_crs, BUILTIN
from ..alignment import time as _time
from ..alignment import space as _space


@xr.register_dataset_accessor("mx")
class MxAccessor:
    """``ds.mx`` accessor — space/time predicates and alignment operations.

    Reads ``mlwp_space_trait`` and ``mlwp_time_trait`` from ``ds.attrs`` on
    every use (xarray caches the accessor per object, so reading them once
    would go stale when attrs change). Raises ``KeyError`` when a trait is
    needed but missing. Never modifies ``ds``; operations return new datasets.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset the accessor is attached to.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def _trait(self, attr: str) -> str:
        """Return the value of the trait attribute ``attr`` from ``ds.attrs``.

        Parameters
        ----------
        attr : str
            Name of the trait attribute to read.

        Returns
        -------
        str
            The trait value stored in ``ds.attrs[attr]``.

        Raises
        ------
        KeyError
            If the dataset has no ``attr`` attribute.
        """
        try:
            return self._ds.attrs[attr]
        except KeyError:
            raise KeyError(
                f"Dataset has no '{attr}' attribute; load it with mlwp-data-loaders "
                "or set it with mxalign.utils.traits."
            ) from None

    @property
    def _space(self) -> Space:
        """The space trait of the dataset.

        Returns
        -------
        Space
            The space trait read from ``ds.attrs``.

        Raises
        ------
        KeyError
            If the dataset has no space trait attribute.
        """
        return Space(self._trait(SPACE_TRAIT_ATTR))

    @property
    def _time(self) -> Time:
        """The time trait of the dataset.

        Returns
        -------
        Time
            The time trait read from ``ds.attrs``.

        Raises
        ------
        KeyError
            If the dataset has no time trait attribute.
        """
        return Time(self._trait(TIME_TRAIT_ATTR))

    # --- Space predicates ---

    def is_grid(self) -> bool:
        """Return whether the dataset has the GRID space trait.

        Returns
        -------
        bool
            ``True`` if the space trait is ``Space.GRID``.

        Raises
        ------
        KeyError
            If the dataset has no space trait attribute.
        """
        return self._space == Space.GRID

    def is_point(self) -> bool:
        """Return whether the dataset has the POINT space trait.

        Returns
        -------
        bool
            ``True`` if the space trait is ``Space.POINT``.

        Raises
        ------
        KeyError
            If the dataset has no space trait attribute.
        """
        return self._space == Space.POINT

    # --- Time predicates ---

    def is_forecast(self) -> bool:
        """Return whether the dataset has the FORECAST time trait.

        Returns
        -------
        bool
            ``True`` if the time trait is ``Time.FORECAST``.

        Raises
        ------
        KeyError
            If the dataset has no time trait attribute.
        """
        return self._time == Time.FORECAST

    def is_observation(self) -> bool:
        """Return whether the dataset has the OBSERVATION time trait.

        Returns
        -------
        bool
            ``True`` if the time trait is ``Time.OBSERVATION``.

        Raises
        ------
        KeyError
            If the dataset has no time trait attribute.
        """
        return self._time == Time.OBSERVATION

    # --- Space operations ---

    def add_crs(self, crs: str | dict | ccrs.Projection) -> xr.Dataset:
        """Attach a Cartopy CRS to ``ds.attrs["crs"]``.

        Parameters
        ----------
        crs : str or dict or ccrs.Projection
            Name of a builtin projection (key into
            :data:`mxalign.utils.projections.BUILTIN`, case-insensitive), a
            config dict with ``projection``, ``kws_projection`` and optionally
            ``kws_globe`` keys (see
            :func:`mxalign.utils.projections.create_cartopy_crs`), or a Cartopy
            CRS object.

        Returns
        -------
        xr.Dataset
            A copy of the dataset with ``attrs["crs"]`` set.

        Raises
        ------
        ValueError
            If the dataset is a point dataset, or ``crs`` is a name that is
            not a builtin projection.
        KeyError
            If the dataset has no space trait attribute, or a config dict
            lacks the ``projection`` or ``kws_projection`` key.
        """
        if self.is_point():
            raise ValueError("Cannot add CRS to a point dataset")
        if isinstance(crs, str):
            try:
                crs = BUILTIN[crs.lower()]
            except KeyError:
                raise ValueError(f"crs: {crs} not found in supported projections")
        if isinstance(crs, dict):
            crs = create_cartopy_crs(
                projection=crs["projection"],
                kws_projection=crs["kws_projection"],
                kws_globe=crs.get("kws_globe", None),
            )
        return self._ds.assign_attrs({"crs": crs})

    def add_grid_mapping(self, grid_mapping: str | dict) -> xr.Dataset:
        """Attach a grid-mapping dict to ``ds.attrs["grid_mapping"]``.

        Parameters
        ----------
        grid_mapping : str or dict
            Name of a builtin projection (key into
            :data:`mxalign.utils.projections.BUILTIN`, case-insensitive) whose
            ``kws_grid`` is used, or a grid-mapping dict (``nx``, ``ny``,
            ``lon_ll``, ``lat_ll``, ``dx``, ``dy``, ...).

        Returns
        -------
        xr.Dataset
            A copy of the dataset with ``attrs["grid_mapping"]`` set.

        Raises
        ------
        ValueError
            If the dataset is a point dataset, or ``grid_mapping`` is a name
            that is not a builtin mapping.
        KeyError
            If the dataset has no space trait attribute.
        """
        if self.is_point():
            raise ValueError("Cannot add grid mapping to a point dataset")
        if isinstance(grid_mapping, str):
            try:
                grid_mapping = BUILTIN[grid_mapping.lower()]["kws_grid"]
            except KeyError:
                raise ValueError(
                    f"grid mapping: {grid_mapping} not found in supported mappings"
                )
        return self._ds.assign_attrs({"grid_mapping": grid_mapping})

    def add_xy(self, crs: str | dict | ccrs.Projection | None = None) -> xr.Dataset:
        """Project ``longitude``/``latitude`` to ``xc``/``yc`` coordinates.

        Parameters
        ----------
        crs : str or dict or ccrs.Projection, optional
            CRS to attach first (see :meth:`add_crs`). If ``None``, the CRS
            already stored in ``ds.attrs["crs"]`` is used.

        Returns
        -------
        xr.Dataset
            The dataset with ``xc``/``yc`` coordinates along ``grid_index``
            (grid) or ``point_index`` (point). Returned unchanged if it already
            has ``xc`` and ``yc`` coordinates.

        Raises
        ------
        ValueError
            If no CRS is given or stored, if ``longitude``/``latitude`` are
            dimensions of the dataset, or if the dataset is neither grid nor
            point.
        KeyError
            If the dataset has no space trait attribute.
        """
        ds = self.add_crs(crs) if crs is not None else self._ds

        crs_obj = ds.attrs.get("crs", None)
        if crs_obj is None:
            raise ValueError("No CRS provided and no CRS found in dataset attributes")

        if {"longitude", "latitude"}.issubset(ds.dims):
            raise ValueError(
                "Cannot add x/y coordinates to a GRID dataset that has longitude/latitude dimensions"
            )
        elif {"xc", "yc"}.issubset(ds.coords):
            return ds
        else:
            xyz = crs_obj.transform_points(
                x=ds["longitude"].values,
                y=ds["latitude"].values,
                src_crs=ccrs.PlateCarree(),
            )

        if self.is_grid():
            return ds.assign_coords(
                xc=("grid_index", xyz[:, 0]), yc=("grid_index", xyz[:, 1])
            )
        elif self.is_point():
            return ds.assign_coords(
                xc=("point_index", xyz[:, 0]), yc=("point_index", xyz[:, 1])
            )
        else:
            raise ValueError("Dataset does not have expected spatial properties")

    def is_stacked(self) -> bool:
        """Return whether the spatial dims are already collapsed into ``grid_index``.

        Returns
        -------
        bool
            ``False`` if the dataset has ``xc``/``yc`` or
            ``longitude``/``latitude`` dimensions, ``True`` if it has a
            ``grid_index`` dimension.

        Raises
        ------
        ValueError
            If the dataset has none of these dimensions.
        """
        if {"xc", "yc"}.issubset(self._ds.dims) or {"longitude", "latitude"}.issubset(
            self._ds.dims
        ):
            return False
        elif "grid_index" in self._ds.dims:
            return True
        else:
            raise ValueError("Dataset does not have expected dimensions for GRID")

    def stack(self) -> xr.Dataset:
        """Collapse 2-D spatial dims into a flat ``grid_index`` dimension.

        Returns
        -------
        xr.Dataset
            The dataset with ``yc``/``xc`` (or ``latitude``/``longitude``)
            stacked into ``grid_index``; returned unchanged if already stacked.

        Raises
        ------
        ValueError
            If the dataset is a point dataset or has no stackable dimensions.
        KeyError
            If the dataset has no space trait attribute.
        """
        if self.is_point():
            raise ValueError("POINT datasets cannot be stacked")
        if self.is_stacked():
            return self._ds
        else:
            if {"xc", "yc"}.issubset(self._ds.dims):
                dims_to_stack = ["yc", "xc"]
            elif {"latitude", "longitude"}.issubset(self._ds.dims):
                dims_to_stack = ["latitude", "longitude"]
            else:
                raise ValueError("Could not find correct dimensions to stack")
        return self._ds.stack({"grid_index": dims_to_stack}).reset_index("grid_index")

    def unstack(
        self, crs: str | dict | ccrs.Projection | None = None, **kwargs: float | None
    ) -> xr.Dataset:
        """Restore a flat ``grid_index`` dim to 2-D ``xc``/``yc`` dimensions.

        Grid parameters are taken from ``**kwargs`` where given, and otherwise
        from ``ds.attrs["grid_mapping"]``.

        Parameters
        ----------
        crs : str or dict or ccrs.Projection, optional
            CRS to attach first (see :meth:`add_crs`). If not given, the CRS
            already stored in ``ds.attrs["crs"]`` is used.
        **kwargs : float or None
            Grid parameters ``nx``, ``ny``, ``lon_ll``, ``lat_ll``, ``dx`` and
            ``dy``, overriding the values in ``ds.attrs["grid_mapping"]``.

        Returns
        -------
        xr.Dataset
            The unstacked dataset, with ``attrs["grid_mapping"]`` set to the
            grid parameters used; returned unchanged if not stacked.

        Raises
        ------
        ValueError
            If the dataset is a point dataset, a grid parameter is ``None``, or
            ``nx * ny`` does not match the size of ``grid_index``.
        KeyError
            If a grid parameter is neither given nor found in
            ``ds.attrs["grid_mapping"]``, or the dataset has no space trait
            or ``crs`` attribute.
        """
        if self.is_point():
            raise ValueError("POINT datasets cannot be unstacked")
        if not self.is_stacked():
            return self._ds
        else:
            ds = self.add_crs(crs) if crs else self._ds
            kws_mindex: dict[str, int | float] = {}
            for key in ["nx", "ny", "lon_ll", "lat_ll", "dx", "dy"]:
                value = kwargs.get(key)
                if value is None:
                    try:
                        value = ds.attrs["grid_mapping"][key]
                    except KeyError:
                        raise KeyError(
                            f"Did not find a value for {key} in dataset attributes, please provide it as an argument"
                        )
                if value is None:
                    raise ValueError(f"Value for {key} cannot be None")
                if key in ("nx", "ny"):
                    kws_mindex[key] = int(value)
                else:
                    kws_mindex[key] = float(value)

            mindex = _create_multiindex(ds, **kws_mindex)
            mcoords = xr.Coordinates.from_pandas_multiindex(mindex, "grid_index")
            ds_mindex = ds.assign_coords(mcoords)
            ds_mindex.attrs["grid_mapping"] = kws_mindex
            return ds_mindex.unstack()

    # --- Time operations ---

    def add_valid_time(self) -> xr.Dataset:
        """Add a ``valid_time`` coordinate (``reference_time + lead_time``) to forecasts.

        Returns
        -------
        xr.Dataset
            The dataset with a 2-D ``valid_time`` coordinate if it is a
            forecast; otherwise the dataset unchanged.

        Raises
        ------
        KeyError
            If the dataset has no time trait attribute.
        """
        if self.is_forecast():
            return _time._add_valid_time(self._ds)
        return self._ds

    # --- Alignment ---

    def align_time_with(
        self,
        ds2: xr.Dataset,
        lead_time: str | list[str | np.timedelta64] | np.timedelta64 | None = None,
    ) -> xr.Dataset:
        """Align this dataset's time axis to match ds2.

        Always uses "reference" semantics: self is reindexed to ds2's time
        coordinates, with NaN-fill for times not present in self. ds2 is never
        modified. For symmetric inner-join behaviour across multiple datasets use
        the module-level ``align_time`` function instead.

        Parameters
        ----------
        ds2 : xr.Dataset
            The reference dataset to align to.
        lead_time : str or np.timedelta64 or list, optional
            For Forecast→Observation: "shortest" (default) | "longest" | specific value or list.
            For Forecast→Forecast: "reference" (default) | "intersection" | "union".
            Ignored for observation→* cases.

        Returns
        -------
        xr.Dataset
            This dataset aligned to the time coordinates of ``ds2``.

        Raises
        ------
        ValueError
            If the time traits cannot be matched, or ``lead_time`` is invalid
            for the forecast case.
        KeyError
            If either dataset has no time trait attribute.
        """
        if self.is_forecast() and ds2.mx.is_observation():
            return _time.align_forecast_to_observation(
                self._ds, ds2, lead_time="shortest" if lead_time is None else lead_time
            )
        elif self.is_observation() and ds2.mx.is_forecast():
            return _time.align_observation_to_forecast(self._ds, ds2)
        elif self.is_observation() and ds2.mx.is_observation():
            return _time.align_observation_to_observation(self._ds, ds2)
        elif self.is_forecast() and ds2.mx.is_forecast():
            return _time.align_forecast_to_forecast(
                self._ds, ds2, lead_time="reference" if lead_time is None else lead_time
            )
        else:
            raise ValueError("Cannot align datasets with unknown time properties")

    def align_space_with(self, ds2: xr.Dataset, **kwargs: object) -> xr.Dataset:
        """Align this dataset's spatial grid to match ds2.

        Always uses "reference" semantics: self is interpolated or reindexed to
        ds2's spatial coordinates. ds2 is never modified.

        Parameters
        ----------
        ds2 : xr.Dataset
            The reference dataset to align to.
        **kwargs : object
            Passed through to the pairwise alignment function. For grid→point
            alignment this includes ``method``, the interpolator name (one of
            "xarray" or "delaunay", default "xarray"), with the remaining
            keywords passed to the interpolator. Ignored for grid→grid.

        Returns
        -------
        xr.Dataset
            This dataset aligned to the spatial coordinates of ``ds2``.

        Raises
        ------
        NotImplementedError
            For point→point and point→grid alignment, and for grid→grid
            alignment of grids that differ (regridding).
        ValueError
            If the space traits are not compatible.
        KeyError
            If either dataset has no space trait attribute.
        """
        if self.is_grid():
            if ds2.mx.is_grid():
                return _space.align_grid_grid(self._ds, ds2, **kwargs)
            elif ds2.mx.is_point():
                return _space.align_grid_point(self._ds, ds2, **kwargs)
        elif self.is_point():
            if ds2.mx.is_point():
                raise NotImplementedError("Point-to-point alignment not implemented")
            elif ds2.mx.is_grid():
                raise NotImplementedError("Point-to-grid alignment not implemented")
        raise ValueError("Datasets do not have compatible spatial properties")


def _create_multiindex(
    ds: xr.Dataset,
    nx: int,
    ny: int,
    lon_ll: float,
    lat_ll: float,
    dx: float,
    dy: float,
    **kwargs: float,
) -> pd.MultiIndex:
    """Build the ``(yc, xc)`` MultiIndex for ``ds.grid_index`` from grid parameters.

    Parameters
    ----------
    ds : xr.Dataset
        Stacked dataset with a ``grid_index`` dimension and a Cartopy CRS in
        ``ds.attrs["crs"]``.
    nx : int
        Number of grid points in the x direction.
    ny : int
        Number of grid points in the y direction.
    lon_ll : float
        Longitude of the lower-left grid point.
    lat_ll : float
        Latitude of the lower-left grid point.
    dx : float
        Grid spacing in the x direction, in CRS units.
    dy : float
        Grid spacing in the y direction, in CRS units.
    **kwargs : float
        Ignored; allows passing a full grid-mapping dict.

    Returns
    -------
    pd.MultiIndex
        Product index of ``yc`` and ``xc`` projected coordinates.

    Raises
    ------
    ValueError
        If ``nx * ny`` does not match the size of ``grid_index``.
    KeyError
        If the dataset has no ``crs`` attribute.
    """
    from pandas import MultiIndex

    if ds.sizes["grid_index"] != nx * ny:
        raise ValueError(
            f"Size of grid_index ({ds.sizes['grid_index']}) does not match nx*ny ({nx * ny})"
        )

    crs = ds.attrs["crs"]
    x_ll, y_ll = crs.transform_point(x=lon_ll, y=lat_ll, src_crs=ccrs.PlateCarree())

    xc = x_ll + np.arange(nx) * dx
    yc = y_ll + np.arange(ny) * dy

    return MultiIndex.from_product([yc, xc], names=["yc", "xc"])
