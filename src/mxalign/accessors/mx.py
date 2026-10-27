import numpy as np
import xarray as xr
import cartopy.crs as ccrs

from mlwp_data_specs.api import SPACE_TRAIT_ATTR, TIME_TRAIT_ATTR
from mlwp_data_specs.specs.traits.spatial_coordinate import Space
from mlwp_data_specs.specs.traits.time_coordinate import Time

from ..utils.projections import create_cartopy_crs, BUILTIN
from . import time as _time
from . import space as _space


@xr.register_dataset_accessor("mx")
class MxAccessor:
    """``ds.mx`` accessor — space/time predicates and alignment operations.

    Reads ``mlwp_space_trait`` and ``mlwp_time_trait`` from ``ds.attrs`` on
    every use (xarray caches the accessor per object, so reading them once
    would go stale when attrs change). Raises ``KeyError`` when a trait is
    needed but missing. Never modifies ``ds``; operations return new datasets.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def _trait(self, attr: str) -> str:
        try:
            return self._ds.attrs[attr]
        except KeyError:
            raise KeyError(
                f"Dataset has no '{attr}' attribute; load it with mlwp-data-loaders "
                "or set it with mxalign.utils.traits."
            ) from None

    @property
    def _space(self) -> Space:
        return Space(self._trait(SPACE_TRAIT_ATTR))

    @property
    def _time(self) -> Time:
        return Time(self._trait(TIME_TRAIT_ATTR))

    # --- Space predicates ---

    def is_grid(self) -> bool:
        return self._space == Space.GRID

    def is_point(self) -> bool:
        return self._space == Space.POINT

    # --- Time predicates ---

    def is_forecast(self) -> bool:
        return self._time == Time.FORECAST

    def is_observation(self) -> bool:
        return self._time == Time.OBSERVATION

    # --- Space operations ---

    def add_crs(self, crs: str | dict | ccrs.Projection) -> xr.Dataset:
        """Attach a Cartopy CRS to ``ds.attrs["crs"]``; accepts a name, config dict, or CRS object."""
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
        """Attach a grid-mapping dict to ``ds.attrs["grid_mapping"]``; accepts a builtin name or dict."""
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
        """Project ``longitude``/``latitude`` to ``xc``/``yc`` coordinates using the dataset CRS."""
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
        """Return ``True`` if the spatial dims are already collapsed into ``grid_index``."""
        if {"xc", "yc"}.issubset(self._ds.dims) or {"longitude", "latitude"}.issubset(
            self._ds.dims
        ):
            return False
        elif "grid_index" in self._ds.dims:
            return True
        else:
            raise ValueError("Dataset does not have expected dimensions for GRID")

    def stack(self) -> xr.Dataset:
        """Collapse 2-D spatial dims into a flat ``grid_index`` dimension."""
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
        self, crs: str | dict | ccrs.Projection | None = None, **kwargs
    ) -> xr.Dataset:
        """Restore a flat ``grid_index`` dim to 2-D ``xc``/``yc`` using grid-mapping metadata."""
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
        """Add a ``valid_time`` coordinate (``reference_time + lead_time``) to forecast datasets."""
        if self.is_forecast():
            return _time._add_valid_time(self._ds)
        return self._ds

    # --- Alignment ---

    def align_time_with(
        self, ds2: xr.Dataset, lead_time: str | list | np.timedelta64 | None = None
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
        lead_time : str or timedelta or list, optional
            For Forecast→Observation: "shortest" (default) | "longest" | specific value or list.
            For Forecast→Forecast: "reference" (default) | "intersection" | "union".
            Ignored for observation→* cases.
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

    def align_space_with(self, ds2: xr.Dataset, **kwargs) -> xr.Dataset:
        """Align this dataset's spatial grid to match ds2.

        Always uses "reference" semantics: self is interpolated or reindexed to
        ds2's spatial coordinates. ds2 is never modified.

        Parameters
        ----------
        ds2 : xr.Dataset
            The reference dataset to align to.
        **kwargs
            ``method`` selects the interpolator for grid→point alignment: one
            of "xarray" or "delaunay" (default "xarray"). All other keywords
            are passed through to the interpolator. Ignored for grid→grid.
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
    **kwargs,
):
    """Build the ``(yc, xc)`` MultiIndex for ``ds.grid_index`` from grid-mapping parameters."""
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
