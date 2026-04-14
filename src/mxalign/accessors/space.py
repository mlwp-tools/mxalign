import xarray as xr
import cartopy.crs as ccrs
import numpy as np

from ..properties.properties import Space
from ..properties.utils import properties_from_attrs

from ..utils.projections import create_cartopy_crs, BUILTIN

# Tolerance in degrees that the coordinates of two grids can differ while still being interpreted as the same grid.
# 0.0001 degrees ~ 10m at 45 deg latitude
COORD_TOLERANCE = 0.0001


@xr.register_dataset_accessor("space")
class SpaceAccessor:
    def __init__(self, ds):
        self._space = properties_from_attrs(ds).space
        self._ds = ds

    def is_grid(self):
        return self._space == Space.GRID

    def is_point(self):
        return self._space == Space.POINT

    def add_crs(self, crs):
        if self.is_point():
            raise ValueError("Cannot add CRS to a point dataset")
        if isinstance(crs, str):
            try:
                crs = BUILTIN[crs.lower()]
            except KeyError:
                raise ValueError("crs: {crs} not found in supported projections")
        if isinstance(crs, dict):
            crs = create_cartopy_crs(
                projection=crs["projection"],
                kws_projection=crs["kws_projection"],
                kws_globe=crs.get("kws_globe", None),
            )
        return self._ds.assign_attrs({"crs": crs})

    def add_grid_mapping(self, grid_mapping: str | dict):
        if self.is_point():
            raise ValueError("Cannot add grid mapping to a point dataset")
        if isinstance(grid_mapping, str):
            try:
                grid_mapping = BUILTIN[grid_mapping.lower()]["kws_grid"]
            except KeyError:
                raise ValueError(
                    "grid mapping: {grid_mapping} not found in supported mappings"
                )
        return self._ds.assign_attrs({"grid_mapping": grid_mapping})

    def add_xy(self, crs=None):
        if crs is not None:
            self._ds = self.add_crs(crs)

        crs = self._ds.attrs.get("crs", None)

        if crs is None:
            raise ValueError("No CRS provided and no CRS found in dataset attributes")

        if {"longitude", "latitude"}.issubset(self._ds.dims):
            raise ValueError(
                "Cannot add x/y coordinates to a GRID dataset that has longitude/latitude dimensions"
            )
        elif {"xc", "yc"}.issubset(self._ds.coords):
            return self._ds
        else:
            xyz = crs.transform_points(
                x=self._ds["longitude"].values,
                y=self._ds["latitude"].values,
                src_crs=ccrs.PlateCarree(),
            )

        if self.is_grid():
            ds_out = self._ds.assign_coords(
                xc=("grid_index", xyz[:, 0]), yc=("grid_index", xyz[:, 1])
            )
        elif self.is_point():
            ds_out = self._ds.assign_coords(
                xc=("point_index", xyz[:, 0]), yc=("point_index", xyz[:, 1])
            )
        else:
            raise ValueError("Dataset does not have expected spatial properties")

        return ds_out

    def is_stacked(self):
        if {"xc", "yc"}.issubset(self._ds.dims) or {"longitude", "latitude"}.issubset(
            self._ds.dims
        ):
            return False
        elif "grid_index" in self._ds.dims:
            return True
        else:
            raise ValueError("Dataset does not have expected dimensions for GRID")

    def stack(self):
        if self.is_point():
            raise ValueError("POINT datasets cannot be stacked")
        if self.is_stacked():
            return self._ds
        else:
            if {"xc", "yc"}.issubset(self._ds.dims):
                dims_to_stack = ["yc", "xc"]
            elif {"lat", "lon"}.issubset(self._ds.dims):
                dims_to_stack = ["lat", "lon"]
            else:
                raise ValueError("Could not find correct dimensions to stack")
        return self._ds.stack({"grid_index": dims_to_stack}).reset_index("grid_index")

    def unstack(self, crs=None, **kwargs):
        if self.is_point():
            raise ValueError("POINT datasets cannot be unstacked")
        if not self.is_stacked():
            return self._ds
        else:
            if crs:
                self.add_crs(crs)
            kws_mindex = dict.fromkeys(["nx", "ny", "lon_ll", "lat_ll", "dx", "dy"])
            for key in kws_mindex.keys():
                value = kwargs.get(key, None)
                if value is None:
                    try:
                        value = self._ds.attrs["grid_mapping"][key]
                    except KeyError:
                        raise KeyError(
                            f"Did not find a value for {key} in the dataset attributes, please provide it as an argument"
                        )
                kws_mindex[key] = value

            mindex = self._create_multiindex(**kws_mindex)
            mcoords = xr.Coordinates.from_pandas_multiindex(mindex, "grid_index")
            ds_mindex = self._ds.assign_coords(mcoords)
            ds_mindex.attrs["grid_mapping"] = kws_mindex
            return ds_mindex.unstack()

    def _create_multiindex(self, nx, ny, lon_ll, lat_ll, dx, dy, **kwargs):
        from pandas import MultiIndex

        if self._ds.sizes["grid_index"] != nx * ny:
            raise ValueError(
                f"Size of grid_index ({self._ds.sizes['grid_index']}) does not match product of nx and ny ({nx * ny})"
            )

        crs = self._ds.attrs["crs"]
        x_ll, y_ll = crs.transform_point(x=lon_ll, y=lat_ll, src_crs=ccrs.PlateCarree())

        xc = x_ll + np.arange(nx) * dx
        yc = y_ll + np.arange(ny) * dy

        mindex = MultiIndex.from_product([yc, xc], names=["yc", "xc"])

        return mindex

    def align_with(self, ds, **kwargs):
        if self.is_grid():
            if ds.space.is_grid():
                return _align_grid_grid(self._ds, ds, **kwargs)
            elif ds.space.is_point():
                return _align_grid_point(self._ds, ds, **kwargs)
        elif self.is_point():
            if ds.space.is_point():
                return _align_point_point(self._ds, ds, **kwargs)
            elif ds.space.is_grid():
                return _align_point_grid(self._ds, ds, **kwargs)
        else:
            raise ValueError("Datasets do not have compatible spatial properties")


def _align_grid_grid(ds1, ds2, **kwargs):
    if np.array_equal(
        ds1["longitude"].values, ds2["longitude"].values
    ) and np.array_equal(ds1["latitude"].values, ds2["latitude"].values):
        return ds1, ds2
    elif np.allclose(
        ds1["longitude"].values, ds2["longitude"].values, atol=COORD_TOLERANCE
    ) and np.allclose(
        ds1["latitude"].values, ds2["latitude"].values, atol=COORD_TOLERANCE
    ):
        print(
            f"Some lat-lon coordinates differ. But the difference is smaller than {COORD_TOLERANCE} degrees, considering both grids as equal"
        )
        return ds1, ds2
    else:
        raise NotImplementedError("Regridding not implemented")


def _align_grid_point(ds1, ds2, **kwargs):
    from ..interpolations.interpolate import interpolate

    method = kwargs.pop("method", "xarray")
    ds1 = interpolate(ds1, ds2, method, **kwargs)

    return ds1, ds2


def _align_point_point(ds1, ds2, **kwargs):
    raise NotImplementedError("Point selection not implemented")


def _align_point_grid(ds1, ds2, **kwargs):
    raise NotImplementedError("Gridding of Point datanot implemented")
