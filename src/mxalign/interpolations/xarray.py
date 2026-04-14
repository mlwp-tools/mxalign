from .base import BaseInterpolator
from .registry import register_interpolator
from ..properties.properties import Space

import xarray as xr


@register_interpolator
class XarrayInterpolator(BaseInterpolator):
    name = "xarray"
    source_space = Space.GRID
    target_space = Space.POINT

    def _interpolate(self, source_dataset):

        if "latitude" in source_dataset.dims and "longitude" in source_dataset.dims:
            ds_out = self._interpolate_from_latlon(source_dataset)

        else:
            if source_dataset.space.is_stacked():
                try:
                    source_dataset = source_dataset.space.unstack()
                except ValueError:
                    raise ValueError(
                        "Cannot unstack dataset, dataset must be unstacked to use xarray interpolation"
                    )
            ds_out = self._interpolate_from_xcyc(source_dataset)
        return ds_out

    def _interpolate_from_xcyc(self, source_dataset):
        import cartopy.crs as ccrs

        try:
            crs = source_dataset.attrs["crs"]
        except KeyError:
            raise KeyError("Source dataset does not have a crs-attribute")

        xyz = crs.transform_points(
            x=self.target_dataset["longitude"].values,
            y=self.target_dataset["latitude"].values,
            src_crs=ccrs.PlateCarree(),
        )

        x = xr.DataArray(xyz[:, 0], dims="point_index")

        y = xr.DataArray(xyz[:, 1], dims="point_index")

        ds_out = source_dataset.interp(xc=x, yc=y, **self.options)
        # ).assing_coords(
        #     longitude=self.target_dataset["longitude"],
        #     latitude=self.target_dataset["latitude"]
        # )

        return ds_out

    def _interpolate_from_latlon(self, source_dataset):
        longitude = self.target_dataset["longitude"]
        latitude = self.target_dataset["latitude"]
        ds_out = source_dataset.interp(
            longitude=longitude, latitude=latitude, **self.options
        )

        return ds_out
