"""Grid-to-point interpolator based on ``xarray.Dataset.interp``."""

from .base import BaseInterpolator
from .registry import register_interpolator
from mlwp_data_specs.specs.traits.spatial_coordinate import Space

import xarray as xr


@register_interpolator
class XarrayInterpolator(BaseInterpolator):
    """Interpolate a grid dataset to points with ``xarray.Dataset.interp``.

    Lat/lon-dimension grids are interpolated directly in longitude/latitude;
    projected grids are interpolated in ``xc``/``yc`` after projecting the
    target points with the source CRS (stacked grids are unstacked first).

    Takes the constructor arguments of :class:`BaseInterpolator`: a point
    ``target_dataset`` with ``longitude`` and ``latitude`` coordinates, and
    ``**options`` that are passed to ``xarray.Dataset.interp`` (e.g.
    ``method``).
    """

    name = "xarray"
    source_space = Space.GRID
    target_space = Space.POINT

    def _interpolate(self, source_dataset: xr.Dataset) -> xr.Dataset:
        """Interpolate a grid dataset to the target points.

        Parameters
        ----------
        source_dataset : xr.Dataset
            Grid dataset with ``longitude``/``latitude`` dimensions, ``xc``/``yc``
            dimensions, or a stacked ``grid_index`` dimension.

        Returns
        -------
        xr.Dataset
            The source interpolated to the target points.

        Raises
        ------
        ValueError
            If a stacked source cannot be unstacked, or has unexpected
            dimensions.
        KeyError
            If a projected source has no ``crs`` attribute.
        """
        if "latitude" in source_dataset.dims and "longitude" in source_dataset.dims:
            ds_out = self._interpolate_from_latlon(source_dataset)

        else:
            if source_dataset.mx.is_stacked():
                try:
                    source_dataset = source_dataset.mx.unstack()
                except ValueError:
                    raise ValueError(
                        "Cannot unstack dataset, dataset must be unstacked to use xarray interpolation"
                    )
            ds_out = self._interpolate_from_xcyc(source_dataset)
        return ds_out

    def _interpolate_from_xcyc(self, source_dataset: xr.Dataset) -> xr.Dataset:
        """Interpolate a projected (xc/yc) source to target lon/lat points.

        Parameters
        ----------
        source_dataset : xr.Dataset
            Grid dataset with ``xc``/``yc`` dimensions and a Cartopy CRS in
            ``attrs["crs"]``.

        Returns
        -------
        xr.Dataset
            The source interpolated along a ``point_index`` dimension, with
            the target ``longitude``/``latitude`` coordinates.

        Raises
        ------
        KeyError
            If the source dataset has no ``crs`` attribute.
        """
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
        # Assign the target DataArrays (not .values) so the point_index index and
        # any other point coords (e.g. station IDs) are carried over
        ds_out = ds_out.assign_coords(
            longitude=self.target_dataset["longitude"],
            latitude=self.target_dataset["latitude"],
        )
        return ds_out

    def _interpolate_from_latlon(self, source_dataset: xr.Dataset) -> xr.Dataset:
        """Interpolate a lat/lon-dim source directly to target lon/lat points.

        Parameters
        ----------
        source_dataset : xr.Dataset
            Grid dataset with ``longitude`` and ``latitude`` dimensions.

        Returns
        -------
        xr.Dataset
            The source interpolated to the target points.
        """
        longitude = self.target_dataset["longitude"]
        latitude = self.target_dataset["latitude"]
        ds_out = source_dataset.interp(
            longitude=longitude, latitude=latitude, **self.options
        )
        return ds_out
