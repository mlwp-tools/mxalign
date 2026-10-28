"""Base class shared by all interpolators."""

import xarray as xr
from mlwp_data_specs.specs.traits.spatial_coordinate import Space
from ...utils.traits import update_space_trait

# Attributes set by ds.mx.add_crs / add_grid_mapping that describe a source grid
GRID_ONLY_ATTRS = ("crs", "grid_mapping")


class BaseInterpolator:
    """Base class for all interpolators.

    Subclasses must set ``name``, ``source_space``, and ``target_space``, and
    implement ``_interpolate``. Subclasses must also copy ``source_dataset.attrs``
    to the output so time/uncertainty traits are preserved.

    Parameters
    ----------
    target_dataset : xr.Dataset
        Dataset whose spatial locations the source is interpolated to.
    **options : object
        Interpolator-specific options, stored as ``self.options``.
    """

    name: str = "base"
    source_space: Space | None = None
    target_space: Space | None = None

    def __init__(self, target_dataset: xr.Dataset, **options: object) -> None:
        self.target_dataset = target_dataset
        self.options = options
        # TODO: Check the properties

    def interpolate(
        self, source_dataset: xr.Dataset | xr.DataArray
    ) -> xr.Dataset | xr.DataArray:
        """Run interpolation and set the output space trait to ``target_space``.

        For point targets, the grid-only attributes ``crs`` and
        ``grid_mapping`` are dropped from the output.

        Parameters
        ----------
        source_dataset : xr.Dataset or xr.DataArray
            Data to interpolate to the target locations.

        Returns
        -------
        xr.Dataset or xr.DataArray
            The interpolated data with the space trait set to ``target_space``.
        """
        ds_out = self._interpolate(source_dataset)
        if self.target_space == Space.POINT:
            # Grid-only attrs don't apply to points (and a cartopy CRS can't be saved)
            ds_out.attrs = {
                k: v for k, v in ds_out.attrs.items() if k not in GRID_ONLY_ATTRS
            }
        return update_space_trait(ds_out, self.target_space)

    def _interpolate(
        self, source_dataset: xr.Dataset | xr.DataArray
    ) -> xr.Dataset | xr.DataArray:
        """Perform the interpolation; subclasses must override this.

        Parameters
        ----------
        source_dataset : xr.Dataset or xr.DataArray
            Data to interpolate to the target locations.

        Returns
        -------
        xr.Dataset or xr.DataArray
            The interpolated data.
        """
        pass
