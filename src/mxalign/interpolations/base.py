import xarray as xr
from mlwp_data_specs.specs.traits.spatial_coordinate import Space
from ..utils.traits import update_space_trait


class BaseInterpolator:
    """Base class for all interpolators."""

    name: str = "base"
    source_space: Space | None = None
    target_space: Space | None = None

    def __init__(self, target_dataset, **options):
        self.target_dataset = target_dataset
        self.options = options
        # TODO: Check the properties

    # def supports(self, src: Properties, tgt: Properties):

    def interpolate(
        self, source_dataset: xr.Dataset | xr.DataArray
    ) -> xr.Dataset | xr.DataArray:
        ds_out = self._interpolate(source_dataset)
        return update_space_trait(ds_out, self.target_space)

    def _interpolate(
        self, source_dataset: xr.Dataset | xr.DataArray
    ) -> xr.Dataset | xr.DataArray:
        pass
