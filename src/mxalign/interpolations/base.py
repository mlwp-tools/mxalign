import xarray as xr
from ..properties.properties import Space
from ..properties.utils import update_space_property


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
        return update_space_property(ds_out, self.target_space)

    def _interpolate(
        self, source_dataset: xr.Dataset | xr.DataArray
    ) -> xr.Dataset | xr.DataArray:
        pass
