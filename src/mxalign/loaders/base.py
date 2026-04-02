from abc import ABC, abstractmethod

from .registry import register_loader
from ..properties.properties import Properties, Space, Time, Uncertainty
from ..properties.validation import validate_dataset
from ..properties.utils import properties_to_attrs


class BaseLoader(ABC):
    """Base class for all loaders."""

    name: str = "base"

    space: Space | None = None
    time: Time | None = None
    uncertainty: Uncertainty | None = None

    def __init__(self, files, variables=None, grid_mapping=None, **kwargs):
        self.files = files
        self.variables = [variables] if isinstance(variables, str) else variables
        self.grid_mapping = grid_mapping
        self.kwargs = kwargs

    def load(self):
        ds = self._load()
        if self.variables:
            ds = self._select_variables(ds)

        properties = self._get_properties(ds)
        validate_dataset(ds, properties)

        ds.attrs["properties"] = properties_to_attrs(properties)

        if self.grid_mapping:
            ds = self._add_grid_mapping(ds)

        # Make sure all the coordinates are loaded
        for coord in ds.coords:
            ds[coord] = ds[coord].compute()

        return ds

    @abstractmethod
    def _load(self): ...

    def _select_variables(self, ds):
        return ds[self.variables]

    def _add_grid_mapping(self, ds):
        ds = ds.space.add_crs(self.grid_mapping)
        ds = ds.space.add_grid_mapping(self.grid_mapping)
        return ds

    def _get_properties(self, ds):
        properties = Properties(
            space=self.space, time=self.time, uncertainty=self.uncertainty
        )
        return properties


@register_loader
class MxAlignLoader(BaseLoader):
    name = "mxalign"

    space = None
    time = None
    uncertainty = None

    def _load(self):
        import xarray as xr

        files = [self.files] if isinstance(self.files, str) else self.files

        ds = xr.open_mfdataset(files, chunks="auto", **self.kwargs)
        if "code" in ds.dims:
            ds = ds.rename_dims({"code": "point_index"}).transpose(
                "valid_time", "point_index"
            )
        return ds

    def _get_properties(self, ds):
        if "reference_time" in ds.dims and "lead_time" in ds.dims:
            time = Time.FORECAST
        elif "valid_time" in ds.dims:
            time = Time.OBSERVATION
        else:
            raise ValueError("Unknown temporal dimensions")

        if "grid_index" in ds.dims or "xc" in ds.dims or "latitude" in ds.dims:
            space = Space.GRID
        elif "point_index" in ds.dims:
            space = Space.POINT
        else:
            raise ValueError("Unknown spatial dimensions")

        if "member" in ds.dims:
            uncertainty = Uncertainty.ENSEMBLE
        elif "quantile" in ds.dims:
            uncertainty = Uncertainty.QUANTILE
        else:
            uncertainty = Uncertainty.DETERMINISTIC

        return Properties(space=space, time=time, uncertainty=uncertainty)
