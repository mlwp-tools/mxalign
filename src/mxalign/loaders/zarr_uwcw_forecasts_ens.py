import numpy as np
import xarray as xr

from .registry import register_loader
from ..properties.properties import Space, Time, Uncertainty
from .base import BaseLoader


@register_loader
class ZarrUWCWForecastsEnsLoader(BaseLoader):
    """
    Loader for UWC-WEST forecasts zarr files — ensemble version.

    Expects one zarr file per member. Stacks them along a 'member' dimension.

    Config example:
      loader: zarr_uwcw_forecasts_ens
      files:
        - /path/to/zarr_mbr000.zarr
        - /path/to/zarr_mbr001.zarr
        - ...
      variables: ["2t", "10u", "10v", "msl"]
    """

    name = "zarr_uwcw_forecasts_ens"

    space = Space.GRID
    time = Time.FORECAST
    uncertainty = Uncertainty.ENSEMBLE

    def _load(self):
        files = [self.files] if isinstance(self.files, str) else self.files

        member_datasets = []
        for member_idx, zarr_path in enumerate(files):
            ds = self._load_single_zarr(zarr_path)
            ds = ds.expand_dims("member").assign_coords(
                member=("member", [member_idx])
            )
            member_datasets.append(ds)

        ds = xr.concat(member_datasets, dim="member")
        ds = ds.chunk({"member": -1, "grid_index": -1})

        if "filter_reference_times" in self.kwargs:
            ref_filter = np.array(self.kwargs["filter_reference_times"], dtype="datetime64[ns]")
            ds = ds.sel(reference_time=ref_filter)

        return ds

    def _load_single_zarr(self, zarr_path):
        ds = xr.open_zarr(zarr_path, chunks="auto", consolidated=False, mask_and_scale=False)
        ds = ds.chunk({"grid_index": -1})

        ds = ds.assign_coords(
            reference_time=ds["reference_time"].values.astype("datetime64[ns]"),
            lead_time=ds["lead_time"].values.astype("timedelta64[h]").astype("timedelta64[ns]"),
        )

        if "valid_time" in ds.coords:
            ds = ds.drop_vars("valid_time")

        return ds
