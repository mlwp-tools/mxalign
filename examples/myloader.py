"""Example custom loader for SYNOP observations, used by the example pipeline config."""

import xarray as xr


def load_dataset(file: str) -> xr.Dataset:
    """Load a SYNOP observation netCDF file and attach the mlwp traits.

    Parameters
    ----------
    file : str
        Path to the netCDF file with a ``code`` station dimension.

    Returns
    -------
    xr.Dataset
        Observation point dataset with a ``point_index`` dimension, mlwp
        traits in its attributes, and CF attributes on its coordinates.
    """
    ds = xr.open_mfdataset([file], engine="h5netcdf")

    ds = ds.rename_dims(code="point_index")

    ds.attrs["mlwp_time_trait"] = "observation"
    ds.attrs["mlwp_space_trait"] = "point"
    ds.attrs["mlwp_uncertainty_trait"] = "deterministic"

    ds.coords["valid_time"].attrs["standard_name"] = "time"
    ds.coords["latitude"].attrs.update(
        {"standard_name": "latitude", "units": "degrees_north"}
    )
    ds.coords["longitude"].attrs.update(
        {"standard_name": "longitude", "units": "degrees_east"}
    )

    return ds
