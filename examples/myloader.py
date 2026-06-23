import xarray as xr


def load_dataset(file):
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
