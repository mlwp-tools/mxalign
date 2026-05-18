import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs

from mlwp_data_specs.api import SPACE_TRAIT_ATTR, TIME_TRAIT_ATTR
from mlwp_data_specs.specs.traits.spatial_coordinate import Space
from mlwp_data_specs.specs.traits.time_coordinate import Time

from ..utils.projections import create_cartopy_crs, BUILTIN

COORD_TOLERANCE = 0.0001


@xr.register_dataset_accessor("mx")
class MxAccessor:
    def __init__(self, ds):
        self._space = Space(ds.attrs[SPACE_TRAIT_ATTR])
        self._time = Time(ds.attrs[TIME_TRAIT_ATTR])
        self._ds = ds

    # --- Space predicates ---

    def is_grid(self):
        return self._space == Space.GRID

    def is_point(self):
        return self._space == Space.POINT

    # --- Time predicates ---

    def is_forecast(self):
        return self._time == Time.FORECAST

    def is_observation(self):
        return self._time == Time.OBSERVATION

    # --- Space operations ---

    def add_crs(self, crs):
        if self.is_point():
            raise ValueError("Cannot add CRS to a point dataset")
        if isinstance(crs, str):
            try:
                crs = BUILTIN[crs.lower()]
            except KeyError:
                raise ValueError(f"crs: {crs} not found in supported projections")
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
                    f"grid mapping: {grid_mapping} not found in supported mappings"
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
            return self._ds.assign_coords(
                xc=("grid_index", xyz[:, 0]), yc=("grid_index", xyz[:, 1])
            )
        elif self.is_point():
            return self._ds.assign_coords(
                xc=("point_index", xyz[:, 0]), yc=("point_index", xyz[:, 1])
            )
        else:
            raise ValueError("Dataset does not have expected spatial properties")

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
                self._ds = self.add_crs(crs)
            kws_mindex = dict.fromkeys(["nx", "ny", "lon_ll", "lat_ll", "dx", "dy"])
            for key in kws_mindex.keys():
                value = kwargs.get(key, None)
                if value is None:
                    try:
                        value = self._ds.attrs["grid_mapping"][key]
                    except KeyError:
                        raise KeyError(
                            f"Did not find a value for {key} in dataset attributes, please provide it as an argument"
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
                f"Size of grid_index ({self._ds.sizes['grid_index']}) does not match nx*ny ({nx * ny})"
            )

        crs = self._ds.attrs["crs"]
        x_ll, y_ll = crs.transform_point(x=lon_ll, y=lat_ll, src_crs=ccrs.PlateCarree())

        xc = x_ll + np.arange(nx) * dx
        yc = y_ll + np.arange(ny) * dy

        return MultiIndex.from_product([yc, xc], names=["yc", "xc"])

    # --- Time operations ---

    def add_valid_time(self):
        if self.is_forecast():
            valid_time = (
                self._ds["reference_time"].values[:, np.newaxis]
                + self._ds["lead_time"].values
            )
            return self._ds.assign_coords(
                {"valid_time": (["reference_time", "lead_time"], valid_time)}
            )
        return self._ds

    # --- Alignment ---

    def align_time_with(self, ds2, lead_time="shortest"):
        """Align this dataset's time axis to match ds2.

        Always uses "reference" semantics: self is reindexed to ds2's time
        coordinates, with NaN-fill for times not present in self. ds2 is never
        modified. For symmetric inner-join behaviour across multiple datasets use
        the module-level ``align_time`` function instead.

        Parameters
        ----------
        ds2 : xr.Dataset
            The reference dataset to align to.
        lead_time : str or timedelta or list
            For Forecast→Observation: "shortest" | "longest" | specific value or list.
            For Forecast→Forecast: "reference" | "intersection" | "union" (default "reference").
            Ignored for observation→* cases.
        """
        if self.is_forecast() and ds2.mx.is_observation():
            return _align_forecast_to_observation(self._ds, ds2, lead_time=lead_time)
        elif self.is_observation() and ds2.mx.is_forecast():
            return _align_observation_to_forecast(self._ds, ds2)
        elif self.is_observation() and ds2.mx.is_observation():
            return _align_observation_to_observation(self._ds, ds2)
        elif self.is_forecast() and ds2.mx.is_forecast():
            ff_lead_time = (
                lead_time
                if lead_time in ("reference", "intersection", "union")
                else "reference"
            )
            return _align_forecast_to_forecast(self._ds, ds2, lead_time=ff_lead_time)
        else:
            raise ValueError("Cannot align datasets with unknown time properties")

    def align_space_with(self, ds2, **kwargs):
        """Align this dataset's spatial grid to match ds2."""
        if self.is_grid():
            if ds2.mx.is_grid():
                return _align_grid_grid(self._ds, ds2, **kwargs)
            elif ds2.mx.is_point():
                return _align_grid_point(self._ds, ds2, **kwargs)
        elif self.is_point():
            if ds2.mx.is_point():
                raise NotImplementedError("Point-to-point alignment not implemented")
            elif ds2.mx.is_grid():
                raise NotImplementedError("Point-to-grid alignment not implemented")
        raise ValueError("Datasets do not have compatible spatial properties")


# ---------------------------------------------------------------------------
# Temporal alignment helpers
# ---------------------------------------------------------------------------


def _add_valid_time(ds_fcst):
    valid_time = (
        ds_fcst["reference_time"].values[:, np.newaxis] + ds_fcst["lead_time"].values
    )
    return ds_fcst.assign_coords(
        {"valid_time": (["reference_time", "lead_time"], valid_time)}
    )


def _align_forecast_to_observation(ds_fcst, ds_obs, lead_time="shortest"):
    ds_with_vt = _add_valid_time(ds_fcst)
    ds_stacked = ds_with_vt.stack(time=["reference_time", "lead_time"]).reset_index(
        "time"
    )

    vt_vals = ds_stacked.valid_time.values
    lt_vals = ds_stacked.lead_time.values

    if lead_time in ("shortest", "longest"):
        df = pd.DataFrame({"vt": vt_vals, "lt": lt_vals})
        agg = "min" if lead_time == "shortest" else "max"
        is_extreme = (df.groupby("vt")["lt"].transform(agg) == df["lt"]).values
        # Among entries that match the extreme lead_time, keep first per valid_time
        # (handles ties: same vt + same extreme lt appearing via different ref_times)
        extreme_positions = np.where(is_extreme)[0]
        _, first_in_group = np.unique(vt_vals[extreme_positions], return_index=True)
        positions = extreme_positions[first_in_group]
    elif isinstance(lead_time, (list, np.ndarray)):
        lt_set = set(np.asarray(lead_time).tolist())
        seen_vt = set()
        positions = []
        for i, (vt, lt) in enumerate(zip(vt_vals, lt_vals)):
            if lt in lt_set and vt not in seen_vt:
                positions.append(i)
                seen_vt.add(vt)
        positions = np.array(positions)
    else:
        # single lead_time value — filter directly
        positions = np.where(lt_vals == lead_time)[0]

    ds_1d = ds_stacked.isel(time=positions)
    ds_1d = ds_1d.swap_dims({"time": "valid_time"})
    ds_1d = ds_1d.drop_vars(
        [v for v in ["reference_time", "lead_time", "time"] if v in ds_1d.coords]
    )
    ds_1d = ds_1d.transpose("valid_time", ...)

    ds_1d = ds_1d.reindex(valid_time=ds_obs.valid_time)
    ds_1d.attrs[TIME_TRAIT_ATTR] = Time.OBSERVATION.value
    return ds_1d


def _align_observation_to_forecast(ds_obs, ds_fcst):
    ds_fcst_with_vt = _add_valid_time(ds_fcst)
    valid_time_2d = ds_fcst_with_vt["valid_time"]  # shape (reference_time, lead_time)

    # Reindex obs onto all unique fcst valid_times (NaN-fills fcst valid_times not in obs)
    fcst_vt_flat = np.unique(valid_time_2d.values.ravel())
    obs_reindexed = ds_obs.reindex(valid_time=fcst_vt_flat)

    # sel with a 2D DataArray indexer broadcasts 1D obs → (reference_time, lead_time)
    ds_out = obs_reindexed.sel(valid_time=valid_time_2d)

    ds_out.attrs[TIME_TRAIT_ATTR] = Time.FORECAST.value
    return ds_out


def _align_observation_to_observation(ds1, ds2):
    return ds1.reindex(valid_time=ds2.valid_time)


def _align_forecast_to_forecast(ds1, ds2, lead_time="reference"):
    ds_out = ds1.reindex(reference_time=ds2.reference_time)

    # Align lead_time
    if lead_time == "reference":
        ds_out = ds_out.reindex(lead_time=ds2.lead_time)
    elif lead_time == "intersection":
        common_lt = np.intersect1d(ds_out.lead_time.values, ds2.lead_time.values)
        ds_out = ds_out.sel(lead_time=common_lt)
    elif lead_time == "union":
        all_lt = np.union1d(ds_out.lead_time.values, ds2.lead_time.values)
        ds_out = ds_out.reindex(lead_time=all_lt)
    else:
        raise ValueError(f"Unknown lead_time option for F→F alignment: {lead_time!r}")

    # Refresh valid_time
    if "valid_time" in ds_out.coords:
        ds_out = ds_out.drop_vars("valid_time")
    return _add_valid_time(ds_out)


# ---------------------------------------------------------------------------
# Spatial alignment helpers
# ---------------------------------------------------------------------------


def _align_grid_grid(ds1, ds2, **kwargs):
    if np.array_equal(
        ds1["longitude"].values, ds2["longitude"].values
    ) and np.array_equal(ds1["latitude"].values, ds2["latitude"].values):
        return ds1
    elif np.allclose(
        ds1["longitude"].values, ds2["longitude"].values, atol=COORD_TOLERANCE
    ) and np.allclose(
        ds1["latitude"].values, ds2["latitude"].values, atol=COORD_TOLERANCE
    ):
        print(
            f"Some lat-lon coordinates differ but within {COORD_TOLERANCE}°, treating as equal"
        )
        return ds1
    else:
        raise NotImplementedError("Regridding not implemented")


def _align_grid_point(ds1, ds2, **kwargs):
    from ..interpolations.interpolate import interpolate

    method = kwargs.pop("method", "xarray")
    return interpolate(ds1, ds2, method, **kwargs)
