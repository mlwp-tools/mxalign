import json
import tempfile

import xarray as xr

from mxalign.properties.properties import Properties, Space, Time, Uncertainty
from mxalign.properties.utils import properties_from_attrs, set_properties_attrs


class TestPropertiesAttrs:
    def test_properties_are_stored_in_netcdf_compatible_attrs(self):
        ds = xr.Dataset()
        props = Properties(
            space=Space.POINT,
            time=Time.OBSERVATION,
            uncertainty=Uncertainty.DETERMINISTIC,
        )

        ds = set_properties_attrs(ds, props)

        assert "properties" not in ds.attrs
        assert ds.attrs["properties.space"] == "point"
        assert ds.attrs["properties.time"] == "observation"
        assert ds.attrs["properties.uncertainty"] == "deterministic"

        with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
            ds.to_netcdf(tmp.name)
            with xr.open_dataset(tmp.name) as ds_loaded:
                assert properties_from_attrs(ds_loaded) == props

    def test_properties_can_still_be_read_from_legacy_format(self):
        ds = xr.Dataset()
        ds.attrs["properties"] = {
            "space": "point",
            "time": "observation",
            "uncertainty": "deterministic",
        }
        assert properties_from_attrs(ds) == Properties(
            space=Space.POINT,
            time=Time.OBSERVATION,
            uncertainty=Uncertainty.DETERMINISTIC,
        )

    def test_properties_can_be_read_from_legacy_json_string(self):
        ds = xr.Dataset()
        ds.attrs["properties"] = json.dumps(
            {"space": "point", "time": "observation", "uncertainty": "deterministic"}
        )
        assert properties_from_attrs(ds) == Properties(
            space=Space.POINT,
            time=Time.OBSERVATION,
            uncertainty=Uncertainty.DETERMINISTIC,
        )
