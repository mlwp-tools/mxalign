import json
import tempfile
import unittest

import xarray as xr

from mxalign.properties.properties import Properties, Space, Time, Uncertainty
from mxalign.properties.utils import properties_from_attrs, set_properties_attrs


class TestPropertiesAttrs(unittest.TestCase):
    def test_properties_are_stored_in_netcdf_compatible_attrs(self):
        ds = xr.Dataset()
        props = Properties(
            space=Space.POINT,
            time=Time.OBSERVATION,
            uncertainty=Uncertainty.DETERMINISTIC,
        )

        ds = set_properties_attrs(ds, props)

        self.assertNotIn("properties", ds.attrs)
        self.assertEqual(ds.attrs["properties_space"], "point")
        self.assertEqual(ds.attrs["properties_time"], "observation")
        self.assertEqual(ds.attrs["properties_uncertainty"], "deterministic")

        with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
            ds.to_netcdf(tmp.name)
            with xr.open_dataset(tmp.name) as ds_loaded:
                self.assertEqual(properties_from_attrs(ds_loaded), props)

    def test_properties_can_still_be_read_from_legacy_format(self):
        ds = xr.Dataset()
        ds.attrs["properties"] = {
            "space": "point",
            "time": "observation",
            "uncertainty": "deterministic",
        }
        self.assertEqual(
            properties_from_attrs(ds),
            Properties(
                space=Space.POINT,
                time=Time.OBSERVATION,
                uncertainty=Uncertainty.DETERMINISTIC,
            ),
        )

    def test_properties_can_be_read_from_legacy_json_string(self):
        ds = xr.Dataset()
        ds.attrs["properties"] = json.dumps(
            {"space": "point", "time": "observation", "uncertainty": "deterministic"}
        )
        self.assertEqual(
            properties_from_attrs(ds),
            Properties(
                space=Space.POINT,
                time=Time.OBSERVATION,
                uncertainty=Uncertainty.DETERMINISTIC,
            ),
        )


if __name__ == "__main__":
    unittest.main()
