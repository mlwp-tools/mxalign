from pathlib import Path

import pytest
from pydantic import ValidationError

from mxalign.pipeline.config import load_config

EXAMPLE_CONFIG = Path(__file__).parent.parent / "examples" / "config.yaml"

DATES = dict(start="2020-01-01", end="2020-01-02", period="1D", range="12h", step="12h")


def _config(**sections) -> dict:
    return {
        "datasets": {"fc": {"loader": "x", "files": "/d/fc.nc"}},
        "alignment": {"reference": "fc"},
        **sections,
    }


def test_example_config_validates():
    config = load_config(str(EXAMPLE_CONFIG))
    assert config.alignment.reference == "cerra"
    assert config.alignment.space.interpolation == {"method": "delaunay"}
    assert config.verification.metrics["rmse"].kwargs == {
        "dim": ["reference_time", "grid_index"],
        "skipna": True,
    }
    # files patterns are expanded over the dates section
    assert len(config.datasets["la-1024-ea-00"].files) > 1


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ValidationError, match="not_a_section"):
        load_config(_config(not_a_section={"reference": "fc"}))


def test_missing_alignment_reference_is_rejected():
    config = _config()
    config["alignment"] = {"broadcast_nans": False}
    with pytest.raises(ValidationError, match="alignment.reference"):
        load_config(config)


def test_unknown_save_datasets_option_is_rejected():
    save = {"method": "to_netcdf", "path": "out.nc", "datasets": "some"}
    with pytest.raises(ValidationError, match="alignment.save.datasets"):
        load_config(_config(alignment={"reference": "fc", "save": save}))


def test_extra_dataset_keys_are_passed_to_loader():
    config = _config()
    config["datasets"]["fc"]["chunks"] = {"time": 1}
    assert load_config(config).datasets["fc"].kwargs == {"chunks": {"time": 1}}


def test_dataset_dates_override_top_level_dates():
    config = _config(dates=DATES)
    config["datasets"]["fc"] = {
        "loader": "x",
        "files": "/d/{reference_time:%Y%m%d}.nc",
        "dates": {"end": "2020-01-03"},
    }
    assert load_config(config).datasets["fc"].files == [
        "/d/20200101.nc",
        "/d/20200102.nc",
        "/d/20200103.nc",
    ]


def test_incomplete_dates_are_rejected():
    config = _config()
    config["datasets"]["fc"]["dates"] = {"start": "2020-01-01"}
    with pytest.raises(ValidationError, match="dates is missing end, period"):
        load_config(config)
