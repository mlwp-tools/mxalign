"""Tests for date-based file path substitution."""

import datetime

import pytest
import xarray as xr

from mxalign.pipeline.config import load_config
from mxalign.pipeline.dates import Dates
from mxalign.pipeline.save import DatasetPath

# 3 reference times (daily) × 3 lead times (0, 12, 24h)
DATES = {
    "start": "2020-01-01",
    "end": "2020-01-03",
    "period": "1D",
    "range": "24h",
    "step": "12h",
}


def test_substitute_reference_time() -> None:
    """Check that ``reference_time`` placeholders expand to one path per run."""
    paths = Dates(**DATES).substitute("/d/{reference_time:%Y%m%d%H}.nc")
    assert paths == ["/d/2020010100.nc", "/d/2020010200.nc", "/d/2020010300.nc"]


def test_substitute_reference_and_lead_time() -> None:
    """Check that reference and lead time placeholders expand as a product."""
    paths = Dates(**DATES).substitute("/d/{reference_time:%Y%m%d}_+{lead_time:03d}.nc")
    assert len(paths) == 9
    assert paths[:3] == [
        "/d/20200101_+000.nc",
        "/d/20200101_+012.nc",
        "/d/20200101_+024.nc",
    ]


def test_substitute_valid_time_is_deduplicated_and_sorted() -> None:
    """Check that ``valid_time`` placeholders give sorted, unique paths."""
    paths = Dates(**DATES).substitute("/d/{valid_time:%Y%m%d%H}.nc")
    # 2020-01-01T00 … 2020-01-04T00 every 12h; overlaps between references collapse
    assert len(paths) == 7
    assert paths[0] == "/d/2020010100.nc"
    assert paths[-1] == "/d/2020010400.nc"


def test_substitute_without_fields_returns_single_path() -> None:
    """Check that a pattern without placeholders yields a single path."""
    assert Dates(**DATES).substitute("/d/store.zarr") == ["/d/store.zarr"]


def test_substitute_lead_time_requires_whole_hours() -> None:
    """Check that a sub-hourly step raises when substituting ``lead_time``."""
    dates = Dates(**{**DATES, "step": "30min"})
    with pytest.raises(ValueError, match="whole hours"):
        dates.substitute("/d/{lead_time}.nc")


def test_config_expands_files_from_dates_section() -> None:
    """Check that config loading expands file patterns over the dates section."""
    # YAML parses unquoted timestamps to datetime objects
    config = load_config(
        {
            "dates": {**DATES, "start": datetime.datetime(2020, 1, 1)},
            "datasets": {
                "fc": {"loader": "x", "files": "/d/{reference_time:%Y%m%d}.nc"}
            },
            "alignment": {"reference": "fc"},
        }
    )
    assert config.datasets["fc"].files == [
        "/d/20200101.nc",
        "/d/20200102.nc",
        "/d/20200103.nc",
    ]


def test_dataset_path_substitute(ds_fcst: xr.Dataset) -> None:
    """Check that ``DatasetPath`` fills name and date fields from the dataset.

    Parameters
    ----------
    ds_fcst : xr.Dataset
        Point forecast fixture.
    """
    path = DatasetPath("model", ds_fcst).substitute("{name}_{year}{month:02d}.nc")
    assert path == "model_202001.nc"
