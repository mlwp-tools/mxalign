"""Tests for the pipeline ``Runner``."""

from pathlib import Path

import pytest
import xarray as xr
from pydantic import ValidationError

from mxalign.pipeline.runner import Runner


def _runner(datasets: dict, alignment: dict) -> Runner:
    """Build a ``Runner`` with pre-loaded datasets and an alignment section.

    Parameters
    ----------
    datasets : dict
        Datasets by name, assigned directly instead of being loaded.
    alignment : dict
        The ``alignment`` section of the config.

    Returns
    -------
    Runner
        Runner ready to transform or align the given datasets.
    """
    runner = Runner({"datasets": {}, "alignment": alignment})
    runner.datasets = datasets
    return runner


def test_align_runs_time_alignment_without_time_section(
    ds_fcst: xr.Dataset, ds_obs: xr.Dataset
) -> None:
    """Check that ``align`` runs time alignment when no time section is given.

    Parameters
    ----------
    ds_fcst : xr.Dataset
        Point forecast fixture.
    ds_obs : xr.Dataset
        Point observation fixture.
    """
    runner = _runner(
        {"fcst": ds_fcst, "obs": ds_obs},
        {"reference": "obs", "broadcast_nans": False},
    )
    runner.align()
    # Global time alignment reshapes the observation reference onto the forecast grid
    assert runner.datasets["obs"].mx.is_forecast()


def test_align_time_rejects_unknown_options() -> None:
    """Check that unknown ``alignment.time`` options are rejected."""
    with pytest.raises(ValidationError, match="alignment.time.return_as"):
        Runner(
            {
                "datasets": {},
                "alignment": {"reference": "obs", "time": {"return_as": "forecast"}},
            }
        )


def test_transform_datasets_without_transformations_section(
    ds_obs: xr.Dataset,
) -> None:
    """Check that datasets pass through unchanged without a transformations section.

    Parameters
    ----------
    ds_obs : xr.Dataset
        Point observation fixture.
    """
    runner = _runner({"obs": ds_obs}, {"reference": "obs"})
    runner.transform_datasets()
    assert runner.datasets["obs"] is ds_obs


def test_align_saves_merged_datasets(ds_obs: xr.Dataset, tmp_path: Path) -> None:
    """Check that ``align`` saves a single merged file with ``datasets: merge``.

    Parameters
    ----------
    ds_obs : xr.Dataset
        Point observation fixture.
    tmp_path : Path
        Temporary directory to save the output in.
    """
    runner = _runner(
        {"a": ds_obs, "b": ds_obs.copy()},
        {
            "reference": "a",
            "broadcast_nans": False,
            "save": {
                "method": "to_netcdf",
                "datasets": "merge",
                "path": str(tmp_path / "{name}.nc"),
            },
        },
    )
    runner.align()
    assert (tmp_path / "merged.nc").exists()
