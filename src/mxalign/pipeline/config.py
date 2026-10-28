"""Schema for the pipeline YAML config.

Sections and keys not listed here are rejected, except where a model allows
extra keys (``extra="allow"``); those are passed on as keyword arguments to
the function the section configures (loader, transformation, metric, save).
"""

from datetime import datetime
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from .dates import Dates


class _Strict(BaseModel):
    """Base model that rejects unknown keys."""

    model_config = ConfigDict(extra="forbid")


class _PassThrough(BaseModel):
    """Model whose unknown keys are kept in ``model_extra`` as function kwargs."""

    model_config = ConfigDict(extra="allow")

    @property
    def kwargs(self) -> dict:
        """Return the extra (unknown) keys of the model as a keyword-argument dict.

        Returns
        -------
        dict
            A copy of ``model_extra``; empty if there are no extra keys.
        """
        return dict(self.model_extra or {})


class DatesConfig(_Strict):
    """Date range used to expand date placeholders in ``files`` patterns.

    The placeholders are ``{reference_time}``, ``{lead_time}`` and
    ``{valid_time}``. Every field is optional so a dataset can override part
    of the top-level ``dates`` section; after merging, all five must be set.

    Attributes
    ----------
    start : datetime or str or None
        First reference time of the range.
    end : datetime or str or None
        Last reference time of the range.
    period : str or None
        Step between successive reference times (e.g. ``"1D"``).
    range : str or None
        Forecast horizon; lead times run from 0 to ``range`` inclusive.
    step : str or None
        Increment between lead times (e.g. ``"1h"``).
    """

    start: datetime | str | None = None
    end: datetime | str | None = None
    period: str | None = None
    range: str | None = None
    step: str | None = None


class DatasetConfig(_PassThrough):
    """One entry of ``datasets``; extra keys go to ``load_and_validate_dataset``.

    Attributes
    ----------
    loader : str
        Name of the ``mlwp_data_loaders`` loader used to open the files.
    files : str or list of str
        File path(s) or path pattern(s); patterns are expanded over ``dates``.
    variables : str or list of str or None
        Variable(s) to keep after loading; all variables if None.
    grid_mapping : str or dict or None
        Grid mapping (CRS) to attach to the dataset; none if None.
    dates : DatesConfig or None
        Dataset-specific date range, merged over the top-level ``dates``.
    """

    loader: str
    files: str | list[str]
    variables: str | list[str] | None = None
    grid_mapping: str | dict | None = None
    dates: DatesConfig | None = None


class TransformationConfig(_PassThrough):
    """One entry of ``transformations``; extra keys go to the transformation.

    Entries are keyed by transformation name.

    Attributes
    ----------
    datasets : list of str or None
        Names of the datasets to transform; all datasets if None.
    """

    datasets: list[str] | None = None


class TimeAlignmentConfig(_Strict):
    """Options for the time alignment step.

    Attributes
    ----------
    lead_time : str or list of str or None
        How lead times of the other forecasts are matched to the reference
        forecast (``"reference"``, ``"intersection"`` or ``"union"``); passed
        on to :func:`mxalign.align_time`.
    """

    lead_time: str | list[str] | None = None


class SpaceAlignmentConfig(_Strict):
    """Interpolator/regridder kwargs, chosen by the kind of space alignment needed.

    Attributes
    ----------
    interpolation : dict or None
        Keyword arguments used when a grid is interpolated to points.
    regrid : dict or None
        Keyword arguments used when a grid is regridded to another grid.
    """

    interpolation: dict | None = None
    regrid: dict | None = None


class SaveConfig(_PassThrough):
    """Output for ``ds.<method>(path, **kwargs)``; extra keys go to the save method.

    Attributes
    ----------
    method : str
        Name of the dataset method used to save (e.g. ``"to_netcdf"``).
    path : str
        Output path, possibly containing placeholders.
    """

    method: str
    path: str


class AlignmentSaveConfig(SaveConfig):
    """Output settings for the aligned datasets.

    Attributes
    ----------
    method : str
        Name of the dataset method used to save (e.g. ``"to_netcdf"``).
    path : str
        Output path pattern; may contain ``{name}``, ``{year}``, ``{month}``
        and ``{day}`` placeholders.
    datasets : {"all", "merge"}
        Save each dataset separately (``"all"``) or concatenated along a new
        ``model`` dimension (``"merge"``).
    """

    datasets: Literal["all", "merge"] = "all"


class AlignmentConfig(_Strict):
    """The ``alignment`` section of the pipeline config.

    Attributes
    ----------
    reference : str
        Name of the dataset that the others are aligned to.
    broadcast_nans : bool
        Whether to broadcast NaNs across all datasets after alignment.
    time : TimeAlignmentConfig or None
        Options for time alignment (always performed).
    space : SpaceAlignmentConfig or None
        Options for space alignment; skipped if None.
    save : AlignmentSaveConfig or None
        Where to save the aligned datasets; not saved if None.
    """

    reference: str
    broadcast_nans: bool = True
    time: TimeAlignmentConfig | None = None
    space: SpaceAlignmentConfig | None = None
    save: AlignmentSaveConfig | None = None


class MetricConfig(_PassThrough):
    """One entry of ``verification.metrics``; extra keys go to the metric function.

    Attributes
    ----------
    function : str
        Dotted import path of the metric function.
    inputs : dict of {str : {"forecast", "reference"}}
        Mapping of function argument names to the dataset role passed to them.
    """

    function: str
    inputs: dict[str, Literal["forecast", "reference"]]


class VerificationConfig(_Strict):
    """The ``verification`` section of the pipeline config.

    Attributes
    ----------
    reference : str
        Name of the dataset used as reference for the metrics.
    metrics : dict of {str : MetricConfig} or None
        Metrics to compute, keyed by metric name.
    save : SaveConfig or None
        Where to save the computed metrics; not saved if None.
    """

    reference: str
    metrics: dict[str, MetricConfig] | None = None
    save: SaveConfig | None = None


class PipelineConfig(_Strict):
    """Top-level pipeline config.

    Attributes
    ----------
    dates : DatesConfig or None
        Default date range used to expand dataset ``files`` patterns.
    datasets : dict of {str : DatasetConfig}
        Datasets to load, keyed by name.
    transformations : dict of {str : TransformationConfig} or None
        Transformations to apply, keyed by transformation name.
    alignment : AlignmentConfig
        Alignment settings.
    verification : VerificationConfig or None
        Verification settings; verification is skipped if None.
    """

    dates: DatesConfig | None = None
    datasets: dict[str, DatasetConfig]
    transformations: dict[str, TransformationConfig] | None = None
    alignment: AlignmentConfig
    verification: VerificationConfig | None = None

    @model_validator(mode="after")
    def _expand_files(self) -> "PipelineConfig":
        """Expand each dataset's ``files`` pattern(s) over its (merged) ``dates``.

        Returns
        -------
        PipelineConfig
            The config itself, with ``files`` of each dataset that has dates
            replaced by the expanded list of paths.

        Raises
        ------
        ValueError
            If the merged ``dates`` of a dataset are non-empty but incomplete.
        """
        top = self.dates.model_dump(exclude_none=True) if self.dates else {}
        for name, ds in self.datasets.items():
            own = ds.dates.model_dump(exclude_none=True) if ds.dates else {}
            dates = {**top, **own}
            if not dates:
                continue
            missing = set(DatesConfig.model_fields) - dates.keys()
            if missing:
                raise ValueError(
                    f"datasets.{name}: dates is missing {', '.join(sorted(missing))}"
                )
            patterns = [ds.files] if isinstance(ds.files, str) else ds.files
            dates = Dates(**dates)
            ds.files = [path for p in patterns for path in dates.substitute(p)]
        return self


def load_config(config: str | dict) -> PipelineConfig:
    """Validate a pipeline config given as a YAML file path or a dict.

    Parameters
    ----------
    config : str or dict
        Path to a YAML config file, or the already-parsed config.

    Returns
    -------
    PipelineConfig
        The validated config.

    Raises
    ------
    pydantic.ValidationError
        If the config does not match the schema.
    """
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)
    return PipelineConfig.model_validate(config)
