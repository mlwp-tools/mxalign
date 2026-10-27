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
    model_config = ConfigDict(extra="forbid")


class _PassThrough(BaseModel):
    """Model whose unknown keys are kept in ``model_extra`` as function kwargs."""

    model_config = ConfigDict(extra="allow")

    @property
    def kwargs(self) -> dict:
        return dict(self.model_extra or {})


class DatesConfig(_Strict):
    """Date range used to expand ``{reference_time}``/``{lead_time}``/``{valid_time}`` in ``files``.

    Every field is optional so a dataset can override part of the top-level
    ``dates`` section; after merging, all five must be set.
    """

    start: datetime | str | None = None
    end: datetime | str | None = None
    period: str | None = None
    range: str | None = None
    step: str | None = None


class DatasetConfig(_PassThrough):
    """One entry of ``datasets``; extra keys go to ``load_and_validate_dataset``."""

    loader: str
    files: str | list[str]
    variables: str | list[str] | None = None
    grid_mapping: str | dict | None = None
    dates: DatesConfig | None = None


class TransformationConfig(_PassThrough):
    """One entry of ``transformations`` (keyed by transformation name); extra keys go to the transformation."""

    datasets: list[str] | None = None


class TimeAlignmentConfig(_Strict):
    lead_time: str | list[str] | None = None


class SpaceAlignmentConfig(_Strict):
    """Interpolator/regridder kwargs, chosen by the kind of space alignment needed."""

    interpolation: dict | None = None
    regrid: dict | None = None


class SaveConfig(_PassThrough):
    """Output for ``ds.<method>(path, **kwargs)``; extra keys go to the save method."""

    method: str
    path: str


class AlignmentSaveConfig(SaveConfig):
    datasets: Literal["all", "merge"] = "all"


class AlignmentConfig(_Strict):
    reference: str
    broadcast_nans: bool = True
    time: TimeAlignmentConfig | None = None
    space: SpaceAlignmentConfig | None = None
    save: AlignmentSaveConfig | None = None


class MetricConfig(_PassThrough):
    """One entry of ``verification.metrics``; extra keys go to the metric function."""

    function: str
    inputs: dict[str, Literal["forecast", "reference"]]


class VerificationConfig(_Strict):
    reference: str
    metrics: dict[str, MetricConfig] | None = None
    save: SaveConfig | None = None


class PipelineConfig(_Strict):
    dates: DatesConfig | None = None
    datasets: dict[str, DatasetConfig]
    transformations: dict[str, TransformationConfig] | None = None
    alignment: AlignmentConfig
    verification: VerificationConfig | None = None

    @model_validator(mode="after")
    def _expand_files(self) -> "PipelineConfig":
        """Expand each dataset's ``files`` pattern(s) over its (merged) ``dates``."""
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
    """Validate a pipeline config given as a YAML file path or a dict."""
    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)
    return PipelineConfig.model_validate(config)
