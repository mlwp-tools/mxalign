"""Align and verify weather forecast and observation datasets with xarray."""

from . import accessor, alignment, pipeline
from .alignment.interpolations.interpolate import interpolate
from .alignment.interpolations.registry import (
    available_interpolations,
    register_interpolator,
)
from .alignment.space import align_space
from .alignment.time import align_time
from .pipeline.transformations.registry import (
    available_transformations,
    register_transformation,
)
from .pipeline.transformations.transform import transform

__all__ = [
    "accessor",
    "align_space",
    "align_time",
    "alignment",
    "available_interpolations",
    "available_transformations",
    "interpolate",
    "pipeline",
    "register_interpolator",
    "register_transformation",
    "transform",
]
