from enum import Enum
from dataclasses import dataclass


class Space(str, Enum):
    GRID = "grid"
    POINT = "point"


class Time(str, Enum):
    FORECAST = "forecast"
    OBSERVATION = "observation"


class Uncertainty(str, Enum):
    DETERMINISTIC = "deterministic"
    ENSEMBLE = "ensemble"
    QUANTILE = "quantile"


@dataclass(frozen=True)
class Properties:
    space: Space
    time: Time
    uncertainty: Uncertainty = Uncertainty.DETERMINISTIC
