from dataclasses import dataclass, field
from typing import Callable
from .properties import Space, Time, Uncertainty


@dataclass
class PropertySpec:
    dim_variants: list[set[str]] = field(default_factory=list)
    required_coords: set[str] = field(default_factory=set)
    optional_dims: set[str] = field(default_factory=set)
    optional_coords: set[str] = field(default_factory=set)
    validators: list[Callable] = field(default_factory=list)


SPACE_SPECS = {
    Space.GRID: PropertySpec(
        dim_variants=[
            {"xc", "yc"},
            {"grid_index"},
            {"longitude", "latitude"},
        ],
        required_coords={"longitude", "latitude"},
        optional_coords={"xc", "yc"},
        optional_dims={"member"},
    ),
    Space.POINT: PropertySpec(
        dim_variants=[
            {"point_index"},
        ],
        required_coords={"longitude", "latitude"},
        optional_coords={"code", "elevation", "name", "country"},
    ),
}
TIME_SPECS = {
    Time.FORECAST: PropertySpec(
        dim_variants=[{"reference_time", "lead_time"}],
        required_coords={"reference_time", "lead_time"},
        optional_coords={"valid_time"},
    ),
    Time.OBSERVATION: PropertySpec(
        dim_variants=[{"valid_time"}],
        required_coords={"valid_time"},
    ),
}

UNCERTAINTY_SPECS = {
    Uncertainty.DETERMINISTIC: PropertySpec(),
    Uncertainty.ENSEMBLE: PropertySpec(
        dim_variants=[{"member"}], required_coords={"member"}
    ),
    Uncertainty.QUANTILE: PropertySpec(
        dim_variants=[{"quantile"}], required_coords={"quantile"}
    ),
}
