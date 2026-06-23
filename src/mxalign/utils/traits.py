import xarray as xr

from mlwp_data_specs.api import (
    TIME_TRAIT_ATTR,
    SPACE_TRAIT_ATTR,
    UNCERTAINTY_TRAIT_ATTR,
)

from mlwp_data_specs.specs.traits.spatial_coordinate import Space
from mlwp_data_specs.specs.traits.spatial_coordinate import (
    validate_dataset as validate_space_dataset,
)

from mlwp_data_specs.specs.traits.time_coordinate import Time
from mlwp_data_specs.specs.traits.time_coordinate import (
    validate_dataset as validate_time_dataset,
)

from mlwp_data_specs.specs.traits.uncertainty import Uncertainty
from mlwp_data_specs.specs.traits.uncertainty import (
    validate_dataset as validate_uncertainty_dataset,
)


def update_space_trait(ds: xr.Dataset, new_trait: Space) -> xr.Dataset:
    """Validate and set the space trait attribute on ``ds`` in-place."""
    validate_space_dataset(ds, trait=new_trait)
    ds.attrs[SPACE_TRAIT_ATTR] = new_trait.value
    return ds


def update_time_trait(ds: xr.Dataset, new_trait: Time) -> xr.Dataset:
    """Validate and set the time trait attribute on ``ds`` in-place."""
    validate_time_dataset(ds, trait=new_trait)
    ds.attrs[TIME_TRAIT_ATTR] = new_trait.value
    return ds


def update_uncertainty_trait(ds: xr.Dataset, new_trait: Uncertainty) -> xr.Dataset:
    """Validate and set the uncertainty trait attribute on ``ds`` in-place."""
    validate_uncertainty_dataset(ds, trait=new_trait)
    ds.attrs[UNCERTAINTY_TRAIT_ATTR] = new_trait.value
    return ds
