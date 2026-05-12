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

from mlwp_data_specs.specs.traits.uncertainty import (
    validate_dataset as validate_uncertainty_dataset,
)


def update_space_trait(ds, new_trait: Space):
    validate_space_dataset(ds, trait=new_trait)
    ds.attrs[SPACE_TRAIT_ATTR] = new_trait.value
    return ds


def update_time_trait(ds, new_trait: Time):
    validate_time_dataset(ds, trait=new_trait)
    ds.attrs[TIME_TRAIT_ATTR] = new_trait.value
    return ds


def update_uncertainty_trait(ds, new_trait: Time):
    validate_uncertainty_dataset(ds, trait=new_trait)
    ds.attrs[UNCERTAINTY_TRAIT_ATTR] = new_trait.value
    return ds
