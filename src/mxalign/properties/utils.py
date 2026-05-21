import json

from .properties import Properties, Space, Time, Uncertainty
from .validation import validate_time_dataset, validate_space_dataset

SPACE_ATTR = "properties.space"
TIME_ATTR = "properties.time"
UNCERTAINTY_ATTR = "properties.uncertainty"


def properties_to_attrs(prop: Properties) -> dict:
    return {
        SPACE_ATTR: prop.space.value,
        TIME_ATTR: prop.time.value,
        UNCERTAINTY_ATTR: prop.uncertainty.value,
    }


def properties_from_attrs(ds) -> Properties:
    attrs = ds.attrs
    old_attrs = attrs.get("properties", {})
    if isinstance(old_attrs, str):
        try:
            old_attrs = json.loads(old_attrs)
        except json.JSONDecodeError:
            old_attrs = {}
    if not isinstance(old_attrs, dict):
        old_attrs = {}

    space = attrs.get(SPACE_ATTR, old_attrs.get("space"))
    time = attrs.get(TIME_ATTR, old_attrs.get("time"))
    uncertainty = attrs.get(UNCERTAINTY_ATTR, old_attrs.get("uncertainty"))

    return Properties(
        space=Space(space),
        time=Time(time),
        uncertainty=Uncertainty(uncertainty or Uncertainty.DETERMINISTIC),
    )


def set_properties_attrs(ds, prop: Properties):
    ds.attrs.update(properties_to_attrs(prop))
    ds.attrs.pop("properties", None)
    return ds


def update_space_property(ds, prop: Space):
    old_props = properties_from_attrs(ds)
    new_props = Properties(
        space=prop,
        time=old_props.time,
        uncertainty=old_props.uncertainty,
    )
    validate_space_dataset(ds, new_props)
    return set_properties_attrs(ds, new_props)


def update_time_property(ds, prop: Time):
    old_props = properties_from_attrs(ds)
    new_props = Properties(
        space=old_props.space,
        time=prop,
        uncertainty=old_props.uncertainty,
    )
    validate_time_dataset(ds, new_props)
    return set_properties_attrs(ds, new_props)
