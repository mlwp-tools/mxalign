from .specs import SPACE_SPECS, TIME_SPECS, UNCERTAINTY_SPECS


def _validate_dims(ds, variants):
    if not variants:
        return

    ds_dims = set(ds.dims)

    for variant in variants:
        if variant.issubset(ds_dims):
            return

    raise ValueError(f"Dataset dims {ds_dims} do not match allowed variants {variants}")


def _validate_coords(ds, required_coords, axis):
    missing = required_coords - set(ds.coords)
    if missing:
        raise ValueError(f"{axis}: missing required coordinates {missing}")


# TIME
def validate_time_dataset(ds, properties):
    time_spec = TIME_SPECS[properties.time.value]
    _validate_dims(ds, time_spec.dim_variants)
    _validate_coords(ds, time_spec.required_coords, "time")


# SPACE
def validate_space_dataset(ds, properties):
    space_spec = SPACE_SPECS[properties.space.value]
    _validate_dims(ds, space_spec.dim_variants)
    _validate_coords(ds, space_spec.required_coords, "space")
    validate_time_dataset(ds, properties)


# UNCERTAINTY
def validate_uncertainty_dataset(ds, properties):
    uncertainty_spec = UNCERTAINTY_SPECS[properties.uncertainty.value]
    _validate_dims(ds, uncertainty_spec.dim_variants)
    _validate_coords(ds, uncertainty_spec.required_coords, "uncertainty")


def validate_dataset(ds, properties):
    validate_time_dataset(ds, properties)
    validate_space_dataset(ds, properties)
    validate_uncertainty_dataset(ds, properties)
