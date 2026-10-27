import pytest
import xarray as xr

from mxalign.pipeline.transformations.registry import (
    available_transformations,
    register_transformation,
)
from mxalign.pipeline.transformations.transform import transform


@register_transformation("_test_double")
def _double(ds):
    return ds * 2


def test_registered_transformation_is_available():
    assert "_test_double" in available_transformations()


def test_transform_applies_function_without_mutating_input():
    ds = xr.Dataset({"t2m": ("x", [1.0, 2.0, 3.0])})
    out = transform("_test_double", ds)
    assert out["t2m"].values.tolist() == [2.0, 4.0, 6.0]
    assert ds["t2m"].values.tolist() == [1.0, 2.0, 3.0]


def test_unknown_transformation_raises():
    with pytest.raises(ValueError, match="Unknown transformation"):
        transform("does_not_exist", xr.Dataset())


@pytest.mark.parametrize(
    "container", [lambda ds: {"a": ds}, lambda ds: [ds]], ids=["dict", "list"]
)
def test_transform_returns_same_container_type(container):
    datasets = container(xr.Dataset({"t2m": ("x", [1.0])}))
    out = transform("_test_double", datasets)
    assert type(out) is type(datasets)
