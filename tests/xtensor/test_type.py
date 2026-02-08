import re

import pytest

from pytensor import as_symbolic, shared
from pytensor.compile import SharedVariable


pytest.importorskip("xarray")
pytestmark = pytest.mark.filterwarnings("error")

import numpy as np
from xarray import DataArray

from pytensor.graph.basic import Constant, equal_computations
from pytensor.tensor import as_tensor, specify_shape, tensor
from pytensor.xtensor import xtensor
from pytensor.xtensor.type import (
    XTensorConstant,
    XTensorSharedVariable,
    XTensorType,
    as_xtensor,
    xtensor_constant,
    xtensor_shared,
)


def test_xtensortype():
    x1 = XTensorType(dtype="float64", dims=("a", "b"), shape=(2, 3))
    x2 = XTensorType(dtype="float64", dims=("a", "b"), shape=(2, 3))
    x3 = XTensorType(dtype="float64", dims=("a", "b"), shape=(None, 3))
    y1 = XTensorType(dtype="float64", dims=("c", "d"), shape=(4, 5))
    z1 = XTensorType(dtype="float32", dims=("a", "b"), shape=(2, 3))

    assert x1 == x2 and x1.is_super(x2) and x2.is_super(x1)
    assert x1 != x3 and not x1.is_super(x3) and x3.is_super(x1)
    assert x1 != y1 and not x1.is_super(y1) and not y1.is_super(x1)
    assert x1 != z1 and not x1.is_super(z1) and not z1.is_super(x1)


def test_xtensortype_filter_variable():
    x = xtensor("x", dims=("a", "b"), shape=(2, 3))

    y1 = xtensor("y1", dims=("a", "b"), shape=(2, 3))
    assert x.type.filter_variable(y1) is y1

    y2 = xtensor("y2", dims=("b", "a"), shape=(3, 2))
    expected_y2 = y2.transpose()
    assert equal_computations([x.type.filter_variable(y2)], [expected_y2])

    y3 = xtensor("y3", dims=("b", "a"), shape=(3, None))
    expected_y3 = as_xtensor(
        specify_shape(y3.transpose().values, (2, 3)), dims=("a", "b")
    )
    assert equal_computations([x.type.filter_variable(y3)], [expected_y3])

    # Cases that fail
    with pytest.raises(TypeError):
        y4 = xtensor("y4", dims=("a", "b"), shape=(3, 2))
        x.type.filter_variable(y4)

    with pytest.raises(TypeError):
        y5 = xtensor("y5", dims=("a", "c"), shape=(2, 3))
        x.type.filter_variable(y5)

    with pytest.raises(TypeError):
        y6 = xtensor("y6", dims=("a", "b", "c"), shape=(2, 3, 4))
        x.type.filter_variable(y6)

    with pytest.raises(TypeError):
        y7 = xtensor("y7", dims=("a", "b"), shape=(2, 3), dtype="int32")
        x.type.filter_variable(y7)

    z1 = tensor("z1", shape=(2, None))
    expected_z1 = as_xtensor(specify_shape(z1, (2, 3)), dims=("a", "b"))
    assert equal_computations([x.type.filter_variable(z1)], [expected_z1])

    # Cases that fail
    with pytest.raises(TypeError):
        z2 = tensor("z2", shape=(3, 2))
        x.type.filter_variable(z2)

    with pytest.raises(TypeError):
        z3 = tensor("z3", shape=(1, 2, 3))
        x.type.filter_variable(z3)

    with pytest.raises(TypeError):
        z4 = tensor("z4", shape=(2, 3), dtype="int32")
        x.type.filter_variable(z4)


def test_xtensortype_filter_variable_constant():
    x = xtensor("x", dims=("a", "b"), shape=(2, 3), dtype="float32")

    valid_x = np.zeros((2, 3), dtype="float32")
    res = x.type.filter_variable(valid_x)
    assert isinstance(res, XTensorConstant) and res.type == x.type

    # Upcasting allowed
    valid_x = np.zeros((2, 3), dtype="float16")
    res = x.type.filter_variable(valid_x)
    assert isinstance(res, XTensorConstant) and res.type == x.type

    valid_x = np.zeros((2, 3), dtype="int16")
    res = x.type.filter_variable(valid_x)
    assert isinstance(res, XTensorConstant) and res.type == x.type

    # Downcasting not allowed
    invalid_x = np.zeros((2, 3), dtype="float64")
    with pytest.raises(TypeError):
        x.type.filter_variable(invalid_x)

    invalid_x = np.zeros((2, 3), dtype="int32")
    with pytest.raises(TypeError):
        x.type.filter_variable(invalid_x)

    # non_array types are fine
    valid_x = [[0, 0, 0], [0, 0, 0]]
    res = x.type.filter_variable(valid_x)
    assert isinstance(res, XTensorConstant) and res.type == x.type


@pytest.mark.parametrize(
    "constant_constructor", (as_symbolic, as_xtensor, xtensor_constant)
)
def test_xtensor_constant(constant_constructor):
    x = constant_constructor(DataArray(np.ones((2, 3)), dims=("a", "b")))
    assert isinstance(x, Constant)
    assert isinstance(x, XTensorConstant)
    assert x.type == XTensorType(dtype="float64", dims=("a", "b"), shape=(2, 3))

    if constant_constructor is not as_symbolic:
        # We should be able to pass numpy arrays if we pass dims
        y = as_xtensor(np.ones((2, 3)), dims=("a", "b"))
        assert y.type == x.type
        assert x.signature() == y.signature()
        assert x.equals(y)
        x_eval = x.eval()
        assert isinstance(x.eval(), np.ndarray)
        np.testing.assert_array_equal(x_eval, y.eval(), strict=True)

        z = as_xtensor(np.ones((3, 2)), dims=("b", "a"))
        assert z.type != x.type
        assert z.signature() != x.signature()
        assert not x.equals(z)
        np.testing.assert_array_equal(x_eval, z.eval().T, strict=True)


@pytest.mark.parametrize("shared_constructor", (shared, xtensor_shared))
def test_xtensor_shared(shared_constructor):
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype="int64")
    xarr = DataArray(arr, dims=("a", "b"), name="xarr")
    shared_xarr = shared_constructor(xarr)
    assert isinstance(shared_xarr, SharedVariable)
    assert isinstance(shared_xarr, XTensorSharedVariable)
    assert shared_xarr.type == XTensorType(
        dtype="int64", dims=("a", "b"), shape=(None, None)
    )
    assert xarr.name == "xarr"

    shared_rrax = shared_constructor(xarr, shape=(2, None), name="rrax")
    assert isinstance(shared_rrax, XTensorSharedVariable)
    assert shared_rrax.type == XTensorType(
        dtype="int64", dims=("a", "b"), shape=(2, None)
    )
    assert shared_rrax.name == "rrax"

    if shared_constructor == xtensor_shared:
        # We should be able to pass numpy arrays, if we pass dims
        with pytest.raises(TypeError):
            shared_constructor(arr)
        shared_arr = shared_constructor(arr, dims=("a", "b"))
        assert isinstance(shared_arr, XTensorSharedVariable)
        assert shared_arr.type == shared_xarr.type

    # Test get and set_value
    retrieved_value = shared_xarr.get_value()
    assert isinstance(retrieved_value, np.ndarray)
    np.testing.assert_allclose(retrieved_value, xarr.to_numpy())

    shared_xarr.set_value(xarr[::-1])
    np.testing.assert_allclose(shared_xarr.get_value(), xarr[::-1].to_numpy())

    # Test dims in different order
    shared_xarr.set_value(xarr[::-1].T)
    np.testing.assert_allclose(shared_xarr.get_value(), xarr[::-1].to_numpy())

    with pytest.raises(ValueError):
        shared_xarr.set_value(xarr.rename(b="c"))

    shared_xarr.set_value(arr[::-1])
    np.testing.assert_allclose(shared_xarr.get_value(), arr[::-1])


def test_as_tensor():
    x = xtensor("x", dims=("a", "b"), shape=(2, 3))

    with pytest.raises(
        TypeError,
        match="PyTensor forbids automatic conversion of XTensorVariable to TensorVariable",
    ):
        as_tensor(x)

    x_pt = as_tensor(x, allow_xtensor_conversion=True)
    assert equal_computations([x_pt], [x.values])


def test_minimum_compile():
    from pytensor.compile.mode import Mode

    x = xtensor("x", dims=("a", "b"), shape=(2, 3))
    y = x.transpose()
    minimum_mode = Mode(linker="py", optimizer="minimum_compile")
    result = y.eval({"x": np.ones((2, 3))}, mode=minimum_mode)
    np.testing.assert_array_equal(result, np.ones((3, 2)))


def test_isel_missing_dims():
    x = xtensor("x", dims=("a", "b"), shape=(2, 3))

    # Check valid case works
    assert x.isel(b=0).dims == ("a",)

    with pytest.raises(ValueError):
        x.isel(c=0)

    with pytest.warns(
        UserWarning,
        match=re.escape("Dimension c does not exist. Expected one of ('a', 'b')"),
    ):
        x.isel(c=0, missing_dims="warn")

    x.isel(c=0, missing_dims="ignore").dims == ("a", "b")
