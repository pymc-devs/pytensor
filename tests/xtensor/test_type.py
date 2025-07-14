# ruff: noqa: E402
import pytest


pytest.importorskip("xarray")

import numpy as np
from xarray import DataArray

from pytensor.graph.basic import equal_computations
from pytensor.tensor import as_tensor, tensor
from pytensor.xtensor import xtensor
from pytensor.xtensor.type import XTensorType, as_xtensor, dim


def test_xtensortype():
    a = dim("a", size=2)
    b = dim("b", size=3)
    x1 = XTensorType(dtype="float64", dims=(a.type, b.type))
    x2 = XTensorType(dtype="float64", dims=(a.type, b.type))

    a = dim("a", size=None)
    x3 = XTensorType(dtype="float64", dims=(a.type, b.type))

    c = dim("c", size=4)
    d = dim("d", size=5)
    y1 = XTensorType(dtype="float64", dims=(c.type, d.type))
    z1 = XTensorType(dtype="float32", dims=(a.type, b.type))

    assert x1 == x2 and x1.is_super(x2) and x2.is_super(x1)
    assert x1 != x3 and not x1.is_super(x3) and x3.is_super(x1)
    assert x1 != y1 and not x1.is_super(y1) and not y1.is_super(x1)
    assert x1 != z1 and not x1.is_super(z1) and not z1.is_super(x1)


def test_xtensortype_filter_variable():
    a = dim("a", size=2)
    b = dim("b", size=3)
    x = xtensor("x", dims=(a, b))

    y1 = xtensor("y1", dims=(a, b))
    assert x.type.filter_variable(y1) is y1

    y2 = xtensor("y2", dims=(b, a))
    expected_y2 = y2.transpose()
    assert equal_computations([x.type.filter_variable(y2)], [expected_y2])

    # Cases that fail
    with pytest.raises(TypeError):
        b_ = dim("b", size=None)
        y4 = xtensor("y4", dims=(a, b_))
        x.type.filter_variable(y4)

    with pytest.raises(TypeError):
        c = dim("c", size=3)
        y5 = xtensor("y5", dims=(a, c))
        x.type.filter_variable(y5)

    with pytest.raises(TypeError):
        y6 = xtensor("y6", dims=(a, b, c))
        x.type.filter_variable(y6)

    with pytest.raises(TypeError):
        y7 = xtensor("y7", dims=(a, b), dtype="int32")
        x.type.filter_variable(y7)

    # Cases that fail
    with pytest.raises(TypeError):
        z2 = tensor("z2", shape=(2, 3))
        # Maybe we could allow this one?
        x.type.filter_variable(z2)

    with pytest.raises(TypeError):
        z2 = tensor("z2", shape=(2, None))
        x.type.filter_variable(z2)

    with pytest.raises(TypeError):
        z2 = tensor("z2", shape=(3, 2))
        x.type.filter_variable(z2)

    with pytest.raises(TypeError):
        z3 = tensor("z3", shape=(1, 2, 3))
        x.type.filter_variable(z3)

    with pytest.raises(TypeError):
        z4 = tensor("z4", shape=(2, 3), dtype="int32")
        x.type.filter_variable(z4)


def test_xtensor_constant():
    x = as_xtensor(DataArray(np.ones((2, 3)), dims=("a", "b")))
    assert x.type == XTensorType(dtype="float64", dims=("a", "b"), shape=(2, 3))

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


def test_as_tensor():
    a = dim("a", size=2)
    b = dim("b", size=3)
    x = xtensor("x", dims=(a, b))

    with pytest.raises(
        TypeError,
        match="PyTensor forbids automatic conversion of XTensorVariable to TensorVariable",
    ):
        as_tensor(x)

    x_pt = as_tensor(x, allow_xtensor_conversion=True)
    assert equal_computations([x_pt], [x.values])


def test_minimum_compile():
    from pytensor.compile.mode import Mode

    a = dim("a", size=2)
    b = dim("b", size=3)
    x = xtensor("x", dims=(a, b))
    y = x.transpose()
    minimum_mode = Mode(linker="py", optimizer="minimum_compile")
    result = y.eval({"x": np.ones((2, 3))}, mode=minimum_mode)
    np.testing.assert_array_equal(result, np.ones((3, 2)))
