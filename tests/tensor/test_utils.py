import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.type import matrix
from pytensor.tensor.utils import (
    as_tensor_variables,
    hash_from_ndarray,
    shape_of_variables,
)


@pytest.mark.parametrize("dtype", ["int8", "uint64", "float16", "float32", "complex64"])
@pytest.mark.parametrize("literal", [2, 2.5, 2j])
@pytest.mark.parametrize("input_kind", ["variable", "array", "scalar"])
@config.change_flags(cast_policy="numpy+floatX", floatX="float64")
def test_as_tensor_variables_weak_scalars(dtype, literal, input_kind):
    value = np.asarray(1, dtype=dtype)
    inp = {
        "variable": lambda: pt.scalar(dtype=dtype),
        "array": lambda: value,
        "scalar": lambda: value[()],
    }[input_kind]()
    expected_dtype = np.result_type(value, literal)
    for inputs in ((inp, literal), (literal, inp)):
        tensors = as_tensor_variables(*inputs)
        strong, weak = tensors if inputs[0] is inp else reversed(tensors)
        assert strong.dtype == dtype
        assert weak.dtype == expected_dtype
        assert weak.data == np.asarray(literal, dtype=expected_dtype)
        if input_kind == "variable":
            assert strong is inp


@pytest.mark.parametrize("floatX", ["float32", "float64"])
@config.change_flags(cast_policy="numpy+floatX")
def test_as_tensor_variables_floatX(floatX):
    with config.change_flags(floatX=floatX):
        assert [x.dtype for x in as_tensor_variables(1, 2.5)] == ["int64", floatX]
        assert as_tensor_variables(pt.ivector(), 2.5)[1].dtype == floatX
        assert as_tensor_variables(pt.fvector(), pt.ivector(), 2)[2].dtype == floatX
        assert as_tensor_variables(pt.fvector(), 2.5)[1].dtype == "float32"
        assert as_tensor_variables(pt.fvector(), np.float64(2.5))[1].dtype == "float64"


@config.change_flags(cast_policy="custom", floatX="float64")
def test_as_tensor_variables_custom():
    x, literal = as_tensor_variables(pt.fvector(), 1.1)
    assert x.dtype == "float32"
    assert literal.dtype == "float64"


@config.change_flags(cast_policy="numpy+floatX")
def test_as_tensor_variables_integer_overflow():
    with pytest.raises(OverflowError):
        as_tensor_variables(pt.bvector(), 128)
    with pytest.raises(OverflowError):
        as_tensor_variables(pt.tensor(dtype="uint8", shape=()), -1)


def test_as_tensor_variables_bool():
    with pytest.raises(TypeError, match="Cannot cast True or False"):
        as_tensor_variables(pt.fvector(), True)


def test_hash_from_ndarray():
    hashes = []
    x = np.random.random((5, 5))

    for data in [
        -2,
        -1,
        0,
        1,
        2,
        np.zeros((1, 5)),
        np.zeros((1, 6)),
        # Data buffer empty but different shapes
        np.zeros((1, 0)),
        np.zeros((2, 0)),
        # Same data buffer and shapes but different strides
        np.arange(25).reshape(5, 5),
        np.arange(25).reshape(5, 5).T,
        # Same data buffer, shapes and strides but different dtypes
        np.zeros((5, 5), dtype="uint32"),
        np.zeros((5, 5), dtype="int32"),
        # Test slice
        x,
        x[1:],
        x[:4],
        x[1:3],
        x[::2],
        x[::-1],
    ]:
        data = np.asarray(data)
        hashes.append(hash_from_ndarray(data))

    assert len(set(hashes)) == len(hashes)

    # test that different type of views and their copy give the same hash
    assert hash_from_ndarray(x[1:]) == hash_from_ndarray(x[1:].copy())
    assert hash_from_ndarray(x[1:3]) == hash_from_ndarray(x[1:3].copy())
    assert hash_from_ndarray(x[:4]) == hash_from_ndarray(x[:4].copy())
    assert hash_from_ndarray(x[::2]) == hash_from_ndarray(x[::2].copy())
    assert hash_from_ndarray(x[::-1]) == hash_from_ndarray(x[::-1].copy())


class TestShapeOfVariables:
    def test_simple(self):
        x = matrix("x")
        y = x + x
        fgraph = FunctionGraph([x], [y], clone=False)
        shapes = shape_of_variables(fgraph, {x: (5, 5)})
        assert shapes == {x: (5, 5), y: (5, 5)}

        x = matrix("x")
        y = pt.dot(x, x.T)
        fgraph = FunctionGraph([x], [y], clone=False)
        shapes = shape_of_variables(fgraph, {x: (5, 1)})
        assert shapes[x] == (5, 1)
        assert shapes[y] == (5, 5)

    def test_subtensor(self):
        x = matrix("x")
        subx = x[1:]
        fgraph = FunctionGraph([x], [subx], clone=False)
        shapes = shape_of_variables(fgraph, {x: (10, 10)})
        assert shapes[subx] == (9, 10)

    def test_err(self):
        x = matrix("x")
        subx = x[1:]
        fgraph = FunctionGraph([x], [subx])
        with pytest.raises(ValueError):
            shape_of_variables(fgraph, {x: (10, 10)})
