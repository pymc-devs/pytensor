import numpy as np
import pytest

import pytensor.scalar as ps
import pytensor.scalar.basic as psb
import pytensor.tensor as pt
from pytensor import config
from pytensor.scalar.basic import Composite
from pytensor.tensor.elemwise import Elemwise
from tests.link.numba.test_basic import compare_numba_and_py


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "x, y,test_values",
    [
        (
            pt.lvector(),
            pt.dvector(),
            [np.arange(4, dtype="int64"), np.arange(4, dtype="float64")],
        ),
        (
            pt.dmatrix(),
            pt.lscalar(),
            [np.arange(4, dtype="float64").reshape((2, 2)), np.array(4, dtype="int64")],
        ),
    ],
)
def test_Second(x, y, test_values):
    # We use the `Elemwise`-wrapped version of `Second`
    g = pt.second(x, y)
    inputs = [x, y]
    compare_numba_and_py(
        inputs,
        [g],
        test_values,
    )


@pytest.mark.parametrize(
    "v, min, max,test_values",
    [
        (pt.scalar(), 3.0, 7.0, [np.array(10, dtype=config.floatX)]),
        (pt.scalar(), 3.0, 7.0, [np.array(1, dtype=config.floatX)]),
        (pt.scalar(), 7.0, 3.0, [np.array(10, dtype=config.floatX)]),
    ],
)
def test_Clip(v, min, max, test_values):
    g = ps.clip(v, min, max)
    inputs = [v]

    compare_numba_and_py(
        inputs,
        [g],
        test_values,
    )


@pytest.mark.parametrize(
    "inputs, input_values, scalar_fn",
    [
        (
            [pt.scalar("x"), pt.scalar("y"), pt.scalar("z")],
            [
                np.array(10, dtype=config.floatX),
                np.array(20, dtype=config.floatX),
                np.array(30, dtype=config.floatX),
            ],
            lambda x, y, z: ps.add(x, y, z),
        ),
        (
            [pt.scalar("x"), pt.scalar("y"), pt.scalar("z")],
            [
                np.array(10, dtype=config.floatX),
                np.array(20, dtype=config.floatX),
                np.array(30, dtype=config.floatX),
            ],
            lambda x, y, z: ps.mul(x, y, z),
        ),
        (
            [pt.scalar("x"), pt.scalar("y")],
            [
                np.array(10, dtype=config.floatX),
                np.array(20, dtype=config.floatX),
            ],
            lambda x, y: x + y * 2 + ps.exp(x - y),
        ),
    ],
)
def test_Composite(inputs, input_values, scalar_fn):
    composite_inputs = [ps.ScalarType(config.floatX)(name=i.name) for i in inputs]
    comp_op = Elemwise(Composite(composite_inputs, [scalar_fn(*composite_inputs)]))
    compare_numba_and_py(inputs, [comp_op(*inputs)], input_values)


@pytest.mark.parametrize(
    "v, dtype, test_values",
    [
        (pt.fscalar(), psb.float64, [np.array(1.0, dtype="float32")]),
        (pt.dscalar(), psb.float32, [np.array(1.0, dtype="float64")]),
    ],
)
def test_Cast(v, dtype, test_values):
    g = psb.Cast(dtype)(v)
    inputs = [v]
    compare_numba_and_py(
        inputs,
        [g],
        test_values,
    )


@pytest.mark.parametrize(
    "v, dtype,test_values",
    [
        (pt.iscalar(), psb.float64, [np.array(10, dtype="int32")]),
    ],
)
def test_reciprocal(v, dtype, test_values):
    g = psb.reciprocal(v)
    inputs = [v]
    compare_numba_and_py(
        inputs,
        [g],
        test_values,
    )
