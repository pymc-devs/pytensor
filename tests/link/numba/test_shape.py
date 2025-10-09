import contextlib

import numpy as np
import pytest

from pytensor import Variable, config
from pytensor import tensor as pt
from pytensor.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape
from tests.link.numba.test_basic import compare_numba_and_py


@pytest.mark.parametrize(
    "x, i",
    [
        (np.zeros((20, 3)), 1),
    ],
)
def test_Shape(x, i):
    g = Shape()(pt.as_tensor_variable(x))

    compare_numba_and_py([], [g], [])

    g = Shape_i(i)(pt.as_tensor_variable(x))

    compare_numba_and_py([], [g], [])


@pytest.mark.parametrize(
    "v, shape, ndim",
    [
        ((pt.vector(), np.array([4], dtype=config.floatX)), ((), None), 0),
        ((pt.vector(), np.arange(4, dtype=config.floatX)), ((2, 2), None), 2),
        (
            (pt.vector(), np.arange(4, dtype=config.floatX)),
            (pt.lvector(), np.array([2, 2], dtype="int64")),
            2,
        ),
    ],
)
def test_Reshape(v, shape, ndim):
    v, v_test_value = v
    shape, shape_test_value = shape

    g = Reshape(ndim)(v, shape)
    inputs = [v] if not isinstance(shape, Variable) else [v, shape]
    test_values = (
        [v_test_value]
        if not isinstance(shape, Variable)
        else [v_test_value, shape_test_value]
    )
    compare_numba_and_py(
        inputs,
        [g],
        test_values,
    )


def test_Reshape_scalar():
    v = pt.vector()
    v_test_value = np.array([1.0], dtype=config.floatX)
    g = Reshape(1)(v[0], (1,))

    compare_numba_and_py(
        [v],
        g,
        [v_test_value],
    )


@pytest.mark.parametrize(
    "v, shape, fails",
    [
        (
            (pt.matrix(), np.array([[1.0]], dtype=config.floatX)),
            (1, 1),
            False,
        ),
        (
            (pt.matrix(), np.array([[1.0, 2.0]], dtype=config.floatX)),
            (1, 1),
            True,
        ),
        (
            (pt.matrix(), np.array([[1.0, 2.0]], dtype=config.floatX)),
            (1, None),
            False,
        ),
    ],
)
def test_SpecifyShape(v, shape, fails):
    v, v_test_value = v
    g = SpecifyShape()(v, *shape)
    cm = contextlib.suppress() if not fails else pytest.raises(AssertionError)

    with cm:
        compare_numba_and_py(
            [v],
            [g],
            [v_test_value],
        )
