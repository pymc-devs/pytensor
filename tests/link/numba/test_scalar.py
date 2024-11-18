import numpy as np
import pytest

import pytensor.scalar as ps
import pytensor.scalar.basic as psb
import pytensor.tensor as pt
from pytensor import config
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.scalar.basic import Composite
from pytensor.tensor.elemwise import Elemwise
from tests.link.numba.test_basic import compare_numba_and_py, set_test_value


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "inputs",
    [
        (
            [
                set_test_value(pt.lvector(), np.arange(4, dtype="int64")),
                set_test_value(pt.dvector(), np.arange(4, dtype="float64")),
            ]
        ),
        (
            [
                set_test_value(
                    pt.dmatrix(), np.arange(4, dtype="float64").reshape((2, 2))
                ),
                set_test_value(pt.lscalar(), np.array(4, dtype="int64")),
            ]
        ),
    ],
)
def test_Second(inputs):
    # We use the `Elemwise`-wrapped version of `Second`
    test_values = {k: v for d in inputs for k, v in d.items()}
    inputs = list(test_values.keys())
    g = pt.second(*inputs)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            test_values[i]
            for i in test_values
            if not isinstance(i, SharedVariable | Constant)
        ],
    )


@pytest.mark.parametrize(
    "test_values, min, max",
    [
        (set_test_value(pt.scalar(), np.array(10, dtype=config.floatX)), 3.0, 7.0),
        (set_test_value(pt.scalar(), np.array(1, dtype=config.floatX)), 3.0, 7.0),
        (set_test_value(pt.scalar(), np.array(10, dtype=config.floatX)), 7.0, 3.0),
    ],
)
def test_Clip(test_values, min, max):
    v = next(iter(test_values.keys()))
    g = ps.clip(v, min, max)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            test_values[i]
            for i in test_values
            if not isinstance(i, SharedVariable | Constant)
        ],
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
    out_fg = FunctionGraph(inputs, [comp_op(*inputs)])
    compare_numba_and_py(out_fg, input_values)


@pytest.mark.parametrize(
    "test_values, dtype",
    [
        (set_test_value(pt.fscalar(), np.array(1.0, dtype="float32")), psb.float64),
        (set_test_value(pt.dscalar(), np.array(1.0, dtype="float64")), psb.float32),
    ],
)
def test_Cast(test_values, dtype):
    v = next(iter(test_values.keys()))
    g = psb.Cast(dtype)(v)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            test_values[i]
            for i in test_values
            if not isinstance(i, SharedVariable | Constant)
        ],
    )


@pytest.mark.parametrize(
    "test_values, dtype",
    [
        (set_test_value(pt.iscalar(), np.array(10, dtype="int32")), psb.float64),
    ],
)
def test_reciprocal(test_values, dtype):
    v = next(iter(test_values.keys()))
    g = psb.reciprocal(v)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            test_values[i]
            for i in test_values
            if not isinstance(i, SharedVariable | Constant)
        ],
    )
