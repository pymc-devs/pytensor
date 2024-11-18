import numpy as np
import pytest

import pytensor.scalar as ps
import pytensor.tensor as pt
import pytensor.tensor.basic as ptb
from pytensor import config, function
from pytensor.compile import get_mode
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.scalar import Add
from pytensor.tensor.shape import Unbroadcast
from tests.link.numba.test_basic import (
    compare_numba_and_py,
    compare_shape_dtype,
    set_test_value,
)
from tests.tensor.test_basic import TestAlloc


pytest.importorskip("numba")
from pytensor.link.numba.dispatch import numba_funcify


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "v, shape",
    [
        (0.0, (2, 3)),
        (1.1, (2, 3)),
        (set_test_value(pt.scalar("a"), np.array(10.0, dtype=config.floatX)), (20,)),
        (set_test_value(pt.vector("a"), np.ones(10, dtype=config.floatX)), (20, 10)),
    ],
)
def test_Alloc(v, shape):
    test_values = {}
    if isinstance(v, dict):
        test_values = v
        v = next(iter(v.keys()))
    g = pt.alloc(v, *shape)
    g_fg = FunctionGraph(outputs=[g])

    _, (numba_res,) = compare_numba_and_py(
        g_fg,
        [
            test_values[i]
            for i in test_values
            if not isinstance(i, SharedVariable | Constant)
        ],
    )

    assert numba_res.shape == shape


def test_alloc_runtime_broadcast():
    TestAlloc.check_runtime_broadcast(get_mode("NUMBA"))


def test_AllocEmpty():
    x = pt.empty((2, 3), dtype="float32")
    x_fg = FunctionGraph([], [x])

    # We cannot compare the values in the arrays, only the shapes and dtypes
    compare_numba_and_py(x_fg, [], assert_fn=compare_shape_dtype)


@pytest.mark.parametrize(
    "test_values", [set_test_value(ps.float64(), np.array(1.0, dtype="float64"))]
)
def test_TensorFromScalar(test_values):
    v = next(iter(test_values.keys()))
    g = ptb.TensorFromScalar()(v)
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
    "test_values",
    [
        set_test_value(pt.scalar(), np.array(1.0, dtype=config.floatX)),
    ],
)
def test_ScalarFromTensor(test_values):
    v = next(iter(test_values))
    g = ptb.ScalarFromTensor()(v)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            test_values[i]
            for i in test_values
            if not isinstance(i, SharedVariable | Constant)
        ],
    )


def test_Unbroadcast():
    test_values = set_test_value(pt.row(), np.array([[1.0, 2.0]], dtype=config.floatX))
    v = next(iter(test_values))
    g = Unbroadcast(0)(v)
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
    "inputs, dtype",
    [
        (
            (
                set_test_value(pt.scalar(), np.array(1, dtype=config.floatX)),
                set_test_value(pt.scalar(), np.array(2, dtype=config.floatX)),
                set_test_value(pt.scalar(), np.array(3, dtype=config.floatX)),
            ),
            config.floatX,
        ),
        (
            (
                set_test_value(pt.dscalar(), np.array(1, dtype=np.float64)),
                set_test_value(pt.lscalar(), np.array(3, dtype=np.int32)),
            ),
            "float64",
        ),
        (
            (set_test_value(pt.iscalar(), np.array(1, dtype=np.int32)),),
            "float64",
        ),
        (
            (set_test_value(pt.scalar(dtype=bool), True),),
            bool,
        ),
    ],
)
def test_MakeVector(inputs, dtype):
    test_values = {k: v for d in inputs for k, v in d.items()}
    vals = list(test_values.keys())
    g = ptb.MakeVector(dtype)(*vals)
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
    "inputs, dtype",
    [
        (
            [
                set_test_value(pt.lscalar(), np.array(1)),
                set_test_value(pt.lscalar(), np.array(10)),
                set_test_value(pt.lscalar(), np.array(3)),
            ],
            config.floatX,
        ),
    ],
)
def test_ARange(inputs, dtype):
    test_values = {k: v for d in inputs for k, v in d.items()}
    inputs = list(test_values.keys())
    g = ptb.ARange(dtype)(*inputs)
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
    "inputs, axis",
    [
        (
            (
                set_test_value(
                    pt.matrix(), rng.normal(size=(1, 2)).astype(config.floatX)
                ),
                set_test_value(
                    pt.matrix(), rng.normal(size=(1, 2)).astype(config.floatX)
                ),
            ),
            0,
        ),
        (
            (
                set_test_value(
                    pt.matrix(), rng.normal(size=(2, 1)).astype(config.floatX)
                ),
                set_test_value(
                    pt.matrix(), rng.normal(size=(3, 1)).astype(config.floatX)
                ),
            ),
            0,
        ),
        (
            (
                set_test_value(
                    pt.matrix(), rng.normal(size=(1, 2)).astype(config.floatX)
                ),
                set_test_value(
                    pt.matrix(), rng.normal(size=(1, 2)).astype(config.floatX)
                ),
            ),
            1,
        ),
        (
            (
                set_test_value(
                    pt.matrix(), rng.normal(size=(2, 2)).astype(config.floatX)
                ),
                set_test_value(
                    pt.matrix(), rng.normal(size=(2, 1)).astype(config.floatX)
                ),
            ),
            1,
        ),
    ],
)
def test_Join(inputs, axis):
    test_values = {k: v for d in inputs for k, v in d.items()}
    vals = list(test_values.keys())
    g = pt.join(axis, *vals)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            test_values[i]
            for i in test_values
            if not isinstance(i, SharedVariable | Constant)
        ],
    )


def test_Join_view():
    test_values = {
        pt.matrix(): rng.normal(size=(2, 2)).astype(config.floatX),
        pt.matrix(): rng.normal(size=(2, 2)).astype(config.floatX),
    }
    vals = list(test_values.keys())
    g = ptb.Join(view=1)(1, *vals)
    g_fg = FunctionGraph(outputs=[g])

    with pytest.raises(NotImplementedError):
        compare_numba_and_py(
            g_fg,
            [
                test_values[i]
                for i in test_values
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "n_splits, axis, test_values, sizes",
    [
        (
            0,
            0,
            set_test_value(pt.vector(), rng.normal(size=20).astype(config.floatX)),
            set_test_value(pt.vector(dtype="int64"), []),
        ),
        (
            5,
            0,
            set_test_value(pt.vector(), rng.normal(size=5).astype(config.floatX)),
            set_test_value(
                pt.vector(dtype="int64"), rng.multinomial(5, np.ones(5) / 5)
            ),
        ),
        (
            5,
            0,
            set_test_value(pt.vector(), rng.normal(size=10).astype(config.floatX)),
            set_test_value(
                pt.vector(dtype="int64"), rng.multinomial(10, np.ones(5) / 5)
            ),
        ),
        (
            5,
            -1,
            set_test_value(pt.matrix(), rng.normal(size=(11, 7)).astype(config.floatX)),
            set_test_value(
                pt.vector(dtype="int64"), rng.multinomial(7, np.ones(5) / 5)
            ),
        ),
        (
            5,
            -2,
            set_test_value(pt.matrix(), rng.normal(size=(11, 7)).astype(config.floatX)),
            set_test_value(
                pt.vector(dtype="int64"), rng.multinomial(11, np.ones(5) / 5)
            ),
        ),
    ],
)
def test_Split(n_splits, axis, test_values, sizes):
    values = next(iter(test_values.keys()))
    test_values.update(sizes)
    sizes = next(iter(sizes.keys()))
    g = pt.split(values, sizes, n_splits, axis=axis)
    assert len(g) == n_splits
    if n_splits == 0:
        return
    g_fg = FunctionGraph(outputs=[g] if n_splits == 1 else g)

    compare_numba_and_py(
        g_fg,
        [
            test_values[i]
            for i in test_values
            if not isinstance(i, SharedVariable | Constant)
        ],
    )


def test_Split_view():
    # https://github.com/pymc-devs/pytensor/issues/343
    x1 = pt.matrix("x1")
    x2 = pt.matrix("x2", shape=(None, 1))
    v = pt.vector("v", shape=(2,), dtype=int)
    out = pt.split(x1, v, n_splits=2, axis=1)[0] + x2

    fn = function([x1, x2, v], out, mode="NUMBA")
    # Check that the addition of split[0] and x2 is not in place
    add_op = fn.maker.fgraph.outputs[0].owner.op
    assert isinstance(add_op.scalar_op, Add)
    assert not add_op.inplace_pattern

    rng = np.random.default_rng(123)
    test_x1 = rng.normal(size=(2, 2))
    test_x2 = rng.normal(size=(2, 1))
    test_v = np.array([1, 1])

    np.testing.assert_allclose(
        fn(test_x1, test_x2, test_v).copy(),
        fn(test_x1, test_x2, test_v).copy(),
    )


@pytest.mark.parametrize(
    "test_values, offset",
    [
        (
            set_test_value(
                pt.matrix(), np.arange(10 * 10, dtype=config.floatX).reshape((10, 10))
            ),
            0,
        ),
        (
            set_test_value(
                pt.matrix(), np.arange(10 * 10, dtype=config.floatX).reshape((10, 10))
            ),
            -1,
        ),
        (
            set_test_value(pt.vector(), np.arange(10, dtype=config.floatX)),
            0,
        ),
    ],
)
def test_ExtractDiag(test_values, offset):
    val = next(iter(test_values.keys()))
    g = pt.diag(val, offset)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            test_values[i]
            for i in test_values
            if not isinstance(i, SharedVariable | Constant)
        ],
    )


@pytest.mark.parametrize("k", range(-5, 4))
@pytest.mark.parametrize(
    "axis1, axis2", ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
)
@pytest.mark.parametrize("reverse_axis", (False, True))
def test_ExtractDiag_exhaustive(k, axis1, axis2, reverse_axis):
    from pytensor.link.numba.dispatch.basic import numba_njit

    if reverse_axis:
        axis1, axis2 = axis2, axis1

    x = pt.tensor4("x")
    x_shape = (2, 3, 4, 5)
    x_test = np.arange(np.prod(x_shape)).reshape(x_shape)
    out = pt.diagonal(x, k, axis1, axis2)
    numba_fn = numba_funcify(out.owner.op, out.owner)

    @numba_njit(no_cpython_wrapper=False)
    def wrap(x):
        return numba_fn(x)

    np.testing.assert_allclose(wrap(x_test), np.diagonal(x_test, k, axis1, axis2))


@pytest.mark.parametrize(
    "n, m, k, dtype",
    [
        (set_test_value(pt.lscalar(), np.array(1, dtype=np.int64)), None, 0, None),
        (
            set_test_value(pt.lscalar(), np.array(1, dtype=np.int64)),
            set_test_value(pt.lscalar(), np.array(2, dtype=np.int64)),
            0,
            "float32",
        ),
        (
            set_test_value(pt.lscalar(), np.array(1, dtype=np.int64)),
            set_test_value(pt.lscalar(), np.array(2, dtype=np.int64)),
            1,
            "int64",
        ),
    ],
)
def test_Eye(n, m, k, dtype):
    test_values = n
    if m:
        test_values.update(m)
        m = next(iter(m))
    n = next(iter(n))
    g = pt.eye(n, m, k, dtype=dtype)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            test_values[i]
            for i in test_values
            if not isinstance(i, SharedVariable | Constant)
        ],
    )
