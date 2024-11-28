import numpy as np
import pytest

import pytensor.scalar as ps
import pytensor.tensor as pt
import pytensor.tensor.basic as ptb
from pytensor import config, function
from pytensor.compile import get_mode
from pytensor.graph.basic import Variable
from pytensor.scalar import Add
from pytensor.tensor.shape import Unbroadcast
from tests.link.numba.test_basic import (
    compare_numba_and_py,
    compare_shape_dtype,
)
from tests.tensor.test_basic import TestAlloc


pytest.importorskip("numba")
from pytensor.link.numba.dispatch import numba_funcify


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "v, shape,test_values",
    [
        (0.0, (2, 3), []),
        (1.1, (2, 3), []),
        (pt.scalar("a"), (20,), [np.array(10.0, dtype=config.floatX)]),
        (pt.vector("a"), (20, 10), [np.ones(10, dtype=config.floatX)]),
    ],
)
def test_Alloc(v, shape, test_values):
    g = pt.alloc(v, *shape)
    inputs = [] if not isinstance(v, Variable) else [v]

    _, (numba_res,) = compare_numba_and_py(
        inputs,
        [g],
        test_values,
    )

    assert numba_res.shape == shape


def test_alloc_runtime_broadcast():
    TestAlloc.check_runtime_broadcast(get_mode("NUMBA"))


def test_AllocEmpty():
    x = pt.empty((2, 3), dtype="float32")

    # We cannot compare the values in the arrays, only the shapes and dtypes
    compare_numba_and_py([], [x], [], assert_fn=compare_shape_dtype)


@pytest.mark.parametrize(
    "v,test_values", [(ps.float64(), [np.array(1.0, dtype="float64")])]
)
def test_TensorFromScalar(v, test_values):
    g = ptb.TensorFromScalar()(v)

    compare_numba_and_py(
        [v],
        [g],
        test_values,
    )


@pytest.mark.parametrize(
    "v,test_values",
    [(pt.scalar(), [np.array(1.0, dtype=config.floatX)])],
)
def test_ScalarFromTensor(v, test_values):
    g = ptb.ScalarFromTensor()(v)

    compare_numba_and_py(
        [v],
        [g],
        test_values,
    )


def test_Unbroadcast():
    v = pt.row()
    g = Unbroadcast(0)(v)

    compare_numba_and_py(
        [v],
        [g],
        [np.array([[1.0, 2.0]], dtype=config.floatX)],
    )


@pytest.mark.parametrize(
    "vals, dtype, test_values",
    [
        (
            (
                pt.scalar(),
                pt.scalar(),
                pt.scalar(),
            ),
            config.floatX,
            [
                np.array(1, dtype=config.floatX),
                np.array(2, dtype=config.floatX),
                np.array(3, dtype=config.floatX),
            ],
        ),
        (
            (
                pt.dscalar(),
                pt.lscalar(),
            ),
            "float64",
            [np.array(1, dtype=np.float64), np.array(3, dtype=np.int32)],
        ),
        ((pt.iscalar(),), "float64", [np.array(1, dtype=np.int32)]),
        ((pt.scalar(dtype=bool),), bool, [True]),
    ],
)
def test_MakeVector(vals, dtype, test_values):
    g = ptb.MakeVector(dtype)(*vals)

    compare_numba_and_py(
        vals,
        [g],
        test_values,
    )


@pytest.mark.parametrize(
    "start, stop, step, dtype, test_values",
    [
        (
            pt.lscalar(),
            pt.lscalar(),
            pt.lscalar(),
            config.floatX,
            [np.array(1), np.array(10), np.array(3)],
        ),
    ],
)
def test_ARange(start, stop, step, dtype, test_values):
    g = ptb.ARange(dtype)(start, stop, step)
    inputs = [start, stop, step]

    compare_numba_and_py(
        inputs,
        [g],
        test_values,
    )


@pytest.mark.parametrize(
    "vals, axis,test_values",
    [
        (
            (
                pt.matrix(),
                pt.matrix(),
            ),
            0,
            [
                rng.normal(size=(1, 2)).astype(config.floatX),
                rng.normal(size=(1, 2)).astype(config.floatX),
            ],
        ),
        (
            (
                pt.matrix(),
                pt.matrix(),
            ),
            0,
            [
                rng.normal(size=(2, 1)).astype(config.floatX),
                rng.normal(size=(3, 1)).astype(config.floatX),
            ],
        ),
        (
            (
                pt.matrix(),
                pt.matrix(),
            ),
            1,
            [
                rng.normal(size=(1, 2)).astype(config.floatX),
                rng.normal(size=(1, 2)).astype(config.floatX),
            ],
        ),
        (
            (
                pt.matrix(),
                pt.matrix(),
            ),
            1,
            [
                rng.normal(size=(2, 2)).astype(config.floatX),
                rng.normal(size=(2, 1)).astype(config.floatX),
            ],
        ),
    ],
)
def test_Join(vals, axis, test_values):
    g = pt.join(axis, *vals)

    compare_numba_and_py(
        vals,
        [g],
        test_values,
    )


def test_Join_view():
    vals = (
        pt.matrix(),
        pt.matrix(),
    )
    g = ptb.Join(view=1)(1, *vals)

    with pytest.raises(NotImplementedError):
        compare_numba_and_py(
            vals,
            [g],
            [
                rng.normal(size=(2, 2)).astype(config.floatX),
                rng.normal(size=(2, 2)).astype(config.floatX),
            ],
        )


@pytest.mark.parametrize(
    "n_splits, axis, values, sizes,test_values",
    [
        (
            0,
            0,
            pt.vector(),
            pt.vector(dtype="int64"),
            [rng.normal(size=20).astype(config.floatX), []],
        ),
        (
            5,
            0,
            pt.vector(),
            pt.vector(dtype="int64"),
            [
                rng.normal(size=5).astype(config.floatX),
                rng.multinomial(5, np.ones(5) / 5),
            ],
        ),
        (
            5,
            0,
            pt.vector(),
            pt.vector(dtype="int64"),
            [
                rng.normal(size=10).astype(config.floatX),
                rng.multinomial(10, np.ones(5) / 5),
            ],
        ),
        (
            5,
            -1,
            pt.matrix(),
            pt.vector(dtype="int64"),
            [
                rng.normal(size=(11, 7)).astype(config.floatX),
                rng.multinomial(7, np.ones(5) / 5),
            ],
        ),
        (
            5,
            -2,
            pt.matrix(),
            pt.vector(dtype="int64"),
            [
                rng.normal(size=(11, 7)).astype(config.floatX),
                rng.multinomial(11, np.ones(5) / 5),
            ],
        ),
    ],
)
def test_Split(n_splits, axis, values, sizes, test_values):
    g = pt.split(values, sizes, n_splits, axis=axis)
    assert len(g) == n_splits
    if n_splits == 0:
        return
    inputs = [values, sizes]

    compare_numba_and_py(
        inputs,
        [g] if n_splits == 1 else g,
        test_values,
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
    "val, offset,test_values",
    [
        (pt.matrix(), 0, [np.arange(10 * 10, dtype=config.floatX).reshape((10, 10))]),
        (pt.matrix(), -1, [np.arange(10 * 10, dtype=config.floatX).reshape((10, 10))]),
        (pt.vector(), 0, [np.arange(10, dtype=config.floatX)]),
    ],
)
def test_ExtractDiag(val, offset, test_values):
    g = pt.diag(val, offset)

    compare_numba_and_py(
        [val],
        [g],
        test_values,
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
    "n, m, k, dtype,test_values",
    [
        (pt.lscalar(), None, 0, None, [np.array(1, dtype=np.int64)]),
        (
            pt.lscalar(),
            pt.lscalar(),
            0,
            "float32",
            [np.array(1, dtype=np.int64), np.array(2, dtype=np.int64)],
        ),
        (
            pt.lscalar(),
            pt.lscalar(),
            1,
            "int64",
            [np.array(1, dtype=np.int64), np.array(2, dtype=np.int64)],
        ),
    ],
)
def test_Eye(n, m, k, dtype, test_values):
    g = pt.eye(n, m, k, dtype=dtype)
    inputs = [n, m] if m is not None else [n]

    compare_numba_and_py(
        inputs,
        [g],
        test_values,
    )
