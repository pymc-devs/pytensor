import numpy as np
import pytest

import pytensor.scalar as ps
import pytensor.tensor as pt
import pytensor.tensor.basic as ptb
from pytensor import config, function
from pytensor.compile import get_mode
from pytensor.scalar import Add
from tests.link.numba.test_basic import (
    compare_numba_and_py,
    compare_shape_dtype,
)
from tests.tensor.test_basic import check_alloc_runtime_broadcast


pytest.importorskip("numba")


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "v, shape",
    [
        (0.0, (2, 3)),
        (1.1, (2, 3)),
        ((pt.scalar("a"), np.array(10.0, dtype=config.floatX)), (20,)),
        ((pt.vector("a"), np.ones(10, dtype=config.floatX)), (20, 10)),
    ],
)
def test_Alloc(v, shape):
    v, v_test = v if isinstance(v, tuple) else (v, None)
    g = pt.alloc(v, *shape)

    _, (numba_res,) = compare_numba_and_py(
        [v] if v_test is not None else [],
        [g],
        [v_test] if v_test is not None else [],
    )

    assert numba_res.shape == shape


def test_alloc_runtime_broadcast():
    check_alloc_runtime_broadcast(get_mode("NUMBA"))


def test_AllocEmpty():
    x = pt.empty((2, 3), dtype="float32")

    # We cannot compare the values in the arrays, only the shapes and dtypes
    compare_numba_and_py([], x, [], assert_fn=compare_shape_dtype)


def test_TensorFromScalar():
    v, v_test = ps.float64(), np.array(1.0, dtype="float64")
    g = ptb.TensorFromScalar()(v)
    compare_numba_and_py(
        [v],
        g,
        [v_test],
    )


def test_ScalarFromTensor():
    v, v_test = pt.scalar(), np.array(1.0, dtype=config.floatX)
    g = ptb.ScalarFromTensor()(v)
    compare_numba_and_py(
        [v],
        g,
        [v_test],
    )


@pytest.mark.parametrize(
    "vals, dtype",
    [
        (
            (
                (pt.scalar(), np.array(1, dtype=config.floatX)),
                (pt.scalar(), np.array(2, dtype=config.floatX)),
                (pt.scalar(), np.array(3, dtype=config.floatX)),
            ),
            config.floatX,
        ),
        (
            (
                (pt.dscalar(), np.array(1, dtype=np.float64)),
                (pt.lscalar(), np.array(3, dtype=np.int32)),
            ),
            "float64",
        ),
        (
            ((pt.iscalar(), np.array(1, dtype=np.int32)),),
            "float64",
        ),
        (
            ((pt.scalar(dtype=bool), True),),
            bool,
        ),
    ],
)
def test_MakeVector(vals, dtype):
    vals, vals_test = zip(*vals, strict=True)
    g = ptb.MakeVector(dtype)(*vals)

    compare_numba_and_py(
        vals,
        [g],
        vals_test,
    )


def test_ARange():
    start, start_test = pt.lscalar(), np.array(1)
    stop, stop_tset = pt.lscalar(), np.array(10)
    step, step_test = pt.lscalar(), np.array(3)
    dtype = config.floatX

    g = ptb.ARange(dtype)(start, stop, step)

    compare_numba_and_py(
        [start, stop, step],
        g,
        [start_test, stop_tset, step_test],
    )


@pytest.mark.parametrize(
    "vals, axis",
    [
        (
            (
                (pt.matrix(), rng.normal(size=(1, 2)).astype(config.floatX)),
                (pt.matrix(), rng.normal(size=(1, 2)).astype(config.floatX)),
            ),
            0,
        ),
        (
            (
                (pt.matrix(), rng.normal(size=(2, 1)).astype(config.floatX)),
                (pt.matrix(), rng.normal(size=(3, 1)).astype(config.floatX)),
            ),
            0,
        ),
        (
            (
                (pt.matrix(), rng.normal(size=(1, 2)).astype(config.floatX)),
                (pt.matrix(), rng.normal(size=(1, 2)).astype(config.floatX)),
            ),
            1,
        ),
        (
            (
                (pt.matrix(), rng.normal(size=(2, 2)).astype(config.floatX)),
                (pt.matrix(), rng.normal(size=(2, 1)).astype(config.floatX)),
            ),
            1,
        ),
    ],
)
def test_Join(vals, axis):
    vals, vals_test = zip(*vals, strict=True)
    g = pt.join(axis, *vals)

    compare_numba_and_py(
        vals,
        g,
        vals_test,
    )


@pytest.mark.parametrize(
    "n_splits, axis, values, sizes",
    [
        (
            0,
            0,
            (pt.vector(), rng.normal(size=20).astype(config.floatX)),
            (pt.vector(dtype="int64"), []),
        ),
        (
            5,
            0,
            (pt.vector(), rng.normal(size=5).astype(config.floatX)),
            (pt.vector(dtype="int64"), rng.multinomial(5, np.ones(5) / 5)),
        ),
        (
            5,
            0,
            (pt.vector(), rng.normal(size=10).astype(config.floatX)),
            (pt.vector(dtype="int64"), rng.multinomial(10, np.ones(5) / 5)),
        ),
        (
            5,
            -1,
            (pt.matrix(), rng.normal(size=(11, 7)).astype(config.floatX)),
            (pt.vector(dtype="int64"), rng.multinomial(7, np.ones(5) / 5)),
        ),
        (
            5,
            -2,
            (pt.matrix(), rng.normal(size=(11, 7)).astype(config.floatX)),
            (pt.vector(dtype="int64"), rng.multinomial(11, np.ones(5) / 5)),
        ),
    ],
)
def test_Split(n_splits, axis, values, sizes):
    values, values_test = values
    sizes, sizes_test = sizes
    g = pt.split(values, sizes, n_splits, axis=axis)
    assert len(g) == n_splits
    if n_splits == 0:
        return

    compare_numba_and_py(
        [values, sizes],
        g,
        [values_test, sizes_test],
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
    "val, offset",
    [
        (
            (pt.matrix(), np.arange(10 * 10, dtype=config.floatX).reshape((10, 10))),
            0,
        ),
        (
            (pt.matrix(), np.arange(10 * 10, dtype=config.floatX).reshape((10, 10))),
            -1,
        ),
        (
            (pt.vector(), np.arange(10, dtype=config.floatX)),
            0,
        ),
    ],
)
def test_ExtractDiag(val, offset):
    val, val_test = val
    g = pt.diag(val, offset)

    compare_numba_and_py(
        [val],
        g,
        [val_test],
    )


@pytest.mark.parametrize("k", (-5, -1, 0, 1, 4))
@pytest.mark.parametrize("axis1, axis2", ((0, 1), (0, 3), (1, 2), (2, 1), (2, 3)))
def test_ExtractDiag_exhaustive(k, axis1, axis2):
    x = pt.tensor4("x")
    x_shape = (2, 3, 4, 5)
    x_test = np.arange(np.prod(x_shape)).reshape(x_shape)
    out = pt.diagonal(x, k, axis1, axis2)

    compare_numba_and_py(
        [x],
        out,
        [x_test],
    )


@pytest.mark.parametrize(
    "n, m, k, dtype",
    [
        ((pt.lscalar(), np.array(1, dtype=np.int64)), None, 0, None),
        (
            (pt.lscalar(), np.array(1, dtype=np.int64)),
            (pt.lscalar(), np.array(2, dtype=np.int64)),
            0,
            "float32",
        ),
        (
            (pt.lscalar(), np.array(1, dtype=np.int64)),
            (pt.lscalar(), np.array(2, dtype=np.int64)),
            1,
            "int64",
        ),
    ],
)
def test_Eye(n, m, k, dtype):
    n, n_test = n
    m, m_test = m if m is not None else (None, None)
    g = pt.eye(n, m, k, dtype=dtype)

    compare_numba_and_py(
        [n, m] if m is not None else [n],
        g,
        [n_test, m_test] if m is not None else [n_test],
    )


@pytest.mark.parametrize(
    "input_data",
    [np.array([1, 0, 3]), np.array([[0, 1], [2, 0]]), np.array([[0, 0], [0, 0]])],
)
def test_Nonzero(input_data):
    a = pt.tensor("a", shape=(None,) * input_data.ndim)

    graph_outputs = pt.nonzero(a)

    compare_numba_and_py(
        graph_inputs=[a], graph_outputs=graph_outputs, test_inputs=[input_data]
    )
