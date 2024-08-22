import contextlib

import numpy as np
import pytest
import scipy.special

import pytensor
import pytensor.tensor as pt
import pytensor.tensor.inplace as pti
import pytensor.tensor.math as ptm
from pytensor import config, function
from pytensor.compile import get_mode
from pytensor.compile.ops import deep_copy_op
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.gradient import grad
from pytensor.graph.basic import Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import All, Any, Max, Mean, Min, Prod, ProdWithoutZeros, Sum
from pytensor.tensor.special import LogSoftmax, Softmax, SoftmaxGrad
from tests.link.numba.test_basic import (
    compare_numba_and_py,
    scalar_my_multi_out,
    set_test_value,
)
from tests.tensor.test_elemwise import TestElemwise


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "inputs, input_vals, output_fn, exc",
    [
        (
            [pt.vector()],
            [rng.uniform(size=100).astype(config.floatX)],
            lambda x: pt.gammaln(x),
            None,
        ),
        (
            [pt.vector()],
            [rng.standard_normal(100).astype(config.floatX)],
            lambda x: pt.sigmoid(x),
            None,
        ),
        (
            [pt.vector()],
            [rng.standard_normal(100).astype(config.floatX)],
            lambda x: pt.log1mexp(x),
            None,
        ),
        (
            [pt.vector()],
            [rng.standard_normal(100).astype(config.floatX)],
            lambda x: pt.erf(x),
            None,
        ),
        (
            [pt.vector()],
            [rng.standard_normal(100).astype(config.floatX)],
            lambda x: pt.erfc(x),
            None,
        ),
        (
            [pt.vector()],
            [rng.standard_normal(100).astype(config.floatX)],
            lambda x: pt.erfcx(x),
            None,
        ),
        (
            [pt.vector() for i in range(4)],
            [rng.standard_normal(100).astype(config.floatX) for i in range(4)],
            lambda x, y, x1, y1: (x + y) * (x1 + y1) * y,
            None,
        ),
        (
            [pt.matrix(), pt.scalar()],
            [rng.normal(size=(2, 2)).astype(config.floatX), 0.0],
            lambda a, b: pt.switch(a, b, a),
            None,
        ),
        (
            [pt.scalar(), pt.scalar()],
            [
                np.array(1.0, dtype=config.floatX),
                np.array(1.0, dtype=config.floatX),
            ],
            lambda x, y: pti.add_inplace(deep_copy_op(x), deep_copy_op(y)),
            None,
        ),
        (
            [pt.vector(), pt.vector()],
            [
                rng.standard_normal(100).astype(config.floatX),
                rng.standard_normal(100).astype(config.floatX),
            ],
            lambda x, y: pti.add_inplace(deep_copy_op(x), deep_copy_op(y)),
            None,
        ),
        (
            [pt.vector(), pt.vector()],
            [
                rng.standard_normal(100).astype(config.floatX),
                rng.standard_normal(100).astype(config.floatX),
            ],
            lambda x, y: scalar_my_multi_out(x, y),
            None,
        ),
    ],
)
def test_Elemwise(inputs, input_vals, output_fn, exc):
    outputs = output_fn(*inputs)

    out_fg = FunctionGraph(
        outputs=[outputs] if not isinstance(outputs, list) else outputs
    )

    cm = contextlib.suppress() if exc is None else pytest.raises(exc)
    with cm:
        compare_numba_and_py(out_fg, input_vals)


@pytest.mark.xfail(reason="Logic had to be reversed due to surprising segfaults")
def test_elemwise_runtime_broadcast():
    TestElemwise.check_runtime_broadcast(get_mode("NUMBA"))


def test_elemwise_speed(benchmark):
    x = pt.dmatrix("y")
    y = pt.dvector("z")

    out = np.exp(2 * x * y + y)

    rng = np.random.default_rng(42)

    x_val = rng.normal(size=(200, 500))
    y_val = rng.normal(size=500)

    func = function([x, y], out, mode="NUMBA")
    func = func.vm.jit_fn
    (out,) = func(x_val, y_val)
    np.testing.assert_allclose(np.exp(2 * x_val * y_val + y_val), out)

    benchmark(func, x_val, y_val)


@pytest.mark.parametrize(
    "v, new_order",
    [
        # `{'drop': [], 'shuffle': [], 'augment': [0, 1]}`
        (
            set_test_value(
                pt.lscalar(name="a"),
                np.array(1, dtype=np.int64),
            ),
            ("x", "x"),
        ),
        # I.e. `a_pt.T`
        # `{'drop': [], 'shuffle': [1, 0], 'augment': []}`
        (
            set_test_value(
                pt.matrix("a"), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)
            ),
            (1, 0),
        ),
        # `{'drop': [], 'shuffle': [0, 1], 'augment': [2]}`
        (
            set_test_value(
                pt.matrix("a"), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)
            ),
            (1, 0, "x"),
        ),
        # `{'drop': [1], 'shuffle': [2, 0], 'augment': [0, 2, 4]}`
        (
            set_test_value(
                pt.tensor(dtype=config.floatX, shape=(None, 1, None), name="a"),
                np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=config.floatX),
            ),
            ("x", 2, "x", 0, "x"),
        ),
        # I.e. `a_pt.dimshuffle((0,))`
        # `{'drop': [1], 'shuffle': [0], 'augment': []}`
        (
            set_test_value(
                pt.tensor(dtype=config.floatX, shape=(None, 1), name="a"),
                np.array([[1.0], [2.0], [3.0], [4.0]], dtype=config.floatX),
            ),
            (0,),
        ),
        (
            set_test_value(
                pt.tensor(dtype=config.floatX, shape=(None, 1), name="a"),
                np.array([[1.0], [2.0], [3.0], [4.0]], dtype=config.floatX),
            ),
            (0,),
        ),
        (
            set_test_value(
                pt.tensor(dtype=config.floatX, shape=(1, 1, 1), name="a"),
                np.array([[[1.0]]], dtype=config.floatX),
            ),
            (),
        ),
    ],
)
def test_Dimshuffle(v, new_order):
    g = v.dimshuffle(new_order)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, SharedVariable | Constant)
        ],
    )


def test_Dimshuffle_returns_array():
    x = pt.vector("x", shape=(1,))
    y = 2 * x.dimshuffle([])
    func = pytensor.function([x], y, mode="NUMBA")
    out = func(np.zeros(1, dtype=config.floatX))
    assert out.ndim == 0


def test_Dimshuffle_non_contiguous():
    """The numba impl of reshape doesn't work with
    non-contiguous arrays, make sure we work around thpt."""
    x = pt.dvector()
    idx = pt.vector(dtype="int64")
    op = DimShuffle(input_ndim=1, new_order=[])
    out = op(pt.specify_shape(x[idx][::2], (1,)))
    func = pytensor.function([x, idx], out, mode="NUMBA")
    assert func(np.zeros(3), np.array([1])).ndim == 0


@pytest.mark.parametrize(
    "careduce_fn, axis, v",
    [
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            0,
            set_test_value(pt.vector(), np.arange(3, dtype=config.floatX)),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: All(axis)(x),
            0,
            set_test_value(pt.vector(), np.arange(3, dtype=config.floatX)),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Any(axis)(x),
            0,
            set_test_value(pt.vector(), np.arange(3, dtype=config.floatX)),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Mean(axis)(x),
            0,
            set_test_value(pt.vector(), np.arange(3, dtype=config.floatX)),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Mean(axis)(x),
            0,
            set_test_value(
                pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            0,
            set_test_value(
                pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            (0, 1),
            set_test_value(
                pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            (1, 0),
            set_test_value(
                pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            None,
            set_test_value(
                pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            1,
            set_test_value(
                pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Prod(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            0,
            set_test_value(pt.vector(), np.arange(3, dtype=config.floatX)),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: ProdWithoutZeros(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            0,
            set_test_value(pt.vector(), np.arange(3, dtype=config.floatX)),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Prod(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            0,
            set_test_value(
                pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Prod(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            1,
            set_test_value(
                pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Max(axis)(x),
            None,
            set_test_value(
                pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Max(axis)(x),
            None,
            set_test_value(
                pt.lmatrix(), np.arange(3 * 2, dtype=np.int64).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Min(axis)(x),
            None,
            set_test_value(
                pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Min(axis)(x),
            None,
            set_test_value(
                pt.lmatrix(), np.arange(3 * 2, dtype=np.int64).reshape((3, 2))
            ),
        ),
    ],
)
def test_CAReduce(careduce_fn, axis, v):
    g = careduce_fn(v, axis=axis)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, SharedVariable | Constant)
        ],
    )


def test_scalar_Elemwise_Clip():
    a = pt.scalar("a")
    b = pt.scalar("b")

    z = pt.switch(1, a, b)
    c = pt.clip(z, 1, 3)
    c_fg = FunctionGraph(outputs=[c])

    compare_numba_and_py(c_fg, [1, 1])


@pytest.mark.parametrize(
    "dy, sm, axis, exc",
    [
        (
            set_test_value(
                pt.matrix(), np.array([[1, 1, 1], [0, 0, 0]], dtype=config.floatX)
            ),
            set_test_value(pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            None,
            None,
        ),
        (
            set_test_value(
                pt.matrix(), np.array([[1, 1, 1], [0, 0, 0]], dtype=config.floatX)
            ),
            set_test_value(pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            0,
            None,
        ),
        (
            set_test_value(
                pt.matrix(), np.array([[1, 1, 1], [0, 0, 0]], dtype=config.floatX)
            ),
            set_test_value(pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            1,
            None,
        ),
    ],
)
def test_SoftmaxGrad(dy, sm, axis, exc):
    g = SoftmaxGrad(axis=axis)(dy, sm)
    g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


def test_SoftMaxGrad_constant_dy():
    dy = pt.constant(np.zeros((3,), dtype=config.floatX))
    sm = pt.vector(shape=(3,))

    g = SoftmaxGrad(axis=None)(dy, sm)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(g_fg, [np.ones((3,), dtype=config.floatX)])


@pytest.mark.parametrize(
    "x, axis, exc",
    [
        (
            set_test_value(pt.vector(), rng.random(size=(2,)).astype(config.floatX)),
            None,
            None,
        ),
        (
            set_test_value(pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            None,
            None,
        ),
        (
            set_test_value(pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            0,
            None,
        ),
    ],
)
def test_Softmax(x, axis, exc):
    g = Softmax(axis=axis)(x)
    g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "x, axis, exc",
    [
        (
            set_test_value(pt.vector(), rng.random(size=(2,)).astype(config.floatX)),
            None,
            None,
        ),
        (
            set_test_value(pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            0,
            None,
        ),
        (
            set_test_value(pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            1,
            None,
        ),
    ],
)
def test_LogSoftmax(x, axis, exc):
    g = LogSoftmax(axis=axis)(x)
    g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "x, axes, exc",
    [
        (
            set_test_value(pt.dscalar(), np.array(0.0, dtype="float64")),
            [],
            None,
        ),
        (
            set_test_value(pt.dvector(), rng.random(size=(3,)).astype("float64")),
            [0],
            None,
        ),
        (
            set_test_value(pt.dmatrix(), rng.random(size=(3, 2)).astype("float64")),
            [0],
            None,
        ),
        (
            set_test_value(pt.dmatrix(), rng.random(size=(3, 2)).astype("float64")),
            [0, 1],
            None,
        ),
    ],
)
def test_Max(x, axes, exc):
    g = ptm.Max(axes)(x)

    if isinstance(g, list):
        g_fg = FunctionGraph(outputs=g)
    else:
        g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "x, axes, exc",
    [
        (
            set_test_value(pt.dscalar(), np.array(0.0, dtype="float64")),
            [],
            None,
        ),
        (
            set_test_value(pt.dvector(), rng.random(size=(3,)).astype("float64")),
            [0],
            None,
        ),
        (
            set_test_value(pt.dmatrix(), rng.random(size=(3, 2)).astype("float64")),
            [0],
            None,
        ),
        (
            set_test_value(pt.dmatrix(), rng.random(size=(3, 2)).astype("float64")),
            [0, 1],
            None,
        ),
    ],
)
def test_Argmax(x, axes, exc):
    g = ptm.Argmax(axes)(x)

    if isinstance(g, list):
        g_fg = FunctionGraph(outputs=g)
    else:
        g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize("size", [(10, 10), (1000, 1000), (10000, 10000)])
@pytest.mark.parametrize("axis", [0, 1])
def test_logsumexp_benchmark(size, axis, benchmark):
    X = pt.matrix("X")
    X_max = pt.max(X, axis=axis, keepdims=True)
    X_max = pt.switch(pt.isinf(X_max), 0, X_max)
    X_lse = pt.log(pt.sum(pt.exp(X - X_max), axis=axis, keepdims=True)) + X_max

    rng = np.random.default_rng(23920)
    X_val = rng.normal(size=size)

    X_lse_fn = pytensor.function([X], X_lse, mode="NUMBA")

    # JIT compile first
    _ = X_lse_fn(X_val)
    res = benchmark(X_lse_fn, X_val)
    exp_res = scipy.special.logsumexp(X_val, axis=axis, keepdims=True)
    np.testing.assert_array_almost_equal(res, exp_res)


def test_fused_elemwise_benchmark(benchmark):
    rng = np.random.default_rng(123)
    size = 100_000
    x = pytensor.shared(rng.normal(size=size), name="x")
    mu = pytensor.shared(rng.normal(size=size), name="mu")

    logp = -((x - mu) ** 2) / 2
    grad_logp = grad(logp.sum(), x)

    func = pytensor.function([], [logp, grad_logp], mode="NUMBA")
    # JIT compile first
    func()
    benchmark(func)


def test_elemwise_out_type():
    # Create a graph with an elemwise
    # Ravel failes if the elemwise output type is reported incorrectly
    x = pt.matrix()
    y = (2 * x).ravel()

    # Pass in the input as mutable, to trigger the inplace rewrites
    func = pytensor.function([pytensor.In(x, mutable=True)], y, mode="NUMBA")

    # Apply it to a numpy array that is neither C or F contigous
    x_val = np.broadcast_to(np.zeros((3,)), (6, 3))

    assert func(x_val).shape == (18,)
