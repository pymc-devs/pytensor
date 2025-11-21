import contextlib

import numpy as np
import pytest
import scipy.special

import pytensor
import pytensor.tensor as pt
import pytensor.tensor.math as ptm
from pytensor import config, function
from pytensor.compile import get_mode
from pytensor.compile.ops import deep_copy_op
from pytensor.gradient import grad
from pytensor.scalar import Composite, float64
from pytensor.scalar import add as scalar_add
from pytensor.tensor import blas, tensor
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.math import All, Any, Max, Min, Prod, ProdWithoutZeros, Sum
from pytensor.tensor.special import LogSoftmax, Softmax, SoftmaxGrad
from tests.link.numba.test_basic import (
    compare_numba_and_py,
    numba_mode,
    scalar_my_multi_out,
)
from tests.tensor.test_elemwise import (
    careduce_benchmark_tester,
    check_elemwise_runtime_broadcast,
    dimshuffle_benchmark,
)


rng = np.random.default_rng(42849)

add_inplace = Elemwise(scalar_add, {0: 0})


@pytest.mark.parametrize(
    "inputs, input_vals, output_fn",
    [
        (
            [pt.vector()],
            [rng.uniform(size=100).astype(config.floatX)],
            lambda x: pt.gammaln(x),
        ),
        (
            [pt.vector()],
            [rng.standard_normal(100).astype(config.floatX)],
            lambda x: pt.sigmoid(x),
        ),
        (
            [pt.vector()],
            [rng.standard_normal(100).astype(config.floatX)],
            lambda x: pt.log1mexp(x),
        ),
        (
            [pt.vector()],
            [rng.standard_normal(100).astype(config.floatX)],
            lambda x: pt.erf(x),
        ),
        (
            [pt.vector()],
            [rng.standard_normal(100).astype(config.floatX)],
            lambda x: pt.erfc(x),
        ),
        (
            [pt.vector()],
            [rng.standard_normal(100).astype(config.floatX)],
            lambda x: pt.erfcx(x),
        ),
        (
            [pt.vector() for i in range(4)],
            [rng.standard_normal(100).astype(config.floatX) for i in range(4)],
            lambda x, y, x1, y1: (x + y) * (x1 + y1) * y,
        ),
        (
            [pt.matrix(), pt.scalar()],
            [rng.normal(size=(2, 2)).astype(config.floatX), 0.0],
            lambda a, b: pt.switch(a, b, a),
        ),
        (
            [pt.scalar(), pt.scalar()],
            [
                np.array(1.0, dtype=config.floatX),
                np.array(1.0, dtype=config.floatX),
            ],
            lambda x, y: add_inplace(deep_copy_op(x), deep_copy_op(y)),
        ),
        (
            [pt.vector(), pt.vector()],
            [
                rng.standard_normal(100).astype(config.floatX),
                rng.standard_normal(100).astype(config.floatX),
            ],
            lambda x, y: add_inplace(deep_copy_op(x), deep_copy_op(y)),
        ),
        (
            [pt.vector(), pt.vector()],
            [
                rng.standard_normal(100).astype(config.floatX),
                rng.standard_normal(100).astype(config.floatX),
            ],
            lambda x, y: scalar_my_multi_out(x, y),
        ),
    ],
    ids=[
        "gammaln",
        "sigmoid",
        "log1mexp",
        "erf",
        "erfc",
        "erfcx",
        "complex_arithmetic",
        "switch",
        "add_inplace_scalar",
        "add_inplace_vector",
        "scalar_multi_out",
    ],
)
def test_Elemwise(inputs, input_vals, output_fn):
    outputs = output_fn(*inputs)
    if not isinstance(outputs, tuple | list):
        outputs = [outputs]

    compare_numba_and_py(
        inputs,
        outputs,
        input_vals,
        inplace=outputs[0].owner.op.destroy_map,
    )


@pytest.mark.xfail(reason="Logic had to be reversed due to surprising segfaults")
def test_elemwise_runtime_broadcast():
    check_elemwise_runtime_broadcast(get_mode("NUMBA"))


@pytest.mark.parametrize(
    "v, new_order",
    [
        # `{'drop': [], 'shuffle': [], 'augment': [0, 1]}`
        (
            (
                pt.lscalar(name="a"),
                np.array(1, dtype=np.int64),
            ),
            ("x", "x"),
        ),
        # I.e. `a_pt.T`
        # `{'drop': [], 'shuffle': [1, 0], 'augment': []}`
        (
            (pt.matrix("a"), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)),
            (1, 0),
        ),
        # `{'drop': [], 'shuffle': [0, 1], 'augment': [2]}`
        (
            (pt.matrix("a"), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)),
            (1, 0, "x"),
        ),
        # `{'drop': [1], 'shuffle': [2, 0], 'augment': [0, 2, 4]}`
        (
            (
                pt.tensor(dtype=config.floatX, shape=(None, 1, None), name="a"),
                np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=config.floatX),
            ),
            ("x", 2, "x", 0, "x"),
        ),
        # I.e. `a_pt.dimshuffle((0,))`
        # `{'drop': [1], 'shuffle': [0], 'augment': []}`
        (
            (
                pt.tensor(dtype=config.floatX, shape=(None, 1), name="a"),
                np.array([[1.0], [2.0], [3.0], [4.0]], dtype=config.floatX),
            ),
            (0,),
        ),
        (
            (
                pt.tensor(dtype=config.floatX, shape=(None, 1), name="a"),
                np.array([[1.0], [2.0], [3.0], [4.0]], dtype=config.floatX),
            ),
            (0,),
        ),
        (
            (
                pt.tensor(dtype=config.floatX, shape=(1, 1, 1), name="a"),
                np.array([[[1.0]]], dtype=config.floatX),
            ),
            (),
        ),
    ],
)
def test_Dimshuffle(v, new_order):
    v, v_test_value = v
    g = v.dimshuffle(new_order)
    compare_numba_and_py(
        [v],
        [g],
        [v_test_value],
    )


def test_Dimshuffle_returns_array():
    x = pt.vector("x", shape=(1,))
    y = 2 * x.dimshuffle([])
    func = pytensor.function([x], y, mode="NUMBA")
    out = func(np.zeros(1, dtype=config.floatX))
    assert out.ndim == 0


def test_Dimshuffle_non_contiguous():
    """The numba impl of reshape doesn't work with
    non-contiguous arrays, make sure we work around that."""
    x = pt.dvector()
    idx = pt.vector(dtype="int64")
    op = DimShuffle(input_ndim=1, new_order=[])
    out = op(pt.specify_shape(x[idx][::2], (1,)))
    func = pytensor.function([x, idx], out, mode="NUMBA")
    assert func(np.zeros(3), np.array([1])).ndim == 0


def test_Dimshuffle_squeeze_errors():
    x = pt.tensor3("x", shape=(4, None, 5))
    out = pt.squeeze(x, axis=1)
    assert out.type.shape == (4, 5)
    fn = function([x], out, mode=numba_mode)
    with pytest.raises(
        ValueError, match="Attempting to squeeze axes with size not equal to one"
    ):
        fn(np.zeros((4, 2, 5)))


@pytest.mark.parametrize(
    "careduce_fn, axis, v",
    [
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            0,
            (pt.vector(), np.arange(3, dtype=config.floatX)),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: All(axis)(x),
            0,
            (pt.vector(dtype="bool"), np.array([False, True, False])),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Any(axis)(x),
            0,
            (pt.vector(dtype="bool"), np.array([False, True, False])),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            0,
            (pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            (0, 1),
            (pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            (1, 0),
            (pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            None,
            (pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Sum(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            1,
            (pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Prod(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            (),  # Empty axes would normally be rewritten away, but we want to test it still works
            (pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Prod(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            None,
            (
                pt.scalar(),
                np.array(99.0, dtype=config.floatX),
            ),  # Scalar input would normally be rewritten away, but we want to test it still works
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Prod(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            0,
            (pt.vector(), np.arange(3, dtype=config.floatX)),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: ProdWithoutZeros(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            0,
            (pt.vector(), np.arange(3, dtype=config.floatX)),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Prod(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            0,
            (pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Prod(
                axis=axis, dtype=dtype, acc_dtype=acc_dtype
            )(x),
            1,
            (pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Max(axis)(x),
            None,
            (pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Max(axis)(x),
            None,
            (pt.lmatrix(), np.arange(3 * 2, dtype=np.int64).reshape((3, 2))),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Min(axis)(x),
            None,
            (pt.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))),
        ),
        (
            lambda x, axis=None, dtype=None, acc_dtype=None: Min(axis)(x),
            None,
            (pt.lmatrix(), np.arange(3 * 2, dtype=np.int64).reshape((3, 2))),
        ),
    ],
)
def test_CAReduce(careduce_fn, axis, v):
    v, v_test_value = v
    g = careduce_fn(v, axis=axis)

    fn, _ = compare_numba_and_py(
        [v],
        [g],
        [v_test_value],
    )
    # Confirm CAReduce is in the compiled function
    # fn.dprint()
    [node] = fn.maker.fgraph.apply_nodes
    assert isinstance(node.op, CAReduce)


def test_scalar_Elemwise_Clip():
    a = pt.scalar("a")
    b = pt.scalar("b")
    inputs = [a, b]

    z = pt.switch(1, a, b)
    c = pt.clip(z, 1, 3)

    compare_numba_and_py(inputs, [c], [1, 1])


@pytest.mark.parametrize(
    "dy, sm, axis, exc",
    [
        (
            (pt.matrix(), np.array([[1, 1, 1], [0, 0, 0]], dtype=config.floatX)),
            (pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            None,
            None,
        ),
        (
            (pt.matrix(), np.array([[1, 1, 1], [0, 0, 0]], dtype=config.floatX)),
            (pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            0,
            None,
        ),
        (
            (pt.matrix(), np.array([[1, 1, 1], [0, 0, 0]], dtype=config.floatX)),
            (pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            1,
            None,
        ),
    ],
)
def test_SoftmaxGrad(dy, sm, axis, exc):
    dy, dy_test_value = dy
    sm, sm_test_value = sm
    g = SoftmaxGrad(axis=axis)(dy, sm)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            [dy, sm],
            [g],
            [dy_test_value, sm_test_value],
        )


def test_SoftMaxGrad_constant_dy():
    dy = pt.constant(np.zeros((3,), dtype=config.floatX))
    sm = pt.vector(shape=(3,))
    inputs = [sm]

    g = SoftmaxGrad(axis=None)(dy, sm)

    compare_numba_and_py(inputs, [g], [np.ones((3,), dtype=config.floatX)])


@pytest.mark.parametrize(
    "x, axis, exc",
    [
        (
            (pt.vector(), rng.random(size=(2,)).astype(config.floatX)),
            None,
            None,
        ),
        (
            (pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            None,
            None,
        ),
        (
            (pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            0,
            None,
        ),
    ],
)
def test_Softmax(x, axis, exc):
    x, x_test_value = x
    g = Softmax(axis=axis)(x)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            [x],
            [g],
            [x_test_value],
        )


@pytest.mark.parametrize(
    "x, axis, exc",
    [
        (
            (pt.vector(), rng.random(size=(2,)).astype(config.floatX)),
            None,
            None,
        ),
        (
            (pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            0,
            None,
        ),
        (
            (pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
            1,
            None,
        ),
    ],
)
def test_LogSoftmax(x, axis, exc):
    x, x_test_value = x
    g = LogSoftmax(axis=axis)(x)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            [x],
            [g],
            [x_test_value],
        )


@pytest.mark.parametrize(
    "x, axes, exc",
    [
        (
            (pt.dscalar(), np.array(0.0, dtype="float64")),
            [],
            None,
        ),
        (
            (pt.dvector(), rng.random(size=(3,)).astype("float64")),
            [0],
            None,
        ),
        (
            (pt.dmatrix(), rng.random(size=(3, 2)).astype("float64")),
            [0],
            None,
        ),
        (
            (pt.dmatrix(), rng.random(size=(3, 2)).astype("float64")),
            [0, 1],
            None,
        ),
    ],
)
def test_Max(x, axes, exc):
    x, x_test_value = x
    g = ptm.Max(axes)(x)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            [x],
            [g],
            [x_test_value],
        )


@pytest.mark.parametrize(
    "x, axes, exc",
    [
        (
            (pt.dscalar(), np.array(0.0, dtype="float64")),
            [],
            None,
        ),
        (
            (pt.dvector(), rng.random(size=(3,)).astype("float64")),
            [0],
            None,
        ),
        (
            (pt.dmatrix(), rng.random(size=(3, 2)).astype("float64")),
            [0],
            None,
        ),
        (
            (pt.dmatrix(), rng.random(size=(3, 2)).astype("float64")),
            [0, 1],
            None,
        ),
        (
            (pt.dmatrix(), rng.random(size=(3, 2)).astype("float64")),
            None,
            None,
        ),
    ],
)
def test_Argmax(x, axes, exc):
    x, x_test_value = x
    g = ptm.Argmax(axes)(x)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            [x],
            [g],
            [x_test_value],
        )


def test_elemwise_inplace_out_type():
    # Create a graph with an elemwise
    # Ravel failes if the elemwise output type is reported incorrectly
    x = pt.matrix()
    y = (2 * x).ravel()

    # Pass in the input as mutable, to trigger the inplace rewrites
    func = pytensor.function([pytensor.In(x, mutable=True)], y, mode="NUMBA")

    # Apply it to a numpy array that is neither C or F contigous
    x_val = np.broadcast_to(np.zeros((3,)), (6, 3))

    assert func(x_val).shape == (18,)


def test_elemwise_multiple_inplace_outs():
    x = pt.vector()
    y = pt.vector()

    x_ = pt.scalar_from_tensor(x[0])
    y_ = pt.scalar_from_tensor(y[0])
    out_ = x_ + 1, y_ + 1

    composite_op = Composite([x_, y_], out_)
    elemwise_op = Elemwise(composite_op, inplace_pattern={0: 0, 1: 1})
    out = elemwise_op(x, y)

    fn = function([x, y], out, mode="NUMBA", accept_inplace=True)
    x_test = np.array([1, 2, 3], dtype=config.floatX)
    y_test = np.array([4, 5, 6], dtype=config.floatX)
    out1, out2 = fn(x_test, y_test)
    assert out1 is x_test
    assert out2 is y_test
    np.testing.assert_allclose(out1, [2, 3, 4])
    np.testing.assert_allclose(out2, [5, 6, 7])


def test_scalar_loop():
    a = float64("a")
    scalar_loop = pytensor.scalar.ScalarLoop([a], [a + a])

    x = pt.tensor("x", shape=(3,))
    elemwise_loop = Elemwise(scalar_loop)(3, x)

    with pytest.warns(UserWarning, match="object mode"):
        compare_numba_and_py(
            [x],
            [elemwise_loop],
            (np.array([1, 2, 3], dtype="float64"),),
        )


class TestsBenchmark:
    def test_elemwise_speed(self, benchmark):
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

    def test_fused_elemwise_benchmark(self, benchmark):
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

    @pytest.mark.parametrize("size", [(10, 10), (1000, 1000), (10000, 10000)])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_logsumexp_benchmark(self, size, axis, benchmark):
        X = pt.matrix("X")
        X_max = pt.max(X, axis=axis, keepdims=True)
        X_max = pt.switch(pt.isinf(X_max), 0, X_max)
        X_lse = pt.log(pt.sum(pt.exp(X - X_max), axis=axis, keepdims=True)) + X_max

        rng = np.random.default_rng(23920)
        X_val = rng.normal(size=size)

        X_lse_fn = pytensor.function([X], X_lse, mode="NUMBA")

        # JIT compile first
        res = X_lse_fn(X_val)
        exp_res = scipy.special.logsumexp(X_val, axis=axis, keepdims=True)
        np.testing.assert_array_almost_equal(res, exp_res)
        benchmark(X_lse_fn, X_val)

    @pytest.mark.parametrize(
        "axis",
        (0, 1, 2, (0, 1), (0, 2), (1, 2), None),
        ids=lambda x: f"axis={x}",
    )
    @pytest.mark.parametrize(
        "c_contiguous",
        (True, False),
        ids=lambda x: f"c_contiguous={x}",
    )
    def test_numba_careduce_benchmark(self, axis, c_contiguous, benchmark):
        return careduce_benchmark_tester(
            axis, c_contiguous, mode="NUMBA", benchmark=benchmark
        )

    @pytest.mark.parametrize("c_contiguous", (True, False))
    def test_dimshuffle(self, c_contiguous, benchmark):
        dimshuffle_benchmark("NUMBA", c_contiguous, benchmark)


@pytest.mark.parametrize(
    "x, y",
    [
        (
            (pt.matrix(), rng.random(size=(3, 2)).astype(config.floatX)),
            (pt.vector(), rng.random(size=(2,)).astype(config.floatX)),
        ),
        (
            (pt.matrix(dtype="float64"), rng.random(size=(3, 2)).astype("float64")),
            (pt.vector(dtype="float32"), rng.random(size=(2,)).astype("float32")),
        ),
        (
            (pt.lmatrix(), rng.poisson(size=(3, 2))),
            (pt.fvector(), rng.random(size=(2,)).astype("float32")),
        ),
        (
            (pt.lvector(), rng.random(size=(2,)).astype(np.int64)),
            (pt.lvector(), rng.random(size=(2,)).astype(np.int64)),
        ),
        (
            (pt.vector(dtype="int16"), rng.random(size=(2,)).astype(np.int16)),
            (pt.vector(dtype="uint8"), rng.random(size=(2,)).astype(np.uint8)),
        ),
        # Viewing the array with 2 last dimensions as complex128 means
        # the first entry will be real part and the second entry the imaginary part
        (
            (
                pt.matrix(dtype="complex128"),
                rng.random(size=(5, 4, 2)).view("complex128").squeeze(-1),
            ),
            (
                pt.matrix(dtype="complex128"),
                rng.random(size=(4, 3, 2)).view("complex128").squeeze(-1),
            ),
        ),
        (
            (pt.matrix(dtype="int64"), rng.random(size=(5, 4)).astype("int64")),
            (
                pt.matrix(dtype="complex128"),
                rng.random(size=(4, 3, 2)).view("complex128").squeeze(-1),
            ),
        ),
    ],
)
def test_Dot(x, y):
    x, x_test_value = x
    y, y_test_value = y

    g = ptm.dot(x, y)

    compare_numba_and_py(
        [x, y],
        [g],
        [x_test_value, y_test_value],
    )


@pytest.mark.parametrize(
    "x, y, exc",
    [
        (
            (
                pt.dtensor3(),
                rng.random(size=(2, 3, 3)).astype("float64"),
            ),
            (
                pt.dtensor3(),
                rng.random(size=(2, 3, 3)).astype("float64"),
            ),
            None,
        ),
        (
            (
                pt.dtensor3(),
                rng.random(size=(2, 3, 3)).astype("float64"),
            ),
            (
                pt.ltensor3(),
                rng.poisson(size=(2, 3, 3)).astype("int64"),
            ),
            None,
        ),
    ],
)
def test_BatchedDot(x, y, exc):
    x, x_test_value = x
    y, y_test_value = y

    g = blas.BatchedDot()(x, y)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            [x, y],
            g,
            [x_test_value, y_test_value],
        )


@pytest.mark.parametrize("dtype", ("float64", "float32", "mixed"))
def test_mat_vec_dot_performance(dtype, benchmark):
    A = tensor("A", shape=(512, 512), dtype="float64" if dtype == "mixed" else dtype)
    x = tensor("x", shape=(512,), dtype="float32" if dtype == "mixed" else dtype)
    out = ptm.dot(A, x)

    fn = function([A, x], out, mode="NUMBA", trust_input=True)

    rng = np.random.default_rng(948)
    A_test = rng.standard_normal(size=A.type.shape, dtype=A.type.dtype)
    x_test = rng.standard_normal(size=x.type.shape, dtype=x.type.dtype)
    np.testing.assert_allclose(fn(A_test, x_test), np.dot(A_test, x_test), atol=1e-4)
    benchmark(fn, A_test, x_test)
