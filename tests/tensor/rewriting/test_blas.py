import numpy as np
import pytest

from pytensor import config, function
from pytensor import tensor as pt
from pytensor.compile import get_default_mode
from pytensor.graph import FunctionGraph, ancestors
from pytensor.tensor import (
    col,
    dscalar,
    dvector,
    matmul,
    matrix,
    mul,
    neg,
    row,
    scalar,
    sqrt,
    tensor,
    vector,
    vectorize,
)
from pytensor.tensor.blas import BatchedDot
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.rewriting.blas import (
    _as_scalar,
    _factor_canonicalized,
    _gemm_canonicalize,
    _is_real_matrix,
    res_is_a,
    specialize_matmul_to_batched_dot,
)


def XYZab():
    return matrix(), matrix(), matrix(), scalar(), scalar()


@pytest.mark.skipif(
    config.mode == "FAST_COMPILE", reason="Test requires specialization rewrites"
)
@pytest.mark.parametrize("aligned", (True, False))
def test_specialize_matmul_to_batched_dot(aligned):
    signature = BatchedDot.gufunc_signature
    rewrite = specialize_matmul_to_batched_dot.__name__

    def core_pt(x, y):
        return matmul(x, y)

    def core_np(x, y):
        return np.matmul(x, y)

    x = tensor(shape=(7, 5, 3, 3))
    if aligned:
        y = tensor(shape=(7, 5, 3, 3))
    else:
        y = tensor(shape=(5, 3, 3))

    out = vectorize(core_pt, signature=signature)(x, y)

    assert (
        sum(
            isinstance(var.owner.op, BatchedDot)
            for var in ancestors([out])
            if var.owner
        )
        == 0
    )

    vectorize_pt = function(
        [x, y],
        out,
        mode=get_default_mode().including(rewrite),
    )

    assert (
        sum(
            isinstance(var.owner.op, BatchedDot)
            for var in ancestors(vectorize_pt.maker.fgraph.outputs)
            if var.owner
        )
        == 1
    )

    x_test = np.random.normal(size=x.type.shape).astype(x.type.dtype)
    y_test = np.random.normal(size=y.type.shape).astype(y.type.dtype)
    vectorize_np = np.vectorize(core_np, signature=signature)
    np.testing.assert_allclose(
        vectorize_pt(x_test, y_test),
        vectorize_np(x_test, y_test),
    )


def test_gemm_factor():
    X, Y = matrix("X"), matrix("Y")

    assert [(1.0, X), (1.0, Y)] == _factor_canonicalized([(1.0, X), (1.0, Y)])
    assert [(2.0, X)] == _factor_canonicalized([(1.0, X), (1.0, X)])


def test_gemm_canonicalize():
    X, Y, Z, a, b = (
        matrix("X"),
        matrix("Y"),
        matrix("Z"),
        scalar("a"),
        scalar("b"),
    )
    c, d = scalar("c"), scalar("d")
    u = row("u")
    v = vector("v")
    w = col("w")

    can = []
    fg = FunctionGraph([X, Y, Z], [X + Y + Z], clone=False)
    _gemm_canonicalize(fg, fg.outputs[0], 1.0, can, 0)
    assert can == [(1.0, X), (1.0, Y), (1.0, Z)]

    can = []
    fg = FunctionGraph([X, Y, u], [X + Y + u], clone=False)
    _gemm_canonicalize(fg, fg.outputs[0], 1.0, can, 0)
    assert can == [(1.0, X), (1.0, Y), (1.0, u)], can

    can = []
    fg = FunctionGraph([X, Y, v], [X + Y + v], clone=False)
    _gemm_canonicalize(fg, fg.outputs[0], 1.0, can, 0)
    # [(1.0, X), (1.0, Y), (1.0, InplaceDimShuffle{x,0}(v))]
    assert can[:2] == [(1.0, X), (1.0, Y)]
    assert isinstance(can[2], tuple)
    assert len(can[2]) == 2
    assert can[2][0] == 1.0
    assert can[2][1].owner
    assert isinstance(can[2][1].owner.op, DimShuffle)
    assert can[2][1].owner.inputs == [v]

    can = []
    fg = FunctionGraph([X, Y, w], [X + Y + w], clone=False)
    _gemm_canonicalize(fg, fg.outputs[0], 1.0, can, 0)
    assert can == [(1.0, X), (1.0, Y), (1.0, w)], can

    can = []
    fg = FunctionGraph([a, X, Y, b, Z, c], [a * X + Y - b * Z * c], clone=False)
    _gemm_canonicalize(fg, fg.outputs[0], 1.0, can, 0)
    assert can[0] == (a, X)
    assert can[1] == (1.0, Y)
    assert can[2][0].owner.op == mul
    assert can[2][0].owner.inputs[0].owner.op == neg
    assert can[2][0].owner.inputs[0].owner.inputs[0] == c
    assert can[2][0].owner.inputs[1] == b

    can = []
    fg = FunctionGraph(
        [a, X, Y, b, Z, c, d], [(-d) * X - (a * X + Y - b * Z * c)], clone=False
    )
    _gemm_canonicalize(fg, fg.outputs[0], 1.0, can, 0)
    assert can[0][0].owner.op == neg
    assert can[0][0].owner.inputs[0] == d
    assert can[0][1] == X
    assert can[1][0].owner.op == neg
    assert can[1][0].owner.inputs[0] == a
    assert can[2] == (-1.0, Y)
    assert can[3][0].owner.op == mul
    assert can[3][0].owner.inputs == [c, b]


def test_res_is_a():
    _X, _Y, _Z, a, _b = XYZab()

    assert not res_is_a(None, a, sqrt)
    assert not res_is_a(None, a + a, sqrt)
    assert res_is_a(None, sqrt(a + a), sqrt)

    sqrt_term = sqrt(a + a)
    fg = FunctionGraph([a], [2 * sqrt_term], clone=False)
    assert res_is_a(fg, sqrt_term, sqrt, 2)
    assert not res_is_a(fg, sqrt_term, sqrt, 0)


class TestAsScalar:
    def test_basic(self):
        # Test that it works on scalar constants
        a = pt.constant(2.5)
        b = pt.constant(np.asarray([[[0.5]]]))
        b2 = b.dimshuffle()
        assert b2.ndim == 0
        d_a = DimShuffle(input_ndim=0, new_order=[])(a)
        d_b = DimShuffle(input_ndim=3, new_order=[0, 2, 1])(b)
        d_a2 = DimShuffle(input_ndim=0, new_order=["x", "x", "x"])(a)

        assert _as_scalar(a) == a
        assert _as_scalar(b) != b
        assert _as_scalar(d_a) != d_a
        assert _as_scalar(d_b) != d_b
        assert _as_scalar(d_a2) != d_a2

    def test_basic_1(self):
        # Test that it fails on nonscalar constants
        a = pt.constant(np.ones(5))
        assert _as_scalar(a) is None
        assert _as_scalar(DimShuffle(input_ndim=1, new_order=[0, "x"])(a)) is None

    def test_basic_2(self):
        # Test that it works on scalar variables
        a = dscalar()
        d_a = DimShuffle(input_ndim=0, new_order=[])(a)
        d_a2 = DimShuffle(input_ndim=0, new_order=["x", "x"])(a)

        assert _as_scalar(a) is a
        assert _as_scalar(d_a) is a
        assert _as_scalar(d_a2) is a

    def test_basic_3(self):
        # Test that it fails on nonscalar variables
        a = matrix()
        assert _as_scalar(a) is None
        assert _as_scalar(DimShuffle(input_ndim=2, new_order=[0, "x", 1])(a)) is None


class TestRealMatrix:
    def test_basic(self):
        assert _is_real_matrix(DimShuffle(input_ndim=2, new_order=[1, 0])(matrix()))
        assert not _is_real_matrix(
            DimShuffle(input_ndim=1, new_order=["x", 0])(dvector())
        )
