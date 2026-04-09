import numpy as np
import pytest
from numpy.testing import assert_allclose

import pytensor
from pytensor import function
from pytensor import tensor as pt
from pytensor.compile import get_default_mode
from pytensor.configdefaults import config
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.tensor._linalg.constructors import BlockDiagonal
from pytensor.tensor._linalg.decomposition.cholesky import Cholesky, cholesky
from pytensor.tensor._linalg.inverse import MatrixInverse, MatrixPinv, inv, pinv
from pytensor.tensor._linalg.inverse import inv as matrix_inverse
from pytensor.tensor._linalg.products import KroneckerProduct
from pytensor.tensor._linalg.solve.general import Solve
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.rewriting.linalg import inv_to_solve
from pytensor.tensor.type import dmatrix, matrix, vector
from tests import unittest_tools as utt
from tests.test_rop import break_op


def test_matrix_inverse_pushforward_pullback():
    rtol = 1e-7 if config.floatX == "float64" else 1e-5
    mx = matrix("mx")
    mv = matrix("mv")
    v = vector("v")
    y = MatrixInverse()(mx).sum(axis=0)

    yv = pytensor.gradient.pushforward(y, mx, mv, use_op_pushforward=True)
    pushforward_f = function([mx, mv], yv)

    yv_via_lop = pytensor.gradient.pushforward(y, mx, mv, use_op_pushforward=False)
    pushforward_via_pullback_f = function([mx, mv], yv_via_lop)

    sy, _ = pytensor.scan(
        lambda i, y, x, v: (pytensor.gradient.grad(y[i], x) * v).sum(),
        sequences=pt.arange(y.shape[0]),
        non_sequences=[y, mx, mv],
    )
    scan_f = function([mx, mv], sy)

    rng = np.random.default_rng(utt.fetch_seed())
    vx = np.asarray(rng.standard_normal((4, 4)), pytensor.config.floatX)
    vv = np.asarray(rng.standard_normal((4, 4)), pytensor.config.floatX)

    v_ref = scan_f(vx, vv)
    np.testing.assert_allclose(pushforward_f(vx, vv), v_ref, rtol=rtol)
    np.testing.assert_allclose(pushforward_via_pullback_f(vx, vv), v_ref, rtol=rtol)

    with pytest.raises(ValueError):
        pytensor.gradient.pushforward(
            clone_replace(y, replace={mx: break_op(mx)}),
            mx,
            mv,
            use_op_pushforward=True,
        )

    vv = np.asarray(rng.uniform(size=(4,)), pytensor.config.floatX)
    yv = pytensor.gradient.pullback(y, mx, v)
    pullback_f = function([mx, v], yv)

    sy = pytensor.gradient.grad((v * y).sum(), mx)
    scan_f = function([mx, v], sy)

    v_ref = scan_f(vx, vv)
    v = pullback_f(vx, vv)
    np.testing.assert_allclose(v, v_ref, rtol=rtol)


def test_transpose_of_inv():
    X = matrix("X")
    Y = matrix_inverse(X)
    Z = Y.transpose()
    f = pytensor.function([X], Z)
    if config.mode != "FAST_COMPILE":
        for node in f.maker.fgraph.toposort():
            if isinstance(node.op, MatrixInverse):
                assert isinstance(node.inputs[0].owner.op, DimShuffle)
            if isinstance(node.op, DimShuffle):
                assert node.inputs[0].name == "X"


def test_inv_to_solve():
    A = dmatrix("A")
    b = dmatrix("b")
    node = matrix_inverse(A).dot(b).owner
    [out] = inv_to_solve.transform(None, node)
    assert isinstance(out.owner.op, Blockwise) and isinstance(
        out.owner.op.core_op, Solve
    )


@pytest.mark.parametrize("inv_op_1", [inv, pinv])
@pytest.mark.parametrize("inv_op_2", [inv, pinv])
def test_inf_of_inv(inv_op_1, inv_op_2):
    x = pt.matrix("x")
    inv_x = inv_op_1(x)
    x_again = inv_op_2(inv_x)
    rewritten_out = rewrite_graph(x_again)
    assert rewritten_out == x


@pytest.mark.parametrize("inv_op", [inv, pinv])
def test_inv_of_diag_from_eye(inv_op):
    x = pt.eye(10)
    x_inv = inv_op(x)
    f_rewritten = function([], x_inv, mode="FAST_RUN")
    nodes = f_rewritten.maker.fgraph.apply_nodes

    # Rewrite Test
    valid_inverses = (MatrixInverse, MatrixPinv)
    assert not any(isinstance(node.op, valid_inverses) for node in nodes)

    # Value Test
    x_test = np.eye(10)
    x_inv_val = np.linalg.inv(x_test)
    rewritten_val = f_rewritten()

    assert_allclose(
        x_inv_val,
        rewritten_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


@pytest.mark.parametrize(
    "shape",
    [(), (7,), (7, 7), (5, 7, 7)],
    ids=["scalar", "vector", "matrix", "batched"],
)
@pytest.mark.parametrize("inv_op", [inv, pinv])
def test_inv_of_diag_from_eye_mul(shape, inv_op):
    # Initializing x based on scalar/vector/matrix
    x = pt.tensor("x", shape=shape)
    x_diag = pt.eye(7) * x
    # Calculating inverse using pt.linalg.inv
    x_inv = inv_op(x_diag)

    # REWRITE TEST
    f_rewritten = function([x], x_inv, mode="FAST_RUN")
    nodes = f_rewritten.maker.fgraph.apply_nodes

    valid_inverses = (MatrixInverse, MatrixPinv)
    assert not any(isinstance(node.op, valid_inverses) for node in nodes)

    # NUMERIC VALUE TEST
    if len(shape) == 0:
        x_test = np.array(np.random.rand()).astype(config.floatX)
    elif len(shape) == 1:
        x_test = np.random.rand(*shape).astype(config.floatX)
    else:
        x_test = np.random.rand(*shape).astype(config.floatX)
    x_test_matrix = np.eye(7) * x_test
    inverse_matrix = np.linalg.inv(x_test_matrix)
    rewritten_inverse = f_rewritten(x_test)

    atol = rtol = 1e-3 if config.floatX == "float32" else 1e-8
    assert_allclose(
        inverse_matrix,
        rewritten_inverse,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("inv_op", [inv, pinv])
def test_inv_of_diag_to_diag_reciprocal(inv_op):
    x = pt.dvector("x")
    x_diag = pt.diag(x)
    x_inv = inv_op(x_diag)

    # REWRITE TEST
    f_rewritten = function([x], x_inv, mode="FAST_RUN")
    nodes = f_rewritten.maker.fgraph.apply_nodes

    valid_inverses = (MatrixInverse, MatrixPinv)
    assert not any(isinstance(node.op, valid_inverses) for node in nodes)

    # NUMERIC VALUE TEST
    x_test = np.random.rand(10)
    x_test_matrix = np.eye(10) * x_test
    inverse_matrix = np.linalg.inv(x_test_matrix)
    rewritten_inverse = f_rewritten(x_test)

    atol = rtol = 1e-3 if config.floatX == "float32" else 1e-8
    assert_allclose(
        inverse_matrix,
        rewritten_inverse,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    "constructor", [pt.dmatrix, pt.tensor3], ids=["not_batched", "batched"]
)
@pytest.mark.parametrize(
    "f_op, f",
    [
        (MatrixInverse, pt.linalg.inv),
        (Cholesky, cholesky),
        (MatrixPinv, pt.linalg.pinv),
    ],
    ids=["inv", "cholesky", "pinv"],
)
@pytest.mark.parametrize(
    "g_op, g",
    [(BlockDiagonal, pt.linalg.block_diag), (KroneckerProduct, pt.linalg.kron)],
    ids=["block_diag", "kron"],
)
def test_lift_linalg_of_expanded_matrices(constructor, f_op, f, g_op, g):
    rng = np.random.default_rng(sum(map(ord, "lift_through_linalg")))

    if pytensor.config.floatX.endswith("32"):
        pytest.skip("Test is flaky at half precision")

    A, B = list(map(constructor, "ab"))
    X = f(g(A, B))

    f1 = pytensor.function(
        [A, B], X, mode=get_default_mode().including("lift_linalg_of_expanded_matrices")
    )

    f2 = pytensor.function(
        [A, B], X, mode=get_default_mode().excluding("lift_linalg_of_expanded_matrices")
    )

    all_apply_nodes = f1.maker.fgraph.apply_nodes
    f_ops = [
        x for x in all_apply_nodes if isinstance(getattr(x.op, "core_op", x.op), f_op)
    ]
    g_ops = [
        x for x in all_apply_nodes if isinstance(getattr(x.op, "core_op", x.op), g_op)
    ]

    assert len(f_ops) == 2
    assert len(g_ops) == 1

    test_vals = [rng.normal(size=(3,) * A.ndim).astype(config.floatX) for _ in range(2)]
    test_vals = [x @ np.swapaxes(x, -1, -2) for x in test_vals]

    np.testing.assert_allclose(f1(*test_vals), f2(*test_vals), atol=1e-8)
