from functools import partial

import numpy as np
import pytest
import scipy.linalg
from numpy.testing import assert_allclose

import pytensor
from pytensor import function
from pytensor import tensor as pt
from pytensor.compile import get_default_mode
from pytensor.configdefaults import config
from pytensor.tensor import swapaxes
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import _allclose, dot, matmul
from pytensor.tensor.nlinalg import Det, MatrixInverse, matrix_inverse
from pytensor.tensor.rewriting.linalg import inv_as_solve
from pytensor.tensor.slinalg import (
    Cholesky,
    Solve,
    SolveBase,
    SolveTriangular,
    cho_solve,
    cholesky,
    solve,
    solve_triangular,
)
from pytensor.tensor.type import dmatrix, matrix, tensor, vector
from tests import unittest_tools as utt
from tests.test_rop import break_op


def test_rop_lop():
    mx = matrix("mx")
    mv = matrix("mv")
    v = vector("v")
    y = MatrixInverse()(mx).sum(axis=0)

    yv = pytensor.gradient.Rop(y, mx, mv)
    rop_f = function([mx, mv], yv)

    sy, _ = pytensor.scan(
        lambda i, y, x, v: (pytensor.gradient.grad(y[i], x) * v).sum(),
        sequences=pt.arange(y.shape[0]),
        non_sequences=[y, mx, mv],
    )
    scan_f = function([mx, mv], sy)

    rng = np.random.default_rng(utt.fetch_seed())
    vx = np.asarray(rng.standard_normal((4, 4)), pytensor.config.floatX)
    vv = np.asarray(rng.standard_normal((4, 4)), pytensor.config.floatX)

    v1 = rop_f(vx, vv)
    v2 = scan_f(vx, vv)

    assert _allclose(v1, v2), f"ROP mismatch: {v1} {v2}"

    raised = False
    try:
        pytensor.gradient.Rop(
            pytensor.clone_replace(y, replace={mx: break_op(mx)}), mx, mv
        )
    except ValueError:
        raised = True
    if not raised:
        raise Exception(
            "Op did not raised an error even though the function"
            " is not differentiable"
        )

    vv = np.asarray(rng.uniform(size=(4,)), pytensor.config.floatX)
    yv = pytensor.gradient.Lop(y, mx, v)
    lop_f = function([mx, v], yv)

    sy = pytensor.gradient.grad((v * y).sum(), mx)
    scan_f = function([mx, v], sy)

    v1 = lop_f(vx, vv)
    v2 = scan_f(vx, vv)
    assert _allclose(v1, v2), f"LOP mismatch: {v1} {v2}"


def test_transinv_to_invtrans():
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


def test_generic_solve_to_solve_triangular():
    A = matrix("A")
    x = matrix("x")

    L = cholesky(A, lower=True)
    U = cholesky(A, lower=False)
    b1 = solve(L, x)
    b2 = solve(U, x)
    f = pytensor.function([A, x], b1)

    rng = np.random.default_rng(97)
    X = rng.normal(size=(10, 10)).astype(config.floatX)
    X = X @ X.T
    X_chol = np.linalg.cholesky(X)
    eye = np.eye(10, dtype=config.floatX)

    if config.mode != "FAST_COMPILE":
        toposort = f.maker.fgraph.toposort()
        op_list = [node.op for node in toposort]

        assert not any(isinstance(op, Solve) for op in op_list)
        assert any(isinstance(op, SolveTriangular) for op in op_list)

        assert_allclose(
            f(X, eye) @ X_chol, eye, atol=1e-8 if config.floatX.endswith("64") else 1e-4
        )

    f = pytensor.function([A, x], b2)

    if config.mode != "FAST_COMPILE":
        toposort = f.maker.fgraph.toposort()
        op_list = [node.op for node in toposort]
        assert not any(isinstance(op, Solve) for op in op_list)
        assert any(isinstance(op, SolveTriangular) for op in op_list)
        assert_allclose(
            f(X, eye).T @ X_chol,
            eye,
            atol=1e-8 if config.floatX.endswith("64") else 1e-4,
        )


def test_matrix_inverse_solve():
    A = dmatrix("A")
    b = dmatrix("b")
    node = matrix_inverse(A).dot(b).owner
    [out] = inv_as_solve.transform(None, node)
    assert isinstance(out.owner.op, Blockwise) and isinstance(
        out.owner.op.core_op, Solve
    )


@pytest.mark.parametrize("tag", ("lower", "upper", None))
@pytest.mark.parametrize("cholesky_form", ("lower", "upper"))
@pytest.mark.parametrize("product", ("lower", "upper", None))
@pytest.mark.parametrize("op", (dot, matmul))
def test_cholesky_ldotlt(tag, cholesky_form, product, op):
    transform_removes_chol = tag is not None and product == tag
    transform_transposes = transform_removes_chol and cholesky_form != tag

    ndim = 2 if op == dot else 3
    A = tensor("L", shape=(None,) * ndim)
    if tag:
        setattr(A.tag, tag + "_triangular", True)

    if product == "lower":
        M = op(A, swapaxes(A, -1, -2))
    elif product == "upper":
        M = op(swapaxes(A, -1, -2), A)
    else:
        M = A

    C = cholesky(M, lower=(cholesky_form == "lower"))
    f = pytensor.function([A], C, mode=get_default_mode().including("cholesky_ldotlt"))

    no_cholesky_in_graph = not any(
        isinstance(node.op, Cholesky)
        or (isinstance(node.op, Blockwise) and isinstance(node.op.core_op, Cholesky))
        for node in f.maker.fgraph.apply_nodes
    )

    assert no_cholesky_in_graph == transform_removes_chol

    if transform_transposes:
        expected_order = (1, 0) if ndim == 2 else (0, 2, 1)
        assert any(
            isinstance(node.op, DimShuffle) and node.op.new_order == expected_order
            for node in f.maker.fgraph.apply_nodes
        )

    # Test some concrete value through f
    # there must be lower triangular (f assumes they are)
    Avs = [
        np.eye(1, dtype=pytensor.config.floatX),
        np.eye(10, dtype=pytensor.config.floatX),
        np.array([[2, 0], [1, 4]], dtype=pytensor.config.floatX),
    ]
    if not tag:
        # these must be positive def
        Avs.extend(
            [
                np.ones((4, 4), dtype=pytensor.config.floatX)
                + np.eye(4, dtype=pytensor.config.floatX),
            ]
        )

    cholesky_vect_fn = np.vectorize(
        partial(scipy.linalg.cholesky, lower=(cholesky_form == "lower")),
        signature="(a, a)->(a, a)",
    )

    for Av in Avs:
        if tag == "upper":
            Av = Av.T

        if product == "lower":
            Mv = Av.dot(Av.T)
        elif product == "upper":
            Mv = Av.T.dot(Av)
        else:
            Mv = Av

        if ndim == 3:
            Av = np.broadcast_to(Av, (5, *Av.shape))
            Mv = np.broadcast_to(Mv, (5, *Mv.shape))

        np.testing.assert_allclose(
            cholesky_vect_fn(Mv),
            f(Av),
        )


def test_local_det_chol():
    X = matrix("X")
    L = pt.linalg.cholesky(X)
    det_X = pt.linalg.det(X)

    f = function([X], [L, det_X])

    nodes = f.maker.fgraph.toposort()
    assert not any(isinstance(node, Det) for node in nodes)

    # This previously raised an error (issue #392)
    f = function([X], [L, det_X, X])
    nodes = f.maker.fgraph.toposort()
    assert not any(isinstance(node, Det) for node in nodes)


def test_psd_solve_with_chol():
    X = matrix("X")
    X.tag.psd = True
    X_inv = pt.linalg.solve(X, pt.identity_like(X))

    f = function([X], X_inv, mode="FAST_RUN")

    nodes = f.maker.fgraph.apply_nodes

    assert not any(isinstance(node.op, Solve) for node in nodes)
    assert any(isinstance(node.op, Cholesky) for node in nodes)
    assert any(isinstance(node.op, SolveTriangular) for node in nodes)

    # Numeric test
    rng = np.random.default_rng(sum(map(ord, "test_psd_solve_with_chol")))

    L = rng.normal(size=(5, 5)).astype(config.floatX)
    X_psd = L @ L.T
    X_psd_inv = f(X_psd)
    assert_allclose(
        X_psd_inv,
        np.linalg.inv(X_psd),
        atol=1e-4 if config.floatX == "float32" else 1e-8,
        rtol=1e-4 if config.floatX == "float32" else 1e-8,
    )


class TestBatchedVectorBSolveToMatrixBSolve:
    rewrite_name = "batched_vector_b_solve_to_matrix_b_solve"

    @staticmethod
    def any_vector_b_solve(fn):
        return any(
            (
                isinstance(node.op, Blockwise)
                and isinstance(node.op.core_op, SolveBase)
                and node.op.core_op.b_ndim == 1
            )
            for node in fn.maker.fgraph.apply_nodes
        )

    @pytest.mark.parametrize("solve_op", (solve, solve_triangular, cho_solve))
    def test_valid_cases(self, solve_op):
        rng = np.random.default_rng(sum(map(ord, solve_op.__name__)))

        a = tensor(shape=(None, None))
        b = tensor(shape=(None, None, None))

        if solve_op is cho_solve:
            # cho_solves expects a tuple (a, lower) as the first input
            out = solve_op((a, True), b, b_ndim=1)
        else:
            out = solve_op(a, b, b_ndim=1)

        mode = get_default_mode().excluding(self.rewrite_name)
        ref_fn = pytensor.function([a, b], out, mode=mode)
        assert self.any_vector_b_solve(ref_fn)

        mode = get_default_mode().including(self.rewrite_name)
        opt_fn = pytensor.function([a, b], out, mode=mode)
        assert not self.any_vector_b_solve(opt_fn)

        test_a = rng.normal(size=(3, 3)).astype(config.floatX)
        test_b = rng.normal(size=(7, 5, 3)).astype(config.floatX)
        np.testing.assert_allclose(
            opt_fn(test_a, test_b),
            ref_fn(test_a, test_b),
            rtol=1e-7 if config.floatX == "float64" else 1e-5,
        )

    def test_invalid_batched_a(self):
        rng = np.random.default_rng(sum(map(ord, self.rewrite_name)))

        # Rewrite is not applicable if a has batched dims
        a = tensor(shape=(None, None, None))
        b = tensor(shape=(None, None, None))

        out = solve(a, b, b_ndim=1)

        mode = get_default_mode().including(self.rewrite_name)
        opt_fn = pytensor.function([a, b], out, mode=mode)
        assert self.any_vector_b_solve(opt_fn)

        ref_fn = np.vectorize(np.linalg.solve, signature="(m,m),(m)->(m)")

        test_a = rng.normal(size=(5, 3, 3)).astype(config.floatX)
        test_b = rng.normal(size=(7, 5, 3)).astype(config.floatX)
        np.testing.assert_allclose(
            opt_fn(test_a, test_b),
            ref_fn(test_a, test_b),
            rtol=1e-7 if config.floatX == "float64" else 1e-5,
        )
