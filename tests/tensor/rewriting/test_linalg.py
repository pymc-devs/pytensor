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
from pytensor.graph import ancestors
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.tensor import swapaxes
from pytensor.tensor.blockwise import Blockwise, BlockwiseWithCoreShape
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import dot, matmul
from pytensor.tensor.nlinalg import (
    SVD,
    Det,
    KroneckerProduct,
    MatrixInverse,
    MatrixPinv,
    SLogDet,
    inv,
    matrix_inverse,
    svd,
)
from pytensor.tensor.rewriting.linalg import inv_as_solve
from pytensor.tensor.slinalg import (
    BlockDiagonal,
    Cholesky,
    CholeskySolve,
    Solve,
    SolveBase,
    SolveTriangular,
    TriangularInv,
    cho_solve,
    cholesky,
    lu,
    qr,
    solve,
    solve_triangular,
)
from pytensor.tensor.type import dmatrix, matrix, tensor, vector
from tests import unittest_tools as utt
from tests.test_rop import break_op


def test_matrix_inverse_rop_lop():
    rtol = 1e-7 if config.floatX == "float64" else 1e-5
    mx = matrix("mx")
    mv = matrix("mv")
    v = vector("v")
    y = MatrixInverse()(mx).sum(axis=0)

    yv = pytensor.gradient.Rop(y, mx, mv, use_op_rop_implementation=True)
    rop_f = function([mx, mv], yv)

    yv_via_lop = pytensor.gradient.Rop(y, mx, mv, use_op_rop_implementation=False)
    rop_via_lop_f = function([mx, mv], yv_via_lop)

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
    np.testing.assert_allclose(rop_f(vx, vv), v_ref, rtol=rtol)
    np.testing.assert_allclose(rop_via_lop_f(vx, vv), v_ref, rtol=rtol)

    with pytest.raises(ValueError):
        pytensor.gradient.Rop(
            pytensor.clone_replace(y, replace={mx: break_op(mx)}),
            mx,
            mv,
            use_op_rop_implementation=True,
        )

    vv = np.asarray(rng.uniform(size=(4,)), pytensor.config.floatX)
    yv = pytensor.gradient.Lop(y, mx, v)
    lop_f = function([mx, v], yv)

    sy = pytensor.gradient.grad((v * y).sum(), mx)
    scan_f = function([mx, v], sy)

    v_ref = scan_f(vx, vv)
    v = lop_f(vx, vv)
    np.testing.assert_allclose(v, v_ref, rtol=rtol)


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
        or (
            isinstance(node.op, Blockwise | BlockwiseWithCoreShape)
            and isinstance(node.op.core_op, Cholesky)
        )
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
            rtol=1e-6 if config.floatX == "float32" else 1e-7,
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
                isinstance(node.op, Blockwise | BlockwiseWithCoreShape)
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


@pytest.mark.parametrize(
    "constructor", [pt.dmatrix, pt.tensor3], ids=["not_batched", "batched"]
)
@pytest.mark.parametrize(
    "f_op, f",
    [
        (MatrixInverse, pt.linalg.inv),
        (Cholesky, pt.linalg.cholesky),
        (MatrixPinv, pt.linalg.pinv),
    ],
    ids=["inv", "cholesky", "pinv"],
)
@pytest.mark.parametrize(
    "g_op, g",
    [(BlockDiagonal, pt.linalg.block_diag), (KroneckerProduct, pt.linalg.kron)],
    ids=["block_diag", "kron"],
)
def test_local_lift_through_linalg(constructor, f_op, f, g_op, g):
    rng = np.random.default_rng(sum(map(ord, "lift_through_linalg")))

    if pytensor.config.floatX.endswith("32"):
        pytest.skip("Test is flaky at half precision")

    A, B = list(map(constructor, "ab"))
    X = f(g(A, B))

    f1 = pytensor.function(
        [A, B], X, mode=get_default_mode().including("local_lift_through_linalg")
    )

    f2 = pytensor.function(
        [A, B], X, mode=get_default_mode().excluding("local_lift_through_linalg")
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


@pytest.mark.parametrize(
    "shape",
    [(), (7,), (1, 7), (7, 1), (7, 7), (3, 7, 7)],
    ids=["scalar", "vector", "row_vec", "col_vec", "matrix", "batched_input"],
)
def test_det_diag_from_eye_mul(shape):
    # Initializing x based on scalar/vector/matrix
    x = pt.tensor("x", shape=shape)
    y = pt.eye(7) * x

    # Calculating determinant value using pt.linalg.det
    z_det = pt.linalg.det(y)

    # REWRITE TEST
    f_rewritten = function([x], z_det, mode="FAST_RUN")
    nodes = f_rewritten.maker.fgraph.apply_nodes

    assert not any(
        isinstance(node.op, Det) or isinstance(getattr(node.op, "core_op", None), Det)
        for node in nodes
    )

    # NUMERIC VALUE TEST
    if len(shape) == 0:
        x_test = np.array(np.random.rand()).astype(config.floatX)
    elif len(shape) == 1:
        x_test = np.random.rand(*shape).astype(config.floatX)
    else:
        x_test = np.random.rand(*shape).astype(config.floatX)

    x_test_matrix = np.eye(7) * x_test
    det_val = np.linalg.det(x_test_matrix)
    rewritten_val = f_rewritten(x_test)

    assert_allclose(
        det_val,
        rewritten_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


def test_det_diag_from_diag():
    x = pt.tensor("x", shape=(None,))
    x_diag = pt.diag(x)
    y = pt.linalg.det(x_diag)

    # REWRITE TEST
    f_rewritten = function([x], y, mode="FAST_RUN")
    nodes = f_rewritten.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, Det) for node in nodes)

    # NUMERIC VALUE TEST
    x_test = np.random.rand(7).astype(config.floatX)
    x_test_matrix = np.eye(7) * x_test
    det_val = np.linalg.det(x_test_matrix)
    rewritten_val = f_rewritten(x_test)

    assert_allclose(
        det_val,
        rewritten_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


def test_dont_apply_det_diag_rewrite_for_1_1():
    x = pt.matrix("x")
    x_diag = pt.eye(1, 1) * x
    y = pt.linalg.det(x_diag)
    f_rewritten = function([x], y, mode="FAST_RUN")

    nodes = f_rewritten.maker.fgraph.apply_nodes

    assert any(isinstance(node.op, Det) for node in nodes)

    # Numeric Value test
    x_test = np.random.normal(size=(3, 3)).astype(config.floatX)
    x_test_matrix = np.eye(1, 1) * x_test
    det_val = np.linalg.det(x_test_matrix)
    rewritten_val = f_rewritten(x_test)

    assert_allclose(
        det_val,
        rewritten_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


def test_det_diag_incorrect_for_rectangle_eye():
    x = pt.matrix("x")
    x_diag = pt.eye(7, 5) * x
    with pytest.raises(ValueError, match="Determinant not defined"):
        pt.linalg.det(x_diag)


def test_svd_uv_merge():
    a = matrix("a")
    s_1 = svd(a, full_matrices=False, compute_uv=False)
    _, s_2, _ = svd(a, full_matrices=False, compute_uv=True)
    _, s_3, _ = svd(a, full_matrices=True, compute_uv=True)
    u_4, _s_4, _v_4 = svd(a, full_matrices=True, compute_uv=True)
    # `grad` will introduces an SVD Op with compute_uv=True
    # full_matrices = True is not supported for grad of svd
    gs = pt.grad(pt.sum(s_1), a)

    # 1. compute_uv=False needs rewriting with compute_uv=True
    f_1 = pytensor.function([a], gs)
    nodes = f_1.maker.fgraph.apply_nodes
    svd_counter = 0
    for node in nodes:
        if isinstance(node.op, SVD):
            assert node.op.compute_uv
            svd_counter += 1
    assert svd_counter == 1

    # 2. compute_uv=True needs rewriting with compute=False, reuse node
    f_2 = pytensor.function([a], [s_1, s_2])
    nodes = f_2.maker.fgraph.apply_nodes
    svd_counter = 0
    for node in nodes:
        if isinstance(node.op, SVD):
            assert not node.op.compute_uv
            svd_counter += 1
    assert svd_counter == 1

    # 3. compute_uv=True needs rewriting with compute=False, create new node
    # full_matrices needs to retain the value
    f_3 = pytensor.function([a], [s_2])
    nodes = f_3.maker.fgraph.apply_nodes
    svd_counter = 0
    for node in nodes:
        if isinstance(node.op, SVD):
            assert not node.op.compute_uv
            svd_counter += 1
    assert svd_counter == 1

    # Case 2 of 3. for a different full_matrices
    f_4 = pytensor.function([a], [s_3])
    nodes = f_4.maker.fgraph.apply_nodes
    svd_counter = 0
    for node in nodes:
        if isinstance(node.op, SVD):
            assert not node.op.compute_uv
            assert node.op.full_matrices
            svd_counter += 1
    assert svd_counter == 1

    # 4. No rewrite should happen
    f_5 = pytensor.function([a], [u_4])
    nodes = f_5.maker.fgraph.apply_nodes
    svd_counter = 0
    for node in nodes:
        if isinstance(node.op, SVD):
            assert node.op.full_matrices
            assert node.op.compute_uv
            svd_counter += 1
    assert svd_counter == 1


def get_pt_function(x, op_name):
    return getattr(pt.linalg, op_name)(x)


@pytest.mark.parametrize("inv_op_1", ["inv", "pinv"])
@pytest.mark.parametrize("inv_op_2", ["inv", "pinv"])
def test_inv_inv_rewrite(inv_op_1, inv_op_2):
    x = pt.matrix("x")
    op1 = get_pt_function(x, inv_op_1)
    op2 = get_pt_function(op1, inv_op_2)
    rewritten_out = rewrite_graph(op2)
    assert rewritten_out == x


@pytest.mark.parametrize("inv_op", ["inv", "pinv"])
def test_inv_eye_to_eye(inv_op):
    x = pt.eye(10)
    x_inv = get_pt_function(x, inv_op)
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
@pytest.mark.parametrize("inv_op", ["inv", "pinv"])
def test_inv_diag_from_eye_mul(shape, inv_op):
    # Initializing x based on scalar/vector/matrix
    x = pt.tensor("x", shape=shape)
    x_diag = pt.eye(7) * x
    # Calculating inverse using pt.linalg.inv
    x_inv = get_pt_function(x_diag, inv_op)

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


@pytest.mark.parametrize("inv_op", ["inv", "pinv"])
def test_inv_diag_from_diag(inv_op):
    x = pt.dvector("x")
    x_diag = pt.diag(x)
    x_inv = get_pt_function(x_diag, inv_op)

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


def test_diag_blockdiag_rewrite():
    n_matrices = 10
    matrix_size = (5, 5)
    sub_matrices = pt.tensor("sub_matrices", shape=(n_matrices, *matrix_size))
    bd_output = pt.linalg.block_diag(*[sub_matrices[i] for i in range(n_matrices)])
    diag_output = pt.diag(bd_output)
    f_rewritten = function([sub_matrices], diag_output, mode="FAST_RUN")

    # Rewrite Test
    nodes = f_rewritten.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, BlockDiagonal) for node in nodes)

    # Value Test
    sub_matrices_test = np.random.rand(n_matrices, *matrix_size).astype(config.floatX)
    bd_output_test = scipy.linalg.block_diag(
        *[sub_matrices_test[i] for i in range(n_matrices)]
    )
    diag_output_test = np.diag(bd_output_test)
    rewritten_val = f_rewritten(sub_matrices_test)
    assert_allclose(
        diag_output_test,
        rewritten_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


def test_det_blockdiag_rewrite():
    n_matrices = 100
    matrix_size = (5, 5)
    sub_matrices = pt.tensor("sub_matrices", shape=(n_matrices, *matrix_size))
    bd_output = pt.linalg.block_diag(*[sub_matrices[i] for i in range(n_matrices)])
    det_output = pt.linalg.det(bd_output)
    f_rewritten = function([sub_matrices], det_output, mode="FAST_RUN")

    # Rewrite Test
    nodes = f_rewritten.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, BlockDiagonal) for node in nodes)

    # Value Test
    sub_matrices_test = np.random.rand(n_matrices, *matrix_size).astype(config.floatX)
    bd_output_test = scipy.linalg.block_diag(
        *[sub_matrices_test[i] for i in range(n_matrices)]
    )
    det_output_test = np.linalg.det(bd_output_test)
    rewritten_val = f_rewritten(sub_matrices_test)
    assert_allclose(
        det_output_test,
        rewritten_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


def test_slogdet_blockdiag_rewrite():
    n_matrices = 10
    matrix_size = (5, 5)
    sub_matrices = pt.tensor("sub_matrices", shape=(n_matrices, *matrix_size))
    bd_output = pt.linalg.block_diag(*[sub_matrices[i] for i in range(n_matrices)])
    sign_output, logdet_output = pt.linalg.slogdet(bd_output)
    f_rewritten = function(
        [sub_matrices], [sign_output, logdet_output], mode="FAST_RUN"
    )

    # Rewrite Test
    nodes = f_rewritten.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, BlockDiagonal) for node in nodes)

    # Value Test
    sub_matrices_test = np.random.rand(n_matrices, *matrix_size).astype(config.floatX)
    bd_output_test = scipy.linalg.block_diag(
        *[sub_matrices_test[i] for i in range(n_matrices)]
    )
    sign_output_test, logdet_output_test = np.linalg.slogdet(bd_output_test)
    rewritten_sign_val, rewritten_logdet_val = f_rewritten(sub_matrices_test)
    assert_allclose(
        sign_output_test,
        rewritten_sign_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )
    assert_allclose(
        logdet_output_test,
        rewritten_logdet_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


def test_diag_kronecker_rewrite():
    a, b = pt.dmatrices("a", "b")
    kron_prod = pt.linalg.kron(a, b)
    diag_kron_prod = pt.diag(kron_prod)
    f_rewritten = function([a, b], diag_kron_prod, mode="FAST_RUN")

    # Rewrite Test
    nodes = f_rewritten.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, KroneckerProduct) for node in nodes)

    # Value Test
    a_test, b_test = np.random.rand(2, 20, 20)
    kron_prod_test = np.kron(a_test, b_test)
    diag_kron_prod_test = np.diag(kron_prod_test)
    rewritten_val = f_rewritten(a_test, b_test)
    assert_allclose(
        diag_kron_prod_test,
        rewritten_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


def test_det_kronecker_rewrite():
    a, b = pt.dmatrices("a", "b")
    kron_prod = pt.linalg.kron(a, b)
    det_output = pt.linalg.det(kron_prod)
    f_rewritten = function([a, b], [det_output], mode="FAST_RUN")

    # Rewrite Test
    nodes = f_rewritten.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, KroneckerProduct) for node in nodes)

    # Value Test
    a_test, b_test = np.random.rand(2, 20, 20)
    kron_prod_test = np.kron(a_test, b_test)
    det_output_test = np.linalg.det(kron_prod_test)
    rewritten_det_val = f_rewritten(a_test, b_test)
    assert_allclose(
        det_output_test,
        rewritten_det_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


def test_slogdet_kronecker_rewrite():
    a, b = pt.dmatrices("a", "b")
    kron_prod = pt.linalg.kron(a, b)
    sign_output, logdet_output = pt.linalg.slogdet(kron_prod)
    f_rewritten = function([a, b], [sign_output, logdet_output], mode="FAST_RUN")

    # Rewrite Test
    nodes = f_rewritten.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, KroneckerProduct) for node in nodes)

    # Value Test
    a_test, b_test = np.random.rand(2, 20, 20)
    kron_prod_test = np.kron(a_test, b_test)
    sign_output_test, logdet_output_test = np.linalg.slogdet(kron_prod_test)
    rewritten_sign_val, rewritten_logdet_val = f_rewritten(a_test, b_test)
    assert_allclose(
        sign_output_test,
        rewritten_sign_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )
    assert_allclose(
        logdet_output_test,
        rewritten_logdet_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


def test_cholesky_eye_rewrite():
    x = pt.eye(10)
    L = pt.linalg.cholesky(x)
    f_rewritten = function([], L, mode="FAST_RUN")
    nodes = f_rewritten.maker.fgraph.apply_nodes

    # Rewrite Test
    assert not any(isinstance(node.op, Cholesky) for node in nodes)

    # Value Test
    x_test = np.eye(10)
    L = np.linalg.cholesky(x_test)
    rewritten_val = f_rewritten()

    assert_allclose(
        L,
        rewritten_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


@pytest.mark.parametrize(
    "shape",
    [(), (7,), (7, 7), (5, 7, 7)],
    ids=["scalar", "vector", "matrix", "batched"],
)
def test_cholesky_diag_from_eye_mul(shape):
    # Initializing x based on scalar/vector/matrix
    x = pt.tensor("x", shape=shape)
    y = pt.eye(7) * x
    # Performing cholesky decomposition using pt.linalg.cholesky
    z_cholesky = pt.linalg.cholesky(y)

    # REWRITE TEST
    f_rewritten = function([x], z_cholesky, mode="FAST_RUN")
    nodes = f_rewritten.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, Cholesky) for node in nodes)

    # NUMERIC VALUE TEST
    if len(shape) == 0:
        x_test = np.array(np.random.rand()).astype(config.floatX)
    elif len(shape) == 1:
        x_test = np.random.rand(*shape).astype(config.floatX)
    else:
        x_test = np.random.rand(*shape).astype(config.floatX)
    x_test_matrix = np.eye(7) * x_test
    cholesky_val = np.linalg.cholesky(x_test_matrix)
    rewritten_val = f_rewritten(x_test)

    assert_allclose(
        cholesky_val,
        rewritten_val,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


def test_cholesky_diag_from_diag():
    x = pt.dvector("x")
    x_diag = pt.diag(x)
    x_cholesky = pt.linalg.cholesky(x_diag)

    # REWRITE TEST
    f_rewritten = function([x], x_cholesky, mode="FAST_RUN")
    nodes = f_rewritten.maker.fgraph.apply_nodes

    assert not any(isinstance(node.op, Cholesky) for node in nodes)

    # NUMERIC VALUE TEST
    x_test = np.random.rand(10)
    x_test_matrix = np.eye(10) * x_test
    cholesky_val = np.linalg.cholesky(x_test_matrix)
    rewritten_cholesky = f_rewritten(x_test)

    assert_allclose(
        cholesky_val,
        rewritten_cholesky,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


def test_rewrite_cholesky_diag_to_sqrt_diag_not_applied():
    # Case 1 : y is not a diagonal matrix because of k = -1
    x = pt.tensor("x", shape=(7, 7))
    y = pt.eye(7, k=-1) * x
    z_cholesky = pt.linalg.cholesky(y)

    # REWRITE TEST (should not be applied)
    f_rewritten = function([x], z_cholesky, mode="FAST_RUN")
    nodes = f_rewritten.maker.fgraph.apply_nodes
    assert any(isinstance(node.op, Cholesky) for node in nodes)


def test_slogdet_specialization():
    x, a = pt.dmatrix("x"), np.random.rand(20, 20)
    det_x, det_a = pt.linalg.det(x), np.linalg.det(a)
    log_abs_det_x, log_abs_det_a = pt.log(pt.abs(det_x)), np.log(np.abs(det_a))
    log_det_x, log_det_a = pt.log(det_x), np.log(det_a)
    sign_det_x, sign_det_a = pt.sign(det_x), np.sign(det_a)
    exp_det_x = pt.exp(det_x)

    # REWRITE TESTS
    # sign(det(x))
    f = function([x], [sign_det_x], mode="FAST_RUN")
    nodes = f.maker.fgraph.apply_nodes
    assert len([node for node in nodes if isinstance(node.op, SLogDet)]) == 1
    assert not any(isinstance(node.op, Det) for node in nodes)
    rw_sign_det_a = f(a)
    assert_allclose(
        sign_det_a,
        rw_sign_det_a,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )

    # log(abs(det(x)))
    f = function([x], [log_abs_det_x], mode="FAST_RUN")
    nodes = f.maker.fgraph.apply_nodes
    assert len([node for node in nodes if isinstance(node.op, SLogDet)]) == 1
    assert not any(isinstance(node.op, Det) for node in nodes)
    rw_log_abs_det_a = f(a)
    assert_allclose(
        log_abs_det_a,
        rw_log_abs_det_a,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )

    # log(det(x))
    f = function([x], [log_det_x], mode="FAST_RUN")
    nodes = f.maker.fgraph.apply_nodes
    assert len([node for node in nodes if isinstance(node.op, SLogDet)]) == 1
    assert not any(isinstance(node.op, Det) for node in nodes)
    rw_log_det_a = f(a)
    assert_allclose(
        log_det_a,
        rw_log_det_a,
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )

    # More than 1 valid function
    f = function([x], [sign_det_x, log_abs_det_x], mode="FAST_RUN")
    nodes = f.maker.fgraph.apply_nodes
    assert len([node for node in nodes if isinstance(node.op, SLogDet)]) == 1
    assert not any(isinstance(node.op, Det) for node in nodes)

    # Other functions (rewrite shouldnt be applied to these)
    # Only invalid functions
    f = function([x], [exp_det_x], mode="FAST_RUN")
    nodes = f.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, SLogDet) for node in nodes)

    # Invalid + Valid function
    f = function([x], [exp_det_x, sign_det_x], mode="FAST_RUN")
    nodes = f.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, SLogDet) for node in nodes)


@pytest.mark.parametrize(
    "a_batch_shape", [(), (5,)], ids=lambda x: f"a_batch_shape={x}"
)
@pytest.mark.parametrize(
    "b_batch_shape", [(), (5,)], ids=lambda x: f"b_batch_shape={x}"
)
@pytest.mark.parametrize("b_ndim", (1, 2), ids=lambda x: f"b_ndim={x}")
@pytest.mark.parametrize(
    "op, fn, extra_kwargs",
    [
        (Solve, pt.linalg.solve, {}),
        (SolveTriangular, pt.linalg.solve_triangular, {}),
        (SolveTriangular, pt.linalg.solve_triangular, {"unit_diagonal": True}),
        (CholeskySolve, pt.linalg.cho_solve, {}),
    ],
)
def test_scalar_solve_to_division_rewrite(
    op, fn, extra_kwargs, b_ndim, a_batch_shape, b_batch_shape
):
    def solve_op_in_graph(graph):
        return any(
            isinstance(var.owner.op, SolveBase)
            or (
                isinstance(var.owner.op, Blockwise)
                and isinstance(var.owner.op.core_op, SolveBase)
            )
            for var in ancestors(graph)
            if var.owner
        )

    rng = np.random.default_rng(
        [
            sum(map(ord, "scalar_solve_to_division_rewrite")),
            b_ndim,
            *a_batch_shape,
            1,
            *b_batch_shape,
        ]
    )

    a = pt.tensor("a", shape=(*a_batch_shape, 1, 1), dtype="float64")
    b = pt.tensor("b", shape=(*b_batch_shape, *([None] * b_ndim)), dtype="float64")

    if op is CholeskySolve:
        # cho_solve expects a tuple (c, lower) as the first input
        c = fn((pt.linalg.cholesky(a), True), b, b_ndim=b_ndim, **extra_kwargs)
    else:
        c = fn(a, b, b_ndim=b_ndim, **extra_kwargs)

    assert solve_op_in_graph([c])
    f = function([a, b], c, mode="FAST_RUN")
    assert not solve_op_in_graph(f.maker.fgraph.outputs)

    a_val = rng.normal(size=(*a_batch_shape, 1, 1)).astype(pytensor.config.floatX)
    b_core_shape = (1, 5) if b_ndim == 2 else (1,)
    b_val = rng.normal(size=(*b_batch_shape, *b_core_shape)).astype(
        pytensor.config.floatX
    )

    if op is CholeskySolve:
        # Avoid sign ambiguity in solve
        a_val = a_val**2

    if extra_kwargs.get("unit_diagonal", False):
        a_val = np.ones_like(a_val)

    signature = "(n,m),(m)->(n)" if b_ndim == 1 else "(n,m),(m,k)->(n,k)"
    c_val = np.vectorize(np.linalg.solve, signature=signature)(a_val, b_val)
    np.testing.assert_allclose(
        f(a_val, b_val), c_val, rtol=1e-7 if config.floatX == "float64" else 1e-5
    )


def _check_op_in_graph(fgraph, op_type, present=True):
    """Helper to check if an Op is in a graph."""

    # We use type() instead of isinstance() to avoid matching subclasses
    # (e.g., finding TriangularInv when we're looking for MatrixInverse).
    found = any(
        type(node.op) is op_type
        or (hasattr(node.op, "core_op") and type(node.op.core_op) is op_type)
        for node in fgraph.apply_nodes
    )
    if present:
        assert found, f"{op_type.__name__} not found in graph"
    else:
        assert not found, f"{op_type.__name__} unexpectedly found in graph"


rewrite_cases = {
    "lower_tag": (
        lambda x: (setattr(x.tag, "lower_triangular", True), x)[-1],
        lambda a: np.tril(a),
    ),
    "upper_tag": (
        lambda x: (setattr(x.tag, "upper_triangular", True), x)[-1],
        lambda a: np.triu(a),
    ),
    "tri": (
        lambda x: pt.tri(x.shape[0], x.shape[1], k=0, dtype=x.dtype),
        lambda a: np.tri(N=a.shape[0], M=a.shape[1], k=0, dtype=a.dtype),
    ),
    "tril": (
        lambda x: pt.tril(x),
        lambda a: np.tril(a),
    ),
    "triu": (
        lambda x: pt.triu(x),
        lambda a: np.triu(a),
    ),
    "cholesky": (
        lambda x: cholesky(x),
        lambda a: np.linalg.cholesky(a),
    ),
    "lu_L": (
        lambda x: lu(x)[1],
        lambda a: scipy.linalg.lu(a)[1],
    ),
    "lu_U": (
        lambda x: lu(x)[2],
        lambda a: scipy.linalg.lu(a)[2],
    ),
    "qr_R": (
        lambda x: qr(x)[1],
        lambda a: np.linalg.qr(a)[1],
    ),
}


@pytest.mark.parametrize("case", rewrite_cases.keys())
def test_inv_to_triangular_inv_rewrite(case):
    """
    Tests the rewrite of inv(triangular) -> TriangularInv.
    """
    x = matrix("x", dtype=config.floatX)
    build_tri, _ = rewrite_cases[case]
    x_tri = build_tri(x)
    y_inv = inv(x_tri)

    # Check graph BEFORE compilation
    pre_compile_fgraph = FunctionGraph([x], [y_inv], clone=False)
    _check_op_in_graph(pre_compile_fgraph, MatrixInverse, present=True)
    _check_op_in_graph(pre_compile_fgraph, TriangularInv, present=False)

    # Trigger the rewrite
    f = function([x], y_inv)

    # Check graph AFTER compilation
    post_compile_fgraph = f.maker.fgraph
    _check_op_in_graph(post_compile_fgraph, TriangularInv, present=True)
    _check_op_in_graph(post_compile_fgraph, MatrixInverse, present=False)

    # Check numerical correctness
    a = np.random.rand(5, 5)
    a = (np.dot(a, a.T) + np.eye(5)).astype(
        config.floatX
    )  # Make positive definite for Cholesky
    if case == "lower_tag":
        a = np.tril(a)
    elif case == "upper_tag":
        a = np.triu(a)
    pytensor_result = f(a)
    _, numpy_tri_func = rewrite_cases[case]
    numpy_result = np.linalg.inv(numpy_tri_func(a))
    atol = rtol = 1e-8 if config.floatX.endswith("64") else 1e-4
    np.testing.assert_allclose(pytensor_result, numpy_result, rtol=rtol, atol=atol)
