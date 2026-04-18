import numpy as np
import pytest
from numpy.testing import assert_allclose

import pytensor
from pytensor import function, scan
from pytensor import tensor as pt
from pytensor.compile import get_default_mode
from pytensor.configdefaults import config
from pytensor.gradient import grad
from pytensor.graph import ancestors
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.scan.op import Scan
from pytensor.tensor.blockwise import Blockwise, BlockwiseWithCoreShape
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky, cholesky
from pytensor.tensor.linalg.decomposition.lu import LUFactor
from pytensor.tensor.linalg.solvers.core import SolveBase
from pytensor.tensor.linalg.solvers.general import Solve, solve
from pytensor.tensor.linalg.solvers.linear_control import (
    solve_sylvester,
)
from pytensor.tensor.linalg.solvers.psd import CholeskySolve, cho_solve
from pytensor.tensor.linalg.solvers.triangular import SolveTriangular, solve_triangular
from pytensor.tensor.linalg.solvers.tridiagonal import (
    LUFactorTridiagonal,
    SolveLUFactorTridiagonal,
)
from pytensor.tensor.rewriting.linalg.solvers import (
    reuse_decomposition_multiple_solves,
    scan_split_non_sequence_decomposition_and_solve,
)
from pytensor.tensor.type import matrix, tensor
from tests.unittest_tools import assert_equal_computations


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


def test_psd_solve_with_chol():
    """Test that solve(A, b) with PSD A gets rewritten to cholesky + cho_solve."""
    A = matrix("A")
    b = matrix("b")
    A_psd = pt.specify_assumptions(A, positive_definite=True)
    out = pt.linalg.solve(A_psd, b)

    rewritten = rewrite_graph(out, include=("canonicalize", "stabilize", "specialize"))

    L = cholesky(A_psd)
    expected = cho_solve((L, True), b, b_ndim=2)

    assert_equal_computations([rewritten], [expected])


def test_paired_triangular_solves_to_cho_solve():
    """Test that paired triangular solves from Cholesky get fused into cho_solve."""
    A = matrix("A")
    b = matrix("b")

    # Manually create the pattern: solve_triangular(L.T, solve_triangular(L, b))
    L = pt.linalg.cholesky(A, lower=True)
    Li_b = pt.linalg.solve_triangular(L, b, lower=True, b_ndim=2)
    x = pt.linalg.solve_triangular(L.mT, Li_b, lower=False, b_ndim=2)

    rewritten = rewrite_graph(x, include=("canonicalize", "stabilize", "specialize"))
    expected = cho_solve((L, True), b, b_ndim=2)

    assert_equal_computations([rewritten], [expected])


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
def test_scalar_solve_to_division(
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
        c = fn((cholesky(a), True), b, b_ndim=b_ndim, **extra_kwargs)
    else:
        c = fn(a, b, b_ndim=b_ndim, **extra_kwargs)

    assert solve_op_in_graph([c])
    f = pytensor.function([a, b], c, mode="FAST_RUN")
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


class DecompSolveOpCounter:
    def __init__(self, solve_op, decomp_op, solve_op_value: float = 1.0):
        self.solve_op = solve_op
        self.decomp_op = decomp_op
        self.solve_op_value = solve_op_value

    def check_node_op_or_core_op(self, node, op):
        return isinstance(node.op, op) or (
            isinstance(node.op, Blockwise | BlockwiseWithCoreShape)
            and isinstance(node.op.core_op, op)
        )

    def count_vanilla_solve_nodes(self, nodes) -> int:
        return sum(self.check_node_op_or_core_op(node, Solve) for node in nodes)

    def count_decomp_nodes(self, nodes) -> int:
        return sum(
            self.check_node_op_or_core_op(node, self.decomp_op) for node in nodes
        )

    def count_solve_nodes(self, nodes) -> int:
        count = sum(
            self.solve_op_value * self.check_node_op_or_core_op(node, self.solve_op)
            for node in nodes
        )
        return int(count)


LUOpCounter = DecompSolveOpCounter(
    solve_op=SolveTriangular,
    decomp_op=LUFactor,
    # Each rewrite introduces two triangular solves, so count them as 1/2 each
    solve_op_value=0.5,
)

TriDiagLUOpCounter = DecompSolveOpCounter(
    solve_op=SolveLUFactorTridiagonal, decomp_op=LUFactorTridiagonal, solve_op_value=1.0
)

CholeskyOpCounter = DecompSolveOpCounter(
    solve_op=CholeskySolve, decomp_op=Cholesky, solve_op_value=1.0
)


@pytest.mark.parametrize("transposed", (False, True))
@pytest.mark.parametrize(
    "assume_a, counter",
    (
        ("gen", LUOpCounter),
        ("tridiagonal", TriDiagLUOpCounter),
        ("pos", CholeskyOpCounter),
    ),
)
def test_lu_decomposition_reused_forward_and_gradient(assume_a, counter, transposed):
    rewrite_name = reuse_decomposition_multiple_solves.__name__
    mode = get_default_mode()

    A = tensor("A", shape=(3, 3))
    b = tensor("b", shape=(3, 4))

    x = solve(A, b, assume_a=assume_a, transposed=transposed)
    grad_x_wrt_A = grad(x.sum(), A)
    fn_no_opt = function(
        [A, b],
        [x, grad_x_wrt_A],
        mode=mode.excluding(rewrite_name, "psd_solve_to_chol_solve"),
    )
    no_opt_nodes = fn_no_opt.maker.fgraph.apply_nodes
    assert counter.count_vanilla_solve_nodes(no_opt_nodes) == 2
    assert counter.count_decomp_nodes(no_opt_nodes) == 0
    assert counter.count_solve_nodes(no_opt_nodes) == 0

    fn_opt = function(
        [A, b],
        [x, grad_x_wrt_A],
        mode=mode.including(rewrite_name).excluding("psd_solve_to_chol_solve"),
    )
    opt_nodes = fn_opt.maker.fgraph.apply_nodes
    assert counter.count_vanilla_solve_nodes(opt_nodes) == 0
    assert counter.count_decomp_nodes(opt_nodes) == 1
    assert counter.count_solve_nodes(opt_nodes) == 2

    # Make sure results are correct
    rng = np.random.default_rng(31)
    A_test = rng.random(A.type.shape, dtype=A.type.dtype)
    if assume_a == "pos":
        A_test = A_test @ A_test.T  # Ensure positive definite for Cholesky

    b_test = rng.random(b.type.shape, dtype=b.type.dtype)
    resx0, resg0 = fn_no_opt(A_test, b_test)
    resx1, resg1 = fn_opt(A_test, b_test)
    rtol = 1e-7 if config.floatX == "float64" else 1e-4
    np.testing.assert_allclose(resx0, resx1, rtol=rtol)
    np.testing.assert_allclose(resg0, resg1, rtol=rtol)


@pytest.mark.parametrize("transposed", (False, True))
@pytest.mark.parametrize(
    "assume_a, counter",
    (
        ("gen", LUOpCounter),
        ("tridiagonal", TriDiagLUOpCounter),
        ("pos", CholeskyOpCounter),
    ),
)
def test_lu_decomposition_reused_blockwise(assume_a, counter, transposed):
    rewrite_name = reuse_decomposition_multiple_solves.__name__
    mode = get_default_mode()

    A = tensor("A", shape=(3, 3))
    b = tensor("b", shape=(2, 3, 4))

    x = solve(A, b, assume_a=assume_a, transposed=transposed)
    fn_no_opt = function(
        [A, b], [x], mode=mode.excluding(rewrite_name, "psd_solve_to_chol_solve")
    )
    no_opt_nodes = fn_no_opt.maker.fgraph.apply_nodes
    assert counter.count_vanilla_solve_nodes(no_opt_nodes) == 1
    assert counter.count_decomp_nodes(no_opt_nodes) == 0
    assert counter.count_solve_nodes(no_opt_nodes) == 0

    fn_opt = function(
        [A, b],
        [x],
        mode=mode.including(rewrite_name).excluding("psd_solve_to_chol_solve"),
    )
    opt_nodes = fn_opt.maker.fgraph.apply_nodes
    assert counter.count_vanilla_solve_nodes(opt_nodes) == 0
    assert counter.count_decomp_nodes(opt_nodes) == 1
    assert counter.count_solve_nodes(opt_nodes) == 1

    # Make sure results are correct
    rng = np.random.default_rng(31)
    A_test = rng.random(A.type.shape, dtype=A.type.dtype)
    if assume_a == "pos":
        A_test = A_test @ A_test.T  # Ensure positive definite for Cholesky

    b_test = rng.random(b.type.shape, dtype=b.type.dtype)
    resx0 = fn_no_opt(A_test, b_test)
    resx1 = fn_opt(A_test, b_test)
    rtol = 1e-7 if config.floatX == "float64" else 1e-4
    np.testing.assert_allclose(resx0, resx1, rtol=rtol)


@pytest.mark.parametrize("transposed", (False, True))
@pytest.mark.parametrize(
    "assume_a, counter",
    (
        ("gen", LUOpCounter),
        ("tridiagonal", TriDiagLUOpCounter),
        ("pos", CholeskyOpCounter),
    ),
)
def test_lu_decomposition_reused_scan(assume_a, counter, transposed):
    rewrite_name = scan_split_non_sequence_decomposition_and_solve.__name__
    mode = get_default_mode()

    A = tensor("A", shape=(3, 3))
    x0 = tensor("b", shape=(3, 4))

    xs = scan(
        lambda xtm1, A: solve(A, xtm1, assume_a=assume_a, transposed=transposed),
        outputs_info=[x0],
        non_sequences=[A],
        n_steps=10,
        return_updates=False,
        mode=get_default_mode().excluding("psd_solve_to_chol_solve"),
    )

    fn_no_opt = function(
        [A, x0],
        [xs],
        mode=mode.excluding(rewrite_name),
    )
    [no_opt_scan_node] = [
        node for node in fn_no_opt.maker.fgraph.apply_nodes if isinstance(node.op, Scan)
    ]
    no_opt_nodes = no_opt_scan_node.op.fgraph.apply_nodes
    assert counter.count_vanilla_solve_nodes(no_opt_nodes) == 1
    assert counter.count_decomp_nodes(no_opt_nodes) == 0
    assert counter.count_solve_nodes(no_opt_nodes) == 0

    fn_opt = function([A, x0], [xs], mode=mode.including("scan", rewrite_name))
    [opt_scan_node] = [
        node for node in fn_opt.maker.fgraph.apply_nodes if isinstance(node.op, Scan)
    ]
    opt_nodes = opt_scan_node.op.fgraph.apply_nodes
    assert counter.count_vanilla_solve_nodes(opt_nodes) == 0
    # The LU decomp is outside of the scan!
    assert counter.count_decomp_nodes(opt_nodes) == 0
    assert counter.count_solve_nodes(opt_nodes) == 1

    # Make sure results are correct
    rng = np.random.default_rng(170)
    A_test = rng.random(A.type.shape, dtype=A.type.dtype)
    if assume_a == "pos":
        A_test = A_test @ A_test.T  # Ensure positive definite for Cholesky

    x0_test = rng.random(x0.type.shape, dtype=x0.type.dtype)
    resx0 = fn_no_opt(A_test, x0_test)
    resx1 = fn_opt(A_test, x0_test)
    rtol = 1e-7 if config.floatX == "float64" else 1e-4
    np.testing.assert_allclose(resx0, resx1, rtol=rtol)


class TestDiagonalSolveToDivision:
    @pytest.mark.parametrize("b_ndim", [1, 2], ids=lambda x: f"b_ndim={x}")
    @pytest.mark.parametrize(
        "make_diag",
        [
            pytest.param(lambda d: pt.diag(d), id="alloc_diag"),
            pytest.param(lambda d: pt.eye(5) * d, id="eye_mul"),
        ],
    )
    def test_solve_diag(self, b_ndim, make_diag):
        d = pt.dvector("d", shape=(5,))
        b = pt.tensor("b", shape=(5,) if b_ndim == 1 else (5, 3), dtype="float64")
        D = make_diag(d)
        out = solve(D, b, b_ndim=b_ndim)

        rewritten = rewrite_graph(
            out, include=("canonicalize", "stabilize", "specialize")
        )
        expected = b / d if b_ndim == 1 else b / d[..., :, None]

        assert_equal_computations([rewritten], [expected])

    @pytest.mark.parametrize("b_ndim", [1, 2], ids=lambda x: f"b_ndim={x}")
    def test_solve_triangular_diag(self, b_ndim):
        d = pt.dvector("d")
        b = pt.tensor("b", shape=(5,) if b_ndim == 1 else (5, 3), dtype="float64")
        D = pt.diag(d)
        out = solve_triangular(D, b, lower=True, b_ndim=b_ndim)

        rewritten = rewrite_graph(
            out, include=("canonicalize", "stabilize", "specialize")
        )
        expected = b / d if b_ndim == 1 else b / d[..., :, None]

        assert_equal_computations([rewritten], [expected])

    @pytest.mark.parametrize("b_ndim", [1, 2], ids=lambda x: f"b_ndim={x}")
    def test_solve_triangular_unit_diag(self, b_ndim):
        d = pt.dvector("d")
        b = pt.tensor("b", shape=(5,) if b_ndim == 1 else (5, 3), dtype="float64")
        D = pt.diag(d)
        out = solve_triangular(D, b, lower=True, unit_diagonal=True, b_ndim=b_ndim)

        rewritten = rewrite_graph(
            out, include=("canonicalize", "stabilize", "specialize")
        )

        assert_equal_computations([rewritten], [b])

    @pytest.mark.parametrize("b_ndim", [1, 2], ids=lambda x: f"b_ndim={x}")
    def test_cho_solve_diag(self, b_ndim):
        d = pt.dvector("d", shape=(5,))
        b = pt.tensor("b", shape=(5,) if b_ndim == 1 else (5, 3), dtype="float64")
        L = pt.diag(d)
        out = cho_solve((L, True), b, b_ndim=b_ndim)

        rewritten = rewrite_graph(
            out, include=("canonicalize", "stabilize", "specialize")
        )
        expected = b / pt.square(d) if b_ndim == 1 else b / pt.square(d[..., :, None])

        assert_equal_computations([rewritten], [expected])


@pytest.mark.parametrize(
    "make_diag",
    [
        pytest.param(lambda d: pt.diag(d), id="alloc_diag"),
        pytest.param(lambda d: pt.eye(5) * d, id="eye_mul"),
    ],
)
def test_solve_sylvester_both_diag(make_diag):
    n = 5
    a = pt.dvector("a", shape=(n,))
    b = pt.dvector("b", shape=(n,))
    C = pt.dmatrix("C", shape=(n, n))

    A = make_diag(a)
    B = make_diag(b)
    X = solve_sylvester(A, B, C)

    rewritten = rewrite_graph(X, include=("canonicalize", "stabilize"))

    expected = C / (a[:, None] + b[None, :])

    assert_equal_computations([rewritten], [expected])


def test_orthogonal_solve_to_transpose_matmul():
    n = 5
    rewrites = ("canonicalize", "stabilize", "ShapeOpt")

    Q = pt.dmatrix("Q", shape=(n, n))
    Q_orth = pt.specify_assumptions(Q, orthogonal=True)
    b = pt.dmatrix("b", shape=(n, 3))
    out = solve(Q_orth, b)
    rewritten = rewrite_graph(out, include=rewrites)
    expected = rewrite_graph(Q_orth.mT @ b, include=rewrites)
    assert_equal_computations([rewritten], [expected])
