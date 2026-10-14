import numpy as np
import pytest
import scipy.linalg
from numpy.testing import assert_allclose

import pytensor
from pytensor import function, scan
from pytensor import tensor as pt
from pytensor.assumptions.specify import assume
from pytensor.compile import get_default_mode
from pytensor.configdefaults import config
from pytensor.gradient import grad
from pytensor.graph import ancestors
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.scan.op import Scan
from pytensor.tensor.blockwise import Blockwise, BlockwiseWithCoreShape
from pytensor.tensor.linalg.constructors import BlockDiagonal
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
from pytensor.tensor.math import Dot
from pytensor.tensor.rewriting.linalg.solvers import (
    cho_solve_quadratic_form_to_whitened,
    reuse_decomposition_multiple_solves,
    scan_split_non_sequence_decomposition_and_solve,
)
from pytensor.tensor.type import matrix, tensor
from tests.unittest_tools import RewriteTester, assert_equal_computations


def test_cholesky_factor_solve_becomes_triangular():
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


@pytest.mark.parametrize("declared", ["gen", "sym", "her"])
@pytest.mark.parametrize("b_ndim", [1, 2], ids=lambda x: f"b_ndim={x}")
def test_psd_solve_with_chol(b_ndim, declared):
    """Test that solve(A, b) with PSD A gets rewritten to cholesky + cho_solve."""
    A = matrix("A")
    b = matrix("b") if b_ndim == 2 else pt.vector("b")
    A_psd = assume(A, positive_definite=True)
    out = pt.linalg.solve(A_psd, b, assume_a=declared, b_ndim=b_ndim)

    rewritten = rewrite_graph(out, include=("canonicalize", "stabilize", "specialize"))

    L = cholesky(A_psd)
    expected = cho_solve((L, True), b, b_ndim=b_ndim)

    assert_equal_computations([rewritten], [expected])


@pytest.mark.skipif(
    config.mode == "FAST_COMPILE",
    reason="sharing needs stabilize plus a later merge pass",
)
def test_psd_vector_solve_shares_cholesky():
    A = matrix("A", shape=(5, 5))
    b = pt.vector("b", shape=(5,))

    logdet = 2 * pt.log(pt.diagonal(cholesky(A))).sum()
    quad = b @ pt.linalg.solve(A, b, assume_a="pos", b_ndim=1)
    f = pytensor.function([A, b], logdet + quad)

    core_ops = [
        getattr(node.op, "core_op", node.op) for node in f.maker.fgraph.apply_nodes
    ]
    assert not any(isinstance(op, Solve) for op in core_ops)
    assert sum(isinstance(op, Cholesky) for op in core_ops) == 1


def test_generic_solve_to_solve_triangular_through_expand_dims():
    A = pt.dmatrix("A")
    b = pt.tensor("b", shape=(None, None, None), dtype="float64")

    # A batched b makes Blockwise.make_node wrap cholesky(A) in a left expand_dims
    L = cholesky(A, lower=True)
    rewritten = rewrite_graph(solve(L, b), include=("canonicalize", "stabilize"))
    assert_equal_computations([rewritten], [solve_triangular(L, b, lower=True)])


@pytest.mark.parametrize("batched_b", [False, True], ids=["unbatched", "batched"])
def test_paired_triangular_solves_to_cho_solve(batched_b):
    """Test that paired triangular solves from Cholesky get fused into cho_solve."""
    A = matrix("A")
    # A batched b makes Blockwise.make_node wrap the Cholesky factor in a
    # left expand_dims on both solves
    b = tensor("b", shape=(None,) * (2 + batched_b))

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


@pytest.mark.parametrize("batched", [False, True], ids=["unbatched", "batched"])
@pytest.mark.parametrize("b_ndim", [1, 2], ids=lambda x: f"b_ndim={x}")
def test_solve_of_inv_to_matmul(b_ndim, batched):
    X = pt.dmatrix("X")
    # A batched b makes Blockwise.make_node wrap inv(X) in a left expand_dims
    b = pt.tensor("b", shape=(None,) * (b_ndim + batched), dtype="float64")
    out = solve(pt.linalg.inv(X), b, b_ndim=b_ndim)

    # We include 'stabilize' because solve_of_inv_to_matmul is registered there.
    # Note: rewrite_graph includes 'canonicalize' by default.
    rewritten_out = rewrite_graph(out, include=["stabilize"])

    # Verify the rewrite against stabilized 'X @ b' to ensure structural equality.
    # stabilization lowers 'X @ b' (Matmul) to specific BLAS ops (like Dot).
    expected = pt.matvec(X, b) if b_ndim == 1 else X @ b
    expected = rewrite_graph(expected, include=["stabilize"])
    assert_equal_computations([rewritten_out], [expected])


@pytest.mark.parametrize(
    "b_ndim, solve_fn, expected_op, a_batch, b_batch",
    [
        (1, pt.linalg.solve, Solve, 0, 0),
        (2, pt.linalg.solve, Solve, 4, 4),
        (1, lambda T, b: solve_triangular(T, b, lower=True), SolveTriangular, 0, 0),
        # Batched b with core blocks: Blockwise.make_node wraps the block_diag
        # in a left expand_dims, which the rewrite must see through. The
        # per-block solves are then eagerly split into LU factor + solve,
        # since each core block is broadcast against the batched b
        (2, pt.linalg.solve, LUFactor, 0, 4),
    ],
    ids=["vector_b", "matrix_b_batched", "solve_triangular", "expanded_A"],
)
def test_block_diag_solve_pushdown(b_ndim, solve_fn, expected_op, a_batch, b_batch):
    A_shape = (a_batch, 3, 3) if a_batch else (3, 3)
    B_shape = (a_batch, 2, 2) if a_batch else (2, 2)
    if b_ndim == 1:
        b_shape = (b_batch, 5) if b_batch else (5,)
    else:
        b_shape = (b_batch, 5, 4) if b_batch else (5, 4)
    A = pt.tensor("A", shape=A_shape)
    B = pt.tensor("B", shape=B_shape)
    b_var = pt.tensor("b", shape=b_shape)
    # The expanded_A case expects the eager LU split, which runs at specialize
    f = function(
        [A, B, b_var],
        solve_fn(pt.linalg.block_diag(A, B), b_var),
        mode=get_default_mode().including("specialize"),
    )

    ops = [getattr(n.op, "core_op", n.op) for n in f.maker.fgraph.toposort()]
    assert not any(isinstance(op, BlockDiagonal) for op in ops)
    assert any(isinstance(op, expected_op) for op in ops)
    if expected_op is SolveTriangular:
        assert not any(isinstance(op, Solve) for op in ops)

    rng = np.random.default_rng(0)
    A_v, B_v, b_v = (
        rng.normal(size=A_shape),
        rng.normal(size=B_shape),
        rng.normal(size=b_shape),
    )
    if expected_op is SolveTriangular:
        # Make A and B lower-triangular so the per-block solve_triangular is valid.
        A_v = np.tril(A_v) if not a_batch else np.stack([np.tril(a) for a in A_v])
        B_v = np.tril(B_v) if not a_batch else np.stack([np.tril(b) for b in B_v])
    if a_batch:
        expected = np.stack(
            [
                np.linalg.solve(scipy.linalg.block_diag(A_v[i], B_v[i]), b_v[i])
                for i in range(a_batch)
            ]
        )
    else:
        expected = np.linalg.solve(scipy.linalg.block_diag(A_v, B_v), b_v)
    assert_allclose(f(A_v, B_v, b_v), expected, atol=1e-10)


@pytest.mark.parametrize(
    "second_shape", [(2, 5), (None, None)], ids=["non_square", "unknown_shape"]
)
def test_block_diag_solve_pushdown_bails_when_blocks_not_provably_square(second_shape):
    A = pt.matrix("A", shape=(3, 3))
    second = pt.matrix("second", shape=second_shape)
    y = pt.vector("y")
    f = function([A, second, y], pt.linalg.solve(pt.linalg.block_diag(A, second), y))
    assert [
        n
        for n in f.maker.fgraph.toposort()
        if isinstance(getattr(n.op, "core_op", n.op), BlockDiagonal)
    ]


def test_block_diag_solve_pushdown_both_sides_block_diag():
    A1, A2 = pt.matrix("A1", shape=(3, 3)), pt.matrix("A2", shape=(2, 2))
    B1, B2 = pt.matrix("B1", shape=(3, 3)), pt.matrix("B2", shape=(2, 2))
    A = pt.linalg.block_diag(A1, A2)
    B = pt.linalg.block_diag(B1, B2)
    f = function([A1, A2, B1, B2], pt.linalg.solve(A, B))

    # Each Solve must consume the original A_i and B_i directly (not a slice or
    # the materialized A/B block_diag). The output BlockDiagonal that wraps the
    # per-block solves back together is expected and fine.
    block_names = {"A1", "A2", "B1", "B2"}
    solves = [
        n
        for n in f.maker.fgraph.toposort()
        if isinstance(getattr(n.op, "core_op", n.op), Solve)
    ]
    assert len(solves) == 2
    for n in solves:
        assert {n.inputs[0].name, n.inputs[1].name} <= block_names

    rng = np.random.default_rng(0)
    A1_v, A2_v = rng.normal(size=(3, 3)), rng.normal(size=(2, 2))
    B1_v, B2_v = rng.normal(size=(3, 3)), rng.normal(size=(2, 2))
    expected = np.linalg.solve(
        scipy.linalg.block_diag(A1_v, A2_v),
        scipy.linalg.block_diag(B1_v, B2_v),
    )
    assert_allclose(f(A1_v, A2_v, B1_v, B2_v), expected, atol=1e-10)


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

    passes = ("canonicalize", "stabilize", "specialize")
    rewritten = rewrite_graph(X, include=passes)
    expected = rewrite_graph(C / (a[:, None] + b[None, :]), include=passes)

    assert_equal_computations([rewritten], [expected])


def test_orthogonal_solve_to_transpose_matmul():
    n = 5
    rewrites = ("canonicalize", "stabilize", "ShapeOpt")

    Q = pt.dmatrix("Q", shape=(n, n))
    Q_orth = assume(Q, orthogonal=True)
    b = pt.dmatrix("b", shape=(n, 3))
    out = solve(Q_orth, b)
    rewritten = rewrite_graph(out, include=rewrites)
    expected = rewrite_graph(Q_orth.mT @ b, include=rewrites)
    assert_equal_computations([rewritten], [expected])


def test_psd_fact_does_not_override_tridiagonal_solve():
    """A positive-definite fact must not replace an O(n) tridiagonal solve."""
    A = matrix("A")
    A_pd = assume(A, positive_definite=True)
    b = matrix("b")
    out = pt.linalg.solve(A_pd, b, assume_a="tridiagonal", b_ndim=2)

    result = RewriteTester([A, b], [out], include=["stabilize"])
    result.assert_graph(out)


def test_tridiagonal_fact_does_not_override_declared_pos_solve():
    A = matrix("A")
    A_banded = assume(A, tridiagonal=True)
    b = matrix("b")
    out = pt.linalg.solve(A_banded, b, assume_a="pos", b_ndim=2)

    result = RewriteTester([A, b], [out], include=["stabilize"])
    L = cholesky(A_banded)
    Li_b = solve_triangular(L, b, lower=True, b_ndim=2)
    result.assert_graph(solve_triangular(L.mT, Li_b, lower=False, b_ndim=2))


@pytest.mark.parametrize("declared", ["sym", "her"])
def test_symmetric_fact_leaves_sym_and_her_solves_alone(declared):
    A = matrix("A")
    A_sym = assume(A, symmetric=True)
    b = matrix("b")
    out = solve(A_sym, b, assume_a=declared)

    result = RewriteTester([A, b], [out], include=["stabilize"])
    result.assert_graph(out)


@pytest.mark.parametrize("transposed", [False, True])
def test_bidiagonal_cholesky_factor_solves_as_tridiagonal(transposed):
    A = matrix("A")
    L = cholesky(assume(A, tridiagonal=True, positive_definite=True))
    factor = L.mT if transposed else L
    b = matrix("b")
    out = solve(factor, b)

    result = RewriteTester([A, b], [out], include=["stabilize"])
    result.assert_graph(solve(factor, b, assume_a="tridiagonal"))


def test_bidiagonal_solve_prefers_tridiagonal_over_triangular():
    """A lower bidiagonal matrix is both lower triangular and tridiagonal, and the
    banded solve is O(n) where the triangular one is O(n^2)."""
    A = matrix("A")
    A_bidiagonal = assume(A, lower_triangular=True, tridiagonal=True)
    b = matrix("b")
    out = solve(A_bidiagonal, b)

    result = RewriteTester([A, b], [out], include=["stabilize"])
    result.assert_graph(solve(A_bidiagonal, b, assume_a="tridiagonal"))


@pytest.mark.parametrize("declared", ["gen", "sym", "her"])
def test_solve_upgrades_to_tridiagonal(declared):
    """With both facts, the band outranks positive-definiteness: a tridiagonal
    solve is O(n) while the Cholesky split is O(n^3)."""
    A = matrix("A")
    A_banded_pd = assume(A, tridiagonal=True, positive_definite=True)
    b = matrix("b")
    out = solve(A_banded_pd, b, assume_a=declared)

    result = RewriteTester([A, b], [out], include=["stabilize"])
    result.assert_graph(solve(A_banded_pd, b, assume_a="tridiagonal"))

    rng = np.random.default_rng(5)
    off_diagonal = rng.uniform(0.1, 0.4, 5)
    A_val = (
        2 * np.eye(6) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    ).astype(config.floatX)
    b_val = rng.normal(size=(6, 3)).astype(config.floatX)
    result.assert_eval(A_val, b_val, rtol=1e-7 if config.floatX == "float64" else 1e-4)


@pytest.mark.parametrize("batched_b", [False, True], ids=["vector_b", "batched_b"])
@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
@pytest.mark.parametrize("b_ndim", [1, 2], ids=["vector", "column"])
def test_cho_solve_quadratic_form_to_whitened(lower, batched_b, b_ndim):
    c = pt.dmatrix("c")
    b = pt.tensor("b", shape=(None,) * (1 + batched_b), dtype="float64")

    rhs = b if b_ndim == 1 else b[..., None]
    quad = (rhs * cho_solve((c, lower), rhs, b_ndim=b_ndim)).sum(axis=-b_ndim)
    if b_ndim == 2:
        quad = quad.squeeze(-1)

    result = RewriteTester(
        [c, b], [quad], include=["canonicalize", "stabilize", "specialize"]
    )

    rng = np.random.default_rng(81)
    A = rng.normal(size=(4, 4))
    A = A @ A.T + 4 * np.eye(4)
    c_val = np.linalg.cholesky(A)
    c_val = c_val if lower else c_val.T
    b_val = rng.normal(size=(3, 4) if batched_b else (4,))
    result.assert_eval(c_val, b_val)

    toposort = result.rewr_fg.toposort()
    core_ops = [getattr(node.op, "core_op", node.op) for node in toposort]
    assert not any(isinstance(op, CholeskySolve) for op in core_ops)
    assert sum(isinstance(op, SolveTriangular) for op in core_ops) == 1


@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
def test_triangular_solve_pair_quadratic_form_to_whitened(lower):
    M = pt.dmatrix("M")
    b = pt.dvector("b")
    inner = solve_triangular(M, b, lower=lower, b_ndim=1)
    quad = (b * solve_triangular(M.mT, inner, lower=not lower, b_ndim=1)).sum()

    result = RewriteTester(
        [M, b],
        [quad],
        include=None,
        custom_rewrite=cho_solve_quadratic_form_to_whitened,
    )
    result.assert_graph((inner**2).sum())


@pytest.mark.parametrize(
    "outer_factor, outer_lower, outer_unit_diagonal",
    [
        pytest.param(lambda M, N: N.mT, False, False, id="different_factor"),
        pytest.param(lambda M, N: M.mT, True, False, id="same_triangle"),
        pytest.param(lambda M, N: M, False, False, id="not_transposed"),
        pytest.param(lambda M, N: M.mT, False, True, id="unit_diagonal_mismatch"),
    ],
)
def test_triangular_solve_pair_quadratic_form_not_whitened(
    outer_factor, outer_lower, outer_unit_diagonal
):
    M = pt.dmatrix("M")
    N = pt.dmatrix("N")
    b = pt.dvector("b")
    inner = solve_triangular(M, b, lower=True, b_ndim=1)
    x = solve_triangular(
        outer_factor(M, N),
        inner,
        lower=outer_lower,
        unit_diagonal=outer_unit_diagonal,
        b_ndim=1,
    )
    quad = (b * x).sum()

    result = RewriteTester(
        [M, N, b],
        [quad],
        include=None,
        custom_rewrite=cho_solve_quadratic_form_to_whitened,
    )
    result.assert_graph(quad)


def test_cho_solve_quadratic_form_keeps_sum_dtype():
    c = pt.matrix("c", dtype="float32")
    b = pt.vector("b", dtype="float32")

    quad = (b * cho_solve((c, True), b, b_ndim=1)).sum(dtype="float64")
    rewritten = rewrite_graph(quad, include=("canonicalize", "stabilize", "specialize"))

    expected = rewrite_graph(
        (solve_triangular(c, b, lower=True, b_ndim=1) ** 2).sum(dtype="float64"),
        include=("canonicalize", "stabilize", "specialize"),
    )
    assert_equal_computations([rewritten], [expected])


def test_cho_solve_quadratic_form_with_reused_solve_not_whitened():
    """A solve result with another consumer is left alone, so it is never computed twice."""
    c = pt.dmatrix("c")
    b = pt.dvector("b")
    x = cho_solve((c, True), b, b_ndim=1)
    quad = (b * x).sum()

    result = RewriteTester([c, b], [quad, x], include=["specialize"])
    result.assert_graph(quad, x)


@pytest.mark.parametrize(
    "a_batch_shape, b_batch_shape",
    [
        pytest.param((), (), id="unbatched"),
        pytest.param((), (3,), id="shared_factor"),
        pytest.param((2, 3), (2, 3), id="two_batch_axes"),
    ],
)
def test_vector_quadratic_form_to_whitened(a_batch_shape, b_batch_shape):
    A_raw = pt.tensor("A", shape=(None,) * (len(a_batch_shape) + 2), dtype="float64")
    b = pt.tensor("b", shape=(None,) * (len(b_batch_shape) + 1), dtype="float64")
    A = assume(A_raw, positive_definite=True)
    quad = (
        pt.squeeze(b[..., None, :] @ pt.linalg.inv(A) @ b[..., :, None], axis=(-2, -1))
        if b_batch_shape
        else b @ pt.linalg.inv(A) @ b
    )

    # RewriteTester evaluates with the Python linker, so skip C-only reduction fusion.
    result = RewriteTester(
        [A_raw, b], [quad], include="o3", exclude=["local_careduce_fusion"]
    )

    rng = np.random.default_rng(81)
    M = rng.normal(size=(*a_batch_shape, 4, 4))
    A_val = M @ M.swapaxes(-2, -1) + 4 * np.eye(4)
    result.assert_eval(A_val, rng.normal(size=(*b_batch_shape, 4)))

    core_ops = [
        getattr(node.op, "core_op", node.op) for node in result.rewr_fg.toposort()
    ]
    assert not any(isinstance(op, CholeskySolve | Dot) for op in core_ops)
    assert sum(isinstance(op, SolveTriangular) for op in core_ops) == 1


@pytest.mark.parametrize("wrap", [lambda t: t, pt.log], ids=["plain", "under_log"])
@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
@pytest.mark.parametrize(
    "b_ndim, build",
    [
        pytest.param(1, lambda x, b: pt.dot(x, b), id="vector_dot"),
        pytest.param(1, lambda x, b: b @ x, id="vector_matmul"),
        pytest.param(2, lambda x, b: x.mT @ b, id="column_matmul"),
        pytest.param(2, lambda x, b: pt.dot(b.mT, x), id="column_dot"),
        pytest.param(
            2,
            lambda x, b: pt.expand_dims(pt.squeeze(x, axis=-1), axis=0) @ b,
            id="squeeze_expand",
        ),
    ],
)
def test_vector_quadratic_form_dot_to_whitened(lower, wrap, b_ndim, build):
    c = pt.dmatrix("c")
    b = pt.dvector("b")
    rhs = b if b_ndim == 1 else b[:, None]
    x = cho_solve((c, lower), rhs, b_ndim=b_ndim)
    out = wrap(pt.squeeze(build(x, rhs)))

    result = RewriteTester([c, b], [out], include=["specialize"])
    whitened = solve_triangular(
        c, rhs, lower=lower, trans=0 if lower else 1, b_ndim=b_ndim
    )
    expected = (whitened**2).sum(axis=-b_ndim)
    result.assert_graph(
        rewrite_graph(wrap(pt.squeeze(expected)), include=["specialize"])
    )

    rng = np.random.default_rng(81)
    M = rng.normal(size=(4, 4))
    c_val = np.linalg.cholesky(M @ M.T + 4 * np.eye(4))
    c_val = c_val if lower else c_val.T
    result.assert_eval(c_val, rng.normal(size=4) ** 2)


@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
@pytest.mark.parametrize("b_ndim", [1, 2], ids=["vector", "column"])
@pytest.mark.parametrize(
    "c_batch_shape, b_batch_shape",
    [
        pytest.param((), (3,), id="shared_factor"),
        pytest.param((2, 3), (2, 3), id="two_batch_axes"),
        pytest.param((2, 1), (3,), id="broadcast_batches"),
    ],
)
def test_batched_quadratic_form_matmul_to_whitened(
    lower, b_ndim, c_batch_shape, b_batch_shape
):
    c = pt.tensor(
        "c",
        shape=(*(1 if n == 1 else None for n in c_batch_shape), None, None),
        dtype="float64",
    )
    b = pt.tensor("b", shape=(None,) * (len(b_batch_shape) + 1), dtype="float64")
    rhs = b if b_ndim == 1 else b[..., None]
    x = cho_solve((c, lower), rhs, b_ndim=b_ndim)
    out = pt.vecdot(b, x) if b_ndim == 1 else pt.squeeze(x.mT @ rhs, axis=(-2, -1))
    passes = ["canonicalize", "stabilize", "specialize"]
    result = RewriteTester([c, b], [out], include=passes)

    whitened = solve_triangular(
        c, rhs, lower=lower, trans=0 if lower else 1, b_ndim=b_ndim
    )
    expected = (whitened**2).sum(axis=-b_ndim)
    if b_ndim == 2:
        expected = expected.squeeze(-1)
    result.assert_graph(rewrite_graph(expected, include=passes))

    rng = np.random.default_rng(81)
    M = rng.normal(size=(*c_batch_shape, 4, 4)).astype(c.dtype)
    c_val = np.linalg.cholesky(M @ M.swapaxes(-2, -1) + 4 * np.eye(4))
    if not lower:
        c_val = c_val.swapaxes(-2, -1)
    b_val = rng.normal(size=(*b_batch_shape, 4)).astype(b.dtype)
    result.assert_eval(c_val, b_val)


@pytest.mark.parametrize(
    "x_order, b_order, whiten",
    [
        pytest.param((1, 0, 2, "x"), (1, 0, "x", 2), True, id="permuted_batches"),
        pytest.param((0, 1, 2, "x"), (1, 0, "x", 2), False, id="mismatched_batches"),
        pytest.param(
            (0, 2, 1, "x"), (0, 2, "x", 1), False, id="wrong_contraction_axis"
        ),
    ],
)
def test_batched_quadratic_form_matmul_batch_axes(x_order, b_order, whiten):
    c = pt.tensor4("c")
    b = pt.tensor3("b")
    x = cho_solve((c, True), b, b_ndim=1)
    out = pt.squeeze(b.dimshuffle(b_order) @ x.dimshuffle(x_order), axis=(-2, -1))
    result = RewriteTester(
        [c, b], [out], include=None, custom_rewrite=cho_solve_quadratic_form_to_whitened
    )
    expected = (
        (solve_triangular(c, b, lower=True, b_ndim=1) ** 2).sum(axis=-1).mT
        if whiten
        else out
    )
    result.assert_graph(expected)


@pytest.mark.parametrize("axis", [-2, 0], ids=["vector_axis", "batch_axis"])
def test_batched_quadratic_form_sum_axis(axis):
    c = pt.dtensor4("c")
    b = pt.tensor("b", shape=(None, None, None, 1), dtype="float64")
    out = (b * cho_solve((c, True), b, b_ndim=2)).sum(axis=axis)
    result = RewriteTester(
        [c, b], [out], include=None, custom_rewrite=cho_solve_quadratic_form_to_whitened
    )
    expected = (
        (solve_triangular(c, b, lower=True, b_ndim=2) ** 2).sum(axis=axis)
        if axis == -2
        else out
    )
    result.assert_graph(expected)


@pytest.mark.parametrize("b_ndim", [1, 2], ids=["vector", "column"])
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda x, col, d_col: [pt.squeeze(x.mT @ d_col)], id="bilinear"),
        pytest.param(lambda x, col, d_col: [x.mT @ col], id="matrix_result"),
        pytest.param(
            lambda x, col, d_col: [pt.squeeze(x.mT @ col), x], id="reused_solve"
        ),
    ],
)
def test_vector_quadratic_form_dot_not_whitened(build, b_ndim):
    c = pt.dmatrix("c")
    b = pt.dvector("b")
    d = pt.dvector("d")
    col = b[:, None]
    x = cho_solve((c, True), b if b_ndim == 1 else col, b_ndim=b_ndim)
    outputs = build(x[:, None] if b_ndim == 1 else x, col, d[:, None])

    result = RewriteTester([c, b, d], outputs, include=["specialize"])

    rng = np.random.default_rng(81)
    M = rng.normal(size=(4, 4))
    c_val = np.linalg.cholesky(M @ M.T + 4 * np.eye(4))
    result.assert_eval(c_val, rng.normal(size=4), rng.normal(size=4))

    core_ops = [
        getattr(node.op, "core_op", node.op) for node in result.rewr_fg.toposort()
    ]
    assert any(isinstance(op, CholeskySolve) for op in core_ops)


def test_vector_quadratic_form_length_one():
    A_raw = pt.dmatrix("A")
    b = pt.dvector("b")
    A = assume(A_raw, positive_definite=True)
    quad = b @ pt.linalg.inv(A) @ b

    result = RewriteTester(
        [A_raw, b], [quad], include=["canonicalize", "stabilize", "specialize"]
    )
    result.assert_eval(np.array([[2.5]]), np.array([3.0]))
