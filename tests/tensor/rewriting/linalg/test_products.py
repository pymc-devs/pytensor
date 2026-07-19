import numpy as np
import pytest
import scipy.linalg
from numpy.testing import assert_allclose

from pytensor import function
from pytensor import tensor as pt
from pytensor.assumptions.specify import assume
from pytensor.configdefaults import config
from pytensor.graph import FunctionGraph, ancestors
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.tensor.basic import ExtractDiag, alloc_diag
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.products import KroneckerProduct
from pytensor.tensor.math import Dot
from pytensor.tensor.rewriting.subtensor_lift import extract_diag_lift_pass
from tests.unittest_tools import RewriteTester, assert_equal_computations


def test_nested_blockdiag_fusion():
    x = pt.tensor("x", shape=(3, 3))
    y = pt.tensor("y", shape=(3, 3))
    z = pt.tensor("z", shape=(3, 3))

    inner = BlockDiagonal(2)(x, y)
    outer = BlockDiagonal(2)(inner, z)

    initial_count = sum(
        1
        for node in ancestors([outer])
        if getattr(node, "owner", None) and isinstance(node.owner.op, BlockDiagonal)
    )
    assert initial_count == 2, "Setup failed: expected 2 nested BlockDiagonal ops"

    fgraph = FunctionGraph(inputs=[x, y, z], outputs=[outer])
    rewrite_graph(fgraph, include=("fast_run", "blockdiag_fusion"))

    fused_nodes = [
        node for node in fgraph.toposort() if isinstance(node.op, BlockDiagonal)
    ]
    assert len(fused_nodes) == 1, "Nested BlockDiagonal ops were not fused"

    fused_op = fused_nodes[0].op
    assert fused_op.n_inputs == 3, f"Expected n_inputs=3, got {fused_op.n_inputs}"

    out_shape = fgraph.outputs[0].type.shape
    assert out_shape == (9, 9), f"Unexpected fused output shape: {out_shape}"


def test_deeply_nested_blockdiag_fusion():
    x = pt.tensor("x", shape=(3, 3))
    y = pt.tensor("y", shape=(3, 3))
    z = pt.tensor("z", shape=(3, 3))
    w = pt.tensor("w", shape=(3, 3))

    inner1 = BlockDiagonal(2)(x, y)
    inner2 = BlockDiagonal(2)(inner1, z)
    outer = BlockDiagonal(2)(inner2, w)

    fgraph = FunctionGraph(inputs=[x, y, z, w], outputs=[outer])
    rewrite_graph(fgraph, include=("fast_run", "blockdiag_fusion"))

    fused_block_diag_nodes = [
        node for node in fgraph.apply_nodes if isinstance(node.op, BlockDiagonal)
    ]
    assert len(fused_block_diag_nodes) == 1, (
        f"Expected 1 fused BlockDiagonal, got {len(fused_block_diag_nodes)}"
    )

    fused_block_diag_op = fused_block_diag_nodes[0].op

    assert fused_block_diag_op.n_inputs == 4, (
        f"Expected n_inputs=4 after fusion, got {fused_block_diag_op.n_inputs}"
    )

    out_shape = fgraph.outputs[0].type.shape
    expected_shape = (12, 12)  # 4 blocks of (3x3)
    assert out_shape == expected_shape, (
        f"Unexpected fused output shape: expected {expected_shape}, got {out_shape}"
    )


def test_diag_of_blockdiag():
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


def test_det_of_blockdiag_():
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


def test_slogdet_of_blockdiag():
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


def test_diag_of_kronecker():
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


def test_det_of_kronecker():
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


@pytest.mark.parametrize(
    "make_diag",
    [
        pytest.param(lambda d: pt.diag(d), id="alloc_diag"),
        pytest.param(lambda d: pt.eye(5) * d, id="eye_mul"),
    ],
)
def test_expm_of_diag(make_diag):
    d = pt.dvector("d", shape=(5,))
    D = make_diag(d)
    out = pt.linalg.expm(D)

    passes = ("canonicalize", "stabilize", "specialize")
    rewritten = rewrite_graph(out, include=passes)
    expected = rewrite_graph(alloc_diag(pt.exp(d), axis1=-2, axis2=-1), include=passes)
    assert_equal_computations([rewritten], [expected])


def test_kron_of_diagonal_to_diagonal():
    da = pt.tensor("da", shape=(3, 3))
    db = pt.tensor("db", shape=(4, 4))
    A = assume(da, diagonal=True)
    B = assume(db, diagonal=True)

    out = pt.linalg.kron(A, B)
    f = function([da, db], out, mode="FAST_RUN")

    nodes_batch = f.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, KroneckerProduct) for node in nodes_batch)

    rng = np.random.default_rng()

    da_test = np.diag(rng.normal(size=(3,)))
    db_test = np.diag(rng.normal(size=(4,)))
    expected = np.kron(da_test, db_test)
    assert_allclose(f(da_test, db_test), expected)

    # Batched case
    da_batch = pt.tensor("da_batch", shape=(2, 3, 3))
    db_batch = pt.tensor("db_batch", shape=(2, 4, 4))
    A_batch = assume(da_batch, diagonal=True)
    B_batch = assume(db_batch, diagonal=True)

    signature = "(m,m),(n,n)->(mn,mn)"
    kron_batched = pt.vectorize(lambda a, b: pt.linalg.kron(a, b), signature=signature)
    out_batch = kron_batched(A_batch, B_batch)
    f_batch = function([da_batch, db_batch], out_batch)

    nodes_batch = f_batch.maker.fgraph.apply_nodes
    assert not any(isinstance(node.op, KroneckerProduct) for node in nodes_batch)

    a_diags = np.random.normal(size=(2, 3))
    b_diags = np.random.normal(size=(2, 4))
    vec_diag = np.vectorize(np.diag, signature="(n)->(n,n)")
    da_batch_val = vec_diag(a_diags)
    db_batch_val = vec_diag(b_diags)

    expected_batch = np.vectorize(np.kron, signature=signature)(
        da_batch_val, db_batch_val
    )
    assert_allclose(f_batch(da_batch_val, db_batch_val), expected_batch)


def test_orthogonal_dot_transpose_to_eye():
    n = 5
    rewrites = ("canonicalize", "specialize", "ShapeOpt")

    # 2D: X @ X.T -> eye
    x = pt.dmatrix("x", shape=(n, n))
    x_orth = assume(x, orthogonal=True)
    out_xxt = pt.dot(x_orth, x_orth.T)
    rewritten_xxt = rewrite_graph(out_xxt, include=rewrites)
    expected = pt.as_tensor(np.eye(n, dtype=config.floatX))
    assert_equal_computations([rewritten_xxt], [expected])

    # Batched: X @ X.T -> broadcast_to(eye, batch_shape)
    x_batch = pt.dtensor3("x_batch", shape=(3, n, n))
    x_batch_orth = assume(x_batch, orthogonal=True)
    out_batch = x_batch_orth @ pt.moveaxis(x_batch_orth, -1, -2)
    rewritten_batch = rewrite_graph(out_batch, include=rewrites)

    n64 = np.array(n, dtype="int64")
    b64 = np.array(3, dtype="int64")
    expected_batch = pt.alloc(np.eye(n, dtype=config.floatX), b64, n64, n64)

    assert_equal_computations([rewritten_batch], [expected_batch])


class TestSelectionDotToIndexing:
    n = 6
    idx_val = np.array([0, 2, 5, 1])

    PATTERNS = [
        ("col_gather", lambda S, d: d @ S, lambda n, k: (3, n)),
        ("row_gather", lambda S, d: S.T @ d, lambda n, k: (n, 4)),
        ("row_scatter", lambda S, d: S @ d, lambda n, k: (k, 4)),
        ("col_scatter", lambda S, d: d @ S.T, lambda n, k: (3, k)),
    ]
    SETUPS = [
        ("symbolic", False),
        ("symbolic", True),
        ("constant", False),
        ("opaque", False),
    ]

    @pytest.mark.parametrize(
        "source, batched", SETUPS, ids=["sym", "sym_batched", "const", "opaque"]
    )
    @pytest.mark.parametrize(
        "build, core_shape",
        [(b, c) for _, b, c in PATTERNS],
        ids=[name for name, *_ in PATTERNS],
    )
    def test_rewrites_to_indexing(self, build, core_shape, source, batched):
        n, idx = self.n, self.idx_val
        k = len(idx)
        S_np = np.eye(n)[:, idx]

        if source == "constant":
            S, extra_in, extra_val = pt.eye(n)[:, idx], [], []
        elif source == "opaque":
            s = pt.matrix("s", shape=(n, k))
            S, extra_in, extra_val = assume(s, selection=True), [s], [S_np]
        else:
            iv = pt.lvector("idx")
            S, extra_in, extra_val = pt.eye(n)[:, iv], [iv], [idx]

        shape = (2, *core_shape(n, k)) if batched else core_shape(n, k)
        d = pt.tensor("d", shape=shape)

        f = function([d, *extra_in], build(S, d), mode="FAST_RUN")
        assert not any(
            isinstance(node.op, Dot)
            or (isinstance(node.op, Blockwise) and isinstance(node.op.core_op, Dot))
            for node in f.maker.fgraph.apply_nodes
        )

        dv = np.random.default_rng(0).normal(size=shape)
        assert_allclose(f(dv, *extra_val), build(S_np, dv))

    @pytest.mark.parametrize(
        "idx", [idx_val, np.array([1, 1, 3])], ids=["distinct", "duplicate"]
    )
    @pytest.mark.parametrize("middle", ["diag_q", "general_a"])
    def test_congruence(self, middle, idx):
        # R @ M @ R.T collapses to a pure scatter. The duplicate-index case guards
        # that inc_subtensor's accumulate reproduces the matmul for repeated columns.
        n, k = self.n, len(idx)
        iv = pt.lvector("idx")
        R = pt.eye(n)[:, iv]
        rng = np.random.default_rng(0)
        if middle == "diag_q":
            m = pt.vector("m")
            mv = rng.normal(size=k) ** 2
            M, M_np = pt.diag(m), np.diag(mv)
        else:
            m = pt.matrix("m")
            mv = rng.normal(size=(k, k))
            M, M_np = m, mv

        f = function([m, iv], R @ M @ R.T, mode="FAST_RUN")
        assert not any(
            isinstance(node.op, Dot)
            or (isinstance(node.op, Blockwise) and isinstance(node.op.core_op, Dot))
            for node in f.maker.fgraph.apply_nodes
        )
        R_np = np.eye(n)[:, idx]
        assert_allclose(f(mv, idx), R_np @ M_np @ R_np.T)


def test_gather_matches_upcast_matmul_dtype():
    x = pt.matrix("x", dtype="float32")
    idx = pt.lvector("idx")
    out = x @ pt.eye(5)[:, idx]
    assert out.type.dtype == "float64"

    f = function([x, idx], out, mode="FAST_RUN")
    assert not any(
        isinstance(node.op, Dot)
        or (isinstance(node.op, Blockwise) and isinstance(node.op.core_op, Dot))
        for node in f.maker.fgraph.apply_nodes
    )
    assert f.maker.fgraph.outputs[0].type.dtype == "float64"
    xv = np.random.default_rng(0).normal(size=(3, 5)).astype("float32")
    iv = np.array([0, 2, 4, 1])
    assert_allclose(f(xv, iv), xv @ np.eye(5)[:, iv])


def test_permutation_rides_selection_rewrite():
    n = 5
    pv = np.eye(n)[:, [2, 0, 4, 1, 3]]
    rng = np.random.default_rng(0)
    p = pt.matrix("p", shape=(n, n))
    P = assume(p, permutation=True)
    x = pt.matrix("x")

    def has_matmul(f):
        return any(
            isinstance(node.op, Dot)
            or (isinstance(node.op, Blockwise) and isinstance(node.op.core_op, Dot))
            for node in f.maker.fgraph.apply_nodes
        )

    f_left = function([p, x], P @ x, mode="FAST_RUN")
    assert not has_matmul(f_left)
    xv = rng.normal(size=(n, 3))
    assert_allclose(f_left(pv, xv), pv @ xv)

    f_right = function([p, x], x @ P, mode="FAST_RUN")
    assert not has_matmul(f_right)
    yv = rng.normal(size=(3, n))
    assert_allclose(f_right(pv, yv), yv @ pv)


@pytest.mark.parametrize(
    "perm", [[0, 1, 2, 3], [1, 0, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]], ids=str
)
def test_det_of_permutation(perm):
    from pytensor.tensor.linalg.summary import Det

    pv = np.eye(4)[:, perm]
    p = pt.matrix("p", shape=(4, 4))
    P = assume(p, permutation=True)
    f = function([p], pt.linalg.det(P), mode="FAST_RUN")
    assert not any(isinstance(node.op, Det) for node in f.maker.fgraph.apply_nodes)
    got = f(pv)
    assert abs(got) == 1.0  # exact, not a floating-point approximation
    assert_allclose(got, np.linalg.det(pv))


def test_det_of_non_permutation_selection_is_not_rewritten():
    from pytensor.tensor.linalg.summary import Det

    idx = pt.lvector("idx")
    S = pt.eye(4)[:, idx]
    f = function([idx], pt.linalg.det(S), mode="FAST_RUN")
    assert any(isinstance(node.op, Det) for node in f.maker.fgraph.apply_nodes)
    assert_allclose(f(np.array([0, 1, 1, 3])), 0.0)  # duplicate column -> singular


class TestDiagOfDot:
    """``diag(A @ B)`` is lowered to a paired-arange gather by
    ``local_extract_diag_lift`` and folded into the operands by
    ``local_advanced_subtensor_of_dot``, so it needs the specialize pass.
    """

    rewrite_kw = dict(include=("ShapeOpt", "canonicalize", "specialize"))

    @pytest.mark.parametrize("batch", [(), (3,)], ids=["2d", "batched"])
    @pytest.mark.parametrize("offset", [0, 1, -1, 2, -2], ids=str)
    def test_diag_of_dot(self, batch, offset):
        rng = np.random.default_rng(0)
        n, contracted = 6, 4
        A = pt.tensor("A", shape=(*batch, n, contracted))
        B = pt.tensor("B", shape=(*batch, contracted, n))

        out = pt.diagonal(A @ B, offset=offset, axis1=-2, axis2=-1)
        result = RewriteTester([A, B], [out], **self.rewrite_kw)

        # The main diagonal has a stable closed form; shifted diagonals only differ
        # by slicing, whose exact canonical shape is verified against the oracle below.
        if offset == 0:
            result.assert_graph((A * B.mT).sum(-1))

        result.assert_eval(
            rng.standard_normal((*batch, n, contracted)),
            rng.standard_normal((*batch, contracted, n)),
        )

    @pytest.mark.parametrize("offset", [9, -9], ids=str)
    def test_out_of_range_offset(self, offset):
        """An offset past the matrix gives an empty diagonal, so only values are checked."""
        rng = np.random.default_rng(1)
        A = pt.matrix("A", shape=(6, 6))
        B = pt.matrix("B", shape=(6, 6))
        result = RewriteTester(
            [A, B], [pt.diagonal(A @ B, offset=offset)], **self.rewrite_kw
        )
        result.assert_eval(rng.standard_normal((6, 6)), rng.standard_normal((6, 6)))

    def test_rectangular_product(self):
        """A non-square product exercises the trim to the shorter side."""
        rng = np.random.default_rng(2)
        A = pt.matrix("A", shape=(6, 4))
        B = pt.matrix("B", shape=(4, 3))
        result = RewriteTester([A, B], [pt.diag(A @ B)], **self.rewrite_kw)
        result.assert_eval(rng.standard_normal((6, 4)), rng.standard_normal((4, 3)))

    @pytest.mark.parametrize("offset", [0, 2, -2], ids=str)
    def test_dynamic_shapes(self, offset):
        """With no static shapes the diagonal length stays symbolic, so the size gate
        rests on the paired ``arange``. An offset diagonal indexes with ``arange(d) +
        k``, whose entry count is the ``arange``'s even though the shift can alias
        entries onto each other -- the gate has to read it as a size, not a uniqueness.
        """
        rng = np.random.default_rng(3)
        A = pt.matrix("A")
        B = pt.matrix("B")
        result = RewriteTester(
            [A, B], [pt.diagonal(A @ B, offset=offset)], **self.rewrite_kw
        )
        assert not any(
            isinstance(node.op, Dot | ExtractDiag)
            for node in result.rewr_fg.apply_nodes
        )
        result.assert_eval(rng.standard_normal((6, 4)), rng.standard_normal((4, 6)))

    def test_trace_of_dot(self):
        """``trace`` is ``diagonal(...).sum()``, so it folds through the same path."""
        rng = np.random.default_rng(4)
        A = pt.matrix("A", shape=(6, 4))
        B = pt.matrix("B", shape=(4, 6))
        result = RewriteTester([A, B], [pt.trace(A @ B)], **self.rewrite_kw)
        assert not any(isinstance(node.op, Dot) for node in result.rewr_fg.apply_nodes)
        result.assert_eval(rng.standard_normal((6, 4)), rng.standard_normal((4, 6)))

    def test_diag_of_einsum(self):
        """``einsum`` only inlines to a ``Dot`` during specialize, so the fold has
        to be reachable there and not only in canonicalize."""
        rng = np.random.default_rng(5)
        A = pt.matrix("A", shape=(6, 4))
        B = pt.matrix("B", shape=(4, 6))
        out = pt.diagonal(pt.einsum("ik,kj->ij", A, B))
        result = RewriteTester([A, B], [out], **self.rewrite_kw)
        assert not any(isinstance(node.op, Dot) for node in result.rewr_fg.apply_nodes)
        result.assert_eval(rng.standard_normal((6, 4)), rng.standard_normal((4, 6)))

    def test_single_client_guard(self):
        """The product also feeds ``M.sum()`` so it is materialized regardless; the
        rewrite must not fire and add a redundant elementwise path."""
        A = pt.matrix("A", shape=(5, 5))
        B = pt.matrix("B", shape=(5, 5))
        M = pt.dot(A, B)
        result = RewriteTester(
            [A, B],
            [pt.diag(M) + M.sum()],
            include=None,
            custom_rewrite=extract_diag_lift_pass,
        )
        result.assert_graph(pt.diag(M) + M.sum())

    def test_wrong_axis_guard(self):
        """A diagonal over the batch/row axes, not the contracted axes, has no
        closed form in the operands and must be left alone."""
        A = pt.tensor("A", shape=(4, 5, 5))
        B = pt.tensor("B", shape=(4, 5, 5))
        out = pt.diagonal(A @ B, axis1=0, axis2=1)
        result = RewriteTester([A, B], [out], **self.rewrite_kw)
        assert any(
            isinstance(node.op, ExtractDiag) for node in result.rewr_fg.apply_nodes
        )

    def test_dtype_preserved(self):
        """``sum`` upcasts an int32 accumulator to int64, so the rewrite must cast back."""
        rng = np.random.default_rng(6)
        A = pt.matrix("A", shape=(6, 4), dtype="int32")
        B = pt.matrix("B", shape=(4, 6), dtype="int32")
        result = RewriteTester([A, B], [pt.diag(A @ B)], **self.rewrite_kw)
        result.assert_graph((A * B.mT).sum(-1).astype("int32"))
        result.assert_eval(
            rng.integers(0, 5, (6, 4)).astype("int32"),
            rng.integers(0, 5, (4, 6)).astype("int32"),
        )
