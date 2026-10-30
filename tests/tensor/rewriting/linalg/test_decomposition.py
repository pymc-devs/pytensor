from functools import partial

import numpy as np
import pytest
import scipy.linalg
from numpy.testing import assert_allclose

import pytensor
from pytensor import tensor as pt
from pytensor.assumptions.specify import assume
from pytensor.compile import get_default_mode
from pytensor.compile.mode import get_mode
from pytensor.configdefaults import config
from pytensor.graph import ancestors
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.tensor import swapaxes
from pytensor.tensor.blockwise import Blockwise, BlockwiseWithCoreShape
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky, cholesky
from pytensor.tensor.linalg.decomposition.eigen import Eigh, eigh
from pytensor.tensor.linalg.decomposition.lu import lu, lu_factor
from pytensor.tensor.linalg.decomposition.qr import qr
from pytensor.tensor.linalg.decomposition.schur import QZ, Schur, qz, schur
from pytensor.tensor.linalg.decomposition.svd import SVD, svd
from pytensor.tensor.math import dot, matmul
from pytensor.tensor.type import tensor
from tests.unittest_tools import RewriteTester, assert_equal_computations


@pytest.mark.parametrize("tag", ("lower", "upper", None))
@pytest.mark.parametrize("cholesky_form", ("lower", "upper"))
@pytest.mark.parametrize("product", ("lower", "upper", None))
@pytest.mark.parametrize("op", (dot, matmul))
def test_cholesky_ldotlt(tag, cholesky_form, product, op):
    transform_removes_chol = tag is not None and product == tag
    transform_transposes = transform_removes_chol and cholesky_form != tag

    ndim = 2 if op == dot else 3
    A = tensor("L", shape=(None,) * ndim)
    if tag == "lower":
        A_with_assumption = assume(A, lower_triangular=True)
    elif tag == "upper":
        A_with_assumption = assume(A, upper_triangular=True)
    else:
        A_with_assumption = A

    if product == "lower":
        M = op(A_with_assumption, swapaxes(A_with_assumption, -1, -2))
    elif product == "upper":
        M = op(swapaxes(A_with_assumption, -1, -2), A_with_assumption)
    else:
        M = A_with_assumption

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


def test_cholesky_ldotlt_through_expand_dims():
    # The batched product keeps the expanded Blockwise(Cholesky) from being
    # eagerly unbatched, so only the look-through peel can reach the Dot
    L = tensor("L", shape=(None, None, None))
    L_assumed = assume(L, lower_triangular=True)
    M = pt.expand_dims(matmul(L_assumed, swapaxes(L_assumed, -1, -2)), 0)
    C = cholesky(M, lower=True)

    rewritten = rewrite_graph(C, include=("canonicalize", "stabilize"))
    assert_equal_computations([rewritten], [pt.expand_dims(L, 0)])


def test_svd_uv_merge():
    a = pt.matrix("a")
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


@pytest.mark.parametrize("expanded_node", ["compute_uv", "no_compute_uv"])
def test_svd_uv_merge_through_expand_dims(expanded_node):
    # The compute_uv=False SVD should reuse s from the compute_uv=True SVD of
    # the same matrix (whose u is actually used), whichever of the two sees
    # the expanded alias
    x = pt.dmatrix("x")
    x_uv, x_s_only = (
        (pt.expand_dims(x, 0), x)
        if expanded_node == "compute_uv"
        else (x, pt.expand_dims(x, 0))
    )

    s_only = svd(x_s_only, full_matrices=False, compute_uv=False)
    u, _s, _v = svd(x_uv, full_matrices=False, compute_uv=True)

    rewritten_outs = rewrite_graph(
        [s_only, u],
        include=("canonicalize", "stabilize", "specialize"),
        exclude=("local_eager_useless_unbatched_blockwise",),
    )
    svd_nodes = {
        anc.owner
        for anc in ancestors(rewritten_outs)
        if anc.owner is not None
        and isinstance(anc.owner.op, Blockwise)
        and isinstance(anc.owner.op.core_op, SVD)
    }
    assert len(svd_nodes) == 1


def test_cholesky_eye_rewrite():
    x = pt.eye(10)
    L = cholesky(x)
    f_rewritten = pytensor.function([], L, mode="FAST_RUN")
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
    z_cholesky = cholesky(y)

    # REWRITE TEST
    f_rewritten = pytensor.function([x], z_cholesky, mode="FAST_RUN")
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


def test_cholesky_of_diag():
    x = pt.dvector("x")
    x_diag = pt.diag(x)
    x_cholesky = cholesky(x_diag)

    # REWRITE TEST
    f_rewritten = pytensor.function([x], x_cholesky, mode="FAST_RUN")
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


def test_cholesky_of_diag_not_applied():
    # Case 1 : y is not a diagonal matrix because of k = -1
    x = pt.tensor("x", shape=(7, 7))
    y = pt.eye(7, k=-1) * x
    z_cholesky = cholesky(y)

    # REWRITE TEST (should not be applied)
    f_rewritten = pytensor.function([x], z_cholesky, mode="FAST_RUN")
    nodes = f_rewritten.maker.fgraph.apply_nodes
    assert any(isinstance(node.op, Cholesky) for node in nodes)


@pytest.mark.parametrize(
    "make_diag",
    [
        pytest.param(lambda d: pt.diag(d), id="alloc_diag"),
        pytest.param(lambda d: pt.eye(5) * d, id="eye_mul"),
    ],
)
def test_eigh_of_diag(make_diag):
    d = pt.dvector("d", shape=(5,))
    D = make_diag(d)
    w, v = pt.linalg.eigh(D)

    # Needs ShapeOpt to rewrite away shape graph used to build Eye for v rewrite;
    # specialize fires the extract_diag short-circuits the lift depends on.
    passes = ("canonicalize", "stabilize", "specialize", "ShapeOpt")
    rewritten = rewrite_graph([w, v], include=passes)

    idx = pt.argsort(d)
    expected_w = pt.take_along_axis(d, idx, axis=-1)
    expected_v = pt.eye(5)[idx].mT
    expected = rewrite_graph([expected_w, expected_v], include=passes)

    assert_equal_computations(rewritten, expected)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_eigh_of_batched_diag(dtype):
    x = pt.tensor3("x", shape=(2, 3, 3), dtype=dtype)
    w, v = pt.linalg.eigh(assume(x, diagonal=True))

    result = RewriteTester([x], [w, (v * w[..., None, :]) @ v.mT])
    assert not any(
        isinstance(getattr(node.op, "core_op", None), Eigh)
        for node in result.rewr_fg.apply_nodes
    )
    a = np.zeros((2, 3, 3), dtype=dtype)
    a[:, range(3), range(3)] = [[3.0, 1.0, 2.0], [0.0, 5.0, -1.0]]
    result.assert_eval(a)


def test_eigh_generalized_of_diag_correctness():
    # Numerical correctness of the generalized eigh-of-diag rewrite: V^T B V = I and
    # A V = B V diag(w).
    a_vals = np.array([4.0, 1.0, 9.0, 2.0, 5.0])
    b_vals = np.array([1.0, 2.0, 4.0, 1.0, 0.5])
    a = pt.dvector("a", shape=(5,))
    b = pt.dvector("b", shape=(5,))
    A = pt.diag(a)
    B = pt.diag(b)
    w, v = pt.linalg.eigh(A, B)
    fn = pytensor.function([a, b], [w, v])
    w_val, v_val = fn(a_vals, b_vals)
    expected_eigvals = np.sort(a_vals / b_vals)
    np.testing.assert_allclose(w_val, expected_eigvals)
    A_np = np.diag(a_vals)
    B_np = np.diag(b_vals)
    np.testing.assert_allclose(v_val.T @ B_np @ v_val, np.eye(5), atol=1e-10)
    np.testing.assert_allclose(A_np @ v_val, B_np @ v_val * w_val, atol=1e-10)


@pytest.mark.parametrize(
    "make_diag, include_b",
    [
        (lambda d: pt.diag(d), True),
        (lambda d: pt.eye(5) * d, False),
    ],
    ids=["alloc_diag_with_b", "eye_mul_no_b"],
)
def test_eigvalsh_of_diag(make_diag, include_b):
    d = pt.dvector("d", shape=(5,))
    D = make_diag(d)

    if include_b:
        b = pt.dvector("b", shape=(5,))
        B = make_diag(b)
    else:
        B = None

    w = pt.linalg.eigvalsh(D, B)

    expected = pt.sort(d) if not include_b else pt.sort(d / b)
    rewritten = rewrite_graph(w, include=("canonicalize", "stabilize", "specialize"))

    assert_equal_computations([rewritten], [expected])


def test_svd_of_diag_singular_values():
    d = pt.dvector("d", shape=(5,))
    s = svd(pt.eye(5) * d, compute_uv=False)

    passes = ("canonicalize", "stabilize", "specialize", "ShapeOpt")
    result = RewriteTester([d], [s], include=passes)
    result.assert_graph(*rewrite_graph([pt.sort(pt.abs(d))[::-1]], include=passes))
    result.assert_eval(np.array([1.0, -3.0, 0.0, 2.0, -0.5]))


def test_svd_of_diag():
    x = pt.dtensor3("x", shape=(2, 5, 3))
    u, s, vh = svd(assume(x, diagonal=True), full_matrices=False)

    d = pt.diagonal(x, axis1=-2, axis2=-1)
    idx = pt.argsort(pt.abs(d), axis=-1)[..., ::-1]
    sorted_d = pt.take_along_axis(d, idx, axis=-1)
    phases = pt.where(pt.eq(sorted_d, 0), 1, pt.sign(sorted_d))
    expected = [
        pt.eye(x.shape[-2])[idx].mT,
        pt.take_along_axis(pt.abs(d), idx, axis=-1),
        pt.eye(x.shape[-1])[idx] * phases[..., None],
    ]

    passes = ("canonicalize", "ShapeOpt")
    result = RewriteTester([x], [u, s, vh], include=passes)
    result.assert_graph(*rewrite_graph(expected, include=passes))


@pytest.mark.parametrize("full_matrices", [True, False], ids=["full", "reduced"])
@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((5, 5), "float64"),
        ((2, 5, 3), "float64"),
        ((2, 3, 5), "float64"),
        ((2, 3, 5), "complex128"),
    ],
    ids=["square", "tall", "wide", "wide_complex"],
)
def test_svd_of_diag_values(shape, dtype, full_matrices):
    """U and Vh are unique only up to phases and the basis of the unused columns, so
    values are compared through the reconstruction and orthogonality."""
    *batch, m, n = shape
    k = min(m, n)
    x = pt.tensor("x", shape=shape, dtype=dtype)
    u, s, vh = svd(assume(x, diagonal=True), full_matrices=full_matrices)
    reconstruction = (u[..., :k] * s[..., None, :]) @ vh[..., :k, :]

    rng = np.random.default_rng(0)
    diag_vals = rng.normal(size=(*batch, k)).astype(dtype)
    if dtype == "complex128":
        diag_vals += 1j * rng.normal(size=(*batch, k))
    diag_vals[..., 0] = 0.0
    a = np.zeros(shape, dtype=dtype)
    a[..., range(k), range(k)] = diag_vals
    result = RewriteTester(
        [x],
        [s, reconstruction, u.conj().mT @ u, vh @ vh.conj().mT],
        include=("canonicalize",),
    )
    assert not any(
        isinstance(getattr(node.op, "core_op", None), SVD)
        for node in result.rewr_fg.apply_nodes
    )
    result.assert_eval(a, atol=1e-12)


@pytest.mark.parametrize("compute_uv", [False, True], ids=["s_only", "uv"])
@pytest.mark.parametrize("assumption", ["symmetric", "positive_definite"])
def test_svd_of_symmetric(assumption, compute_uv):
    x = pt.tensor("x", shape=(3, 4, 4))
    x_assumed = assume(x, **{assumption: True})

    if compute_uv:
        u, s, vh = svd(x_assumed)
        out = [s, (u * s[..., None, :]) @ vh]
        w, v = eigh(x)
        abs_w = pt.abs(w)
        idx = pt.argsort(abs_w, axis=-1)[..., ::-1]
        column_idx = pt.expand_dims(idx, -2)
        signs = pt.where(w < 0, -1, 1).astype(w.dtype)
        expected_s = pt.take_along_axis(abs_w, idx, axis=-1)
        expected_u = pt.take_along_axis(v, column_idx, axis=-1)
        expected_vh = pt.take_along_axis(
            v * pt.expand_dims(signs, -2), column_idx, axis=-1
        ).mT
        expected = [expected_s, (expected_u * expected_s[..., None, :]) @ expected_vh]
    else:
        out = [svd(x_assumed, compute_uv=False)]
        expected = [pt.sort(pt.abs(pt.linalg.eigvalsh(x)), axis=-1)[..., ::-1]]

    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 4, 4))
    a = a + a.mT
    if assumption == "positive_definite":
        a = a @ a.mT + np.eye(4)

    result = RewriteTester([x], out)
    result.assert_graph(*rewrite_graph(expected, include=("canonicalize",)))
    result.assert_eval(a)


def test_svd_of_rank_deficient_gram_is_nonnegative():
    """x @ x.T is assumed positive definite but is singular for a tall x, so roundoff
    can give it tiny negative eigenvalues. Singular values must still be >= 0."""
    x = pt.dmatrix("x", shape=(6, 2))
    result = RewriteTester([x], [svd(x @ x.T, compute_uv=False)])
    [s] = result.rewr_fn(np.random.default_rng(0).normal(size=(6, 2)))
    assert (s >= 0).all()


@pytest.mark.parametrize("compute_uv", [False, True], ids=["s_only", "uv"])
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_svd_of_orthogonal(dtype, compute_uv):
    x = pt.tensor("x", shape=(3, 4, 4), dtype=dtype)
    x_orthogonal = assume(x, orthogonal=True)

    if compute_uv:
        u, s, vh = svd(x_orthogonal)
        out = [u, s, vh]
        expected = [
            x,
            pt.ones(x.shape[:-1]),
            pt.broadcast_to(pt.eye(x.shape[-1]), x.shape).astype(dtype),
        ]
    else:
        out = [svd(x_orthogonal, compute_uv=False)]
        expected = [pt.ones(x.shape[:-1])]

    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 4, 4))
    if dtype == "complex128":
        a = a + 1j * rng.normal(size=(3, 4, 4))
    q, _ = np.linalg.qr(a)

    result = RewriteTester([x], out)
    result.assert_graph(*rewrite_graph(expected, include=("canonicalize",)))
    if not compute_uv:
        result.assert_eval(q)


@pytest.mark.parametrize("compute_uv", [False, True], ids=["s_only", "uv"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_svd_of_orthogonal_product(side, compute_uv):
    q = pt.dmatrix("q", shape=(4, 4))
    x = pt.dmatrix("x", shape=(4, 4))
    q_orth = assume(q, orthogonal=True)
    y = q_orth @ x if side == "left" else x @ q_orth

    if compute_uv:
        out = svd(y)
        U, s, Vh = svd(x)
        expected = [q @ U, s, Vh] if side == "left" else [U, s, Vh @ q]
    else:
        out = [svd(y, compute_uv=False)]
        expected = [svd(x, compute_uv=False)]

    result = RewriteTester([q, x], out)
    result.assert_graph(*rewrite_graph(expected, include=("canonicalize",)))


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_svd_of_orthogonal_product_broadcasts_batch(dtype):
    """With a batched Q and an unbatched X, the factors without Q are broadcast."""
    q = pt.tensor("q", shape=(2, 4, 4), dtype=dtype)
    x = pt.tensor("x", shape=(4, 3), dtype=dtype)
    u, s, vh = svd(assume(q, orthogonal=True) @ x, full_matrices=False)

    U, s_x, Vh = svd(pt.expand_dims(x, 0), full_matrices=False)
    expected = [q @ U, pt.broadcast_to(s_x, (2, 3)), pt.broadcast_to(Vh, (2, 3, 3))]

    passes = ("canonicalize", "ShapeOpt")
    result = RewriteTester([q, x], [u, s, vh], include=passes)
    result.assert_graph(*rewrite_graph(expected, include=passes))

    rng = np.random.default_rng(0)
    q_val = rng.normal(size=(2, 4, 4)).astype(dtype)
    x_val = rng.normal(size=(4, 3)).astype(dtype)
    if dtype == "complex128":
        q_val += 1j * rng.normal(size=(2, 4, 4))
        x_val += 1j * rng.normal(size=(4, 3))
    q_val, _ = np.linalg.qr(q_val)
    result = RewriteTester([q, x], [s, (u * s[..., None, :]) @ vh])
    result.assert_eval(q_val, x_val)


@pytest.mark.parametrize("compute_uv", [False, True], ids=["s_only", "uv"])
@pytest.mark.parametrize(
    "make_structured, combine_singular_values",
    [
        (pt.linalg.kron, lambda s_a, s_b: pt.outer(s_a, s_b).ravel()),
        (pt.linalg.block_diag, lambda s_a, s_b: pt.concatenate([s_a, s_b])),
    ],
    ids=["kron", "block_diag"],
)
def test_svd_of_square_structured(make_structured, combine_singular_values, compute_uv):
    a = pt.dmatrix("a", shape=(2, 2))
    b = pt.dmatrix("b", shape=(3, 3))

    if compute_uv:
        out = svd(make_structured(a, b))
        (U_a, s_a, Vh_a), (U_b, s_b, Vh_b) = svd(a), svd(b)
        s_vals = combine_singular_values(s_a, s_b)
        order = pt.argsort(s_vals, axis=-1)[..., ::-1]
        expected = [
            pt.take_along_axis(
                make_structured(U_a, U_b), pt.expand_dims(order, -2), axis=-1
            ),
            pt.take_along_axis(s_vals, order, axis=-1),
            pt.take_along_axis(
                make_structured(Vh_a, Vh_b), pt.expand_dims(order, -1), axis=-2
            ),
        ]
    else:
        out = [svd(make_structured(a, b), compute_uv=False)]
        s_vals = combine_singular_values(
            svd(a, compute_uv=False), svd(b, compute_uv=False)
        )
        expected = [pt.sort(s_vals)[::-1]]

    result = RewriteTester([a, b], out)
    result.assert_graph(*rewrite_graph(expected, include=("canonicalize",)))


@pytest.mark.parametrize("transposed", [False, True], ids=["", "transposed"])
@pytest.mark.parametrize(
    "combine, shapes, full_matrices, dtype",
    [
        (pt.linalg.kron, [(2, 2), (3, 3)], False, "float64"),
        (pt.linalg.kron, [(3, 2), (4, 1)], False, "complex128"),
        (pt.linalg.kron, [(3, 2), (2, 4)], True, "float64"),
        (pt.linalg.kron, [(1, 3), (3, 1)], True, "complex128"),
        (lambda a, b: pt.linalg.kron(a, b)[None], [(2, 3), (2, 2)], True, "float64"),
        (pt.linalg.block_diag, [(3, 2), (4, 1), (2, 2)], False, "complex128"),
        (pt.linalg.block_diag, [(3, 2), (1, 4), (2, 2)], True, "float64"),
        (pt.linalg.block_diag, [(4, 3, 2), (4, 1, 3)], True, "complex128"),
    ],
    ids=[
        "kron_square",
        "kron_tall_reduced",
        "kron_mixed",
        "kron_rank_deficient",
        "kron_expanded",
        "block_diag_tall_reduced",
        "block_diag_mixed",
        "block_diag_batched",
    ],
)
def test_svd_of_structured_values(combine, shapes, full_matrices, dtype, transposed):
    """U and Vh are unique only up to phases and the basis of the zero singular values,
    so values are compared through the reconstruction and orthogonality."""
    factors = [
        pt.tensor(f"x{i}", shape=shape, dtype=dtype) for i, shape in enumerate(shapes)
    ]
    X = combine(*factors)
    if transposed:
        X = X.mT
    u, s, vh = svd(X, full_matrices=full_matrices)
    k = min(X.type.shape[-2:])
    reconstruction = (u[..., :k] * s[..., None, :]) @ vh[..., :k, :]
    outs = [
        s,
        svd(X, compute_uv=False),
        reconstruction,
        u.conj().mT @ u,
        vh @ vh.conj().mT,
    ]

    result = RewriteTester(factors, outs)
    svd_inputs = sorted(
        node.inputs[0].name
        for node in result.rewr_fg.apply_nodes
        if isinstance(getattr(node.op, "core_op", node.op), SVD)
    )
    assert svd_inputs == sorted(factor.name for factor in factors)

    rng = np.random.default_rng(0)
    values = [rng.normal(size=shape).astype(dtype) for shape in shapes]
    if dtype == "complex128":
        for value in values:
            value += 1j * rng.normal(size=value.shape)
    result.assert_eval(*values, atol=1e-10)


@pytest.mark.parametrize("combine", [pt.linalg.kron, pt.linalg.block_diag])
def test_reduced_svd_of_mixed_orientation_structured_is_not_rewritten(combine):
    """A reduced svd needs full factor bases when tall and wide factors mix, which can
    be far larger than the requested output, so the rewrite does not apply."""
    a = pt.dmatrix("a", shape=(300, 2))
    b = pt.dmatrix("b", shape=(2, 4))
    out = svd(combine(a, b), full_matrices=False)

    result = RewriteTester([a, b], out)
    result.assert_graph(*svd(combine(a, b), full_matrices=False))


def test_svd_of_symmetric_skips_complex():
    x = pt.tensor("x", shape=(4, 4), dtype="complex128")
    s = svd(assume(x, symmetric=True), compute_uv=False)

    result = RewriteTester([x], [s])
    result.assert_graph(svd(x, compute_uv=False))


@pytest.mark.parametrize(
    "assumptions, make_expected",
    [
        (dict(orthogonal=True, diagonal=True), lambda x: pt.ones(x.shape[:-1])),
        (
            dict(diagonal=True, symmetric=True),
            lambda x: pt.sort(pt.abs(pt.diagonal(x)), axis=-1)[..., ::-1],
        ),
    ],
    ids=["orthogonal_over_diagonal", "diagonal_over_symmetric"],
)
def test_svd_of_structured_prefers_cheapest_form(assumptions, make_expected):
    x = pt.dmatrix("x", shape=(4, 4))
    s = svd(assume(x, **assumptions), compute_uv=False)

    result = RewriteTester([x], [s])
    result.assert_graph(*rewrite_graph([make_expected(x)], include=("canonicalize",)))


def test_svd_of_transposed_orthogonal_product():
    q = pt.dmatrix("q", shape=(4, 4))
    x = pt.dmatrix("x", shape=(4, 3))
    u, s, vh = svd((assume(q, orthogonal=True) @ x).mT, full_matrices=False)

    rng = np.random.default_rng(0)
    q_val, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    result = RewriteTester([q, x], [s, (u * s[..., None, :]) @ vh])
    result.assert_eval(q_val, rng.normal(size=(4, 3)))


@pytest.mark.parametrize(
    "make_diag, permute_l, p_indices",
    [
        pytest.param(lambda d: pt.diag(d), False, False, id="alloc_diag_default"),
        pytest.param(lambda d: pt.diag(d), True, False, id="alloc_diag_permute_l"),
    ],
)
def test_lu_of_diag(make_diag, permute_l, p_indices):
    n = 5
    d = pt.dvector("d", shape=(n,))
    D = make_diag(d)
    diag_idx = np.diag_indices(n)
    n_64 = np.array(n, dtype="int64")
    simplified_D = pt.zeros((n_64, n_64))[diag_idx].set(d)

    out = lu(D, permute_l=permute_l, p_indices=p_indices)
    rewritten = rewrite_graph(
        out, include=("canonicalize", "stabilize", "specialize", "ShapeOpt")
    )
    if permute_l:
        expected_PL = pt.as_tensor(np.eye(n, dtype="float64"))
        expected = [expected_PL, simplified_D]

    elif p_indices:
        expected_p = pt.as_tensor(np.arange(n, dtype="int32"))
        expected_L = pt.as_tensor(np.eye(n, dtype="float64"))
        expected = [expected_p, expected_L, simplified_D]

    else:
        expected_P = pt.as_tensor(np.eye(n, dtype="float64"))
        expected_L = pt.as_tensor(np.eye(n, dtype="float64"))
        expected = [expected_P, expected_L, simplified_D]

    assert_equal_computations(rewritten, expected)


def test_lu_factor_of_diag():
    n = 5
    d = pt.dvector("d", shape=(n,))
    D = pt.diag(d)
    diag_idx = np.diag_indices(n)
    n_64 = np.array(n, dtype="int64")
    simplified_D = pt.zeros((n_64, n_64))[diag_idx].set(d)

    lu_out, piv_out = lu_factor(D)
    rewritten = rewrite_graph(
        [lu_out, piv_out],
        include=("canonicalize", "stabilize", "specialize", "ShapeOpt"),
    )

    expected_pivots = pt.as_tensor(np.arange(n, dtype="int32"))
    expected = [simplified_D, expected_pivots]

    assert_equal_computations(rewritten, expected)


@pytest.mark.parametrize(
    "make_diag, mode, pivoting",
    [
        pytest.param(lambda d: pt.diag(d), "full", False, id="alloc_diag_full"),
        pytest.param(lambda d: pt.eye(5) * d, "r", False, id="eye_mul_r"),
        pytest.param(
            lambda d: pt.diag(d), "economic", True, id="alloc_diag_economic_pivot"
        ),
        pytest.param(lambda d: pt.diag(d), "raw", False, id="alloc_diag_raw"),
    ],
)
def test_qr_of_diag(make_diag, mode, pivoting):
    n = 5
    d = pt.dvector("d", shape=(n,))
    D = make_diag(d)
    diag_idx = np.diag_indices(n)
    n_64 = np.array(n, dtype="int64")

    out = qr(D, mode=mode, pivoting=pivoting)
    if not isinstance(out, list | tuple):
        out = [out]
    passes = ("canonicalize", "stabilize", "specialize", "ShapeOpt")
    rewritten = rewrite_graph(out, include=passes)

    def diag_of(vals):
        return pt.zeros((n_64, n_64))[diag_idx].set(vals)

    expected_R = diag_of(pt.abs(d))

    if mode == "raw":
        simplified_D = diag_of(d)
        expected_tau = pt.alloc(np.float64(0.0), n_64)
        expected = [simplified_D, expected_tau, simplified_D]
    elif mode == "r":
        expected = [expected_R]
    else:
        expected_Q = diag_of(pt.sign(d))
        expected = [expected_Q, expected_R]

    if pivoting:
        expected.append(pt.as_tensor(np.arange(n, dtype="int32")))

    assert_equal_computations(rewritten, expected)


@pytest.mark.parametrize(
    "make_diag",
    [
        pytest.param(lambda d: pt.diag(d), id="alloc_diag"),
    ],
)
def test_schur_of_diag(make_diag):
    n = 5
    d = pt.dvector("d", shape=(n,))
    D = make_diag(d)
    diag_idx = np.diag_indices(n)
    n_64 = np.array(n, dtype="int64")

    T, Z = schur(D)
    passes = ("canonicalize", "stabilize", "specialize", "ShapeOpt")
    rewritten = rewrite_graph([T, Z], include=passes)

    simplified_D = pt.zeros((n_64, n_64))[diag_idx].set(d)
    expected = [simplified_D, pt.as_tensor(np.eye(n, dtype="float64"))]
    assert_equal_computations(rewritten, expected)


@pytest.mark.parametrize("sort", ["lhp", "rhp", "iuc", "ouc"])
def test_schur_of_diag_sort(sort):
    n = 5
    rng = np.random.default_rng(sum(map(ord, "schur_of_diag_sort")))
    d_var = pt.dvector("d", shape=(n,))
    D = pt.diag(d_var)
    T, Z = schur(D, sort=sort)

    mode = get_mode("FAST_RUN")
    f = pytensor.function([d_var], [T, Z], mode=mode)
    f_no_rewrite = pytensor.function(
        [d_var], [T, Z], mode=mode.excluding("schur_of_diag")
    )

    assert not any(
        isinstance(node.op, Blockwise | BlockwiseWithCoreShape)
        and isinstance(node.op.core_op, Schur)
        for node in f.maker.fgraph.apply_nodes
    )

    d_val = rng.uniform(-2, 2, n)
    T_val, Z_val = f(d_val)
    T_ref, _ = f_no_rewrite(d_val)

    # Order within each sort group (selected / non-selected) is not uniquely
    # determined, so compare sorted diagonals instead of element-wise.
    np.testing.assert_allclose(
        np.sort(np.diag(T_val)), np.sort(np.diag(T_ref)), atol=1e-12
    )
    np.testing.assert_allclose(Z_val @ T_val @ Z_val.T, np.diag(d_val), atol=1e-12)


@pytest.mark.parametrize(
    "make_diag, return_eigenvalues",
    [
        pytest.param(lambda d: pt.diag(d), False, id="alloc_diag_no_eigs"),
        pytest.param(lambda d: pt.diag(d), True, id="alloc_diag_with_eigs"),
    ],
)
def test_qz_of_diag(make_diag, return_eigenvalues):
    n = 5
    a = pt.dvector("a", shape=(n,))
    b = pt.dvector("b", shape=(n,))
    A = make_diag(a)
    B = make_diag(b)
    diag_idx = np.diag_indices(n)
    n_64 = np.array(n, dtype="int64")

    out = qz(A, B, return_eigenvalues=return_eigenvalues)
    passes = ("canonicalize", "stabilize", "specialize", "ShapeOpt")
    rewritten = rewrite_graph(list(out), include=passes)

    def diag_of(vals):
        return pt.zeros((n_64, n_64))[diag_idx].set(vals)

    expected_I = pt.as_tensor(np.eye(n, dtype="float64"))

    if return_eigenvalues:
        expected = [
            diag_of(a),
            diag_of(b),
            a.astype("complex128"),
            b,
            expected_I,
            expected_I,
        ]
    else:
        expected = [diag_of(a), diag_of(b), expected_I, expected_I]

    assert_equal_computations(rewritten, expected)


@pytest.mark.parametrize(
    "sort, return_eigenvalues",
    [
        pytest.param("lhp", False, id="lhp"),
        pytest.param("rhp", False, id="rhp"),
        pytest.param("iuc", False, id="iuc"),
        pytest.param("ouc", True, id="ouc_with_eigs"),
    ],
)
def test_qz_of_diag_sort(sort, return_eigenvalues):
    n = 5
    rng = np.random.default_rng(42)
    a_var = pt.dvector("a", shape=(n,))
    b_var = pt.dvector("b", shape=(n,))
    A = pt.diag(a_var)
    B = pt.diag(b_var)

    out = qz(A, B, sort=sort, return_eigenvalues=return_eigenvalues)
    mode = get_mode("FAST_RUN")
    f = pytensor.function([a_var, b_var], list(out), mode=mode)
    f_no_rewrite = pytensor.function(
        [a_var, b_var], list(out), mode=mode.excluding("qz_of_diag")
    )

    assert not any(
        isinstance(node.op, Blockwise | BlockwiseWithCoreShape)
        and isinstance(node.op.core_op, QZ)
        for node in f.maker.fgraph.apply_nodes
    )

    a_val = rng.uniform(-2, 2, n)
    b_val = rng.uniform(0.5, 2, n)
    result = f(a_val, b_val)
    ref = f_no_rewrite(a_val, b_val)

    if return_eigenvalues:
        AA_val, BB_val, alpha_val, beta_val, Q_val, Z_val = result
        AA_ref, BB_ref, alpha_ref, beta_ref, *_ = ref
        # Order within each sort group is not uniquely determined, so compare
        # sorted diagonals / eigenvalue arrays instead of element-wise.
        np.testing.assert_allclose(
            np.sort(np.diag(AA_val)), np.sort(np.diag(AA_ref)), atol=1e-12
        )
        np.testing.assert_allclose(
            np.sort(np.diag(BB_val)), np.sort(np.diag(BB_ref)), atol=1e-12
        )
        np.testing.assert_allclose(
            np.sort(np.abs(alpha_val)), np.sort(np.abs(alpha_ref)), atol=1e-12
        )
        np.testing.assert_allclose(np.sort(beta_val), np.sort(beta_ref), atol=1e-12)
    else:
        AA_val, BB_val, Q_val, Z_val = result
        AA_ref, BB_ref, *_ = ref
        np.testing.assert_allclose(
            np.sort(np.diag(AA_val)), np.sort(np.diag(AA_ref)), atol=1e-12
        )
        np.testing.assert_allclose(
            np.sort(np.diag(BB_val)), np.sort(np.diag(BB_ref)), atol=1e-12
        )

    np.testing.assert_allclose(Q_val @ AA_val @ Z_val.T, np.diag(a_val), atol=1e-12)
    np.testing.assert_allclose(Q_val @ BB_val @ Z_val.T, np.diag(b_val), atol=1e-12)


def test_eig_to_eigh():
    x = pt.dmatrix("x", shape=(5, 5))
    x_sym = assume(x, symmetric=True)
    w, v = pt.linalg.eig(x_sym)

    rewrites = ("canonicalize", "ShapeOpt")
    w_r, v_r = rewrite_graph([w, v], include=rewrites)

    w_expected, v_expected = eigh(x)
    w_expected = w_expected.astype("complex128")
    v_expected = v_expected.astype("complex128")
    assert_equal_computations([w_r, v_r], [w_expected, v_expected])
