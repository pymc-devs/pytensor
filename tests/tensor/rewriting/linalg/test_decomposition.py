from functools import partial

import numpy as np
import pytest
import scipy.linalg
from numpy.testing import assert_allclose

import pytensor
from pytensor import tensor as pt
from pytensor.compile import get_default_mode
from pytensor.configdefaults import config
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.tensor import swapaxes
from pytensor.tensor.basic import alloc_diag
from pytensor.tensor.blockwise import Blockwise, BlockwiseWithCoreShape
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky, cholesky
from pytensor.tensor.linalg.decomposition.lu import lu, lu_factor
from pytensor.tensor.linalg.decomposition.svd import SVD, svd
from pytensor.tensor.math import dot, matmul
from pytensor.tensor.type import tensor
from tests.unittest_tools import assert_equal_computations


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

    # Needs ShapeOpt to rewrite away shape graph used to build Eye for v rewrite
    rewritten = rewrite_graph([w, v], include=("canonicalize", "stabilize", "ShapeOpt"))

    idx = pt.argsort(d)
    expected_w = d[idx]
    expected_v = pt.as_tensor(np.eye(5))[:, idx]

    assert_equal_computations(rewritten, [expected_w, expected_v])


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
    rewritten = rewrite_graph(w, include=("canonicalize", "stabilize"))

    assert_equal_computations([rewritten], [expected])


@pytest.mark.parametrize(
    "make_diag, compute_uv",
    [(lambda d: pt.diag(d), True), (lambda d: pt.eye(5) * d, False)],
    ids=["alloc_diag_with_uv", "eye_mul_no_uv"],
)
def test_svd_of_diag(make_diag, compute_uv):
    d = pt.dvector("d", shape=(5,))
    D = make_diag(d)
    out = svd(D, compute_uv=compute_uv)
    if not compute_uv:
        out = [out]

    passes = ("canonicalize", "stabilize", "ShapeOpt")
    rewritten = rewrite_graph(out, include=passes)

    abs_d = pt.abs(d)
    idx = pt.argsort(-abs_d)

    if compute_uv:
        expected_s = abs_d[idx]
        expected_u = pt.as_tensor(np.eye(5))[:, idx]
        sorted_signs = pt.sign(d[idx])
        expected_vh = alloc_diag(sorted_signs, axis1=-1, axis2=-2)[:, idx]
        expected = rewrite_graph([expected_u, expected_s, expected_vh], include=passes)
    else:
        expected = rewrite_graph([abs_d[idx]], include=passes)

    assert_equal_computations(rewritten, expected)


@pytest.mark.parametrize(
    "make_diag, permute_l, p_indices",
    [
        pytest.param(lambda d: pt.diag(d), False, False, id="alloc_diag_default"),
        pytest.param(lambda d: pt.diag(d), True, False, id="alloc_diag_permute_l"),
        pytest.param(lambda d: pt.eye(5) * d, False, True, id="eye_mul_p_indices"),
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
