import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytensor import function
from pytensor import tensor as pt
from pytensor.assumptions.specify import assume
from pytensor.configdefaults import config
from pytensor.gradient import grad
from pytensor.graph import rewrite_graph
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg.decomposition import lu, qr, svd
from pytensor.tensor.linalg.decomposition.cholesky import cholesky
from pytensor.tensor.linalg.summary import Det, SLogDet, det, slogdet
from pytensor.tensor.rewriting.linalg.utils import permutation_indices_to_det_sign
from pytensor.tensor.type import matrix
from tests.unittest_tools import RewriteTester, assert_equal_computations


def test_det_of_cholesky():
    X = matrix("X")
    L = cholesky(X)
    det_X = pt.linalg.det(X)

    f = function([X], [L, det_X])
    assert not any(isinstance(node, Det) for node in f.maker.fgraph.apply_nodes)

    # This previously raised an error (issue #392)
    f = function([X], [L, det_X, X])
    assert not any(isinstance(node, Det) for node in f.maker.fgraph.apply_nodes)

    # Test graph that only has det_X
    f = function([X], [det_X])
    assert not any(isinstance(node, Det) for node in f.maker.fgraph.apply_nodes)


@pytest.mark.parametrize(
    "shape",
    [(), (7,), (1, 7), (7, 1), (7, 7), (3, 7, 7)],
    ids=["scalar", "vector", "row_vec", "col_vec", "matrix", "batched_input"],
)
def test_det_of_diag_from_eye_mul(shape):
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


def test_det_of_diag_from_diag():
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


def test_dont_apply_det_of_diag_from_scalar_eye():
    x = pt.matrix("x")
    x_diag = pt.eye(1, 1) * x
    y = pt.linalg.det(x_diag)
    f_rewritten = function([x], y, mode="FAST_RUN")
    f_rewritten.dprint()
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


@pytest.mark.parametrize(
    "side, shape",
    [
        pytest.param("lower_triangular", (5, 5), id="lower"),
        pytest.param("upper_triangular", (5, 5), id="upper"),
        pytest.param("lower_triangular", (3, 5, 5), id="batched_lower"),
    ],
)
def test_det_of_triangular(side, shape):
    x = pt.tensor("x", shape=shape)
    y = pt.linalg.det(assume(x, **{side: True}))

    f = function([x], y, mode="FAST_RUN")
    nodes = f.maker.fgraph.apply_nodes
    assert not any(
        isinstance(node.op, Det) or isinstance(getattr(node.op, "core_op", None), Det)
        for node in nodes
    )

    rng = np.random.default_rng(0)
    tri = np.tril if side == "lower_triangular" else np.triu
    x_test = tri(rng.normal(size=shape)).astype(config.floatX)
    expected = np.linalg.det(x_test)

    assert_allclose(
        expected,
        f(x_test),
        atol=1e-3 if config.floatX == "float32" else 1e-8,
        rtol=1e-3 if config.floatX == "float32" else 1e-8,
    )


def test_det_of_diag_incorrect_for_rectangle_eye():
    x = pt.matrix("x")
    x_diag = pt.eye(7, 5) * x
    with pytest.raises(ValueError, match="Determinant not defined"):
        pt.linalg.det(x_diag)


@pytest.mark.parametrize("batch_dim", [(), (4,)])
def test_slogdet_specialization(batch_dim):
    shape = (*batch_dim, 20, 20)
    x = pt.tensor("x", shape=shape, dtype=config.floatX)
    a = np.random.rand(*shape)
    det_x = pt.linalg.det(x)
    log_abs_det_x = pt.log(pt.abs(det_x))
    log_det_x = pt.log(det_x)
    sign_det_x = pt.sign(det_x)
    exp_det_x = pt.exp(det_x)

    # REWRITE TESTS
    # sign(det(x))
    expected_sign_det_x, expected_log_abs_det_x = Blockwise(SLogDet())(x)

    result = RewriteTester([x], [sign_det_x], include=["stabilize", "specialize"])
    result.assert_graph(expected_sign_det_x)
    result.assert_eval(a)

    # log(abs(det(x)))
    result = RewriteTester([x], [log_abs_det_x], include=["stabilize", "specialize"])
    result.assert_graph(expected_log_abs_det_x)
    result.assert_eval(a)

    # log(det(x))
    expected_sign_value = np.array([-1], dtype=np.int8) if batch_dim else -1
    expected_nan_value = (
        np.array([np.nan], dtype=config.floatX) if batch_dim else np.nan
    )

    expected_log_det_x = pt.where(
        pt.eq(expected_sign_det_x, expected_sign_value),
        expected_nan_value,
        expected_log_abs_det_x,
    )

    result = RewriteTester([x], [log_det_x], include=["stabilize", "specialize"])
    result.assert_graph(expected_log_det_x)
    result.assert_eval(a)

    # More than 1 valid function
    result = RewriteTester(
        [x],
        [sign_det_x, log_abs_det_x],
        include=["stabilize", "specialize"],
    )
    result.assert_graph(expected_sign_det_x, expected_log_abs_det_x)
    result.assert_eval(a)

    # Other functions (rewrite shouldn't be applied to these)
    # Only invalid functions
    result = RewriteTester([x], [exp_det_x], include=["stabilize", "specialize"])
    result.assert_graph(exp_det_x)
    result.assert_eval(a)

    # Invalid + Valid function
    result = RewriteTester(
        [x],
        [exp_det_x, sign_det_x],
        include=["stabilize", "specialize"],
    )
    result.assert_graph(exp_det_x, sign_det_x)
    result.assert_eval(a)


def test_slogdet_specialization_through_expand_dims():
    x = pt.tensor("x", shape=(3, 3), dtype=config.floatX)
    log_det_x = pt.log(pt.expand_dims(det(x), 0))

    # dimshuffle_lift pushes the expand_dims inside the Switch
    sign_det_x, log_abs_det_x = Blockwise(SLogDet())(x)
    expected = pt.where(
        pt.eq(pt.expand_dims(sign_det_x, 0), np.array([-1], dtype=np.int8)),
        np.array([np.nan], dtype=config.floatX),
        pt.expand_dims(log_abs_det_x, 0),
    )

    result = RewriteTester([x], [log_det_x], include=["stabilize", "specialize"])
    result.assert_graph(expected)
    result.assert_eval(np.random.rand(3, 3))


@pytest.mark.parametrize(
    "original_fn, expected_fn",
    [
        pytest.param(
            lambda x: pt.log(pt.prod(pt.abs(x))),
            lambda x: pt.sum(pt.log(pt.abs(x))),
            id="log_prod_abs",
        ),
        pytest.param(
            lambda x: pt.log(pt.prod(pt.exp(x))), lambda x: pt.sum(x), id="log_prod_exp"
        ),
        pytest.param(
            lambda x: pt.log(pt.prod(x**2)),
            lambda x: np.float64(2.0) * pt.sum(pt.log(pt.abs(x))),
            id="log_prod_sqr",
        ),
        pytest.param(
            lambda x: pt.log(pt.abs(pt.prod(x))),
            lambda x: pt.sum(pt.log(pt.abs(x))),
            id="log_abs_prod",
        ),
        pytest.param(
            lambda x: pt.log(pt.abs(pt.sqr(pt.prod(x)))),
            lambda x: np.float64(2.0) * pt.sum(pt.log(pt.abs(x))),
            id="log_abs_sqr_prod",
        ),
        pytest.param(
            lambda x: pt.log(pt.prod(pt.abs(x), axis=0)),
            lambda x: pt.sum(pt.log(pt.abs(x)), axis=0),
            id="log_prod_abs_axis0",
        ),
        pytest.param(
            lambda x: pt.log(pt.prod(pt.exp(x), axis=-1)),
            lambda x: pt.sum(x, axis=-1),
            id="log_prod_exp_axis-1",
        ),
    ],
)
def test_local_log_prod_to_sum_log(original_fn, expected_fn):
    x = pt.tensor("x", shape=(3, 4))
    out = original_fn(x)
    expected = expected_fn(x)
    rewritten = rewrite_graph(out, include=["stabilize", "specialize"])
    assert_equal_computations([rewritten], [expected])


def test_local_log_prod_to_sum_log_unknown_sign():
    """A sign-indefinite operand leaves log(prod(x)) alone."""
    x = pt.tensor("x", shape=(3, 4))
    out = pt.log(pt.prod(x))
    result = RewriteTester([x], [out], include=["stabilize", "specialize"])

    result.assert_graph(out)


def test_slogdet_grad_when_cholesky_factored_det_underflows():
    """The slogdet gradient of a Cholesky-factored matrix must stay exact
    after prod(diag(L)) underflows, in both rewrite orders (#2468)."""
    N = 100
    rng = np.random.default_rng(2468)
    A = rng.normal(size=(N, N))
    C = A @ A.T / N + np.eye(N)

    log_sigma = pt.dscalar("log_sigma")
    K = assume(pt.exp(2 * log_sigma) * C, symmetric=True, positive_definite=True)
    L = pt.linalg.cholesky(K)
    _, logdet = pt.linalg.slogdet(K)

    # d(logdet)/d(log_sigma) = 2 * N exactly, for any log_sigma
    f = function([log_sigma], [grad(logdet, log_sigma), L], mode="FAST_RUN")

    logdet_stab, L_stab = rewrite_graph(
        [logdet, L], include=("canonicalize", "stabilize"), clone=True
    )
    f_stab = function(
        [log_sigma], [grad(logdet_stab, log_sigma), L_stab], mode="FAST_RUN"
    )

    point = -5.0  # det(K) ~ exp(2 * N * point) underflows float64
    assert_allclose(f(point)[0], 2 * N)
    assert_allclose(f_stab(point)[0], 2 * N)


@pytest.mark.parametrize(
    "decomp_fn, sign_op, expected_fn",
    [
        pytest.param(
            lambda x: svd.svd(x, compute_uv=True)[0],
            pt.abs,
            lambda x: pt.prod(svd.svd(x, compute_uv=True)[1], axis=0),
            id="svd_abs",
        ),
        pytest.param(
            lambda x: svd.svd(x, compute_uv=False),
            pt.abs,
            lambda x: pt.prod(svd.svd(x, compute_uv=False), axis=0),
            id="svd_no_uv_abs",
        ),
        pytest.param(
            lambda x: qr.qr(x)[0],
            pt.abs,
            lambda x: pt.prod(pt.diagonal(qr.qr(x)[1], axis1=-2, axis2=-1), axis=-1),
            id="qr_abs",
        ),
        pytest.param(
            lambda x: svd.svd(x, compute_uv=True)[0],
            pt.sqr,
            lambda x: pt.prod(svd.svd(x, compute_uv=True)[1], axis=0),
            id="svd_sqr",
        ),
        pytest.param(
            lambda x: svd.svd(x, compute_uv=False),
            pt.sqr,
            lambda x: pt.prod(svd.svd(x, compute_uv=False), axis=0),
            id="svd_no_uv_sqr",
        ),
        pytest.param(
            lambda x: qr.qr(x)[0],
            pt.sqr,
            lambda x: pt.prod(pt.diagonal(qr.qr(x)[1], axis1=-2, axis2=-1), axis=-1),
            id="qr_sqr",
        ),
    ],
)
def test_det_of_matrix_factorized_elsewhere_abs(decomp_fn, sign_op, expected_fn):
    x = pt.tensor("x", shape=(3, 3))

    decomp_var = decomp_fn(x)
    d = sign_op(det(x))

    decomp_var, d = rewrite_graph(
        [decomp_var, d], include=["canonicalize", "stabilize", "specialize"]
    )
    assert_equal_computations([decomp_var], [decomp_fn(x)])
    assert_equal_computations([d], [sign_op(expected_fn(x))])


@pytest.mark.parametrize(
    "original_fn, expected_fn",
    [
        pytest.param(
            lambda x: det(cholesky(x)),
            lambda x: pt.prod(pt.diagonal(cholesky(x), axis1=-2, axis2=-1), axis=-1),
            id="det_cholesky",
        ),
        pytest.param(
            lambda x: det(lu.lu(x)[-1]),
            lambda x: pt.prod(pt.diagonal(lu.lu(x)[-1], axis1=-2, axis2=-1), axis=-1),
            id="det_lu_U",
        ),
        pytest.param(
            lambda x: det(lu.lu(x)[-2]),
            lambda x: pt.as_tensor(1.0, dtype=x.dtype),
            id="det_lu_L",
        ),
    ],
)
def test_det_of_factorized_matrix(original_fn, expected_fn):
    x = pt.tensor("x", shape=(3, 3))
    out = original_fn(x)
    expected = expected_fn(x)
    rewritten = rewrite_graph(out, include=["stabilize", "specialize"])
    assert_equal_computations([rewritten], [expected])


@pytest.mark.parametrize(
    "original_fn, expected_fn",
    [
        pytest.param(
            lambda x: pt.abs(det(svd.svd(x, compute_uv=True)[0])),
            lambda x: pt.as_tensor(1.0, dtype=x.dtype),
            id="abs_det_svd_U",
        ),
        pytest.param(
            lambda x: pt.abs(det(svd.svd(x, compute_uv=True)[2])),
            lambda x: pt.as_tensor(1.0, dtype=x.dtype),
            id="abs_det_svd_Vt",
        ),
        pytest.param(
            lambda x: pt.abs(det(qr.qr(x)[0])),
            lambda x: pt.as_tensor(1.0, dtype=x.dtype),
            id="abs_det_qr_Q",
        ),
        pytest.param(
            lambda x: pt.sqr(det(svd.svd(x, compute_uv=True)[0])),
            lambda x: pt.as_tensor(1.0, dtype=x.dtype),
            id="sqr_det_svd_U",
        ),
        pytest.param(
            lambda x: pt.sqr(det(svd.svd(x, compute_uv=True)[2])),
            lambda x: pt.as_tensor(1.0, dtype=x.dtype),
            id="sqr_det_svd_Vt",
        ),
        pytest.param(
            lambda x: pt.sqr(det(qr.qr(x)[0])),
            lambda x: pt.as_tensor(1.0, dtype=x.dtype),
            id="sqr_det_qr_Q",
        ),
        pytest.param(
            lambda x: det(qr.qr(x)[1]),
            lambda x: pt.prod(pt.diagonal(qr.qr(x)[1], axis1=-2, axis2=-1), axis=-1),
            id="det_qr_R",
        ),
        pytest.param(
            lambda x: det(qr.qr(x)[0]),
            lambda x: det(qr.qr(x)[0]),
            id="det_qr_Q_no_rewrite",
        ),
    ],
)
def test_det_of_factorized_matrix_special_cases(original_fn, expected_fn):
    x = pt.tensor("x", shape=(3, 3))
    out = original_fn(x)
    expected = expected_fn(x)
    rewritten = rewrite_graph(out, include=["stabilize", "specialize"])
    assert_equal_computations([rewritten], [expected])


def test_det_of_factorized_matrix_through_expand_dims():
    # The batched Cholesky keeps the expanded Blockwise(Det) from being eagerly
    # unbatched, so only the look-through peel can reach the factor
    x = pt.tensor("x", shape=(None, 3, 3))
    L = pt.linalg.cholesky(x)
    d = det(pt.expand_dims(L, 0))

    d_rewritten = rewrite_graph(d, include=["stabilize"])
    expected = pt.expand_dims(pt.prod(pt.diagonal(L, axis1=-2, axis2=-1), axis=-1), 0)
    assert_equal_computations([d_rewritten], [expected])


def test_det_of_inv():
    x = pt.tensor("x", shape=(3, 3))
    out = det(pt.linalg.inv(x))
    expected = pt.as_tensor(1.0, dtype="float64") / det(x)
    rewritten = rewrite_graph(out, include=["canonicalize", "stabilize"])
    assert_equal_computations([rewritten], [expected])


def test_det_of_inv_through_expand_dims():
    x = pt.tensor("x", shape=(3, 3))
    out = det(pt.expand_dims(pt.linalg.inv(x), 0))
    # dimshuffle_lift moves the expand_dims inside the division
    expected = pt.as_tensor([1.0], dtype="float64") / pt.expand_dims(det(x), 0)
    rewritten = rewrite_graph(out, include=["canonicalize", "stabilize"])
    assert_equal_computations([rewritten], [expected])


def test_det_of_matrix_factorized_elsewhere_through_expand_dims():
    x = pt.tensor("x", shape=(3, 3))

    # The Cholesky hangs off an expanded alias of x, the det off x itself
    L = pt.linalg.cholesky(pt.expand_dims(x, 0))
    d = det(x)

    _, d_rewritten = rewrite_graph(
        [L, d], include=["canonicalize", "stabilize", "specialize"]
    )
    expected = pt.sqr(pt.prod(pt.diagonal(L, axis1=-2, axis2=-1), axis=-1).squeeze(0))
    assert_equal_computations([d_rewritten], [expected])


def test_det_of_matrix_factorized_elsewhere_transposed_input():
    x = pt.tensor("x", shape=(3, 3))

    # Both the det and the Cholesky hang off the transpose itself
    L = pt.linalg.cholesky(x.mT)
    d = det(x.mT)

    _, d_rewritten = rewrite_graph(
        [L, d], include=["canonicalize", "stabilize", "specialize"]
    )
    expected = pt.sqr(pt.prod(pt.diag(L), axis=0))
    assert_equal_computations([d_rewritten], [expected])


def test_slogdet_of_inv():
    x = pt.dmatrix("x")
    # slogdet(inv(x)) -> (sign, logabsdet)
    sign_inv, logabsdet_inv = pt.linalg.slogdet(pt.linalg.inv(x))

    # expected: (sign(det(x)), -logabsdet(det(x)))
    # det(inv(x)) = 1/det(x), so sign is same.
    # logabsdet(inv(x)) = log(abs(1/det(x))) = -log(abs(det(x)))
    sign_x, logabsdet_x = pt.linalg.slogdet(x)
    expected_sign = sign_x
    expected_logabsdet = -logabsdet_x

    # We need stabilize for det_of_inv and log_reciprocal
    # and specialize for slogdet_specialization
    rewritten_sign, rewritten_logabsdet = rewrite_graph(
        [sign_inv, logabsdet_inv], include=["canonicalize", "stabilize", "specialize"]
    )

    expected_sign_opt, expected_logabsdet_opt = rewrite_graph(
        [expected_sign, expected_logabsdet],
        include=["canonicalize", "stabilize", "specialize"],
    )

    assert_equal_computations([rewritten_sign], [expected_sign_opt])
    assert_equal_computations([rewritten_logabsdet], [expected_logabsdet_opt])


@pytest.mark.parametrize("factor_kind", ["cholesky", "lu_factor", "lu", "lu_indices"])
@pytest.mark.parametrize(
    "dtype, batched",
    [(config.floatX, False), ("complex128", True)],
    ids=["real", "batched_complex"],
)
def test_det_factor_sharing(factor_kind, batched, dtype):
    x = pt.tensor("x", shape=(None, 3, 3) if batched else (3, 3), dtype=dtype)
    if factor_kind == "cholesky":
        factor = cholesky(x)
        diag = pt.diagonal(factor, axis1=-2, axis2=-1)
        expected_det = pt.sqr(diag.prod(axis=-1))
    else:
        if factor_kind == "lu_factor":
            factor, pivots = lu.lu_factor(x)
            n_swaps = pt.neq(pivots, pt.arange(3)).sum(axis=-1)
            permutation_sign = 1 - 2 * (n_swaps % 2)
        else:
            p, _, factor = lu.lu(x, p_indices=factor_kind == "lu_indices")
            indices = p if factor_kind == "lu_indices" else pt.argmax(p, axis=-1)
            permutation_sign = permutation_indices_to_det_sign(indices)
        permutation_sign = permutation_sign.astype(dtype)
        diag = pt.diagonal(factor, axis1=-2, axis2=-1)
        expected_det = permutation_sign * diag.prod(axis=-1)
    result = RewriteTester(
        [x],
        [factor, det(x)],
        include=["canonicalize", "stabilize", "specialize", "slogdet_specialization"],
    )
    result.assert_graph(
        *rewrite_graph(
            [factor, expected_det],
            include=["stabilize", "specialize"],
        )
    )

    rng = np.random.default_rng(11)
    m = rng.normal(size=(3, 3))
    if dtype.startswith("complex"):
        m = m + 1j * rng.normal(size=(3, 3))
    spd = (m @ m.conj().T + 3 * np.eye(3)).astype(dtype)
    # A swap is odd; a three-cycle is even despite moving every index.
    permutations = [
        (np.eye(3)[perm] * [2, 3, 4]).astype(dtype) for perm in ([1, 0, 2], [1, 2, 0])
    ]
    values = [spd] if factor_kind == "cholesky" else [*permutations, m.astype(dtype)]
    rtol = 1e-5 if dtype == "float32" else 1e-7
    if batched:
        result.assert_eval(np.stack(values), rtol=rtol)
    else:
        for value in values:
            result.assert_eval(value, rtol=rtol)


@pytest.mark.parametrize("factor_kind", ["permuted_lu", "svd", "qr"])
def test_logabsdet_shared_from_sign_free_factor(factor_kind):
    x = pt.tensor("x", shape=(3, 3))
    if factor_kind == "svd":
        factor = svd.svd(x, compute_uv=False)
        expected = pt.log(pt.abs(factor)).sum(axis=-1)
    else:
        if factor_kind == "qr":
            factor = qr.qr(x)[-1]
        else:
            factor = lu.lu(x, permute_l=True)[-1]
        expected = pt.log(pt.abs(pt.diagonal(factor))).sum(axis=-1)
    _, logabsdet = slogdet(x)
    result = RewriteTester(
        [x],
        [factor, logabsdet],
        include=["canonicalize", "stabilize", "specialize", "slogdet_specialization"],
    )
    result.assert_graph(
        *rewrite_graph([factor, expected], include=["stabilize", "specialize"])
    )
    negative = np.eye(3, dtype=config.floatX)[[1, 0, 2]] * np.array(
        [2.0, 3.0, 4.0], dtype=config.floatX
    )
    result.assert_eval(negative, rtol=1e-3 if config.floatX == "float32" else 1e-7)


@pytest.mark.parametrize("summary", [det, slogdet])
def test_permuted_lu_determinant_fallback(summary):
    x = pt.tensor("x", shape=(3, 3))
    u = lu.lu(x, permute_l=True)[-1]
    outputs = [det(x)] if summary is det else list(slogdet(x))
    result = RewriteTester(
        [x],
        [*outputs, u],
        include=["canonicalize", "stabilize", "specialize", "slogdet_specialization"],
    )
    expected = [det(x)] if summary is det else list(Blockwise(SLogDet())(x))
    result.assert_graph(*expected, u)
