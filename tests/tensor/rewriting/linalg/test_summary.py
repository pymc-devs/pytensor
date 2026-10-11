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
from pytensor.tensor.linalg.summary import Det, SLogDet, det
from pytensor.tensor.type import matrix
from tests.unittest_tools import RewriteTester, assert_equal_computations


def core_op_names(nodes):
    return [type(getattr(node.op, "core_op", node.op)).__name__ for node in nodes]


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


@pytest.mark.parametrize(
    "decomp_fn, expected_fn",
    [
        pytest.param(
            lambda x: cholesky(x),
            lambda x: pt.prod(pt.sqr(pt.diag(cholesky(x))), axis=0),
            id="cholesky",
        ),
        pytest.param(
            lambda x: lu.lu_factor(x)[0],
            lambda x: (
                (
                    1 - 2 * (pt.neq(lu.lu_factor(x)[1], np.arange(3)).sum(axis=-1) % 2)
                ).astype(x.dtype)
                * pt.prod(pt.extract_diag(lu.lu_factor(x)[0]), axis=0)
            ),
            id="lu_factor",
        ),
    ],
)
def test_det_of_matrix_factorized_elsewhere(decomp_fn, expected_fn):
    x = pt.tensor("x", shape=(3, 3))

    decomp_var = decomp_fn(x)
    d = det(x)

    decomp_var, d = rewrite_graph(
        [decomp_var, d], include=["canonicalize", "stabilize", "specialize"]
    )
    assert_equal_computations([decomp_var], [decomp_fn(x)])
    assert_equal_computations([d], [expected_fn(x)])


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


def test_det_shared_from_lu_factor_keeps_the_sign():
    x = pt.tensor("x", shape=(3, 3))
    d = det(x)
    factor = lu.lu_factor(x)[0]
    f = function([x], [d, factor], mode="FAST_RUN")

    assert "Det" not in core_op_names(f.maker.fgraph.apply_nodes)

    # An odd permutation makes the determinant negative
    x_val = (np.eye(3)[[1, 0, 2]] * np.array([2.0, 3.0, 4.0])).astype(config.floatX)
    assert_allclose(f(x_val)[0], np.linalg.det(x_val), rtol=1e-5)


def test_signed_det_not_shared_from_lu():
    """The permutation sign is not recoverable from lu outputs, so a signed
    det must keep its own computation."""
    x = pt.tensor("x", shape=(3, 3))
    d = det(x)
    u = lu.lu(x)[-1]

    d_rewritten, _ = rewrite_graph(
        [d, u], include=["canonicalize", "stabilize", "specialize"]
    )
    assert_equal_computations([d_rewritten], [det(x)])


@pytest.mark.parametrize(
    "decomp_fn, sign_op, expected_fn",
    [
        pytest.param(
            lambda x: lu.lu(x)[-1],
            pt.abs,
            lambda x: pt.prod(pt.diagonal(lu.lu(x)[-1], axis1=-2, axis2=-1), axis=-1),
            id="lu_abs",
        ),
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


def test_det_of_inv():
    x = pt.tensor("x", shape=(3, 3))
    out = det(pt.linalg.inv(x))
    expected = pt.as_tensor(1.0, dtype="float64") / det(x)
    rewritten = rewrite_graph(out, include=["canonicalize", "stabilize"])
    assert_equal_computations([rewritten], [expected])


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


@pytest.mark.parametrize(
    "factor_fn, factor_op",
    [
        pytest.param(cholesky, "Cholesky", id="cholesky"),
        pytest.param(lambda x: lu.lu_factor(x)[0], "LUFactor", id="lu_factor"),
    ],
)
@pytest.mark.parametrize("batched", [False, True], ids=["matrix", "batched"])
def test_slogdet_of_matrix_factorized_elsewhere(factor_fn, factor_op, batched):
    x = pt.tensor("x", shape=(None, 3, 3) if batched else (3, 3))
    factor = factor_fn(x)
    sign_x, logabsdet_x = Blockwise(SLogDet())(x)

    result = RewriteTester(
        [x], [factor, sign_x, logabsdet_x], include=["stabilize", "specialize"]
    )

    rng = np.random.default_rng(11)
    M = rng.normal(size=(3, 3))
    spd = (M @ M.T + 3 * np.eye(3)).astype(config.floatX)
    # A permuted matrix with a negative determinant exercises the pivot parity
    # in the shared sign
    negative = np.eye(3, dtype=config.floatX)[[1, 0, 2]] * np.array(
        [2.0, 3.0, 4.0], dtype=config.floatX
    )
    rtol = 1e-3 if config.floatX == "float32" else 1e-7
    values = [spd] if factor_op == "Cholesky" else [spd, negative]
    if batched:
        result.assert_eval(np.stack(values), rtol=rtol)
    else:
        for value in values:
            result.assert_eval(value, rtol=rtol)

    core_ops = core_op_names(result.rewr_fg.toposort())
    assert "SLogDet" not in core_ops
    assert "Det" not in core_ops
    assert core_ops.count(factor_op) == 1


def test_slogdet_shared_from_cholesky_of_complex_matrix():
    """Regression: the shared logabsdet must stay real for complex input."""
    x = pt.tensor("x", shape=(3, 3), dtype="complex128")
    L = cholesky(x)
    sign_x, logabsdet_x = Blockwise(SLogDet())(x)

    result = RewriteTester(
        [x], [L, sign_x, logabsdet_x], include=["stabilize", "specialize"]
    )

    rng = np.random.default_rng(5)
    M = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    result.assert_eval(M @ M.conj().T + 3 * np.eye(3))

    assert "SLogDet" not in core_op_names(result.rewr_fg.toposort())


def test_slogdet_sign_shared_from_lu_factor_of_complex_matrix():
    """Regression: the unit part of a complex factor is u / abs(u), because
    Sign rejects complex input."""
    x = pt.tensor("x", shape=(3, 3), dtype="complex128")
    factor = lu.lu_factor(x)[0]
    sign_x, logabsdet_x = Blockwise(SLogDet())(x)

    result = RewriteTester(
        [x], [factor, sign_x, logabsdet_x], include=["stabilize", "specialize"]
    )

    rng = np.random.default_rng(3)
    result.assert_eval(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))

    assert "SLogDet" not in core_op_names(result.rewr_fg.toposort())


@pytest.mark.parametrize(
    "factor_fn",
    [
        pytest.param(cholesky, id="cholesky"),
        pytest.param(lambda x: lu.lu_factor(x)[0], id="lu_factor"),
    ],
)
def test_log_abs_det_factor_sharing_does_not_underflow(factor_fn):
    """Regression: the shared det must reach log as a product the
    log(abs(prod)) -> sum(log(abs)) stabilization can rewrite, or it
    underflows to -inf."""
    n = 60
    x = pt.tensor("x", shape=(n, n))
    out = pt.log(pt.abs(det(x)))
    f = function([x], [factor_fn(x), out], mode="FAST_RUN")

    core_ops = core_op_names(f.maker.fgraph.toposort())
    assert "Det" not in core_ops
    assert "SLogDet" not in core_ops

    x_val = np.diag(np.full(n, 1e-7)).astype(config.floatX)
    assert np.linalg.det(x_val) == 0.0
    assert_allclose(f(x_val)[1], n * np.log(1e-7), rtol=1e-5)


@pytest.mark.parametrize(
    "logdet_first", [True, False], ids=["logdet_first", "logdet_last"]
)
def test_slogdet_factor_sharing_is_order_independent(logdet_first):
    """Regression for pytensor#2477: node visit order decided whether the
    determinant shared the LU factor or kept a second factorization."""
    A = matrix("A")
    b1 = pt.vector("b1")
    b2 = pt.vector("b2")
    logdet = pt.log(pt.abs(det(A)))
    x1 = pt.linalg.solve(A, b1, b_ndim=1)
    x2 = pt.linalg.solve(A, b2, b_ndim=1)

    # Two solves of the same matrix make reuse_decomposition_multiple_solves
    # introduce the shared LUFactor; a single solve stays a direct Solve
    outputs = [logdet, x1, x2] if logdet_first else [x1, x2, logdet]
    f = function([A, b1, b2], outputs, mode="FAST_RUN")

    core_ops = core_op_names(f.maker.fgraph.toposort())
    assert "SLogDet" not in core_ops
    assert "Det" not in core_ops
    assert core_ops.count("LUFactor") == 1
