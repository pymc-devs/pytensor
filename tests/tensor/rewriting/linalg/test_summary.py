import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytensor import function
from pytensor import tensor as pt
from pytensor.configdefaults import config
from pytensor.graph import rewrite_graph
from pytensor.tensor._linalg.decomposition import lu, qr, svd
from pytensor.tensor._linalg.decomposition.cholesky import cholesky
from pytensor.tensor._linalg.summary import Det, SLogDet, det
from pytensor.tensor.type import matrix
from tests.unittest_tools import assert_equal_computations


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


def test_det_of_diag_incorrect_for_rectangle_eye():
    x = pt.matrix("x")
    x_diag = pt.eye(7, 5) * x
    with pytest.raises(ValueError, match="Determinant not defined"):
        pt.linalg.det(x_diag)


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
            lambda x: pt.sum(pt.log(pt.sqr(x))),
            id="log_prod_sqr",
        ),
        pytest.param(
            lambda x: pt.log(pt.abs(pt.prod(x))),
            lambda x: pt.sum(pt.log(pt.abs(x))),
            id="log_abs_prod",
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


@pytest.mark.parametrize(
    "expected, pos_tag",
    [
        pytest.param(
            lambda x: pt.sum(pt.log(x)),
            True,
            id="local_log_prod_to_sum_log_positive_tag",
        ),
        pytest.param(
            lambda x: pt.log(pt.prod(x)),
            False,
            id="local_log_prod_to_sum_log_no_rewrite",
        ),
    ],
)
def test_local_log_prod_to_sum_log_positive_tag(expected, pos_tag):
    x = pt.tensor("x", shape=(3, 4))
    if pos_tag:
        x.tag.positive = True

    out = pt.log(pt.prod(x))

    rewritten = rewrite_graph(out, include=["stabilize", "specialize"])
    assert_equal_computations([rewritten], [expected(x)])


@pytest.mark.parametrize(
    "decomp_fn, expected_fn",
    [
        pytest.param(
            lambda x: cholesky(x),
            lambda x: pt.sqr(pt.prod(pt.diag(cholesky(x)), axis=0)),
            id="cholesky",
        ),
        pytest.param(
            lambda x: lu.lu(x)[-1],
            lambda x: pt.prod(pt.extract_diag(lu.lu(x)[-1]), axis=0),
            id="lu",
        ),
        pytest.param(
            lambda x: lu.lu_factor(x)[0],
            lambda x: pt.prod(pt.extract_diag(lu.lu_factor(x)[0]), axis=0),
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
