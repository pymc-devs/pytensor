import numpy as np
import pytest
import scipy

from pytensor import function, grad
from pytensor.configdefaults import config
from pytensor.tensor.linalg import Cholesky, cholesky
from pytensor.tensor.type import matrix
from tests import unittest_tools as utt


def check_lower_triangular(pd, ch_f):
    ch = ch_f(pd)
    assert ch[0, pd.shape[1] - 1] == 0
    assert ch[pd.shape[0] - 1, 0] != 0
    assert np.allclose(np.dot(ch, ch.T), pd)
    assert not np.allclose(np.dot(ch.T, ch), pd)


def check_upper_triangular(pd, ch_f):
    ch = ch_f(pd)
    assert ch[4, 0] == 0
    assert ch[0, 4] != 0
    assert np.allclose(np.dot(ch.T, ch), pd)
    assert not np.allclose(np.dot(ch, ch.T), pd)


def test_cholesky():
    rng = np.random.default_rng(utt.fetch_seed())
    r = rng.standard_normal((5, 5)).astype(config.floatX)
    pd = np.dot(r, r.T)
    x = matrix()
    chol = cholesky(x)
    # Check the default.
    ch_f = function([x], chol)
    check_lower_triangular(pd, ch_f)
    # Explicit lower-triangular.
    chol = Cholesky(lower=True)(x)
    ch_f = function([x], chol)
    check_lower_triangular(pd, ch_f)
    # Explicit upper-triangular.
    chol = Cholesky(lower=False)(x)
    ch_f = function([x], chol)
    check_upper_triangular(pd, ch_f)


def test_cholesky_empty():
    empty = np.empty([0, 0], dtype=config.floatX)
    x = matrix()
    chol = cholesky(x)
    ch_f = function([x], chol)
    ch = ch_f(empty)
    assert ch.size == 0
    assert ch.dtype == config.floatX


def test_cholesky_indef():
    x = matrix()
    mat = np.array([[1, 0.2], [0.2, -2]]).astype(config.floatX)

    with pytest.warns(FutureWarning):
        out = cholesky(x, lower=True, on_error="raise")
    chol_f = function([x], out)
    with pytest.raises(scipy.linalg.LinAlgError):
        chol_f(mat)

    out = cholesky(x, lower=True, on_error="nan")
    chol_f = function([x], out)
    assert np.all(np.isnan(chol_f(mat)))


def test_cholesky_grad():
    rng = np.random.default_rng(utt.fetch_seed())
    r = rng.standard_normal((5, 5)).astype(config.floatX)

    # The dots are inside the graph since Cholesky needs separable matrices

    # Check the default.
    utt.verify_grad(lambda r: cholesky(r.dot(r.T)), [r], 3, rng)
    # Explicit lower-triangular.
    utt.verify_grad(
        lambda r: Cholesky(lower=True)(r.dot(r.T)),
        [r],
        3,
        rng,
        abs_tol=0.05,
        rel_tol=0.05,
    )

    # Explicit upper-triangular.
    utt.verify_grad(
        lambda r: Cholesky(lower=False)(r.dot(r.T)),
        [r],
        3,
        rng,
        abs_tol=0.05,
        rel_tol=0.05,
    )


def test_cholesky_grad_indef():
    x = matrix()
    mat = np.array([[1, 0.2], [0.2, -2]]).astype(config.floatX)

    with pytest.warns(FutureWarning):
        out = cholesky(x, lower=True, on_error="raise")
    chol_f = function([x], grad(out.sum(), [x]), mode="FAST_RUN")

    # original cholesky doesn't show up in the grad (if mode="FAST_RUN"), so it does not raise
    assert np.all(np.isnan(chol_f(mat)))

    out = cholesky(x, lower=True, on_error="nan")
    chol_f = function([x], grad(out.sum(), [x]))
    assert np.all(np.isnan(chol_f(mat)))


def test_cholesky_infer_shape():
    x = matrix()
    f_chol = function([x], [cholesky(x).shape, cholesky(x, lower=False).shape])
    if config.mode != "FAST_COMPILE":
        topo_chol = f_chol.maker.fgraph.toposort()
        f_chol.dprint()
        assert not any(
            isinstance(getattr(node.op, "core_op", node.op), Cholesky)
            for node in topo_chol
        )
    for shp in [2, 3, 5]:
        res1, res2 = f_chol(np.eye(shp).astype(x.dtype))
        assert tuple(res1) == (shp, shp)
        assert tuple(res2) == (shp, shp)
