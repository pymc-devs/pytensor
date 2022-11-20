import numpy as np
import numpy.linalg

import pytensor
from pytensor import function
from pytensor import tensor as at
from pytensor.configdefaults import config
from pytensor.sandbox.linalg.ops import inv_as_solve, spectral_radius_bound
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import _allclose
from pytensor.tensor.nlinalg import MatrixInverse, matrix_inverse
from pytensor.tensor.slinalg import Cholesky, Solve, solve
from pytensor.tensor.type import dmatrix, matrix, vector
from tests import unittest_tools as utt
from tests.test_rop import break_op


def test_rop_lop():
    mx = matrix("mx")
    mv = matrix("mv")
    v = vector("v")
    y = matrix_inverse(mx).sum(axis=0)

    yv = pytensor.gradient.Rop(y, mx, mv)
    rop_f = function([mx, mv], yv)

    sy, _ = pytensor.scan(
        lambda i, y, x, v: (pytensor.gradient.grad(y[i], x) * v).sum(),
        sequences=at.arange(y.shape[0]),
        non_sequences=[y, mx, mv],
    )
    scan_f = function([mx, mv], sy)

    rng = np.random.default_rng(utt.fetch_seed())
    vx = np.asarray(rng.standard_normal((4, 4)), pytensor.config.floatX)
    vv = np.asarray(rng.standard_normal((4, 4)), pytensor.config.floatX)

    v1 = rop_f(vx, vv)
    v2 = scan_f(vx, vv)

    assert _allclose(v1, v2), f"ROP mismatch: {v1} {v2}"

    raised = False
    try:
        pytensor.gradient.Rop(
            pytensor.clone_replace(y, replace={mx: break_op(mx)}), mx, mv
        )
    except ValueError:
        raised = True
    if not raised:
        raise Exception(
            "Op did not raised an error even though the function"
            " is not differentiable"
        )

    vv = np.asarray(rng.uniform(size=(4,)), pytensor.config.floatX)
    yv = pytensor.gradient.Lop(y, mx, v)
    lop_f = function([mx, v], yv)

    sy = pytensor.gradient.grad((v * y).sum(), mx)
    scan_f = function([mx, v], sy)

    v1 = lop_f(vx, vv)
    v2 = scan_f(vx, vv)
    assert _allclose(v1, v2), f"LOP mismatch: {v1} {v2}"


def test_spectral_radius_bound():
    tol = 10 ** (-6)
    rng = np.random.default_rng(utt.fetch_seed())
    x = matrix()
    radius_bound = spectral_radius_bound(x, 5)
    f = pytensor.function([x], radius_bound)

    shp = (3, 4)
    m = rng.random(shp)
    m = np.cov(m).astype(config.floatX)
    radius_bound_pytensor = f(m)

    # test the approximation
    mm = m
    for i in range(5):
        mm = np.dot(mm, mm)
    radius_bound_numpy = np.trace(mm) ** (2 ** (-5))
    assert abs(radius_bound_numpy - radius_bound_pytensor) < tol

    # test the bound
    eigen_val = numpy.linalg.eig(m)
    assert (eigen_val[0].max() - radius_bound_pytensor) < tol

    # test type errors
    xx = vector()
    ok = False
    try:
        spectral_radius_bound(xx, 5)
    except TypeError:
        ok = True
    assert ok
    ok = False
    try:
        spectral_radius_bound(x, 5.0)
    except TypeError:
        ok = True
    assert ok

    # test value error
    ok = False
    try:
        spectral_radius_bound(x, -5)
    except ValueError:
        ok = True
    assert ok


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


def test_tag_solve_triangular():
    cholesky_lower = Cholesky(lower=True)
    cholesky_upper = Cholesky(lower=False)
    A = matrix("A")
    x = vector("x")
    L = cholesky_lower(A)
    U = cholesky_upper(A)
    b1 = solve(L, x)
    b2 = solve(U, x)
    f = pytensor.function([A, x], b1)
    if config.mode != "FAST_COMPILE":
        for node in f.maker.fgraph.toposort():
            if isinstance(node.op, Solve):
                assert node.op.assume_a != "gen" and node.op.lower
    f = pytensor.function([A, x], b2)
    if config.mode != "FAST_COMPILE":
        for node in f.maker.fgraph.toposort():
            if isinstance(node.op, Solve):
                assert node.op.assume_a != "gen" and not node.op.lower


def test_matrix_inverse_solve():
    A = dmatrix("A")
    b = dmatrix("b")
    node = matrix_inverse(A).dot(b).owner
    [out] = inv_as_solve.transform(None, node)
    assert isinstance(out.owner.op, Solve)
