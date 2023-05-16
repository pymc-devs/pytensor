import numpy as np
import numpy.linalg
import pytest
import scipy.linalg

import pytensor
from pytensor import function
from pytensor import tensor as at
from pytensor.compile import get_default_mode
from pytensor.configdefaults import config
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import _allclose
from pytensor.tensor.nlinalg import MatrixInverse, matrix_inverse
from pytensor.tensor.rewriting.linalg import inv_as_solve
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


@pytest.mark.parametrize("tag", ("lower", "upper", None))
@pytest.mark.parametrize("cholesky_form", ("lower", "upper"))
@pytest.mark.parametrize("product", ("lower", "upper", None))
def test_cholesky_ldotlt(tag, cholesky_form, product):
    cholesky = Cholesky(lower=(cholesky_form == "lower"))

    transform_removes_chol = tag is not None and product == tag
    transform_transposes = transform_removes_chol and cholesky_form != tag

    A = matrix("L")
    if tag:
        setattr(A.tag, tag + "_triangular", True)

    if product == "lower":
        M = A.dot(A.T)
    elif product == "upper":
        M = A.T.dot(A)
    else:
        M = A

    C = cholesky(M)
    f = pytensor.function([A], C, mode=get_default_mode().including("cholesky_ldotlt"))

    print(f.maker.fgraph.apply_nodes)

    no_cholesky_in_graph = not any(
        isinstance(node.op, Cholesky) for node in f.maker.fgraph.apply_nodes
    )

    assert no_cholesky_in_graph == transform_removes_chol

    if transform_transposes:
        assert any(
            isinstance(node.op, DimShuffle) and node.op.new_order == (1, 0)
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

    for Av in Avs:
        if tag == "upper":
            Av = Av.T

        if product == "lower":
            Mv = Av.dot(Av.T)
        elif product == "upper":
            Mv = Av.T.dot(Av)
        else:
            Mv = Av

        assert np.all(
            np.isclose(
                scipy.linalg.cholesky(Mv, lower=(cholesky_form == "lower")),
                f(Av),
            )
        )
