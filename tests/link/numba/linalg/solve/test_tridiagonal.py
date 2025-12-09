import numpy as np
import pytest
import scipy

from pytensor import In
from pytensor import tensor as pt
from pytensor.tensor._linalg.solve.tridiagonal import (
    LUFactorTridiagonal,
    SolveLUFactorTridiagonal,
)
from pytensor.tensor.blockwise import Blockwise
from tests.link.numba.test_basic import compare_numba_and_py, numba_inplace_mode


@pytest.mark.parametrize("inplace", [False, True], ids=lambda x: f"inplace={x}")
def test_tridiagonal_lu_factor(inplace):
    dl = pt.vector("dl", shape=(4,))
    d = pt.vector("d", shape=(5,))
    du = pt.vector("du", shape=(4,))
    lu_factor_outs = Blockwise(LUFactorTridiagonal())(dl, d, du)

    rng = np.random.default_rng(734)
    dl_test = rng.random(dl.type.shape)
    d_test = rng.random(d.type.shape)
    du_test = rng.random(du.type.shape)

    f, results = compare_numba_and_py(
        [
            In(dl, mutable=inplace),
            In(d, mutable=inplace),
            In(du, mutable=inplace),
        ],
        lu_factor_outs,
        test_inputs=[dl_test, d_test, du_test],
        inplace=True,
        numba_mode=numba_inplace_mode,
        eval_obj_mode=False,
    )

    # Test with contiguous inputs
    dl_test_contig = dl_test.copy()
    d_test_contig = d_test.copy()
    du_test_contig = du_test.copy()
    results_contig = f(dl_test_contig, d_test_contig, du_test_contig)
    for res, res_contig in zip(results, results_contig):
        np.testing.assert_allclose(res, res_contig)
    assert (dl_test_contig == dl_test).all() == (not inplace)
    assert (d_test_contig == d_test).all() == (not inplace)
    assert (du_test_contig == du_test).all() == (not inplace)

    # Test with non-contiguous inputs
    dl_test_not_contig = np.repeat(dl_test, 2)[::2]
    d_test_not_contig = np.repeat(d_test, 2)[::2]
    du_test_not_contig = np.repeat(du_test, 2)[::2]
    results_not_contig = f(dl_test_not_contig, d_test_not_contig, du_test_not_contig)
    for res, res_not_contig in zip(results, results_not_contig):
        np.testing.assert_allclose(res, res_not_contig)
    # Non-contiguous inputs have to be copied so are not modified in place
    assert (dl_test_not_contig == dl_test).all()
    assert (d_test_not_contig == d_test).all()
    assert (du_test_not_contig == du_test).all()


@pytest.mark.parametrize("transposed", [False, True], ids=lambda x: f"transposed={x}")
@pytest.mark.parametrize("inplace", [True, False], ids=lambda x: f"inplace={x}")
@pytest.mark.parametrize("b_ndim", [1, 2], ids=lambda x: f"b_ndim={x}")
def test_tridiagonal_lu_solve(b_ndim, transposed, inplace):
    scipy_gttrf = scipy.linalg.get_lapack_funcs("gttrf")

    dl = pt.tensor("dl", shape=(9,))
    d = pt.tensor("d", shape=(10,))
    du = pt.tensor("du", shape=(9,))
    du2 = pt.tensor("du2", shape=(8,))
    ipiv = pt.tensor("ipiv", shape=(10,), dtype="int32")
    diagonals = [dl, d, du, du2, ipiv]
    b = pt.tensor("b", shape=(10, 25)[:b_ndim])

    x = Blockwise(SolveLUFactorTridiagonal(b_ndim=b.type.ndim, transposed=transposed))(
        *diagonals, b
    )

    rng = np.random.default_rng(787)
    A_test = rng.random((d.type.shape[0], d.type.shape[0]))
    *diagonals_test, _ = scipy_gttrf(
        *(np.diagonal(A_test, offset=o) for o in (-1, 0, 1))
    )
    b_test = rng.random(b.type.shape)

    f, res = compare_numba_and_py(
        [
            *diagonals,
            In(b, mutable=inplace),
        ],
        x,
        test_inputs=[*diagonals_test, b_test],
        inplace=True,
        numba_mode=numba_inplace_mode,
        eval_obj_mode=False,
    )

    # Test with contiguous_inputs
    diagonals_test_contig = [d_test.copy() for d_test in diagonals_test]
    b_test_contig = b_test.copy(order="F")
    res_contig = f(*diagonals_test_contig, b_test_contig)
    assert (res_contig == res).all()
    assert (b_test == b_test_contig).all() == (not inplace)

    # Test with non-contiguous inputs
    diagonals_test_non_contig = [np.repeat(d_test, 2)[::2] for d_test in diagonals_test]
    b_test_non_contig = np.repeat(b_test, 2, axis=0)[::2]
    res_non_contig = f(*diagonals_test_non_contig, b_test_non_contig)
    assert (res_non_contig == res).all()
    # b must be copied when not contiguous so it can't be inplaced
    assert (b_test == b_test_non_contig).all()


def test_cast_needed():
    dl = pt.vector("dl", shape=(4,), dtype="int16")
    d = pt.vector("d", shape=(5,), dtype="float32")
    du = pt.vector("du", shape=(4,), dtype="float64")
    b = pt.vector("b", shape=(5,), dtype="float32")

    lu_factor_outs = LUFactorTridiagonal()(dl, d, du)
    for i, out in enumerate(lu_factor_outs):
        if i == 4:
            assert out.type.dtype == "int32"  # ipiv is int32
        else:
            assert out.type.dtype == "float64"

    lu_solve_out = SolveLUFactorTridiagonal(b_ndim=1, transposed=False)(
        *lu_factor_outs, b
    )
    assert lu_solve_out.type.dtype == "float64"

    compare_numba_and_py(
        [dl, d, du, b],
        lu_solve_out,
        test_inputs=[
            np.array([1, 2, 3, 4], dtype="int16"),
            np.array([1, 2, 3, 4, 5], dtype="float32"),
            np.array([1, 2, 3, 4], dtype="float64"),
            np.array([1, 2, 3, 4, 5], dtype="float32"),
        ],
        eval_obj_mode=False,
    )
