import numpy as np
import pytest

from pytensor import config, function, scan
from pytensor.compile.mode import get_default_mode
from pytensor.gradient import grad
from pytensor.scan.op import Scan
from pytensor.tensor._linalg.solve.rewriting import (
    reuse_lu_decomposition_multiple_solves,
    scan_split_non_sequence_lu_decomposition_solve,
)
from pytensor.tensor._linalg.solve.tridiagonal import (
    LUFactorTridiagonal,
    SolveLUFactorTridiagonal,
)
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg import solve
from pytensor.tensor.slinalg import LUFactor, Solve, SolveTriangular
from pytensor.tensor.type import tensor


def count_vanilla_solve_nodes(nodes) -> int:
    return sum(
        (
            isinstance(node.op, Solve)
            or (isinstance(node.op, Blockwise) and isinstance(node.op.core_op, Solve))
        )
        for node in nodes
    )


def count_lu_decom_nodes(nodes) -> int:
    return sum(
        (
            isinstance(node.op, LUFactor | LUFactorTridiagonal)
            or (
                isinstance(node.op, Blockwise)
                and isinstance(node.op.core_op, LUFactor | LUFactorTridiagonal)
            )
        )
        for node in nodes
    )


def count_lu_solve_nodes(nodes) -> int:
    count = sum(
        (
            # LUFactor uses 2 SolveTriangular nodes, so we count each as 0.5
            0.5
            * (
                isinstance(node.op, SolveTriangular)
                or (
                    isinstance(node.op, Blockwise)
                    and isinstance(node.op.core_op, SolveTriangular)
                )
            )
            or (
                isinstance(node.op, SolveLUFactorTridiagonal)
                or (
                    isinstance(node.op, Blockwise)
                    and isinstance(node.op.core_op, SolveLUFactorTridiagonal)
                )
            )
        )
        for node in nodes
    )
    return int(count)


@pytest.mark.parametrize("transposed", (False, True))
@pytest.mark.parametrize("assume_a", ("gen", "tridiagonal"))
def test_lu_decomposition_reused_forward_and_gradient(assume_a, transposed):
    rewrite_name = reuse_lu_decomposition_multiple_solves.__name__
    mode = get_default_mode()

    A = tensor("A", shape=(3, 3))
    b = tensor("b", shape=(3, 4))

    x = solve(A, b, assume_a=assume_a, transposed=transposed)
    grad_x_wrt_A = grad(x.sum(), A)
    fn_no_opt = function([A, b], [x, grad_x_wrt_A], mode=mode.excluding(rewrite_name))
    no_opt_nodes = fn_no_opt.maker.fgraph.apply_nodes
    assert count_vanilla_solve_nodes(no_opt_nodes) == 2
    assert count_lu_decom_nodes(no_opt_nodes) == 0
    assert count_lu_solve_nodes(no_opt_nodes) == 0

    fn_opt = function([A, b], [x, grad_x_wrt_A], mode=mode.including(rewrite_name))
    opt_nodes = fn_opt.maker.fgraph.apply_nodes
    assert count_vanilla_solve_nodes(opt_nodes) == 0
    assert count_lu_decom_nodes(opt_nodes) == 1
    assert count_lu_solve_nodes(opt_nodes) == 2

    # Make sure results are correct
    rng = np.random.default_rng(31)
    A_test = rng.random(A.type.shape, dtype=A.type.dtype)
    b_test = rng.random(b.type.shape, dtype=b.type.dtype)
    resx0, resg0 = fn_no_opt(A_test, b_test)
    resx1, resg1 = fn_opt(A_test, b_test)
    rtol = 1e-7 if config.floatX == "float64" else 1e-4
    np.testing.assert_allclose(resx0, resx1, rtol=rtol)
    np.testing.assert_allclose(resg0, resg1, rtol=rtol)


@pytest.mark.parametrize("transposed", (False, True))
@pytest.mark.parametrize("assume_a", ("gen", "tridiagonal"))
def test_lu_decomposition_reused_blockwise(assume_a, transposed):
    rewrite_name = reuse_lu_decomposition_multiple_solves.__name__
    mode = get_default_mode()

    A = tensor("A", shape=(3, 3))
    b = tensor("b", shape=(2, 3, 4))

    x = solve(A, b, assume_a=assume_a, transposed=transposed)
    fn_no_opt = function([A, b], [x], mode=mode.excluding(rewrite_name))
    no_opt_nodes = fn_no_opt.maker.fgraph.apply_nodes
    assert count_vanilla_solve_nodes(no_opt_nodes) == 1
    assert count_lu_decom_nodes(no_opt_nodes) == 0
    assert count_lu_solve_nodes(no_opt_nodes) == 0

    fn_opt = function([A, b], [x], mode=mode.including(rewrite_name))
    opt_nodes = fn_opt.maker.fgraph.apply_nodes
    assert count_vanilla_solve_nodes(opt_nodes) == 0
    assert count_lu_decom_nodes(opt_nodes) == 1
    assert count_lu_solve_nodes(opt_nodes) == 1

    # Make sure results are correct
    rng = np.random.default_rng(31)
    A_test = rng.random(A.type.shape, dtype=A.type.dtype)
    b_test = rng.random(b.type.shape, dtype=b.type.dtype)
    resx0 = fn_no_opt(A_test, b_test)
    resx1 = fn_opt(A_test, b_test)
    rtol = rtol = 1e-7 if config.floatX == "float64" else 1e-4
    np.testing.assert_allclose(resx0, resx1, rtol=rtol)


@pytest.mark.parametrize("transposed", (False, True))
@pytest.mark.parametrize("assume_a", ("gen", "tridiagonal"))
def test_lu_decomposition_reused_scan(assume_a, transposed):
    rewrite_name = scan_split_non_sequence_lu_decomposition_solve.__name__
    mode = get_default_mode()

    A = tensor("A", shape=(3, 3))
    x0 = tensor("b", shape=(3, 4))

    xs, _ = scan(
        lambda xtm1, A: solve(A, xtm1, assume_a=assume_a, transposed=transposed),
        outputs_info=[x0],
        non_sequences=[A],
        n_steps=10,
    )

    fn_no_opt = function(
        [A, x0],
        [xs],
        mode=mode.excluding(rewrite_name),
    )
    [no_opt_scan_node] = [
        node for node in fn_no_opt.maker.fgraph.apply_nodes if isinstance(node.op, Scan)
    ]
    no_opt_nodes = no_opt_scan_node.op.fgraph.apply_nodes
    assert count_vanilla_solve_nodes(no_opt_nodes) == 1
    assert count_lu_decom_nodes(no_opt_nodes) == 0
    assert count_lu_solve_nodes(no_opt_nodes) == 0

    fn_opt = function([A, x0], [xs], mode=mode.including("scan", rewrite_name))
    [opt_scan_node] = [
        node for node in fn_opt.maker.fgraph.apply_nodes if isinstance(node.op, Scan)
    ]
    opt_nodes = opt_scan_node.op.fgraph.apply_nodes
    assert count_vanilla_solve_nodes(opt_nodes) == 0
    # The LU decomp is outside of the scan!
    assert count_lu_decom_nodes(opt_nodes) == 0
    assert count_lu_solve_nodes(opt_nodes) == 1

    # Make sure results are correct
    rng = np.random.default_rng(170)
    A_test = rng.random(A.type.shape, dtype=A.type.dtype)
    x0_test = rng.random(x0.type.shape, dtype=x0.type.dtype)
    resx0 = fn_no_opt(A_test, x0_test)
    resx1 = fn_opt(A_test, x0_test)
    rtol = 1e-7 if config.floatX == "float64" else 1e-4
    np.testing.assert_allclose(resx0, resx1, rtol=rtol)


def test_lu_decomposition_reused_preserves_check_finite():
    # Check that the LU decomposition rewrite preserves the check_finite flag
    rewrite_name = reuse_lu_decomposition_multiple_solves.__name__

    A = tensor("A", shape=(2, 2))
    b1 = tensor("b1", shape=(2,))
    b2 = tensor("b2", shape=(2,))

    x1 = solve(A, b1, assume_a="gen", check_finite=True)
    x2 = solve(A, b2, assume_a="gen", check_finite=False)
    fn_opt = function(
        [A, b1, b2], [x1, x2], mode=get_default_mode().including(rewrite_name)
    )
    opt_nodes = fn_opt.maker.fgraph.apply_nodes
    assert count_vanilla_solve_nodes(opt_nodes) == 0
    assert count_lu_decom_nodes(opt_nodes) == 1
    assert count_lu_solve_nodes(opt_nodes) == 2

    # We should get an error if A or b1 is non finite
    A_valid = np.array([[1, 0], [0, 1]], dtype=A.type.dtype)
    b1_valid = np.array([1, 1], dtype=b1.type.dtype)
    b2_valid = np.array([1, 1], dtype=b2.type.dtype)

    assert fn_opt(A_valid, b1_valid, b2_valid)  # Fine
    assert fn_opt(
        A_valid, b1_valid, b2_valid * np.nan
    )  # Should not raise (also fine on most LAPACK implementations?)
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        assert fn_opt(A_valid, b1_valid * np.nan, b2_valid)
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        assert fn_opt(A_valid * np.nan, b1_valid, b2_valid)
