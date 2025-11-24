import numpy as np
import pytest

from pytensor import config, function, scan
from pytensor.compile.mode import get_default_mode
from pytensor.gradient import grad
from pytensor.scan.op import Scan
from pytensor.tensor._linalg.solve.rewriting import (
    reuse_decomposition_multiple_solves,
    scan_split_non_sequence_decomposition_and_solve,
)
from pytensor.tensor._linalg.solve.tridiagonal import (
    LUFactorTridiagonal,
    SolveLUFactorTridiagonal,
)
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg import solve
from pytensor.tensor.slinalg import (
    Cholesky,
    CholeskySolve,
    LUFactor,
    Solve,
    SolveTriangular,
)
from pytensor.tensor.type import tensor


class DecompSolveOpCounter:
    def __init__(self, solve_op, decomp_op, solve_op_value: float = 1.0):
        self.solve_op = solve_op
        self.decomp_op = decomp_op
        self.solve_op_value = solve_op_value

    def check_node_op_or_core_op(self, node, op):
        return isinstance(node.op, op) or (
            isinstance(node.op, Blockwise) and isinstance(node.op.core_op, op)
        )

    def count_vanilla_solve_nodes(self, nodes) -> int:
        return sum(self.check_node_op_or_core_op(node, Solve) for node in nodes)

    def count_decomp_nodes(self, nodes) -> int:
        return sum(
            self.check_node_op_or_core_op(node, self.decomp_op) for node in nodes
        )

    def count_solve_nodes(self, nodes) -> int:
        count = sum(
            self.solve_op_value * self.check_node_op_or_core_op(node, self.solve_op)
            for node in nodes
        )
        return int(count)


LUOpCounter = DecompSolveOpCounter(
    solve_op=SolveTriangular,
    decomp_op=LUFactor,
    # Each rewrite introduces two triangular solves, so count them as 1/2 each
    solve_op_value=0.5,
)

TriDiagLUOpCounter = DecompSolveOpCounter(
    solve_op=SolveLUFactorTridiagonal, decomp_op=LUFactorTridiagonal, solve_op_value=1.0
)

CholeskyOpCounter = DecompSolveOpCounter(
    solve_op=CholeskySolve, decomp_op=Cholesky, solve_op_value=1.0
)


@pytest.mark.parametrize("transposed", (False, True))
@pytest.mark.parametrize(
    "assume_a, counter",
    (
        ("gen", LUOpCounter),
        ("tridiagonal", TriDiagLUOpCounter),
        ("pos", CholeskyOpCounter),
    ),
)
def test_lu_decomposition_reused_forward_and_gradient(assume_a, counter, transposed):
    rewrite_name = reuse_decomposition_multiple_solves.__name__
    mode = get_default_mode()

    A = tensor("A", shape=(3, 3))
    b = tensor("b", shape=(3, 4))

    x = solve(A, b, assume_a=assume_a, transposed=transposed)
    grad_x_wrt_A = grad(x.sum(), A)
    fn_no_opt = function([A, b], [x, grad_x_wrt_A], mode=mode.excluding(rewrite_name))
    no_opt_nodes = fn_no_opt.maker.fgraph.apply_nodes
    assert counter.count_vanilla_solve_nodes(no_opt_nodes) == 2
    assert counter.count_decomp_nodes(no_opt_nodes) == 0
    assert counter.count_solve_nodes(no_opt_nodes) == 0

    fn_opt = function([A, b], [x, grad_x_wrt_A], mode=mode.including(rewrite_name))
    opt_nodes = fn_opt.maker.fgraph.apply_nodes
    assert counter.count_vanilla_solve_nodes(opt_nodes) == 0
    assert counter.count_decomp_nodes(opt_nodes) == 1
    assert counter.count_solve_nodes(opt_nodes) == 2

    # Make sure results are correct
    rng = np.random.default_rng(31)
    A_test = rng.random(A.type.shape, dtype=A.type.dtype)
    if assume_a == "pos":
        A_test = A_test @ A_test.T  # Ensure positive definite for Cholesky

    b_test = rng.random(b.type.shape, dtype=b.type.dtype)
    resx0, resg0 = fn_no_opt(A_test, b_test)
    resx1, resg1 = fn_opt(A_test, b_test)
    rtol = 1e-7 if config.floatX == "float64" else 1e-4
    np.testing.assert_allclose(resx0, resx1, rtol=rtol)
    np.testing.assert_allclose(resg0, resg1, rtol=rtol)


@pytest.mark.parametrize("transposed", (False, True))
@pytest.mark.parametrize(
    "assume_a, counter",
    (
        ("gen", LUOpCounter),
        ("tridiagonal", TriDiagLUOpCounter),
        ("pos", CholeskyOpCounter),
    ),
)
def test_lu_decomposition_reused_blockwise(assume_a, counter, transposed):
    rewrite_name = reuse_decomposition_multiple_solves.__name__
    mode = get_default_mode()

    A = tensor("A", shape=(3, 3))
    b = tensor("b", shape=(2, 3, 4))

    x = solve(A, b, assume_a=assume_a, transposed=transposed)
    fn_no_opt = function([A, b], [x], mode=mode.excluding(rewrite_name))
    no_opt_nodes = fn_no_opt.maker.fgraph.apply_nodes
    assert counter.count_vanilla_solve_nodes(no_opt_nodes) == 1
    assert counter.count_decomp_nodes(no_opt_nodes) == 0
    assert counter.count_solve_nodes(no_opt_nodes) == 0

    fn_opt = function([A, b], [x], mode=mode.including(rewrite_name))
    opt_nodes = fn_opt.maker.fgraph.apply_nodes
    assert counter.count_vanilla_solve_nodes(opt_nodes) == 0
    assert counter.count_decomp_nodes(opt_nodes) == 1
    assert counter.count_solve_nodes(opt_nodes) == 1

    # Make sure results are correct
    rng = np.random.default_rng(31)
    A_test = rng.random(A.type.shape, dtype=A.type.dtype)
    if assume_a == "pos":
        A_test = A_test @ A_test.T  # Ensure positive definite for Cholesky

    b_test = rng.random(b.type.shape, dtype=b.type.dtype)
    resx0 = fn_no_opt(A_test, b_test)
    resx1 = fn_opt(A_test, b_test)
    rtol = 1e-7 if config.floatX == "float64" else 1e-4
    np.testing.assert_allclose(resx0, resx1, rtol=rtol)


@pytest.mark.parametrize("transposed", (False, True))
@pytest.mark.parametrize(
    "assume_a, counter",
    (
        ("gen", LUOpCounter),
        ("tridiagonal", TriDiagLUOpCounter),
        ("pos", CholeskyOpCounter),
    ),
)
def test_lu_decomposition_reused_scan(assume_a, counter, transposed):
    rewrite_name = scan_split_non_sequence_decomposition_and_solve.__name__
    mode = get_default_mode()

    A = tensor("A", shape=(3, 3))
    x0 = tensor("b", shape=(3, 4))

    xs = scan(
        lambda xtm1, A: solve(A, xtm1, assume_a=assume_a, transposed=transposed),
        outputs_info=[x0],
        non_sequences=[A],
        n_steps=10,
        return_updates=False,
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
    assert counter.count_vanilla_solve_nodes(no_opt_nodes) == 1
    assert counter.count_decomp_nodes(no_opt_nodes) == 0
    assert counter.count_solve_nodes(no_opt_nodes) == 0

    fn_opt = function([A, x0], [xs], mode=mode.including("scan", rewrite_name))
    [opt_scan_node] = [
        node for node in fn_opt.maker.fgraph.apply_nodes if isinstance(node.op, Scan)
    ]
    opt_nodes = opt_scan_node.op.fgraph.apply_nodes
    assert counter.count_vanilla_solve_nodes(opt_nodes) == 0
    # The LU decomp is outside of the scan!
    assert counter.count_decomp_nodes(opt_nodes) == 0
    assert counter.count_solve_nodes(opt_nodes) == 1

    # Make sure results are correct
    rng = np.random.default_rng(170)
    A_test = rng.random(A.type.shape, dtype=A.type.dtype)
    if assume_a == "pos":
        A_test = A_test @ A_test.T  # Ensure positive definite for Cholesky

    x0_test = rng.random(x0.type.shape, dtype=x0.type.dtype)
    resx0 = fn_no_opt(A_test, x0_test)
    resx1 = fn_opt(A_test, x0_test)
    rtol = 1e-7 if config.floatX == "float64" else 1e-4
    np.testing.assert_allclose(resx0, resx1, rtol=rtol)


@pytest.mark.parametrize(
    "assume_a, counter",
    (
        ("gen", LUOpCounter),
        ("pos", CholeskyOpCounter),
    ),
)
def test_decomposition_reused_preserves_check_finite(assume_a, counter):
    # Check that the LU decomposition rewrite preserves the check_finite flag
    rewrite_name = reuse_decomposition_multiple_solves.__name__

    A = tensor("A", shape=(2, 2))
    b1 = tensor("b1", shape=(2,))
    b2 = tensor("b2", shape=(2,))

    x1 = solve(A, b1, assume_a=assume_a, check_finite=True)
    x2 = solve(A, b2, assume_a=assume_a, check_finite=False)
    fn_opt = function(
        [A, b1, b2], [x1, x2], mode=get_default_mode().including(rewrite_name)
    )
    opt_nodes = fn_opt.maker.fgraph.apply_nodes
    assert counter.count_vanilla_solve_nodes(opt_nodes) == 0
    assert counter.count_decomp_nodes(opt_nodes) == 1
    assert counter.count_solve_nodes(opt_nodes) == 2

    # We should get an error if A or b1 is non finite
    A_valid = np.array([[1, 0], [0, 1]], dtype=A.type.dtype)
    b1_valid = np.array([1, 1], dtype=b1.type.dtype)
    b2_valid = np.array([1, 1], dtype=b2.type.dtype)

    assert fn_opt(A_valid, b1_valid, b2_valid)  # Fine
    assert fn_opt(
        A_valid, b1_valid, b2_valid * np.nan
    )  # Should not raise (also fine on most LAPACK implementations?)
    err_msg = (
        "(array must not contain infs or NaNs"
        r"|Non-numeric values \(nan or inf\))"
    )
    with pytest.raises((ValueError, np.linalg.LinAlgError), match=err_msg):
        assert fn_opt(A_valid, b1_valid * np.nan, b2_valid)
    with pytest.raises((ValueError, np.linalg.LinAlgError), match=err_msg):
        assert fn_opt(A_valid * np.nan, b1_valid, b2_valid)
