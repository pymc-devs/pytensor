import numpy as np
import pytest

from pytensor import config, function, scan
from pytensor import tensor as pt
from pytensor.compile.mode import get_default_mode
from pytensor.gradient import grad
from pytensor.graph import rewrite_graph
from pytensor.scan.op import Scan
from pytensor.tensor._linalg.solve.rewriting import (
    reuse_decomposition_multiple_solves,
    scan_split_non_sequence_decomposition_and_solve,
)
from pytensor.tensor._linalg.solve.tridiagonal import (
    LUFactorTridiagonal,
    SolveLUFactorTridiagonal,
)
from pytensor.tensor.blockwise import Blockwise, BlockwiseWithCoreShape
from pytensor.tensor.linalg import solve
from pytensor.tensor.nlinalg import det
from pytensor.tensor.slinalg import (
    Cholesky,
    CholeskySolve,
    LUFactor,
    Solve,
    SolveTriangular,
)
from pytensor.tensor.type import tensor
from tests.unittest_tools import assert_equal_computations


class DecompSolveOpCounter:
    def __init__(self, solve_op, decomp_op, solve_op_value: float = 1.0):
        self.solve_op = solve_op
        self.decomp_op = decomp_op
        self.solve_op_value = solve_op_value

    def check_node_op_or_core_op(self, node, op):
        return isinstance(node.op, op) or (
            isinstance(node.op, Blockwise | BlockwiseWithCoreShape)
            and isinstance(node.op.core_op, op)
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
    fn_no_opt = function(
        [A, b],
        [x, grad_x_wrt_A],
        mode=mode.excluding(rewrite_name, "psd_solve_to_chol_solve"),
    )
    no_opt_nodes = fn_no_opt.maker.fgraph.apply_nodes
    assert counter.count_vanilla_solve_nodes(no_opt_nodes) == 2
    assert counter.count_decomp_nodes(no_opt_nodes) == 0
    assert counter.count_solve_nodes(no_opt_nodes) == 0

    fn_opt = function(
        [A, b],
        [x, grad_x_wrt_A],
        mode=mode.including(rewrite_name).excluding("psd_solve_to_chol_solve"),
    )
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
    fn_no_opt = function(
        [A, b], [x], mode=mode.excluding(rewrite_name, "psd_solve_to_chol_solve")
    )
    no_opt_nodes = fn_no_opt.maker.fgraph.apply_nodes
    assert counter.count_vanilla_solve_nodes(no_opt_nodes) == 1
    assert counter.count_decomp_nodes(no_opt_nodes) == 0
    assert counter.count_solve_nodes(no_opt_nodes) == 0

    fn_opt = function(
        [A, b],
        [x],
        mode=mode.including(rewrite_name).excluding("psd_solve_to_chol_solve"),
    )
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
        mode=get_default_mode().excluding("psd_solve_to_chol_solve"),
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
            lambda x: pt.linalg.cholesky(x),
            lambda x: pt.sqr(pt.prod(pt.diag(pt.linalg.cholesky(x)), axis=0)),
            id="cholesky",
        ),
        pytest.param(
            lambda x: pt.linalg.lu(x)[-1],
            lambda x: pt.prod(pt.extract_diag(pt.linalg.lu(x)[-1]), axis=0),
            id="lu",
        ),
        pytest.param(
            lambda x: pt.linalg.lu_factor(x)[0],
            lambda x: pt.prod(pt.extract_diag(pt.linalg.lu_factor(x)[0]), axis=0),
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
            lambda x: pt.linalg.svd(x, compute_uv=True)[0],
            pt.abs,
            lambda x: pt.prod(pt.linalg.svd(x, compute_uv=True)[1], axis=0),
            id="svd_abs",
        ),
        pytest.param(
            lambda x: pt.linalg.svd(x, compute_uv=False),
            pt.abs,
            lambda x: pt.prod(pt.linalg.svd(x, compute_uv=False), axis=0),
            id="svd_no_uv_abs",
        ),
        pytest.param(
            lambda x: pt.linalg.qr(x)[0],
            pt.abs,
            lambda x: pt.prod(
                pt.diagonal(pt.linalg.qr(x)[1], axis1=-2, axis2=-1), axis=-1
            ),
            id="qr_abs",
        ),
        pytest.param(
            lambda x: pt.linalg.svd(x, compute_uv=True)[0],
            pt.sqr,
            lambda x: pt.prod(pt.linalg.svd(x, compute_uv=True)[1], axis=0),
            id="svd_sqr",
        ),
        pytest.param(
            lambda x: pt.linalg.svd(x, compute_uv=False),
            pt.sqr,
            lambda x: pt.prod(pt.linalg.svd(x, compute_uv=False), axis=0),
            id="svd_no_uv_sqr",
        ),
        pytest.param(
            lambda x: pt.linalg.qr(x)[0],
            pt.sqr,
            lambda x: pt.prod(
                pt.diagonal(pt.linalg.qr(x)[1], axis1=-2, axis2=-1), axis=-1
            ),
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
            lambda x: det(pt.linalg.cholesky(x)),
            lambda x: pt.prod(
                pt.diagonal(pt.linalg.cholesky(x), axis1=-2, axis2=-1), axis=-1
            ),
            id="det_cholesky",
        ),
        pytest.param(
            lambda x: det(pt.linalg.lu(x)[-1]),
            lambda x: pt.prod(
                pt.diagonal(pt.linalg.lu(x)[-1], axis1=-2, axis2=-1), axis=-1
            ),
            id="det_lu_U",
        ),
        pytest.param(
            lambda x: det(pt.linalg.lu(x)[-2]),
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
            lambda x: pt.abs(det(pt.linalg.svd(x, compute_uv=True)[0])),
            lambda x: pt.as_tensor(1.0, dtype=x.dtype),
            id="abs_det_svd_U",
        ),
        pytest.param(
            lambda x: pt.abs(det(pt.linalg.svd(x, compute_uv=True)[2])),
            lambda x: pt.as_tensor(1.0, dtype=x.dtype),
            id="abs_det_svd_Vt",
        ),
        pytest.param(
            lambda x: pt.abs(det(pt.linalg.qr(x)[0])),
            lambda x: pt.as_tensor(1.0, dtype=x.dtype),
            id="abs_det_qr_Q",
        ),
        pytest.param(
            lambda x: pt.sqr(det(pt.linalg.svd(x, compute_uv=True)[0])),
            lambda x: pt.as_tensor(1.0, dtype=x.dtype),
            id="sqr_det_svd_U",
        ),
        pytest.param(
            lambda x: pt.sqr(det(pt.linalg.svd(x, compute_uv=True)[2])),
            lambda x: pt.as_tensor(1.0, dtype=x.dtype),
            id="sqr_det_svd_Vt",
        ),
        pytest.param(
            lambda x: pt.sqr(det(pt.linalg.qr(x)[0])),
            lambda x: pt.as_tensor(1.0, dtype=x.dtype),
            id="sqr_det_qr_Q",
        ),
        pytest.param(
            lambda x: det(pt.linalg.qr(x)[1]),
            lambda x: pt.prod(
                pt.diagonal(pt.linalg.qr(x)[1], axis1=-2, axis2=-1), axis=-1
            ),
            id="det_qr_R",
        ),
        pytest.param(
            lambda x: det(pt.linalg.qr(x)[0]),
            lambda x: det(pt.linalg.qr(x)[0]),
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
