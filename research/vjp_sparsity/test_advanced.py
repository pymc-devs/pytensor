"""Representative advanced cases; the full notebook audit lives in gist_audit."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.builders import OpFromGraph
from research.vjp_sparsity.prototype import jacobian_sparsity_vjp
from tests.sparsity.conftest import detected_and_numeric_patterns


a = pt.matrix("a")
b = pt.matrix("b")
rhs = pt.vector("rhs")
batch = pt.tensor3("batch")
x = pt.vector("x")
positive = a + 10 * pt.eye(a.shape[0])
blocks = pt.linalg.block_diag(positive, b + 10 * pt.eye(b.shape[0]))
block_rhs = pt.concatenate([rhs, rhs[:2]])
rng = np.random.default_rng(2)
sparse_stack = rng.normal(size=(2, 4, 3)) * (rng.uniform(size=(2, 4, 3)) < 0.5)
inner = pt.matrix("inner")
wrapped = OpFromGraph([inner], [pt.sin(inner * np.eye(4))])(a)
equations = pt.stack(
    [
        x[0] ** 2 + pt.sin(x[1]),
        x[1] ** 3,
        pt.exp(x[2]) + x[0],
        pt.exp(x[3]) + 3 * x[0],
        x[4] ** 3 + x[1] * x[4],
    ]
)
rosenbrock = pt.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
two_blocks = np.kron(np.eye(2), np.ones((2, 2)))
arrow = np.eye(4)
arrow[-1, :] = arrow[:, -1] = 1

CASES = {
    "slice": ([a[1:3, ::2]], [a]),
    "join/split": ([pt.concatenate([a, rhs[None]])], [a, rhs]),
    "block diagonal": ([pt.linalg.block_diag(a, b)], [a, b]),
    "block diagonal dot": ([blocks @ block_rhs], [a, b, rhs]),
    "batched sparse dot": ([batch @ sparse_stack], [batch]),
    "Kronecker": ([pt.linalg.kron(pt.eye(2), a)], [a]),
    "block diagonal solve": ([pt.linalg.solve(blocks, block_rhs)], [a, b, rhs]),
    "triangular solve": (
        [pt.linalg.solve_triangular(positive, rhs, lower=True)],
        [a, rhs],
    ),
    "unit triangular solve": (
        [pt.linalg.solve_triangular(positive, rhs, lower=True, unit_diagonal=True)],
        [a, rhs],
    ),
    "Cholesky solve lower": ([pt.linalg.cho_solve((positive, True), rhs)], [a, rhs]),
    "Cholesky solve upper": ([pt.linalg.cho_solve((positive, False), rhs)], [a, rhs]),
    "inner graph": ([wrapped], [a]),
    "nonlinear system": ([equations], [x]),
    "Rosenbrock Hessian": ([pt.grad(rosenbrock, x)], [x]),
    "Cholesky arrow last": ([pt.linalg.cholesky(a * arrow + 10 * np.eye(4))], [a]),
    "Cholesky dense": ([pt.linalg.cholesky(positive)], [a]),
    "QR masked": (list(pt.linalg.qr(a * two_blocks, mode="economic")), [a]),
}
CONSERVATIVE_BOUNDS = {"QR masked": 44}


@pytest.fixture(scope="module")
def patterns():
    rng = np.random.default_rng(13)
    values = [
        rng.uniform(0.5, 1.5, shape)
        for shape in ((4, 4), (2, 2), (4,), (2, 4, 4), (8,))
    ]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pt, "jacobian_sparsity", jacobian_sparsity_vjp)
        with pytensor.config.change_flags(mode="FAST_COMPILE"):
            reference = detected_and_numeric_patterns(
                CASES, [a, b, rhs, batch, x], values
            )
    function = pytensor.function(
        [a, b, rhs, batch, x],
        [jacobian_sparsity_vjp(outputs, wrt) for outputs, wrt in CASES.values()],
        on_unused_input="ignore",
    )
    return {
        title: (detected, reference[title][1])
        for title, detected in zip(CASES, function(*values), strict=True)
    }


@pytest.mark.parametrize("title", CASES)
def test_advanced_patterns(patterns, title):
    detected, numeric = patterns[title]
    assert not (numeric & ~detected).any()
    if title in CONSERVATIVE_BOUNDS:
        assert detected.sum() <= CONSERVATIVE_BOUNDS[title]
    else:
        np.testing.assert_array_equal(detected, numeric)


def test_cholesky_first_column_precision():
    matrix = pt.matrix("matrix")
    pattern = jacobian_sparsity_vjp(pt.linalg.cholesky(matrix), matrix)
    function = pytensor.function([matrix], pattern)
    # L[2, 0] = A[2, 0] / sqrt(A[0, 0]); A[1, 0] cannot affect it.
    assert not function(np.eye(3))[6, 3]
