import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.assumptions import UPPER_TRIANGULAR
from pytensor.graph.replace import vectorize_graph
from pytensor.tensor.linalg.solvers.triangular import SolveTriangular
from research.vjp_sparsity.algebra import nonzero_algebra_db
from research.vjp_sparsity.rewriting import infer_nonzero, nonzero_basic_query
from tests.unittest_tools import RewriteTester


@pytest.mark.parametrize("known_upper", [True, False])
def test_projected_conjugate_solve(known_upper):
    raw = pt.matrix("matrix")
    direction = pt.matrix("direction")
    matrix = UPPER_TRIANGULAR.assume(raw) if known_upper else raw
    solve = SolveTriangular(lower=False, b_ndim=2)
    product = pt.dot(matrix, direction)
    projection = pt.tril(product) - pt.diag(pt.diagonal(product) / 2.0)
    original = solve(matrix, solve(matrix, projection.T).T)
    result = RewriteTester(
        [raw, direction],
        [original],
        include=None,
        custom_rewrite=nonzero_algebra_db.query("+basic"),
    )
    if known_upper:
        lower_direction = pt.tril(direction)
        product = pt.dot(matrix, lower_direction)
        complement = pt.triu(product, 1) + pt.diag(pt.diagonal(product) / 2.0)
        expected = solve(matrix, (lower_direction - solve(matrix, complement)).T).T
        result.assert_graph(expected)
    else:
        result.assert_graph(original)

    rng = np.random.default_rng(12)
    for size in (2, 5):
        value = rng.normal(size=(size, size))
        if known_upper:
            value = np.triu(value)
        value += 4 * np.eye(size)
        result.assert_eval(value, rng.normal(size=(size, size)), atol=1e-12)


def test_projected_solve_masks_do_not_loosen():
    rows, columns = pt.lvector("rows"), pt.lvector("columns")
    matrix = UPPER_TRIANGULAR.assume(pt.eye(4)[rows, columns].set(1.0))
    direction = pt.matrix("direction", shape=(4, 4))
    solve = SolveTriangular(lower=False, b_ndim=2)
    product = pt.dot(matrix, direction)
    projection = pt.tril(product) - pt.diag(pt.diagonal(product) / 2.0)
    original = solve(matrix, solve(matrix, projection.T).T)
    masks = [
        infer_nonzero(
            original, direction, query=nonzero_basic_query.excluding("algebra")
        ),
        infer_nonzero(original, direction),
    ]
    batched = vectorize_graph(
        masks, replace={direction: pt.eye(16, dtype="bool").reshape((16, 4, 4))}
    )
    function = pytensor.function([rows, columns], batched)
    upper_rows, upper_columns = np.triu_indices(4, k=1)
    for bits in range(1 << len(upper_rows)):
        selected = bits & (1 << np.arange(len(upper_rows))) != 0
        before, after = function(upper_rows[selected], upper_columns[selected])
        assert not (after & ~before).any()
