import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import optdb
from pytensor.gradient import pullback
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.tensor.rewriting.shape import ShapeOptimizer
from research.vjp_sparsity.algebra import AlgebraRewriter
from research.vjp_sparsity.prototype import jacobian_sparsity_vjp


@pytest.mark.parametrize(
    "mode, shape",
    [
        ("full", (4, 4)),
        ("economic", (4, 4)),
        ("economic", (5, 3)),
        ("economic", (3, 5)),
    ],
    ids=["full square", "economic square", "economic tall", "economic wide"],
)
def test_q_times_r_pullback_simplifies_to_cotangent(mode, shape):
    a = pt.matrix("a", shape=shape)
    q, r = pt.linalg.qr(a, mode=mode)
    product = q @ r
    cotangent = product.type("cotangent")
    [a_bar] = pullback([product], [a], [cotangent])

    fgraph = FunctionGraph([a, cotangent], [a_bar], clone=True)
    ShapeOptimizer().rewrite(fgraph)
    optdb.query("+canonicalize").rewrite(fgraph)
    AlgebraRewriter().rewrite(fgraph)

    assert fgraph.outputs[0] is fgraph.inputs[1]


general = pt.matrix("general", shape=(3, 3))
upper = pt.linalg.qr(general)[1]
rhs = pt.matrix("rhs", shape=(3, 2))


@pytest.mark.parametrize(
    "solved, simplifies",
    [
        (pt.linalg.solve_triangular(upper, upper @ rhs, lower=False), True),
        (pt.linalg.solve(general, general @ rhs), True),
        (pt.linalg.solve_triangular(general, general @ rhs, lower=False), False),
        (pt.linalg.solve(upper, upper @ rhs, assume_a="pos"), False),
        (
            pt.linalg.solve_triangular(
                upper, upper @ rhs, lower=False, unit_diagonal=True
            ),
            False,
        ),
    ],
    ids=[
        "triangular solve of a triangular matrix",
        "general solve",
        "triangular solve ignoring a triangle",
        "positive definite solve ignoring a triangle",
        "unit diagonal solve ignoring the diagonal",
    ],
)
def test_solve_of_product_requires_solver_to_read_every_entry(solved, simplifies):
    fgraph = FunctionGraph([general, rhs], [solved], clone=True)
    AlgebraRewriter().rewrite(fgraph)
    assert (fgraph.outputs[0] is fgraph.inputs[1]) == simplifies


def test_algebra_stage_gives_exact_pattern():
    a = pt.matrix("a", shape=(5, 3))
    q, r = pt.linalg.qr(a, mode="economic")
    query = RewriteDatabaseQuery(include=["basic", "algebra"])
    pattern = jacobian_sparsity_vjp([q @ r], [a], query=query)
    detected = pytensor.function([a], pattern)(np.ones((5, 3)))
    np.testing.assert_array_equal(detected, np.eye(15, dtype=bool))
