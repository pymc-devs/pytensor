import pytest

from pytensor.graph.traversal import (
    apply_ancestors,
    toposort,
    toposort_with_orderings,
    variable_ancestors,
)
from tests.graph.test_basic import MyOp, MyVariable


@pytest.mark.parametrize(
    "func",
    [
        lambda x: all(variable_ancestors([x])),
        lambda x: all(variable_ancestors([x], blockers=[x.clone()])),
        lambda x: all(apply_ancestors([x])),
        lambda x: all(apply_ancestors([x], blockers=[x.clone()])),
        lambda x: all(toposort([x])),
        lambda x: all(toposort([x], blockers=[x.clone()])),
        lambda x: all(toposort_with_orderings([x], orderings={x: []})),
        lambda x: all(
            toposort_with_orderings([x], blockers=[x.clone()], orderings={x: []})
        ),
    ],
    ids=[
        "variable_ancestors",
        "variable_ancestors_with_blockers",
        "apply_ancestors",
        "apply_ancestors_with_blockers)",
        "toposort",
        "toposort_with_blockers",
        "toposort_with_orderings",
        "toposort_with_orderings_and_blockers",
    ],
)
def test_traversal_benchmark(func, benchmark):
    r1 = MyVariable(1)
    out = r1
    for i in range(50):
        out = MyOp(out, out)

    benchmark(func, out)
