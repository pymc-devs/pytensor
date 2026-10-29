"""Show the lifted graph and compare it with the PR under identical cleanup."""

# ruff: noqa: T201

from collections import Counter

import pytensor
import pytensor.tensor as pt
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.graph.traversal import io_toposort
from research.vjp_sparsity.prototype import jacobian_sparsity_vjp
from tests.sparsity.test_elemwise import CASES


def graph_stats(pattern):
    cleaned = rewrite_graph(pattern, include=("ShapeOpt", "canonicalize"), clone=True)
    nodes = io_toposort([], [cleaned])
    floating = sum(
        getattr(output.type, "dtype", "").startswith("float")
        for node in nodes
        for output in node.outputs
    )
    return {
        "nodes": len(nodes),
        "float_outputs": floating,
        "ops": dict(Counter(str(node.op) for node in nodes)),
    }


def main():
    x = pt.vector("x", shape=(3,))
    y = pt.sin(x * [0.0, 2.0, 0.0])
    diagnostics = {}
    jacobian_sparsity_vjp(y, x, diagnostics=diagnostics)
    print("Ordinary VJP:")
    pytensor.dprint(diagnostics["vjp"])
    print("Lifted VJP:")
    pytensor.dprint(diagnostics["lifted_vjp"])

    cases = [
        "matrix * row",
        "matrix * mask",
        "switch(condition, matrix, row)",
        "sin(matrix * mask) * row",
    ]
    comparisons = [
        (
            title,
            graph_stats(jacobian_sparsity_vjp(outputs, wrt)),
            graph_stats(pt.jacobian_sparsity(outputs, wrt)),
        )
        for title, (outputs, wrt) in CASES.items()
    ]
    for title, vjp, pr in comparisons:
        if title in cases:
            print(title)
            print("VJP", vjp)
            print("PR", pr)
    print("Totals over", len(comparisons), "separate case graphs:")
    for key in ("nodes", "float_outputs"):
        print(
            key,
            {
                "VJP": sum(a[key] for _, a, _ in comparisons),
                "PR": sum(b[key] for _, _, b in comparisons),
            },
        )
    print("VJP smaller:", sum(a["nodes"] < b["nodes"] for _, a, b in comparisons))
    print("VJP larger:", sum(a["nodes"] > b["nodes"] for _, a, b in comparisons))


if __name__ == "__main__":
    main()
