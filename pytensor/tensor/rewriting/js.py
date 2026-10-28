"""Graph fusions for the JavaScript backend."""

from pytensor.compile import optdb
from pytensor.compile.builders import OpFromGraph
from pytensor.graph.rewriting.basic import dfs_rewriter, node_rewriter
from pytensor.scalar.basic import Neg
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.linalg.solvers.triangular import SolveTriangular


class NegatedSolveTriangular(OpFromGraph):
    """Keep RHS negation inside the triangular-solve staging step."""

    def __init__(self, solve_op, a_type, b_type):
        a, b = a_type(), b_type()
        self.solve_op = solve_op
        super().__init__([a, b], [solve_op(a, -b)], inline=False)


@node_rewriter([SolveTriangular])
def local_js_negated_solve(fgraph, node):
    a, b = node.inputs
    if (
        b.owner
        and isinstance(b.owner.op, Elemwise)
        and isinstance(b.owner.op.scalar_op, Neg)
        and b.ndim == 2
        and a.dtype == b.dtype == "float64"
        and len(fgraph.clients[b]) == 1
    ):
        source = b.owner.inputs[0]
        op = NegatedSolveTriangular(node.op, a.type, source.type)
        return [op(a, source)]


optdb.register(
    "js_negated_solve",
    dfs_rewriter(local_js_negated_solve),
    "js",
    "symbolic_op_recognition",
    position=101,
)
