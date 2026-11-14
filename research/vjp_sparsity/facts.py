"""Mock of a parameterized assumption: which sides of a matrix are orthonormal.

``ORTHOGONAL`` is a boolean today, so it cannot say that the economic ``Q`` of a tall matrix has
orthonormal columns but not rows. Here the fact's value is a subset of ``{COLUMNS, ROWS}``,
evidence joins by union, and a square shape upgrades either side to both. This is a stand-in
for a payload-carrying key in ``pytensor.assumptions``, not a proposal for its API.
"""

from pytensor.assumptions import ORTHOGONAL, check_assumption
from pytensor.assumptions.specify import SpecifyAssumptions
from pytensor.raise_op import Assert
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.decomposition.qr import QR
from pytensor.tensor.math import Dot


COLUMNS = "columns"
ROWS = "rows"
BOTH = frozenset({COLUMNS, ROWS})
NEITHER = frozenset()


def join(*evidence):
    return frozenset().union(*evidence)


def with_square_shape(variable, sides):
    """A square matrix with orthonormal columns also has orthonormal rows, and vice versa."""
    shape = variable.type.shape[-2:]
    if sides and len(shape) == 2 and shape[0] is not None and shape[0] == shape[1]:
        return BOTH
    return sides


def owner_sides(fgraph, variable, memo):
    node = variable.owner
    if node is None:
        return NEITHER
    op = node.op.core_op if isinstance(node.op, Blockwise) else node.op

    if isinstance(op, QR) and node.outputs.index(variable) == 0 and not op.pivoting:
        return {"full": BOTH, "economic": frozenset({COLUMNS})}.get(op.mode, NEITHER)
    if isinstance(op, DimShuffle) and op.new_order == (1, 0):
        sides = orthonormal_sides(fgraph, node.inputs[0], memo)
        return frozenset({COLUMNS: ROWS, ROWS: COLUMNS}[side] for side in sides)
    if isinstance(op, Assert | SpecifyAssumptions):
        return orthonormal_sides(fgraph, node.inputs[0], memo)
    if isinstance(op, Dot):
        left, right = (orthonormal_sides(fgraph, value, memo) for value in node.inputs)
        return left & right
    return NEITHER


def orthonormal_sides(fgraph, variable, memo=None):
    """The sides of ``variable`` known to be orthonormal: a subset of ``{COLUMNS, ROWS}``."""
    memo = {} if memo is None else memo
    if variable not in memo:
        evidence = [owner_sides(fgraph, variable, memo)]
        if check_assumption(fgraph, variable, ORTHOGONAL):
            evidence.append(BOTH)
        memo[variable] = with_square_shape(variable, join(*evidence))
    return memo[variable]
