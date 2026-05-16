from pytensor.graph.basic import Variable
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import Dot


def _are_matrix_transposes(a: Variable, b: Variable) -> bool:
    """True iff ``a`` and ``b`` are matrix transposes of each other (in either direction)."""
    for y, x in ((a, b), (b, a)):
        match y.owner_op_and_inputs:
            case (DimShuffle() as ds, inner) if ds.is_matrix_transpose and inner is x:
                return True
    return False


def _matmul_inputs(var: Variable) -> tuple[Variable, Variable] | None:
    """If ``var`` is the output of a binary matmul (``Dot`` or ``Blockwise(Dot)``),
    return its two inputs; otherwise ``None``."""
    owner = var.owner
    if owner is None:
        return None
    op = owner.op
    if isinstance(op, Dot) or (
        isinstance(op, Blockwise) and isinstance(op.core_op, Dot)
    ):
        return owner.inputs[0], owner.inputs[1]
    return None


def match_congruence(node) -> Variable | None:
    """Detect a congruence ``M @ S @ M.T`` (or ``M.T @ S @ M``) at a matmul node.

    Both Python associativities are matched: ``(M @ S) @ M.T`` and
    ``M @ (S @ M.T)``, plus their mirrors. ``M`` and ``M.T`` must appear by
    variable identity; ``S`` is the inner matmul's other operand. The outer
    and inner matmuls may each be a plain ``Dot`` or a ``Blockwise(Dot)``
    (i.e. ``matmul``), since Python's ``@`` lowers to the latter.

    Returns the inner ``S`` variable on a match, or ``None``. The caller
    decides which assumption to check on ``S`` (``SYMMETRIC``, ``POSITIVE_DEFINITE``).
    """
    left, right = node.inputs

    # Right-associative: outer(M, inner(S, M.T))  or  outer(M.T, inner(S, M))
    inner = _matmul_inputs(right)
    if inner is not None:
        s, inner_right = inner
        if _are_matrix_transposes(left, inner_right):
            return s

    # Left-associative: outer(inner(M, S), M.T)  or  outer(inner(M.T, S), M)
    inner = _matmul_inputs(left)
    if inner is not None:
        inner_left, s = inner
        if _are_matrix_transposes(inner_left, right):
            return s

    return None
