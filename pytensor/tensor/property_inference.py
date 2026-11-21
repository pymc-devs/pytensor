from typing import TYPE_CHECKING

import numpy as np

from pytensor.graph.basic import Constant
from pytensor.scalar import (
    Cast,
    ScalarMaximum,
    ScalarMinimum,
    Sqr,
    UnaryScalarOp,
)
from pytensor.tensor.basic import (
    ExtractDiag,
    MakeVector,
    ScalarFromTensor,
    TensorFromScalar,
)
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import Prod
from pytensor.tensor.shape import Shape, Shape_i


if TYPE_CHECKING:
    from pytensor.graph.basic import Variable


def is_provably_positive(
    var: "Variable | int | np.integer", strict: bool = True
) -> bool:
    """``True`` when ``var`` can be statically shown to be positive.

    With ``strict=True`` this proves :math:`var > 0`; with ``strict=False`` it
    proves :math:`var \\geq 0`. Proofs through ``Prod`` and ``sqr`` use exact
    arithmetic; floating-point evaluation can underflow to zero.

    Recognized cases:

    - Python ``int`` / ``np.integer``.
    - ``TensorConstant`` / ``ScalarConstant`` whose data clears the bound
      (cached via ``tag.is_positive`` or ``tag.is_non_negative``).
    - ``MakeVector`` of provably positive inputs.
    - View / shape-permutation ops — ``Subtensor``, ``ScalarFromTensor``,
      ``TensorFromScalar``, ``DimShuffle`` — recurse to their input.
    - ``minimum(a, b)`` when both ``a`` and ``b`` are positive.
    - ``maximum(a, b)`` when at least one of ``a``, ``b`` is positive.
    - ``ExtractDiag`` taking the main diagonal of a ``Cholesky`` factor along
      its core matrix axes, which is positive whenever the factorization
      succeeded (``on_error="nan"`` yields NaN instead); any other diagonal
      recurses to the matrix.
    - ``Prod`` of a positive input, with float output and accumulation:
      integer accumulators can overflow even when the output dtype is float.
    - ``sqr`` of a strictly positive input, restricted to float outputs.

    Four further cases prove non-negativity but not strict positivity, so they
    are recognized only when ``strict=False``:

    - Unsigned-integer dtype (a ``uint`` may be 0).
    - ``Shape`` / ``Shape_i`` outputs (a dimension may be 0).
    - ``Cast`` of a non-negative input (a float :math:`0 < x < 1` truncates to 0).
    - A unary scalar op declaring ``non_negative`` (``abs``, ``sqr``, ``exp``, ...),
      restricted to float and unsigned outputs: signed integers wrap on overflow
      (``sqr(int8(12)) == -112``) and complex values are unordered.

    Parameters
    ----------
    var : Variable or int
        The value to test.
    strict : bool
        Prove :math:`> 0` when ``True``, :math:`\\geq 0` when ``False``. Default
        ``True``.
    """
    tag_name = "is_positive" if strict else "is_non_negative"

    if isinstance(var, int | np.integer):
        return bool(var > 0) if strict else bool(var >= 0)
    if not strict and var.type.dtype.startswith("uint"):
        return True
    if isinstance(var, Constant):
        cached: bool | None = getattr(var.tag, tag_name, None)
        if cached is not None:
            return cached
        data = np.asarray(var.data)
        result = bool((data > 0).all()) if strict else bool((data >= 0).all())
        setattr(var.tag, tag_name, result)
        return result
    op = var.owner_op
    if not strict and isinstance(op, Shape | Shape_i):
        return True
    if isinstance(op, MakeVector):
        return all(is_provably_positive(i, strict) for i in var.owner.inputs)

    # Subtensor is imported lazily to avoid circular imports with subtensor.py
    from pytensor.tensor.subtensor import Subtensor

    if isinstance(op, Subtensor | ScalarFromTensor | TensorFromScalar | DimShuffle):
        return is_provably_positive(var.owner.inputs[0], strict)
    if isinstance(op, ExtractDiag):
        from pytensor.tensor.blockwise import Blockwise
        from pytensor.tensor.linalg.decomposition.cholesky import Cholesky

        [matrix] = var.owner.inputs
        matrix_op = matrix.owner_op
        core_op = matrix_op.core_op if isinstance(matrix_op, Blockwise) else matrix_op
        diag_axes = {op.axis1 % matrix.ndim, op.axis2 % matrix.ndim}
        core_matrix_axes = {matrix.ndim - 2, matrix.ndim - 1}
        if (
            isinstance(core_op, Cholesky)
            and op.offset == 0
            and diag_axes == core_matrix_axes
        ):
            return True
        return is_provably_positive(matrix, strict)
    if (
        isinstance(op, Prod)
        and var.type.dtype.startswith("float")
        and op.acc_dtype.startswith("float")
    ):
        return is_provably_positive(var.owner.inputs[0], strict)
    if isinstance(op, Elemwise):
        scalar_op = op.scalar_op
        if not strict and isinstance(scalar_op, Cast):
            return is_provably_positive(var.owner.inputs[0], strict)
        if strict and isinstance(scalar_op, Sqr) and var.type.dtype.startswith("float"):
            return is_provably_positive(var.owner.inputs[0], strict=True)
        if (
            not strict
            and isinstance(scalar_op, UnaryScalarOp)
            and scalar_op.non_negative
            and var.type.dtype.startswith(("float", "uint"))
        ):
            return True
        if isinstance(scalar_op, ScalarMinimum):
            return all(is_provably_positive(i, strict) for i in var.owner.inputs)
        if isinstance(scalar_op, ScalarMaximum):
            return any(is_provably_positive(i, strict) for i in var.owner.inputs)
    return False


def is_provably_non_negative(var: "Variable | int | np.integer") -> bool:
    """``True`` when ``var`` can be statically shown to be non-negative (:math:`\\geq 0`).

    Thin wrapper over :func:`is_provably_positive` with ``strict=False``.
    """
    return is_provably_positive(var, strict=False)
