"""Shared Elemwise inference helpers for keys that denote a structural-zero
pattern (DIAGONAL, LOWER_TRIANGULAR, UPPER_TRIANGULAR).

These keys share the same elementwise propagation rules: any operation that
preserves the "structural zeros remain zero" invariant preserves the key.
"""

from pytensor.scalar.basic import (
    Add,
    Mul,
    Pow,
    Sub,
    TrueDiv,
    UnaryScalarOp,
)
from pytensor.tensor.assumptions.core import AssumptionKey, FactState
from pytensor.tensor.assumptions.utils import true_if
from pytensor.tensor.basic import (
    NotScalarConstantError,
    get_underlying_scalar_constant_value,
)


def elemwise_preserves_zero_pattern(
    key: AssumptionKey, op, feature, node, input_states
) -> list[FactState]:
    """Inference for keys whose defining property is a fixed pattern of zeros.

    The output preserves *key* when:
    - ``Mul``: at least one input has *key* and is a non-broadcasting matrix.
    - ``Add`` / ``Sub``: every input has *key*.
    - ``TrueDiv``: the numerator has *key*.
    - ``Pow``: the base has *key* and the exponent is a positive constant.
    - Zero-preserving unary ops (e.g. ``Neg``, ``Abs``, ``Sin``, ``Sqr``):
      output inherits *key* from the input.
    """
    scalar_op = op.scalar_op

    if isinstance(scalar_op, Mul):
        return true_if(
            any(
                feature.check(inp, key)
                and inp.type.ndim >= 2
                and not any(inp.type.broadcastable[-2:])
                for inp in node.inputs
            )
        )

    if isinstance(scalar_op, (Add, Sub)):
        return true_if(all(feature.check(inp, key) for inp in node.inputs))

    if isinstance(scalar_op, TrueDiv):
        return true_if(feature.check(node.inputs[0], key))

    if isinstance(scalar_op, Pow):
        if not feature.check(node.inputs[0], key):
            return [FactState.UNKNOWN]
        try:
            if get_underlying_scalar_constant_value(node.inputs[1]) > 0:
                return [FactState.TRUE]
        except NotScalarConstantError:
            pass
        return [FactState.UNKNOWN]

    if isinstance(scalar_op, UnaryScalarOp) and scalar_op.preserves_zero:
        return true_if(input_states[0])

    return [FactState.UNKNOWN] * len(node.outputs)
