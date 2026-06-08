from pytensor.assumptions.core import AssumptionKey, FactState, true_if
from pytensor.scalar.basic import (
    Add,
    Mul,
    Pow,
    Sub,
    TrueDiv,
    UnaryScalarOp,
)
from pytensor.tensor.basic import (
    NotScalarConstantError,
    get_underlying_scalar_constant_value,
)
from pytensor.tensor.subtensor import is_provably_positive


def elemwise_preserves_zero_pattern(
    key: AssumptionKey, op, feature, node, input_states
) -> list[FactState]:
    """Inference for keys whose defining property is a fixed pattern of zeros.

    The output preserves *key* when:
    - ``Mul``: at least one such input has *key* and is a non-singleton matrix.
    - ``Add`` / ``Sub``: every such input has *key*, and at least one exists.
    - ``TrueDiv``: the numerator has *key* and is a non-singleton matrix.
    - ``Pow``: the base has *key*, is a non-singleton matrix, and the exponent is provably positive.
    - Zero-preserving unary ops (e.g. ``Neg``, ``Abs``, ``Sin``, ``Sqr``):
      output inherits *key* from the input.
    """

    def _not_singleton_matrix(var) -> bool:
        return var.type.ndim >= 2 and not all(var.type.broadcastable[-2:])

    scalar_op = op.scalar_op

    if isinstance(scalar_op, Mul):
        return true_if(
            any(
                feature.check(inp, key) and _not_singleton_matrix(inp)
                for inp in node.inputs
            )
        )

    if isinstance(scalar_op, (Add, Sub)):
        matrix_inputs = []
        broadcast_inputs = []
        for inp in node.inputs:
            if _not_singleton_matrix(inp):
                matrix_inputs.append(inp)
            else:
                broadcast_inputs.append(inp)
        if not matrix_inputs:
            return [FactState.UNKNOWN]

        # A scalar / vector / (1, 1) input broadcasts to every entry of the matrix,
        # so it adds (or subtracts) its value at every off-diagonal position too.
        # The zero pattern survives only if every such broadcast input is zero.
        for inp in broadcast_inputs:
            try:
                val = get_underlying_scalar_constant_value(inp)
            except NotScalarConstantError:
                return [FactState.UNKNOWN]
            if val != 0:
                return [FactState.UNKNOWN]
        return true_if(all(feature.check(inp, key) for inp in matrix_inputs))

    if isinstance(scalar_op, TrueDiv):
        numerator = node.inputs[0]
        return true_if(
            feature.check(numerator, key) and _not_singleton_matrix(numerator)
        )

    if isinstance(scalar_op, Pow):
        base = node.inputs[0]
        if not (feature.check(base, key) and _not_singleton_matrix(base)):
            return [FactState.UNKNOWN]
        # 0 ** p == 0 for p > 0, so a provably-positive exponent (scalar or
        # elementwise matrix) preserves the base's zero pattern.
        return true_if(is_provably_positive(node.inputs[1]))

    if isinstance(scalar_op, UnaryScalarOp) and scalar_op.preserves_zero:
        return true_if(input_states[0] is FactState.TRUE)

    return [FactState.UNKNOWN] * len(node.outputs)
