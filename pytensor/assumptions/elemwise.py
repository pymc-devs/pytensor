from pytensor.assumptions.core import AssumptionKey, FactState, true_if
from pytensor.scalar.basic import (
    Add,
    Mul,
    Pow,
    Sub,
    Switch,
    TrueDiv,
    UnaryScalarOp,
)
from pytensor.tensor.basic import (
    NotScalarConstantError,
    get_underlying_scalar_constant_value,
)
from pytensor.tensor.subtensor import _is_provably_positive


def zero_preserving_input_groups(node) -> list[tuple[int, ...]]:
    """Return groups of input positions that keep the output zero wherever all are zero.

    The output of ``node`` is zero wherever every input of any one group is zero. An empty
    list means that no such group is known.
    """
    scalar_op = node.op.scalar_op
    n_inputs = len(node.inputs)

    match scalar_op:
        case Mul():
            return [(position,) for position in range(n_inputs)]
        case Add() | Sub():
            return [tuple(range(n_inputs))]
        case TrueDiv():
            return [(0,)]
        case Pow():
            # 0 ** p == 0 for p > 0, so only a positive exponent keeps the base's zeros.
            return [(0,)] if _is_provably_positive(node.inputs[1]) else []
        case Switch():
            return [(1, 2)]
        case UnaryScalarOp() if scalar_op.preserves_zero:
            return [(0,)]
        case _:
            return []


def elemwise_preserves_zero_pattern(
    key: AssumptionKey, op, feature, node, input_states
) -> list[FactState]:
    """Inference for keys whose defining property is a fixed pattern of zeros.

    The output preserves *key* when some group from :func:`zero_preserving_input_groups`
    preserves it. A group of one input needs *key* on a non-singleton matrix, and a group
    of several needs *key* on every matrix among them, with any broadcast input zero.
    Zero-preserving unary ops (e.g. ``Neg``, ``Abs``, ``Sin``, ``Sqr``) inherit *key* from
    the input.
    """

    def _not_singleton_matrix(var) -> bool:
        return var.type.ndim >= 2 and not all(var.type.broadcastable[-2:])

    def _group_preserves_key(inputs) -> bool:
        if len(inputs) == 1:
            [inp] = inputs
            return _not_singleton_matrix(inp) and feature.check(inp, key)

        matrix_inputs = []
        broadcast_inputs = []
        for inp in inputs:
            if _not_singleton_matrix(inp):
                matrix_inputs.append(inp)
            else:
                broadcast_inputs.append(inp)
        if not matrix_inputs:
            return False

        # A scalar / vector / (1, 1) input broadcasts to every entry of the matrix,
        # so it reaches every off-diagonal position too. The zero pattern survives
        # only if every such broadcast input is zero.
        for inp in broadcast_inputs:
            try:
                val = get_underlying_scalar_constant_value(inp)
            except NotScalarConstantError:
                return False
            if val != 0:
                return False
        return all(feature.check(inp, key) for inp in matrix_inputs)

    groups = zero_preserving_input_groups(node)
    if not groups:
        return [FactState.UNKNOWN] * len(node.outputs)

    if isinstance(op.scalar_op, UnaryScalarOp):
        return true_if(input_states[0] is FactState.TRUE)

    return true_if(
        any(
            _group_preserves_key([node.inputs[position] for position in group])
            for group in groups
        )
    )
