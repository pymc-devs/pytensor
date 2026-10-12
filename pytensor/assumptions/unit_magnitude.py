import numpy as np

from pytensor.assumptions.core import (
    UNIT_MAGNITUDE,
    FactState,
    propagate_first,
    register_assumption,
    register_constant_inference,
    true_if,
)
from pytensor.scalar.basic import Abs, Cast, Conj, Mul, Neg, Reciprocal, Sqr
from pytensor.tensor.basic import Alloc
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.shape import Reshape


register_constant_inference(
    UNIT_MAGNITUDE,
    lambda var: FactState.TRUE if np.all(np.abs(var.data) == 1) else FactState.FALSE,
)


@register_assumption(UNIT_MAGNITUDE, Elemwise)
def elemwise_unit_magnitude(key, op, feature, fgraph, node, input_states):
    match op.scalar_op:
        case Abs() | Conj() | Neg() | Reciprocal() | Sqr():
            return true_if(input_states[0] is FactState.TRUE)
        case Mul():
            return true_if(all(state is FactState.TRUE for state in input_states))
        case Cast():
            inp, out = node.inputs[0], node.outputs[0]
            if out.dtype.startswith("uint") or out.dtype == "bool":
                return [FactState.UNKNOWN]
            if inp.dtype.startswith("complex") and not out.dtype.startswith("complex"):
                return [FactState.UNKNOWN]
            return true_if(input_states[0] is FactState.TRUE)
        case _:
            return [FactState.UNKNOWN]


register_assumption(UNIT_MAGNITUDE, Alloc, DimShuffle, Reshape)(propagate_first)
