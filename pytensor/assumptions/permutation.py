import numpy as np

from pytensor.assumptions.alloc import eye_identity_rule
from pytensor.assumptions.core import (
    PERMUTATION,
    FactState,
    all_inputs_have_key,
    propagate_first,
    register_assumption,
    register_constant_inference,
)
from pytensor.assumptions.dimshuffle import left_expand_dims_propagates_matrix_property
from pytensor.assumptions.subtensor import subtensor_propagates_matrix_property
from pytensor.tensor.basic import Eye
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.inverse import MatrixInverse, MatrixPinv
from pytensor.tensor.linalg.products import KroneckerProduct
from pytensor.tensor.math import Dot
from pytensor.tensor.subtensor import Subtensor
from pytensor.tensor.variable import TensorConstant


# A PERMUTATION matrix is a square 0/1 matrix with exactly one 1 in every row and every
# column -- a reordering of the identity's columns. It is a square SELECTION and is
# ORTHOGONAL; unlike a general selection it is closed under transpose and inverse
# (P.T = P^{-1} is again a permutation).


def _permutation_from_constant(var: TensorConstant) -> FactState:
    """Recognize a constant PERMUTATION matrix from its data, caching the scan on
    ``var.tag.is_permutation``."""
    cached: FactState | None = getattr(var.tag, "is_permutation", None)
    if cached is not None:
        return cached

    data = np.asarray(var.data)
    if data.ndim < 2 or data.shape[-1] != data.shape[-2]:
        result = FactState.FALSE
    else:
        with np.errstate(invalid="ignore"):
            if not (data.sum(axis=-2) == 1).all():
                is_permutation = False
            elif not (data.sum(axis=-1) == 1).all():
                is_permutation = False
            elif data.dtype.kind in "ub":
                is_permutation = True
            elif data.dtype.kind == "i":
                is_permutation = data.min(initial=0) >= 0
            else:
                n = data.shape[-1]
                is_permutation = np.count_nonzero(data) == (data.size // n if n else 0)
        result = FactState.TRUE if is_permutation else FactState.FALSE

    var.tag.is_permutation = result
    return result


register_constant_inference(PERMUTATION, _permutation_from_constant)

# eye(n, m, k) is a permutation iff it is the square identity -- the same condition that
# makes it orthogonal.
register_assumption(PERMUTATION, Eye)(eye_identity_rule)
# The symmetric group is closed under matmul and inversion.
register_assumption(PERMUTATION, Dot)(all_inputs_have_key)
register_assumption(PERMUTATION, MatrixInverse)(propagate_first)
register_assumption(PERMUTATION, MatrixPinv)(propagate_first)
register_assumption(PERMUTATION, BlockDiagonal)(all_inputs_have_key)
register_assumption(PERMUTATION, KroneckerProduct)(all_inputs_have_key)
register_assumption(PERMUTATION, Subtensor)(subtensor_propagates_matrix_property)


@register_assumption(PERMUTATION, DimShuffle)
def _dimshuffle(key, op, feature, fgraph, node, input_states):
    """Permutation survives matrix transpose (P.T = P^{-1}) and batch left-expand-dims."""
    if input_states[0] is not FactState.TRUE:
        return [FactState.UNKNOWN]
    if op.is_matrix_transpose or left_expand_dims_propagates_matrix_property(op):
        return [FactState.TRUE]
    return [FactState.UNKNOWN]
