import numpy as np

from pytensor.assumptions.core import (
    SELECTION,
    FactState,
    all_inputs_have_key,
    register_assumption,
    register_constant_inference,
    true_if,
)
from pytensor.assumptions.subtensor import subtensor_propagates_matrix_property
from pytensor.graph.basic import Variable
from pytensor.tensor.basic import Eye
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.products import KroneckerProduct
from pytensor.tensor.subtensor import AdvancedSubtensor, Subtensor
from pytensor.tensor.variable import TensorConstant


def _selection_from_constant(var: TensorConstant) -> FactState:
    """Recognize a constant SELECTION matrix from its data, caching the O(n*k) scan on
    ``var.tag.is_selection``."""
    cached: FactState | None = getattr(var.tag, "is_selection", None)
    if cached is not None:
        return cached

    data = np.asarray(var.data)
    if data.ndim < 2:
        result = FactState.FALSE
    else:
        with np.errstate(invalid="ignore"):
            if not (data.sum(axis=-2) == 1).all():
                is_selection = False
            elif data.dtype.kind in "ub":
                is_selection = True
            elif data.dtype.kind == "i":
                is_selection = data.min(initial=0) >= 0
            else:
                is_selection = bool(((data == 0) | (data == 1)).all())
        result = FactState.TRUE if is_selection else FactState.FALSE

    var.tag.is_selection = result
    return result


register_constant_inference(SELECTION, _selection_from_constant)


def column_selection_index(op, node) -> Variable | None:
    """Return the 1-D integer index of an ``x[..., idx]`` advanced subtensor, or ``None``.

    Matches every axis but the last a full slice, the last an integer vector. Selecting
    columns preserves selection for any ``idx``; selecting rows does not, so only this
    pattern is recognized.
    """
    x = node.inputs[0]
    ndim = x.type.ndim
    idx_list = op.idx_list
    if ndim < 2 or len(idx_list) != ndim:
        return None

    full = slice(None, None, None)
    if any(entry != full for entry in idx_list[:-1]):
        return None

    last = idx_list[-1]
    if not isinstance(last, int) or isinstance(last, bool):
        return None

    index_var: Variable = node.inputs[1:][last]
    if index_var.type.ndim != 1 or not index_var.type.dtype.startswith(("int", "uint")):
        return None
    return index_var


@register_assumption(SELECTION, Eye)
def _eye(key, op, feature, fgraph, node, input_states):
    """``eye(n, m, k)`` is a selection iff ``k == 0`` (main diagonal) and ``n >= m``
    (no trailing zero columns)."""
    n, m, k = node.inputs
    if not isinstance(k, TensorConstant):
        return [FactState.UNKNOWN]
    if k.data.item() != 0:
        return [FactState.FALSE]
    if n is m:
        return [FactState.TRUE]
    if isinstance(n, TensorConstant) and isinstance(m, TensorConstant):
        return true_if(n.data.item() >= m.data.item(), else_false=True)
    rows, cols = node.outputs[0].type.shape[-2:]
    if rows is not None and cols is not None:
        return true_if(rows >= cols, else_false=True)
    return [FactState.UNKNOWN]


register_assumption(SELECTION, BlockDiagonal)(all_inputs_have_key)
register_assumption(SELECTION, KroneckerProduct)(all_inputs_have_key)


@register_assumption(SELECTION, AdvancedSubtensor)
def _advanced_subtensor(key, op, feature, fgraph, node, input_states):
    if (
        input_states[0] is FactState.TRUE
        and column_selection_index(op, node) is not None
    ):
        return [FactState.TRUE]
    return [FactState.UNKNOWN]


@register_assumption(SELECTION, Subtensor)
def _subtensor(key, op, feature, fgraph, node, input_states):
    # Batch-axis indexing (trailing two axes untouched) keeps every per-matrix property.
    states = subtensor_propagates_matrix_property(
        key, op, feature, fgraph, node, input_states
    )
    if states[0] is FactState.TRUE:
        return states

    # A column slice keeps a subset of the one-hot columns -- still a selection.
    if input_states[0] is FactState.TRUE:
        x = node.inputs[0]
        idx_list = op.idx_list
        full = slice(None, None, None)
        if (
            2 <= x.type.ndim == len(idx_list)
            and all(entry == full for entry in idx_list[:-1])
            and isinstance(idx_list[-1], slice)
        ):
            return [FactState.TRUE]

    return [FactState.UNKNOWN]
