from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.rewriting.basic import register_canonicalize, register_specialize
from pytensor.tensor.sort import SortOp
from pytensor.tensor.subtensor import Subtensor, get_constant_idx


def _reversed_axes(node) -> set[int] | None:
    """Axes reversed by a Subtensor of whole-axis slices, else None."""
    indices = get_constant_idx(node.op.idx_list, node.inputs, allow_partial=True)
    if not all(
        isinstance(index, slice)
        and index.start is None
        and index.stop is None
        and index.step in (None, -1)
        for index in indices
    ):
        return None
    return {axis for axis, index in enumerate(indices) if index.step == -1}


@register_canonicalize
@register_specialize
@node_rewriter([SortOp])
def local_sort_under_reduce(fgraph, node):
    """reduce(f(sort(x))) -> reduce(f(x)) when only reductions see the sorted order.

    Every ``CAReduce`` is commutative, so a sort along a reduced axis cannot change its
    result. Between the sort and the reductions there may be ``Elemwise`` ops whose
    other inputs are broadcast along the permuted axes, ``DimShuffle`` ops, and reversals
    of whole axes, which are dropped along with the sort.
    """
    [x] = node.inputs
    [sorted_x] = node.outputs

    # Each reversal adds the axes it flips to the permuted set of everything below it
    reversals = []
    pending = [(sorted_x, {node.op.axis})]
    while pending:
        var, permuted = pending.pop()
        for client, input_index in fgraph.clients[var]:
            match client.op:
                case CAReduce(axis=axis) if axis is None or permuted <= set(axis):
                    pass
                case Elemwise() if len(client.outputs) == 1 and all(
                    inp.type.broadcastable[axis]
                    for i, inp in enumerate(client.inputs)
                    if i != input_index
                    for axis in permuted
                ):
                    pending.append((client.outputs[0], permuted))
                case DimShuffle(new_order=new_order):
                    # A dropped axis is broadcastable, so no permutation can act on it
                    moved = {new_order.index(a) for a in permuted if a in new_order}
                    pending.append((client.outputs[0], moved))
                case Subtensor() if input_index == 0 and (
                    (reversed_axes := _reversed_axes(client)) is not None
                ):
                    reversals.append(client)
                    pending.append((client.outputs[0], permuted | reversed_axes))
                case _:
                    return None

    # Replace the deepest reversals first, so every replaced variable is still in the
    # graph when its turn comes
    replacements = {
        reversal.outputs[0]: reversal.inputs[0] for reversal in reversed(reversals)
    }
    replacements[sorted_x] = x
    return replacements
