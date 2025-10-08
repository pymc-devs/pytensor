from pytensor.graph.basic import Constant
from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
    indices_from_subtensor,
)
from pytensor.tensor.type_other import MakeSlice, SliceType


def check_negative_steps(indices):
    for index in indices:
        if isinstance(index, slice):
            if index.step is not None and index.step < 0:
                raise NotImplementedError(
                    "Negative step sizes are not supported in Pytorch"
                )


@pytorch_funcify.register(Subtensor)
def pytorch_funcify_Subtensor(op, node, **kwargs):
    idx_list = op.idx_list
    _x, *idxs = node.inputs

    if all(isinstance(idx, Constant) for idx in idxs):
        # Use constant indices to avoid graph break
        constant_indices = indices_from_subtensor(
            [int(idx.data) for idx in idxs], idx_list
        )
        check_negative_steps(constant_indices)

        def constant_index_subtensor(x, *_):
            return x[constant_indices]

        return constant_index_subtensor

    # Fallback that will introduce a graph break
    def subtensor(x, *flattened_indices):
        indices = indices_from_subtensor(flattened_indices, idx_list)
        check_negative_steps(indices)
        return x[indices]

    return subtensor


@pytorch_funcify.register(MakeSlice)
def pytorch_funcify_makeslice(op, **kwargs):
    def makeslice(start, stop, step):
        # Torch does not like numpy integers in indexing slices
        return slice(
            None if start is None else int(start),
            None if stop is None else int(stop),
            None if step is None else int(step),
        )

    return makeslice


@pytorch_funcify.register(AdvancedSubtensor1)
@pytorch_funcify.register(AdvancedSubtensor)
def pytorch_funcify_AdvSubtensor(op, node, **kwargs):
    def advsubtensor(x, *indices):
        check_negative_steps(indices)
        return x[indices]

    return advsubtensor


@pytorch_funcify.register(IncSubtensor)
def pytorch_funcify_IncSubtensor(op, node, **kwargs):
    idx_list = op.idx_list
    inplace = op.inplace
    if op.set_instead_of_inc:

        def set_subtensor(x, y, *flattened_indices):
            indices = indices_from_subtensor(flattened_indices, idx_list)
            check_negative_steps(indices)
            if not inplace:
                x = x.clone()
            x[indices] = y
            return x

        return set_subtensor

    else:

        def inc_subtensor(x, y, *flattened_indices):
            indices = indices_from_subtensor(flattened_indices, idx_list)
            check_negative_steps(indices)
            if not inplace:
                x = x.clone()
            x[indices] += y
            return x

        return inc_subtensor


@pytorch_funcify.register(AdvancedIncSubtensor)
@pytorch_funcify.register(AdvancedIncSubtensor1)
def pytorch_funcify_AdvancedIncSubtensor(op, node, **kwargs):
    inplace = op.inplace
    ignore_duplicates = getattr(op, "ignore_duplicates", False)

    if op.set_instead_of_inc:

        def adv_set_subtensor(x, y, *indices):
            check_negative_steps(indices)
            if isinstance(op, AdvancedIncSubtensor1):
                op._check_runtime_broadcasting(node, x, y, indices)
            if not inplace:
                x = x.clone()
            x[indices] = y.type_as(x)
            return x

        return adv_set_subtensor

    elif ignore_duplicates:

        def adv_inc_subtensor_no_duplicates(x, y, *indices):
            check_negative_steps(indices)
            if isinstance(op, AdvancedIncSubtensor1):
                op._check_runtime_broadcasting(node, x, y, indices)
            if not inplace:
                x = x.clone()
            x[indices] += y.type_as(x)
            return x

        return adv_inc_subtensor_no_duplicates

    else:
        if any(isinstance(idx.type, SliceType) for idx in node.inputs[2:]):
            raise NotImplementedError(
                "IncSubtensor with potential duplicates indexes and slice indexing not implemented in PyTorch"
            )

        def adv_inc_subtensor(x, y, *indices):
            # Not needed because slices aren't supported
            # check_negative_steps(indices)
            if not inplace:
                x = x.clone()
            x.index_put_(indices, y.type_as(x), accumulate=True)
            return x

        return adv_inc_subtensor
