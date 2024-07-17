from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
    indices_from_subtensor,
)


@pytorch_funcify.register(Subtensor)
@pytorch_funcify.register(AdvancedSubtensor)
@pytorch_funcify.register(AdvancedSubtensor1)
def pytorch_funcify_Subtensor(op, node, **kwargs):
    idx_list = getattr(op, "idx_list", None)

    def subtensor(x, *ilists):
        indices = indices_from_subtensor(ilists, idx_list)
        new_indices = []

        for i in indices:
            new_start, new_step, new_stop = None, None, None
            if isinstance(i, slice):
                if i.start:
                    new_start = i.start.item()
                if i.step:
                    new_step = i.step.item()
                    if new_step < 0:
                        raise NotImplementedError(
                            "Negative step sizes are not supported in Pytorch"
                        )
                if i.stop:
                    new_stop = i.stop.item()
                new_indices.append(slice(new_start, new_stop, new_step))
            else:
                new_indices.append(i.tolist())

        if len(indices) == 1:
            indices = indices[0]

        return x[tuple(new_indices)]

    return subtensor


@pytorch_funcify.register(IncSubtensor)
@pytorch_funcify.register(AdvancedIncSubtensor1)
def pytorch_funcify_IncSubtensor(op, node, **kwargs):
    idx_list = getattr(op, "idx_list", None)

    if getattr(op, "set_instead_of_inc", False):

        def torch_fn(x, indices, y):
            return x.at[indices].set(y)

    else:

        def torch_fn(x, indices, y):
            return x.at[indices].add(y)

    def incsubtensor(x, y, *ilist, torch_fn=torch_fn, idx_list=idx_list):
        indices = indices_from_subtensor(ilist, idx_list)
        if len(indices) == 1:
            indices = indices[0]

        return torch_fn(x, indices, y)

    return incsubtensor
