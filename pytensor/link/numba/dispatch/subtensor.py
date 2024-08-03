import numpy as np

from pytensor.graph import Type
from pytensor.link.numba.dispatch import numba_funcify
from pytensor.link.numba.dispatch.basic import generate_fallback_impl, numba_njit
from pytensor.link.utils import compile_function_src, unique_name_generator
from pytensor.tensor import TensorType
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)


@numba_funcify.register(Subtensor)
@numba_funcify.register(IncSubtensor)
@numba_funcify.register(AdvancedSubtensor1)
def numba_funcify_default_subtensor(op, node, **kwargs):
    """Create a Python function that assembles and uses an index on an array."""

    unique_names = unique_name_generator(
        ["subtensor", "incsubtensor", "z"], suffix_sep="_"
    )

    def convert_indices(indices, entry):
        if indices and isinstance(entry, Type):
            rval = indices.pop(0)
            return unique_names(rval)
        elif isinstance(entry, slice):
            return (
                f"slice({convert_indices(indices, entry.start)}, "
                f"{convert_indices(indices, entry.stop)}, "
                f"{convert_indices(indices, entry.step)})"
            )
        elif isinstance(entry, type(None)):
            return "None"
        else:
            raise ValueError()

    set_or_inc = isinstance(
        op, IncSubtensor | AdvancedIncSubtensor1 | AdvancedIncSubtensor
    )
    index_start_idx = 1 + int(set_or_inc)

    input_names = [unique_names(v, force_unique=True) for v in node.inputs]
    op_indices = list(node.inputs[index_start_idx:])
    idx_list = getattr(op, "idx_list", None)

    indices_creation_src = (
        tuple(convert_indices(op_indices, idx) for idx in idx_list)
        if idx_list
        else tuple(input_names[index_start_idx:])
    )

    if len(indices_creation_src) == 1:
        indices_creation_src = f"indices = ({indices_creation_src[0]},)"
    else:
        indices_creation_src = ", ".join(indices_creation_src)
        indices_creation_src = f"indices = ({indices_creation_src})"

    if set_or_inc:
        if op.inplace:
            index_prologue = f"z = {input_names[0]}"
        else:
            index_prologue = f"z = np.copy({input_names[0]})"

        if node.inputs[1].ndim == 0:
            # TODO FIXME: This is a hack to get around a weird Numba typing
            # issue.  See https://github.com/numba/numba/issues/6000
            y_name = f"{input_names[1]}.item()"
        else:
            y_name = input_names[1]

        if op.set_instead_of_inc:
            function_name = "setsubtensor"
            index_body = f"z[indices] = {y_name}"
        else:
            function_name = "incsubtensor"
            index_body = f"z[indices] += {y_name}"
    else:
        function_name = "subtensor"
        index_prologue = ""
        index_body = f"z = {input_names[0]}[indices]"

    subtensor_def_src = f"""
def {function_name}({", ".join(input_names)}):
    {index_prologue}
    {indices_creation_src}
    {index_body}
    return np.asarray(z)
    """

    func = compile_function_src(
        subtensor_def_src,
        function_name=function_name,
        global_env=globals() | {"np": np},
    )
    return numba_njit(func, boundscheck=True)


@numba_funcify.register(AdvancedSubtensor)
@numba_funcify.register(AdvancedIncSubtensor)
def numba_funcify_AdvancedSubtensor(op, node, **kwargs):
    idxs = node.inputs[1:] if isinstance(op, AdvancedSubtensor) else node.inputs[2:]
    adv_idxs_dims = [
        idx.type.ndim
        for idx in idxs
        if (isinstance(idx.type, TensorType) and idx.type.ndim > 0)
    ]

    if (
        # Numba does not support indexes with more than one dimension
        # Nor multiple vector indexes
        (len(adv_idxs_dims) > 1 or adv_idxs_dims[0] > 1)
        # The default index implementation does not handle duplicate indices correctly
        or (
            isinstance(op, AdvancedIncSubtensor)
            and not op.set_instead_of_inc
            and not op.ignore_duplicates
        )
    ):
        return generate_fallback_impl(op, node, **kwargs)

    return numba_funcify_default_subtensor(op, node, **kwargs)


@numba_funcify.register(AdvancedIncSubtensor1)
def numba_funcify_AdvancedIncSubtensor1(op, node, **kwargs):
    inplace = op.inplace
    set_instead_of_inc = op.set_instead_of_inc
    x, vals, idxs = node.inputs
    # TODO: Add explicit expand_dims in make_node so we don't need to worry about this here
    broadcast = vals.type.ndim < x.type.ndim or vals.type.broadcastable[0]

    if set_instead_of_inc:
        if broadcast:

            @numba_njit(boundscheck=True)
            def advancedincsubtensor1_inplace(x, val, idxs):
                if val.ndim == x.ndim:
                    core_val = val[0]
                elif val.ndim == 0:
                    # Workaround for https://github.com/numba/numba/issues/9573
                    core_val = val.item()
                else:
                    core_val = val

                for idx in idxs:
                    x[idx] = core_val
                return x

        else:

            @numba_njit(boundscheck=True)
            def advancedincsubtensor1_inplace(x, vals, idxs):
                if not len(idxs) == len(vals):
                    raise ValueError("The number of indices and values must match.")
                for idx, val in zip(idxs, vals, strict=True):
                    x[idx] = val
                return x
    else:
        if broadcast:

            @numba_njit(boundscheck=True)
            def advancedincsubtensor1_inplace(x, val, idxs):
                if val.ndim == x.ndim:
                    core_val = val[0]
                elif val.ndim == 0:
                    # Workaround for https://github.com/numba/numba/issues/9573
                    core_val = val.item()
                else:
                    core_val = val

                for idx in idxs:
                    x[idx] += core_val
                return x

        else:

            @numba_njit(boundscheck=True)
            def advancedincsubtensor1_inplace(x, vals, idxs):
                if not len(idxs) == len(vals):
                    raise ValueError("The number of indices and values must match.")
                for idx, val in zip(idxs, vals, strict=True):
                    x[idx] += val
                return x

    if inplace:
        return advancedincsubtensor1_inplace

    else:

        @numba_njit
        def advancedincsubtensor1(x, vals, idxs):
            x = x.copy()
            return advancedincsubtensor1_inplace(x, vals, idxs)

        return advancedincsubtensor1
