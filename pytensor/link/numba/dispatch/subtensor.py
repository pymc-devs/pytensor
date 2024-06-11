import warnings

import numba
import numpy as np

from pytensor.graph import Type
from pytensor.link.numba.dispatch import numba_funcify
from pytensor.link.numba.dispatch.basic import numba_njit
from pytensor.link.utils import compile_function_src, unique_name_generator
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)


def create_index_func(node, objmode=False):
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
        node.op, IncSubtensor | AdvancedIncSubtensor1 | AdvancedIncSubtensor
    )
    index_start_idx = 1 + int(set_or_inc)

    input_names = [unique_names(v, force_unique=True) for v in node.inputs]
    op_indices = list(node.inputs[index_start_idx:])
    idx_list = getattr(node.op, "idx_list", None)

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
        fn_name = "incsubtensor"
        if node.op.inplace:
            index_prologue = f"z = {input_names[0]}"
        else:
            index_prologue = f"z = np.copy({input_names[0]})"

        if node.inputs[1].ndim == 0:
            # TODO FIXME: This is a hack to get around a weird Numba typing
            # issue.  See https://github.com/numba/numba/issues/6000
            y_name = f"{input_names[1]}.item()"
        else:
            y_name = input_names[1]

        if node.op.set_instead_of_inc:
            index_body = f"z[indices] = {y_name}"
        else:
            index_body = f"z[indices] += {y_name}"
    else:
        fn_name = "subtensor"
        index_prologue = ""
        index_body = f"z = {input_names[0]}[indices]"

    if objmode:
        output_var = node.outputs[0]

        if not set_or_inc:
            # Since `z` is being "created" while in object mode, it's
            # considered an "outgoing" variable and needs to be manually typed
            output_sig = f"z='{output_var.dtype}[{', '.join([':'] * output_var.ndim)}]'"
        else:
            output_sig = ""

        index_body = f"""
    with objmode({output_sig}):
        {index_body}
        """

    subtensor_def_src = f"""
def {fn_name}({", ".join(input_names)}):
    {index_prologue}
    {indices_creation_src}
    {index_body}
    return np.asarray(z)
    """

    return subtensor_def_src


@numba_funcify.register(Subtensor)
@numba_funcify.register(AdvancedSubtensor1)
def numba_funcify_Subtensor(op, node, **kwargs):
    objmode = isinstance(op, AdvancedSubtensor)
    if objmode:
        warnings.warn(
            ("Numba will use object mode to allow run " "AdvancedSubtensor."),
            UserWarning,
        )

    subtensor_def_src = create_index_func(node, objmode=objmode)

    global_env = {"np": np}
    if objmode:
        global_env["objmode"] = numba.objmode

    subtensor_fn = compile_function_src(
        subtensor_def_src, "subtensor", {**globals(), **global_env}
    )

    return numba_njit(subtensor_fn, boundscheck=True)


@numba_funcify.register(IncSubtensor)
def numba_funcify_IncSubtensor(op, node, **kwargs):
    objmode = isinstance(op, AdvancedIncSubtensor)
    if objmode:
        warnings.warn(
            ("Numba will use object mode to allow run " "AdvancedIncSubtensor."),
            UserWarning,
        )

    incsubtensor_def_src = create_index_func(node, objmode=objmode)

    global_env = {"np": np}
    if objmode:
        global_env["objmode"] = numba.objmode

    incsubtensor_fn = compile_function_src(
        incsubtensor_def_src, "incsubtensor", {**globals(), **global_env}
    )

    return numba_njit(incsubtensor_fn, boundscheck=True)


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
                for idx, val in zip(idxs, vals):
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
                for idx, val in zip(idxs, vals):
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
