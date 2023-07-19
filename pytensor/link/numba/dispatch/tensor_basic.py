from textwrap import indent

import numpy as np

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import create_tuple_string, numba_funcify
from pytensor.link.utils import compile_function_src, unique_name_generator
from pytensor.tensor.basic import (
    Alloc,
    AllocEmpty,
    ARange,
    ExtractDiag,
    Eye,
    Join,
    MakeVector,
    ScalarFromTensor,
    Split,
    TensorFromScalar,
)
from pytensor.tensor.shape import Unbroadcast


@numba_funcify.register(AllocEmpty)
def numba_funcify_AllocEmpty(op, node, **kwargs):
    global_env = {
        "np": np,
        "to_scalar": numba_basic.to_scalar,
        "dtype": np.dtype(op.dtype),
    }

    unique_names = unique_name_generator(
        ["np", "to_scalar", "dtype", "allocempty", "scalar_shape"], suffix_sep="_"
    )
    shape_var_names = [unique_names(v, force_unique=True) for v in node.inputs]
    shape_var_item_names = [f"{name}_item" for name in shape_var_names]
    shapes_to_items_src = indent(
        "\n".join(
            [
                f"{item_name} = to_scalar({shape_name})"
                for item_name, shape_name in zip(shape_var_item_names, shape_var_names)
            ]
        ),
        " " * 4,
    )

    alloc_def_src = f"""
def allocempty({", ".join(shape_var_names)}):
{shapes_to_items_src}
    scalar_shape = {create_tuple_string(shape_var_item_names)}
    return np.empty(scalar_shape, dtype)
    """

    alloc_fn = compile_function_src(
        alloc_def_src, "allocempty", {**globals(), **global_env}
    )

    return numba_basic.numba_njit(alloc_fn)


@numba_funcify.register(Alloc)
def numba_funcify_Alloc(op, node, **kwargs):
    global_env = {"np": np, "to_scalar": numba_basic.to_scalar}

    unique_names = unique_name_generator(
        ["np", "to_scalar", "alloc", "val_np", "val", "scalar_shape", "res"],
        suffix_sep="_",
    )
    shape_var_names = [unique_names(v, force_unique=True) for v in node.inputs[1:]]
    shape_var_item_names = [f"{name}_item" for name in shape_var_names]
    shapes_to_items_src = indent(
        "\n".join(
            [
                f"{item_name} = to_scalar({shape_name})"
                for item_name, shape_name in zip(shape_var_item_names, shape_var_names)
            ]
        ),
        " " * 4,
    )

    check_runtime_broadcast = []
    for i, val_static_dim in enumerate(node.inputs[0].type.shape[::-1]):
        if val_static_dim is None:
            check_runtime_broadcast.append(
                f'if val.shape[{-i - 1}] == 1 and scalar_shape[{-i - 1}] != 1: raise ValueError("{Alloc._runtime_broadcast_error_msg}")'
            )
    check_runtime_broadcast_src = indent("\n".join(check_runtime_broadcast), " " * 4)

    alloc_def_src = f"""
def alloc(val, {", ".join(shape_var_names)}):
    val_np = np.asarray(val)
{shapes_to_items_src}
    scalar_shape = {create_tuple_string(shape_var_item_names)}
{check_runtime_broadcast_src}
    res = np.empty(scalar_shape, dtype=val_np.dtype)
    res[...] = val_np
    return res
    """
    alloc_fn = compile_function_src(alloc_def_src, "alloc", {**globals(), **global_env})

    return numba_basic.numba_njit(alloc_fn)


@numba_funcify.register(ARange)
def numba_funcify_ARange(op, **kwargs):
    dtype = np.dtype(op.dtype)

    @numba_basic.numba_njit(inline="always")
    def arange(start, stop, step):
        return np.arange(
            numba_basic.to_scalar(start),
            numba_basic.to_scalar(stop),
            numba_basic.to_scalar(step),
            dtype=dtype,
        )

    return arange


@numba_funcify.register(Join)
def numba_funcify_Join(op, **kwargs):
    view = op.view

    if view != -1:
        # TODO: Where (and why) is this `Join.view` even being used?  From a
        # quick search, the answer appears to be "nowhere", so we should
        # probably just remove it.
        raise NotImplementedError("The `view` parameter to `Join` is not supported")

    @numba_basic.numba_njit
    def join(axis, *tensors):
        return np.concatenate(tensors, numba_basic.to_scalar(axis))

    return join


@numba_funcify.register(Split)
def numba_funcify_Split(op, **kwargs):
    @numba_basic.numba_njit
    def split(tensor, axis, indices):
        # Work around for https://github.com/numba/numba/issues/8257
        axis = axis % tensor.ndim
        axis = numba_basic.to_scalar(axis)
        return np.split(tensor, np.cumsum(indices)[:-1], axis=axis)

    return split


@numba_funcify.register(ExtractDiag)
def numba_funcify_ExtractDiag(op, node, **kwargs):
    view = op.view
    axis1, axis2, offset = op.axis1, op.axis2, op.offset

    if node.inputs[0].type.ndim == 2:

        @numba_basic.numba_njit(inline="always")
        def extract_diag(x):
            out = np.diag(x, k=offset)

            if not view:
                out = out.copy()

            return out

    else:
        axis1p1 = axis1 + 1
        axis2p1 = axis2 + 1
        leading_dims = (slice(None),) * axis1
        middle_dims = (slice(None),) * (axis2 - axis1 - 1)

        @numba_basic.numba_njit(inline="always")
        def extract_diag(x):
            if offset >= 0:
                diag_len = min(x.shape[axis1], max(0, x.shape[axis2] - offset))
            else:
                diag_len = min(x.shape[axis2], max(0, x.shape[axis1] + offset))
            base_shape = x.shape[:axis1] + x.shape[axis1p1:axis2] + x.shape[axis2p1:]
            out_shape = base_shape + (diag_len,)
            out = np.empty(out_shape)

            for i in range(diag_len):
                if offset >= 0:
                    new_entry = x[leading_dims + (i,) + middle_dims + (i + offset,)]
                else:
                    new_entry = x[leading_dims + (i - offset,) + middle_dims + (i,)]
                out[..., i] = new_entry
            return out

    return extract_diag


@numba_funcify.register(Eye)
def numba_funcify_Eye(op, **kwargs):
    dtype = np.dtype(op.dtype)

    @numba_basic.numba_njit(inline="always")
    def eye(N, M, k):
        return np.eye(
            numba_basic.to_scalar(N),
            numba_basic.to_scalar(M),
            numba_basic.to_scalar(k),
            dtype=dtype,
        )

    return eye


@numba_funcify.register(MakeVector)
def numba_funcify_MakeVector(op, node, **kwargs):
    dtype = np.dtype(op.dtype)

    global_env = {"np": np, "to_scalar": numba_basic.to_scalar, "dtype": dtype}

    unique_names = unique_name_generator(
        ["np", "to_scalar"],
        suffix_sep="_",
    )
    input_names = [unique_names(v, force_unique=True) for v in node.inputs]

    def create_list_string(x):
        args = ", ".join([f"to_scalar({i})" for i in x] + ([""] if len(x) == 1 else []))
        return f"[{args}]"

    makevector_def_src = f"""
def makevector({", ".join(input_names)}):
    return np.array({create_list_string(input_names)}, dtype=dtype)
    """

    makevector_fn = compile_function_src(
        makevector_def_src, "makevector", {**globals(), **global_env}
    )

    return numba_basic.numba_njit(makevector_fn)


@numba_funcify.register(Unbroadcast)
def numba_funcify_Unbroadcast(op, **kwargs):
    @numba_basic.numba_njit
    def unbroadcast(x):
        return x

    return unbroadcast


@numba_funcify.register(TensorFromScalar)
def numba_funcify_TensorFromScalar(op, **kwargs):
    @numba_basic.numba_njit(inline="always")
    def tensor_from_scalar(x):
        return np.array(x)

    return tensor_from_scalar


@numba_funcify.register(ScalarFromTensor)
def numba_funcify_ScalarFromTensor(op, **kwargs):
    @numba_basic.numba_njit(inline="always")
    def scalar_from_tensor(x):
        return numba_basic.to_scalar(x)

    return scalar_from_tensor
