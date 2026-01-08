from hashlib import sha256
from textwrap import indent

import numpy as np

from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    register_funcify_and_cache_key,
    register_funcify_default_op_cache_key,
)
from pytensor.link.numba.dispatch.string_codegen import create_tuple_string
from pytensor.tensor.basic import (
    Alloc,
    AllocEmpty,
    ARange,
    ExtractDiag,
    Eye,
    Join,
    MakeVector,
    Nonzero,
    ScalarFromTensor,
    Split,
    TensorFromScalar,
)


@register_funcify_default_op_cache_key(AllocEmpty)
def numba_funcify_AllocEmpty(op, node, **kwargs):
    shape_var_names = [f"sh{i}" for i in range(len(node.inputs))]
    shape_var_item_names = [f"{name}_item" for name in shape_var_names]
    shapes_to_items_src = indent(
        "\n".join(
            f"{item_name} = {shape_name}.item()"
            for item_name, shape_name in zip(
                shape_var_item_names, shape_var_names, strict=True
            )
        ),
        " " * 4,
    )

    alloc_def_src = f"""
def allocempty({", ".join(shape_var_names)}):
{shapes_to_items_src}
    scalar_shape = {create_tuple_string(shape_var_item_names)}
    return np.empty(scalar_shape, dtype)
    """

    alloc_fn = compile_numba_function_src(
        alloc_def_src, "allocempty", globals() | {"np": np, "dtype": np.dtype(op.dtype)}
    )

    return numba_basic.numba_njit(alloc_fn)


@register_funcify_and_cache_key(Alloc)
def numba_funcify_Alloc(op, node, **kwargs):
    shape_var_names = [f"sh{i}" for i in range(len(node.inputs) - 1)]
    shape_var_item_names = [f"{name}_item" for name in shape_var_names]
    shapes_to_items_src = indent(
        "\n".join(
            f"{item_name} = {shape_name}.item()"
            for item_name, shape_name in zip(
                shape_var_item_names, shape_var_names, strict=True
            )
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
    dtype = node.inputs[0].type.dtype
    alloc_def_src = f"""
def alloc(val, {", ".join(shape_var_names)}):
{shapes_to_items_src}
    scalar_shape = {create_tuple_string(shape_var_item_names)}
{check_runtime_broadcast_src}
    res = np.empty(scalar_shape, dtype=np.{dtype})
    res[...] = val
    return res
    """
    alloc_fn = compile_numba_function_src(
        alloc_def_src,
        "alloc",
        globals() | {"np": np},
        write_to_disk=True,
    )

    cache_version = -1
    cache_key = sha256(
        str((type(op), node.inputs[0].type.broadcastable, cache_version)).encode()
    ).hexdigest()
    return numba_basic.numba_njit(alloc_fn), cache_key


@register_funcify_default_op_cache_key(ARange)
def numba_funcify_ARange(op, **kwargs):
    dtype = np.dtype(op.dtype)

    @numba_basic.numba_njit
    def arange(start, stop, step):
        return np.arange(
            start.item(),
            stop.item(),
            step.item(),
            dtype=dtype,
        )

    return arange


@register_funcify_default_op_cache_key(Join)
def numba_funcify_Join(op, **kwargs):
    @numba_basic.numba_njit
    def join(axis, *tensors):
        return np.concatenate(tensors, axis.item())

    return join


@register_funcify_default_op_cache_key(Split)
def numba_funcify_Split(op, **kwargs):
    @numba_basic.numba_njit
    def split(x, axis, sizes):
        if (sizes < 0).any():
            raise ValueError("Split sizes cannot be negative")
        axis = axis.item()
        split_indices = np.cumsum(sizes)
        if split_indices[-1] != x.shape[axis]:
            raise ValueError(
                f"Split sizes sum to {split_indices[-1]}; expected {x.shape[axis]}"
            )
        return np.split(x, split_indices[:-1], axis=axis)

    cache_version = 1
    return split, cache_version


@register_funcify_default_op_cache_key(ExtractDiag)
def numba_funcify_ExtractDiag(op, node, **kwargs):
    view = op.view
    axis1, axis2, offset = op.axis1, op.axis2, op.offset

    if node.inputs[0].type.ndim == 2:

        @numba_basic.numba_njit
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

        @numba_basic.numba_njit
        def extract_diag(x):
            if offset >= 0:
                diag_len = min(x.shape[axis1], max(0, x.shape[axis2] - offset))
            else:
                diag_len = min(x.shape[axis2], max(0, x.shape[axis1] + offset))
            base_shape = x.shape[:axis1] + x.shape[axis1p1:axis2] + x.shape[axis2p1:]
            out_shape = (*base_shape, diag_len)
            out = np.empty(out_shape, dtype=x.dtype)

            for i in range(diag_len):
                if offset >= 0:
                    new_entry = x[(*leading_dims, i, *middle_dims, i + offset)]
                else:
                    new_entry = x[(*leading_dims, i - offset, *middle_dims, i)]
                out[..., i] = new_entry
            return out

    cache_version = 1
    return extract_diag, cache_version


@register_funcify_default_op_cache_key(Eye)
def numba_funcify_Eye(op, **kwargs):
    dtype = np.dtype(op.dtype)

    @numba_basic.numba_njit
    def eye(N, M, k):
        return np.eye(
            N.item(),
            M.item(),
            k.item(),
            dtype=dtype,
        )

    return eye


@register_funcify_default_op_cache_key(MakeVector)
def numba_funcify_MakeVector(op, node, **kwargs):
    dtype = np.dtype(op.dtype)
    input_names = [f"x{i}" for i in range(len(node.inputs))]

    def create_list_string(x):
        args = ", ".join([f"{i}.item()" for i in x] + ([""] if len(x) == 1 else []))
        return f"[{args}]"

    makevector_def_src = f"""
def makevector({", ".join(input_names)}):
    return np.array({create_list_string(input_names)}, dtype=dtype)
    """

    makevector_fn = compile_numba_function_src(
        makevector_def_src,
        "makevector",
        globals() | {"np": np, "dtype": dtype},
    )

    return numba_basic.numba_njit(makevector_fn)


@register_funcify_default_op_cache_key(TensorFromScalar)
def numba_funcify_TensorFromScalar(op, **kwargs):
    @numba_basic.numba_njit
    def tensor_from_scalar(x):
        return np.array(x)

    return tensor_from_scalar


@register_funcify_default_op_cache_key(ScalarFromTensor)
def numba_funcify_ScalarFromTensor(op, **kwargs):
    @numba_basic.numba_njit
    def scalar_from_tensor(x):
        return x.item()

    return scalar_from_tensor


@register_funcify_default_op_cache_key(Nonzero)
def numba_funcify_Nonzero(op, node, **kwargs):
    @numba_basic.numba_njit
    def nonzero(a):
        result_tuple = np.nonzero(a)
        if a.ndim == 1:
            return result_tuple[0]
        return list(result_tuple)

    return nonzero
