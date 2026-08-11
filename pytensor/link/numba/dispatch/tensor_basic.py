from hashlib import sha256
from textwrap import indent

import numpy as np

from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    register_funcify_and_cache_key,
    register_funcify_default_op_cache_key,
)
from pytensor.link.numba.dispatch.string_codegen import (
    CODE_TOKEN,
    build_source_code,
    create_tuple_string,
)
from pytensor.scalar.basic import ScalarType
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
from pytensor.tensor.rewriting.scalarize import ScalarJoin


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
def numba_funcify_Join(op, node, **kwargs):
    axis = op.axis

    if node.outputs[0].type.ndim == 1:
        # A 1-d join allocates the output and writes each input into it
        # element-wise. This matches what ``np.concatenate`` lowers to (an
        # element copy loop into a fresh contiguous buffer) but, unlike
        # ``np.concatenate``, the dispatch can admit scalar/0-d inputs and so
        # avoid boxing a scalar just to concatenate it.
        dtype = np.dtype(node.outputs[0].type.dtype)
        names = [f"i{i}" for i in range(len(node.inputs))]
        length = " + ".join(f"{name}.shape[0]" for name in names)
        code: list[str | CODE_TOKEN] = [
            f"def join({', '.join(names)}):",
            CODE_TOKEN.INDENT,
            f"out = np.empty({length}, dtype=dtype)",
            "offset = 0",
        ]
        for name in names:
            code.extend(
                [
                    f"for k in range({name}.shape[0]):",
                    CODE_TOKEN.INDENT,
                    f"out[offset] = {name}[k]",
                    "offset += 1",
                    CODE_TOKEN.DEDENT,
                ]
            )
        code.extend(["return out", CODE_TOKEN.DEDENT])
        join = compile_numba_function_src(
            build_source_code(code),
            "join",
            global_env=globals() | {"np": np, "dtype": dtype},
        )
        return numba_basic.numba_njit(join), 1

    @numba_basic.numba_njit
    def join(*tensors):
        return np.concatenate(tensors, axis)

    return join, 1


@register_funcify_and_cache_key(ScalarJoin)
def numba_funcify_ScalarJoin(op, node, **kwargs):
    # Same allocate-and-write shape as the 1-d ``Join`` above, but each entry may
    # be a bare scalar (which contributes a single element) rather than an array,
    # so it is written directly without ever being boxed.
    dtype = np.dtype(node.outputs[0].type.dtype)
    is_scalar = [isinstance(inp.type, ScalarType) for inp in node.inputs]
    names = [f"i{i}" for i in range(len(node.inputs))]
    length = " + ".join(
        "1" if scalar else f"{name}.shape[0]"
        for name, scalar in zip(names, is_scalar, strict=True)
    )
    code: list[str | CODE_TOKEN] = [
        f"def scalar_join({', '.join(names)}):",
        CODE_TOKEN.INDENT,
        f"out = np.empty({length}, dtype=dtype)",
        "offset = 0",
    ]
    for name, scalar in zip(names, is_scalar, strict=True):
        if scalar:
            code.extend([f"out[offset] = {name}", "offset += 1"])
        else:
            code.extend(
                [
                    f"for k in range({name}.shape[0]):",
                    CODE_TOKEN.INDENT,
                    f"out[offset] = {name}[k]",
                    "offset += 1",
                    CODE_TOKEN.DEDENT,
                ]
            )
    code.extend(["return out", CODE_TOKEN.DEDENT])
    scalar_join = compile_numba_function_src(
        build_source_code(code),
        "scalar_join",
        global_env=globals() | {"np": np, "dtype": dtype},
    )
    key = sha256(
        str(("ScalarJoin", 1, str(dtype), tuple(is_scalar))).encode()
    ).hexdigest()
    return numba_basic.numba_njit(scalar_join), key


@register_funcify_default_op_cache_key(Split)
def numba_funcify_Split(op, **kwargs):
    axis = op.axis

    @numba_basic.numba_njit
    def split(x, sizes):
        if (sizes < 0).any():
            raise ValueError("Split sizes cannot be negative")
        split_indices = np.cumsum(sizes)
        if split_indices[-1] != x.shape[axis]:
            raise ValueError(
                f"Split sizes sum to {split_indices[-1]}; expected {x.shape[axis]}"
            )
        return np.split(x, split_indices[:-1], axis=axis)

    cache_version = 2
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
