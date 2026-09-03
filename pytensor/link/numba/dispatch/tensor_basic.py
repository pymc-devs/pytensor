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
    r"""Generate one of three fills, chosen by the static shape of the value.

    Examples
    --------

    A 0d value is filled directly, as in ``alloc(scalar, m, n)``:

    .. code-block:: python

        def alloc(val, sh0, sh1):
            sh0_item = sh0.item()
            sh1_item = sh1.item()
            scalar_shape = (sh0_item, sh1_item)
            res = np.empty(scalar_shape, dtype=np.float64)
            res[...] = val
            return res

    A value that broadcasts in every dimension is filled by its only element, as in
    ``alloc(tensor(shape=(1, 1)), m, n)``:

    .. code-block:: python

        def alloc(val, sh0, sh1):
            sh0_item = sh0.item()
            sh1_item = sh1.item()
            scalar_shape = (sh0_item, sh1_item)
            res = np.empty(scalar_shape, dtype=np.float64)
            res[...] = val[0, 0]
            return res

    Any other value is copied by a loop nest that indexes broadcast dimensions with a
    literal 0, as in ``alloc(tensor(shape=(None, 1)), m, n)``. Dimensions that do not
    broadcast are checked first, so that every index in the loop is in bounds:

    .. code-block:: python

        def alloc(val, sh0, sh1):
            sh0_item = sh0.item()
            sh1_item = sh1.item()
            scalar_shape = (sh0_item, sh1_item)
            if val.shape[-2] != scalar_shape[-2]:
                if val.shape[-2] == 1:
                    raise ValueError(...)  # Runtime broadcasting not allowed
                raise ValueError(...)  # Could not broadcast into the requested shape
            res = np.empty(scalar_shape, dtype=np.float64)
            for i0 in range(sh0_item):
                for i1 in range(sh1_item):
                    res[i0, i1] = val[i0, 0]
            return res

    """
    val, *shape_vars = node.inputs
    out_ndim = len(shape_vars)
    # The dimensions of `val` align with the trailing dimensions of the output
    offset = out_ndim - val.type.ndim

    shape_var_names = [f"sh{i}" for i in range(out_ndim)]
    shape_var_item_names = [f"{name}_item" for name in shape_var_names]

    code: list[str | CODE_TOKEN] = [
        f"def alloc(val, {', '.join(shape_var_names)}):",
        CODE_TOKEN.INDENT,
        *(
            f"{item_name} = {shape_name}.item()"
            for item_name, shape_name in zip(
                shape_var_item_names, shape_var_names, strict=True
            )
        ),
        f"scalar_shape = {create_tuple_string(shape_var_item_names)}",
    ]

    mismatch_error_msg = (
        "could not broadcast input array into the shape requested by Alloc"
    )
    for i, val_static_dim in enumerate(val.type.shape[::-1]):
        if val_static_dim == 1:
            # Broadcasts against any output dimension
            continue
        code.extend(
            (
                f"if val.shape[{-i - 1}] != scalar_shape[{-i - 1}]:",
                CODE_TOKEN.INDENT,
            )
        )
        if val_static_dim is None:
            code.append(
                f'if val.shape[{-i - 1}] == 1: raise ValueError("{Alloc._runtime_broadcast_error_msg}")'
            )
        code.extend((f'raise ValueError("{mismatch_error_msg}")', CODE_TOKEN.DEDENT))

    code.append(f"res = np.empty(scalar_shape, dtype=np.{val.type.dtype})")

    # A dimension of `val` is either statically size 1, and so broadcasts, or it
    # matches the output dimension (the checks above reject every other case), so
    # the whole broadcasting pattern is known here. Filling `res` with an explicit
    # loop nest instead of `res[...] = val` matters: Numba lowers array-to-array
    # assignment to a dtype-generic byte-wise copy helper, while the loop nest
    # vectorizes.
    val_idxs = [
        "0" if val.type.shape[d - offset] == 1 else f"i{d}"
        for d in range(offset, out_ndim)
    ]
    if all(idx == "0" for idx in val_idxs):
        # Every dimension broadcasts, so this is a fill by a scalar, which Numba
        # already lowers well. This is also the only branch valid for a 0d output.
        # A 0d value is filled directly, without indexing it into a scalar first,
        # because Numba is not consistent about the distinction and may hand us a
        # scalar that cannot be indexed.
        val_item = f"val[{', '.join(val_idxs)}]" if val_idxs else "val"
        code.append(f"res[...] = {val_item}")
    else:
        for d in range(out_ndim):
            code.extend(
                (f"for i{d} in range({shape_var_item_names[d]}):", CODE_TOKEN.INDENT)
            )
        out_idxs = [f"i{d}" for d in range(out_ndim)]
        code.append(f"res[{', '.join(out_idxs)}] = val[{', '.join(val_idxs)}]")
        code.extend([CODE_TOKEN.DEDENT] * out_ndim)

    code.extend(("return res", CODE_TOKEN.DEDENT))

    alloc_fn = compile_numba_function_src(
        build_source_code(code),
        "alloc",
        globals() | {"np": np},
        write_to_disk=True,
    )

    cache_version = 1
    # The code branches on whether a dimension of the value is statically 1 (it
    # broadcasts), of unknown size (it needs a runtime broadcast check) or of known
    # size, but never on the concrete sizes.
    static_shape_key = tuple(
        dim if dim in (1, None) else "known" for dim in val.type.shape
    )
    cache_key = sha256(
        str((type(op), static_shape_key, cache_version)).encode()
    ).hexdigest()
    # The shape checks above make every index in the fill loop in-bounds by construction
    return numba_basic.numba_njit(alloc_fn, boundscheck=False), cache_key


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
    """Copy each input into a preallocated output with a scalar loop nest.

    ``np.concatenate`` on a tuple of arrays compiles slower for the same result.
    To let LLVM vectorize the inner copy loop, the join axis offset is unsigned
    (provably non-negative) and broadcastable dimensions are indexed with zero.

    For ``join(1, x0, x1, x2)`` on matrices this emits::

        def join(*tensors):
            total = tensors[0].shape[1]
            total += tensors[1].shape[1]
            total += tensors[2].shape[1]
            if tensors[1].shape[0] != tensors[0].shape[0]:
                raise ValueError(_mismatch_msg)
            if tensors[2].shape[0] != tensors[0].shape[0]:
                raise ValueError(_mismatch_msg)
            out = np.empty((tensors[0].shape[0], total), dtype)
            off = np.uint64(0)
            for j0 in range(tensors[0].shape[0]):
                for j1 in range(tensors[0].shape[1]):
                    out[j0, np.uint64(j1) + off] = tensors[0][j0, j1]
            off += np.uint64(tensors[0].shape[1])
            for j0 in range(tensors[1].shape[0]):
                for j1 in range(tensors[1].shape[1]):
                    out[j0, np.uint64(j1) + off] = tensors[1][j0, j1]
            off += np.uint64(tensors[1].shape[1])
            for j0 in range(tensors[2].shape[0]):
                for j1 in range(tensors[2].shape[1]):
                    out[j0, np.uint64(j1) + off] = tensors[2][j0, j1]
            off += np.uint64(tensors[2].shape[1])
            return out
    """
    ndim = node.outputs[0].type.ndim
    ax = op.axis
    names = [f"tensors[{i}]" for i in range(len(node.inputs))]

    shape = [f"tensors[0].shape[{d}]" for d in range(ndim)]
    shape[ax] = "total"
    code: list[str | CODE_TOKEN] = [
        "def join(*tensors):",
        CODE_TOKEN.INDENT,
        f"total = tensors[0].shape[{ax}]",
        *(f"total += {name}.shape[{ax}]" for name in names[1:]),
    ]
    for name in names[1:]:
        for d in range(ndim):
            if d != ax:
                code += [
                    f"if {name}.shape[{d}] != tensors[0].shape[{d}]:",
                    CODE_TOKEN.INDENT,
                    "raise ValueError(_mismatch_msg)",
                    CODE_TOKEN.DEDENT,
                ]
    code += [
        f"out = np.empty({create_tuple_string(shape)}, dtype)",
        "off = np.uint64(0)",
    ]
    # The join axis is never broadcastable here: its static length is the sum.
    static_shape = node.outputs[0].type.shape
    looped = [d for d in range(ndim) if d == ax or static_shape[d] != 1]
    src_idx = ", ".join(f"j{d}" if d in looped else "0" for d in range(ndim))
    dst_idx = ", ".join(
        f"np.uint64(j{d}) + off" if d == ax else (f"j{d}" if d in looped else "0")
        for d in range(ndim)
    )
    for name in names:
        for d in looped:
            code += [f"for j{d} in range({name}.shape[{d}]):", CODE_TOKEN.INDENT]
        code += [f"out[{dst_idx}] = {name}[{src_idx}]"]
        code += [CODE_TOKEN.DEDENT] * len(looped)
        code += [f"off += np.uint64({name}.shape[{ax}])"]
    code += ["return out", CODE_TOKEN.DEDENT]

    join_fn = compile_numba_function_src(
        build_source_code(code),
        "join",
        globals()
        | {
            "np": np,
            "dtype": np.dtype(node.outputs[0].type.dtype),
            "_mismatch_msg": "all the input array dimensions except for the "
            "concatenation axis must match exactly",
        },
    )

    cache_version = 5
    # The loop nests cannot go out of bounds: the output is allocated from the
    # validated input shapes
    return numba_basic.numba_njit(join_fn, boundscheck=False), cache_version


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
