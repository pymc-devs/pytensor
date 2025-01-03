from functools import singledispatch
from textwrap import dedent, indent

import numba
import numpy as np
from numba.core.extending import overload
from numpy.core.numeric import normalize_axis_index, normalize_axis_tuple

from pytensor import config
from pytensor.graph.op import Op
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    create_numba_signature,
    numba_funcify,
    numba_njit,
    use_optimized_cheap_pass,
)
from pytensor.link.numba.dispatch.vectorize_codegen import (
    _jit_options,
    _vectorized,
    encode_literals,
    store_core_outputs,
)
from pytensor.link.utils import compile_function_src
from pytensor.scalar.basic import (
    AND,
    OR,
    XOR,
    Add,
    IntDiv,
    Mul,
    ScalarMaximum,
    ScalarMinimum,
    Sub,
    TrueDiv,
    get_scalar_type,
    scalar_maximum,
)
from pytensor.scalar.basic import add as add_as
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.math import Argmax, MulWithoutZeros, Sum
from pytensor.tensor.special import LogSoftmax, Softmax, SoftmaxGrad


@singledispatch
def scalar_in_place_fn(op: Op, idx: str, res: str, arr: str):
    """Return code for an in-place update on an array using a binary scalar :class:`Op`.

    Parameters
    ----------
    op
        The scalar :class:`Op`
    idx
        The index of `res` that needs to be updated.
    res
        The symbol name for the first input and results/output.
    arr
        The symbol name for the second input.
    """
    raise NotImplementedError()


@scalar_in_place_fn.register(Add)
def scalar_in_place_fn_Add(op, idx, res, arr):
    return f"{res}[{idx}] += {arr}"


@scalar_in_place_fn.register(Sub)
def scalar_in_place_fn_Sub(op, idx, res, arr):
    return f"{res}[{idx}] -= {arr}"


@scalar_in_place_fn.register(Mul)
def scalar_in_place_fn_Mul(op, idx, res, arr):
    return f"{res}[{idx}] *= {arr}"


@scalar_in_place_fn.register(MulWithoutZeros)
def scalar_in_place_fn_MulWithoutZeros(op, idx, res, arr):
    return f"{res}[{idx}] = {arr} if {res}[{idx}] == 0 else ({res}[{idx}] if {arr} == 0 else {res}[{idx}] * {arr})"


@scalar_in_place_fn.register(AND)
def scalar_in_place_fn_AND(op, idx, res, arr):
    return f"{res}[{idx}] &= {arr}"


@scalar_in_place_fn.register(OR)
def scalar_in_place_fn_OR(op, idx, res, arr):
    return f"{res}[{idx}] |= {arr}"


@scalar_in_place_fn.register(XOR)
def scalar_in_place_fn_XOR(op, idx, res, arr):
    return f"{res}[{idx}] ^= {arr}"


@scalar_in_place_fn.register(TrueDiv)
def scalar_in_place_fn_TrueDiv(op, idx, res, arr):
    return f"{res}[{idx}] /= {arr}"


@scalar_in_place_fn.register(IntDiv)
def scalar_in_place_fn_IntDiv(op, idx, res, arr):
    return f"{res}[{idx}] //= {arr}"


@scalar_in_place_fn.register(ScalarMaximum)
def scalar_in_place_fn_ScalarMaximum(op, idx, res, arr):
    return f"""
if {res}[{idx}] < {arr}:
    {res}[{idx}] = {arr}
"""


@scalar_in_place_fn.register(ScalarMinimum)
def scalar_in_place_fn_ScalarMinimum(op, idx, res, arr):
    return f"""
if {res}[{idx}] > {arr}:
    {res}[{idx}] = {arr}
"""


def create_multiaxis_reducer(
    scalar_op,
    identity,
    axes,
    ndim,
    dtype,
    keepdims: bool = False,
):
    r"""Construct a function that reduces multiple axes.

    The functions generated by this function take the following form:

    .. code-block:: python

        def careduce_add(x):
            # For x.ndim == 3 and axes == (0, 1) and scalar_op == "Add"
            x_shape = x.shape
            res_shape = x_shape[2]
            res = np.full(res_shape, numba_basic.to_scalar(0.0), dtype=out_dtype)

            for i0 in range(x_shape[0]):
                for i1 in range(x_shape[1]):
                    for i2 in range(x_shape[2]):
                        res[i2] += x[i0, i1, i2]

            return res

    Parameters
    ==========
    scalar_op:
        The scalar :class:`Op` that performs the desired reduction.
    identity:
        The identity value for the reduction.
    axes:
        The axes to reduce.
    ndim:
        The number of dimensions of the input variable.
    dtype:
        The data type of the result.
    keepdims: boolean, default False
        Whether to keep the reduced dimensions.
    Returns
    =======
    A Python function that can be JITed.

    """
    # if len(axes) == 1:
    #     return create_axis_reducer(scalar_op, identity, axes[0], ndim, dtype)

    axes = normalize_axis_tuple(axes, ndim)
    if keepdims and len(axes) > 1:
        raise NotImplementedError(
            "Cannot keep multiple dimensions when reducing multiple axes"
        )

    careduce_fn_name = f"careduce_{scalar_op}"

    identity = str(identity)
    if identity == "inf":
        identity = "np.inf"
    elif identity == "-inf":
        identity = "-np.inf"

    global_env = {
        "np": np,
        "numba_basic": numba_basic,
        "out_dtype": dtype,
    }
    complete_reduction = len(axes) == ndim
    kept_axis = tuple(i for i in range(ndim) if i not in axes)

    res_indices = []
    arr_indices = []
    for i in range(ndim):
        index_label = f"i{i}"
        arr_indices.append(index_label)
        if i not in axes:
            res_indices.append(index_label)
    res_indices = ", ".join(res_indices) if res_indices else ()
    arr_indices = ", ".join(arr_indices) if arr_indices else ()

    inplace_update_stmt = scalar_in_place_fn(
        scalar_op, res_indices, "res", f"x[{arr_indices}]"
    )

    res_shape = f"({', '.join(f'x_shape[{i}]' for i in kept_axis)})"
    if complete_reduction and ndim > 0:
        # We accumulate on a scalar, not an array
        res_creator = f"np.asarray({identity}).astype(out_dtype).item()"
        inplace_update_stmt = inplace_update_stmt.replace("res[()]", "res")
        return_obj = "np.asarray(res)"
    else:
        res_creator = (
            f"np.full({res_shape}, np.asarray({identity}).item(), dtype=out_dtype)"
        )
        return_obj = "res"

    if keepdims:
        [axis] = axes
        return_obj = f"np.expand_dims({return_obj}, {axis})"

    careduce_def_src = dedent(
        f"""
        def {careduce_fn_name}(x):
            x_shape = x.shape
            res_shape = {res_shape}
            res = {res_creator}
        """
    )
    for axis in range(ndim):
        careduce_def_src += indent(
            f"for i{axis} in range(x_shape[{axis}]):\n",
            " " * (4 + 4 * axis),
        )
    careduce_def_src += indent(inplace_update_stmt, " " * (4 + 4 * ndim))
    careduce_def_src += "\n\n"
    careduce_def_src += indent(f"return {return_obj}", " " * 4)

    careduce_fn = compile_function_src(
        careduce_def_src, careduce_fn_name, {**globals(), **global_env}
    )

    return careduce_fn


def jit_compile_reducer(
    node, fn, *, reduce_to_scalar=False, infer_signature=True, **kwds
):
    """Compile Python source for reduction loops using additional optimizations.

    Parameters
    ==========
    node
        An node from which the signature can be derived.
    fn
        The Python function object to compile.
    reduce_to_scalar: bool, default False
        Whether to reduce output to a scalar (instead of 0d array)
    infer_signature: bool: default True
        Whether to try and infer the function signature from the Apply node.
    kwds
        Extra keywords to be added to the :func:`numba.njit` function.

    Returns
    =======
    A :func:`numba.njit`-compiled function.

    """
    if infer_signature:
        signature = create_numba_signature(node, reduce_to_scalar=reduce_to_scalar)
        args = (signature,)
    else:
        args = ()

    # Eagerly compile the function using increased optimizations.  This should
    # help improve nested loop reductions.
    with use_optimized_cheap_pass():
        res = numba_basic.numba_njit(
            *args,
            boundscheck=False,
            fastmath=config.numba__fastmath,
            **kwds,
        )(fn)

    return res


def create_axis_apply_fn(fn, axis, ndim, dtype):
    axis = normalize_axis_index(axis, ndim)

    reaxis_first = (*(i for i in range(ndim) if i != axis), axis)

    @numba_basic.numba_njit(boundscheck=False)
    def axis_apply_fn(x):
        x_reaxis = x.transpose(reaxis_first)

        res = np.zeros(x_reaxis.shape[:-1], dtype=dtype)
        for m in np.ndindex(res.shape):
            v = fn(x_reaxis[m])
            res[m] = v
        return res

    return axis_apply_fn


@numba_funcify.register(Elemwise)
def numba_funcify_Elemwise(op, node, **kwargs):
    scalar_inputs = [get_scalar_type(dtype=input.dtype)() for input in node.inputs]
    scalar_node = op.scalar_op.make_node(*scalar_inputs)

    scalar_op_fn = numba_funcify(
        op.scalar_op,
        node=scalar_node,
        parent_node=node,
        fastmath=_jit_options["fastmath"],
        **kwargs,
    )

    nin = len(node.inputs)
    nout = len(node.outputs)
    core_op_fn = store_core_outputs(scalar_op_fn, nin=nin, nout=nout)

    input_bc_patterns = tuple(inp.type.broadcastable for inp in node.inputs)
    output_bc_patterns = tuple(out.type.broadcastable for out in node.outputs)
    output_dtypes = tuple(out.type.dtype for out in node.outputs)
    inplace_pattern = tuple(op.inplace_pattern.items())
    core_output_shapes = tuple(() for _ in range(nout))

    # numba doesn't support nested literals right now...
    input_bc_patterns_enc = encode_literals(input_bc_patterns)
    output_bc_patterns_enc = encode_literals(output_bc_patterns)
    output_dtypes_enc = encode_literals(output_dtypes)
    inplace_pattern_enc = encode_literals(inplace_pattern)

    def elemwise_wrapper(*inputs):
        return _vectorized(
            core_op_fn,
            input_bc_patterns_enc,
            output_bc_patterns_enc,
            output_dtypes_enc,
            inplace_pattern_enc,
            (),  # constant_inputs
            inputs,
            core_output_shapes,  # core_shapes
            None,  # size
        )

    # Pure python implementation, that will be used in tests
    def elemwise(*inputs):
        inputs = [np.asarray(input) for input in inputs]
        inputs_bc = np.broadcast_arrays(*inputs)
        shape = inputs[0].shape
        for input, bc in zip(inputs, input_bc_patterns, strict=True):
            for length, allow_bc, iter_length in zip(
                input.shape, bc, shape, strict=True
            ):
                if length == 1 and shape and iter_length != 1 and not allow_bc:
                    raise ValueError("Broadcast not allowed.")

        outputs = [np.empty(shape, dtype=dtype) for dtype in output_dtypes]

        for idx in np.ndindex(shape):
            vals = [input[idx] for input in inputs_bc]
            outs = scalar_op_fn(*vals)
            if not isinstance(outs, tuple):
                outs = (outs,)
            for out, out_val in zip(outputs, outs, strict=True):
                out[idx] = out_val

        outputs_summed = []
        for output, bc in zip(outputs, output_bc_patterns, strict=True):
            axes = tuple(np.nonzero(bc)[0])
            outputs_summed.append(output.sum(axes, keepdims=True))
        if len(outputs_summed) != 1:
            return tuple(outputs_summed)
        return outputs_summed[0]

    @overload(elemwise, jit_options=_jit_options)
    def ov_elemwise(*inputs):
        return elemwise_wrapper

    return elemwise


@numba_funcify.register(Sum)
def numba_funcify_Sum(op, node, **kwargs):
    ndim_input = node.inputs[0].ndim
    axes = op.axis
    if axes is None:
        axes = list(range(node.inputs[0].ndim))
    else:
        axes = normalize_axis_tuple(axes, ndim_input)

    if hasattr(op, "acc_dtype") and op.acc_dtype is not None:
        acc_dtype = op.acc_dtype
    else:
        acc_dtype = node.outputs[0].type.dtype
    np_acc_dtype = np.dtype(acc_dtype)
    out_dtype = np.dtype(node.outputs[0].dtype)

    if ndim_input == len(axes):
        # Slightly faster than `numba_funcify_CAReduce` for this case
        @numba_njit(fastmath=config.numba__fastmath)
        def impl_sum(array):
            return np.asarray(array.sum(), dtype=np_acc_dtype).astype(out_dtype)

    elif len(axes) == 0:
        # These cases should be removed by rewrites!
        @numba_njit(fastmath=config.numba__fastmath)
        def impl_sum(array):
            return np.asarray(array, dtype=out_dtype)

    else:
        impl_sum = numba_funcify_CAReduce(op, node, **kwargs)

    return impl_sum


@numba_funcify.register(CAReduce)
def numba_funcify_CAReduce(op, node, **kwargs):
    axes = op.axis
    if axes is None:
        axes = list(range(node.inputs[0].ndim))

    if hasattr(op, "acc_dtype") and op.acc_dtype is not None:
        acc_dtype = op.acc_dtype
    else:
        acc_dtype = node.outputs[0].type.dtype

    np_acc_dtype = np.dtype(acc_dtype)

    scalar_op_identity = op.scalar_op.identity
    if np_acc_dtype.kind == "i" and not np.isfinite(scalar_op_identity):
        if np.isposinf(scalar_op_identity):
            scalar_op_identity = np.iinfo(np_acc_dtype).max
        else:
            scalar_op_identity = np.iinfo(np_acc_dtype).min

    # Make sure it has the correct dtype
    scalar_op_identity = np.array(scalar_op_identity, dtype=np_acc_dtype)

    ndim = node.inputs[0].ndim
    careduce_py_fn = create_multiaxis_reducer(
        op.scalar_op,
        scalar_op_identity,
        axes,
        ndim,
        np.dtype(node.outputs[0].type.dtype),
    )

    careduce_fn = jit_compile_reducer(node, careduce_py_fn, reduce_to_scalar=False)
    return careduce_fn


@numba_funcify.register(DimShuffle)
def numba_funcify_DimShuffle(op, node, **kwargs):
    shuffle = tuple(op.shuffle)
    transposition = tuple(op.transposition)
    augment = tuple(op.augment)
    inplace = op.inplace

    ndim_new_shape = len(shuffle) + len(augment)

    no_transpose = all(i == j for i, j in enumerate(transposition))
    if no_transpose:

        @numba_basic.numba_njit
        def transpose(x):
            return x

    else:

        @numba_basic.numba_njit
        def transpose(x):
            return np.transpose(x, transposition)

    shape_template = (1,) * ndim_new_shape

    # When `len(shuffle) == 0`, the `shuffle_shape[j]` expression below
    # is typed as `getitem(Tuple(), int)`, which has no implementation
    # (since getting an item from an empty sequence doesn't make sense).
    # To avoid this compile-time error, we omit the expression altogether.
    if len(shuffle) > 0:
        # Use the statically known shape if available
        if all(length is not None for length in node.outputs[0].type.shape):
            shape = node.outputs[0].type.shape

            @numba_basic.numba_njit
            def find_shape(array_shape):
                return shape

        else:

            @numba_basic.numba_njit
            def find_shape(array_shape):
                shape = shape_template
                j = 0
                for i in range(ndim_new_shape):
                    if i not in augment:
                        length = array_shape[j]
                        shape = numba_basic.tuple_setitem(shape, i, length)
                        j = j + 1
                return shape

    else:

        @numba_basic.numba_njit
        def find_shape(array_shape):
            return shape_template

    if ndim_new_shape > 0:

        @numba_basic.numba_njit
        def dimshuffle_inner(x, shuffle):
            x = transpose(x)
            shuffle_shape = x.shape[: len(shuffle)]
            new_shape = find_shape(shuffle_shape)

            # FIXME: Numba's `array.reshape` only accepts C arrays.
            res_reshape = np.reshape(np.ascontiguousarray(x), new_shape)

            if not inplace:
                return res_reshape.copy()
            else:
                return res_reshape

    else:

        @numba_basic.numba_njit
        def dimshuffle_inner(x, shuffle):
            return np.reshape(np.ascontiguousarray(x), ())

    # Without the following wrapper function we would see this error:
    # E   No implementation of function Function(<built-in function getitem>) found for signature:
    # E
    # E    >>> getitem(UniTuple(int64 x 2), slice<a:b>)
    # E
    # E   There are 22 candidate implementations:
    # E      - Of which 22 did not match due to:
    # E      Overload of function 'getitem': File: <numerous>: Line N/A.
    # E        With argument(s): '(UniTuple(int64 x 2), slice<a:b>)':
    # E       No match.
    # ...(on this line)...
    # E           shuffle_shape = res.shape[: len(shuffle)]
    @numba_basic.numba_njit(inline="always")
    def dimshuffle(x):
        return dimshuffle_inner(np.asarray(x), shuffle)

    return dimshuffle


@numba_funcify.register(Softmax)
def numba_funcify_Softmax(op, node, **kwargs):
    x_at = node.inputs[0]
    x_dtype = x_at.type.numpy_dtype
    x_dtype = numba.np.numpy_support.from_dtype(x_dtype)
    axis = op.axis

    if axis is not None:
        axis = normalize_axis_index(axis, x_at.ndim)
        reduce_max_py = create_multiaxis_reducer(
            scalar_maximum, -np.inf, axis, x_at.ndim, x_dtype, keepdims=True
        )
        reduce_sum_py = create_multiaxis_reducer(
            add_as, 0.0, (axis,), x_at.ndim, x_dtype, keepdims=True
        )

        jit_fn = numba_basic.numba_njit(
            boundscheck=False, fastmath=config.numba__fastmath
        )
        reduce_max = jit_fn(reduce_max_py)
        reduce_sum = jit_fn(reduce_sum_py)
    else:
        reduce_max = np.max
        reduce_sum = np.sum

    def softmax_py_fn(x):
        z = reduce_max(x)
        e_x = np.exp(x - z)
        w = reduce_sum(e_x)
        sm = e_x / w
        return sm

    softmax = jit_compile_reducer(node, softmax_py_fn)

    return softmax


@numba_funcify.register(SoftmaxGrad)
def numba_funcify_SoftmaxGrad(op, node, **kwargs):
    sm_at = node.inputs[1]
    sm_dtype = sm_at.type.numpy_dtype
    sm_dtype = numba.np.numpy_support.from_dtype(sm_dtype)

    axis = op.axis
    if axis is not None:
        axis = normalize_axis_index(axis, sm_at.ndim)
        reduce_sum_py = create_multiaxis_reducer(
            add_as, 0.0, (axis,), sm_at.ndim, sm_dtype, keepdims=True
        )

        jit_fn = numba_basic.numba_njit(
            boundscheck=False, fastmath=config.numba__fastmath
        )
        reduce_sum = jit_fn(reduce_sum_py)
    else:
        reduce_sum = np.sum

    def softmax_grad_py_fn(dy, sm):
        dy_times_sm = dy * sm
        sum_dy_times_sm = reduce_sum(dy_times_sm)
        dx = dy_times_sm - sum_dy_times_sm * sm
        return dx

    # The signature inferred by jit_compile_reducer is wrong when dy is a constant (readonly=True)
    softmax_grad = jit_compile_reducer(node, softmax_grad_py_fn, infer_signature=False)

    return softmax_grad


@numba_funcify.register(LogSoftmax)
def numba_funcify_LogSoftmax(op, node, **kwargs):
    x_at = node.inputs[0]
    x_dtype = x_at.type.numpy_dtype
    x_dtype = numba.np.numpy_support.from_dtype(x_dtype)
    axis = op.axis

    if axis is not None:
        axis = normalize_axis_index(axis, x_at.ndim)
        reduce_max_py = create_multiaxis_reducer(
            scalar_maximum,
            -np.inf,
            (axis,),
            x_at.ndim,
            x_dtype,
            keepdims=True,
        )
        reduce_sum_py = create_multiaxis_reducer(
            add_as, 0.0, (axis,), x_at.ndim, x_dtype, keepdims=True
        )

        jit_fn = numba_basic.numba_njit(
            boundscheck=False, fastmath=config.numba__fastmath
        )
        reduce_max = jit_fn(reduce_max_py)
        reduce_sum = jit_fn(reduce_sum_py)
    else:
        reduce_max = np.max
        reduce_sum = np.sum

    def log_softmax_py_fn(x):
        xdev = x - reduce_max(x)
        lsm = xdev - np.log(reduce_sum(np.exp(xdev)))
        return lsm

    log_softmax = jit_compile_reducer(node, log_softmax_py_fn)
    return log_softmax


@numba_funcify.register(Argmax)
def numba_funcify_Argmax(op, node, **kwargs):
    axis = op.axis
    x_at = node.inputs[0]
    x_dtype = x_at.type.numpy_dtype
    x_dtype = numba.np.numpy_support.from_dtype(x_dtype)
    x_ndim = x_at.ndim

    if x_ndim == 0:

        @numba_basic.numba_njit(inline="always")
        def argmax(x):
            return 0

    else:
        axes = tuple(int(ax) for ax in axis)

        # NumPy does not support multiple axes for argmax; this is a
        # work-around
        keep_axes = tuple(i for i in range(x_ndim) if i not in axes)

        reduced_x_ndim = x_ndim - len(axes) + 1
        argmax_axis = create_axis_apply_fn(
            np.argmax, reduced_x_ndim - 1, reduced_x_ndim, np.int64
        )

        reaxis_order = keep_axes + axes
        sl1 = slice(None, len(keep_axes))
        sl2 = slice(len(keep_axes), None)

        @numba_basic.numba_njit
        def argmax(x):
            # Not-reduced axes in front
            transposed_x = np.ascontiguousarray(np.transpose(x, reaxis_order))
            kept_shape = transposed_x.shape[sl1]
            reduced_shape = transposed_x.shape[sl2]
            reduced_size = 1
            for s in reduced_shape:
                reduced_size *= s

            # Numpy.prod returns 1.0 when arg is empty, so we cast it to int64
            # Otherwise reshape would complain citing float arg
            new_shape = (*kept_shape, reduced_size)
            reshaped_x = transposed_x.reshape(new_shape)

            max_idx_res = argmax_axis(reshaped_x)

            return max_idx_res

    return argmax
