from collections.abc import Sequence
from functools import singledispatch
from hashlib import sha256
from itertools import combinations

import numpy as np
from numba.core.extending import overload
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple
from numpy.lib.stride_tricks import as_strided

from pytensor import config
from pytensor.graph.op import Op
from pytensor.link.numba.cache import (
    compile_numba_function_src,
)
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    numba_funcify_and_cache_key,
    register_funcify_and_cache_key,
    register_funcify_default_op_cache_key,
)
from pytensor.link.numba.dispatch.string_codegen import (
    CODE_TOKEN,
    build_source_code,
    create_tuple_string,
)
from pytensor.link.numba.dispatch.vectorize_codegen import (
    _vectorized,
    encode_literals,
    store_core_outputs,
)
from pytensor.scalar.basic import (
    AND,
    OR,
    XOR,
    Add,
    IntDiv,
    Maximum,
    Minimum,
    Mul,
    Sub,
    TrueDiv,
    get_scalar_type,
    maximum,
)
from pytensor.scalar.basic import add as add_as
from pytensor.tensor.blas import BatchedDot
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.math import Argmax, Dot, MulWithoutZeros
from pytensor.tensor.special import LogSoftmax, Softmax, SoftmaxGrad


@singledispatch
def scalar_in_place_fn(
    op: Op, idx: str, res: str, arr: str
) -> Sequence[CODE_TOKEN | str]:
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
    raise NotImplementedError(f"No scalar_in_place_fn implemented for {op}")


@scalar_in_place_fn.register(Add)
def scalar_in_place_fn_Add(op, idx, res, arr):
    return [f"{res}[{idx}] += {arr}"]


@scalar_in_place_fn.register(Sub)
def scalar_in_place_fn_Sub(op, idx, res, arr):
    return [f"{res}[{idx}] -= {arr}"]


@scalar_in_place_fn.register(Mul)
def scalar_in_place_fn_Mul(op, idx, res, arr):
    return [f"{res}[{idx}] *= {arr}"]


@scalar_in_place_fn.register(MulWithoutZeros)
def scalar_in_place_fn_MulWithoutZeros(op, idx, res, arr):
    return [
        f"{res}[{idx}] = {arr} if {res}[{idx}] == 0 else ({res}[{idx}] if {arr} == 0 else {res}[{idx}] * {arr})"
    ]


@scalar_in_place_fn.register(AND)
def scalar_in_place_fn_AND(op, idx, res, arr):
    return [f"{res}[{idx}] &= {arr}"]


@scalar_in_place_fn.register(OR)
def scalar_in_place_fn_OR(op, idx, res, arr):
    return [f"{res}[{idx}] |= {arr}"]


@scalar_in_place_fn.register(XOR)
def scalar_in_place_fn_XOR(op, idx, res, arr):
    return [f"{res}[{idx}] ^= {arr}"]


@scalar_in_place_fn.register(TrueDiv)
def scalar_in_place_fn_TrueDiv(op, idx, res, arr):
    return [f"{res}[{idx}] /= {arr}"]


@scalar_in_place_fn.register(IntDiv)
def scalar_in_place_fn_IntDiv(op, idx, res, arr):
    return [f"{res}[{idx}] //= {arr}"]


@scalar_in_place_fn.register(Maximum)
def scalar_in_place_fn_Maximum(op, idx, res, arr):
    return [
        f"if {res}[{idx}] < {arr}:",
        CODE_TOKEN.INDENT,
        f"{res}[{idx}] = {arr}",
        CODE_TOKEN.DEDENT,
    ]


@scalar_in_place_fn.register(Minimum)
def scalar_in_place_fn_Minimum(op, idx, res, arr):
    return [
        f"if {res}[{idx}] > {arr}:",
        CODE_TOKEN.INDENT,
        f"{res}[{idx}] = {arr}",
        CODE_TOKEN.DEDENT,
    ]


def create_multiaxis_reducer(
    scalar_op,
    *,
    identity,
    axes,
    ndim,
    acc_dtype=None,
    out_dtype,
):
    r"""Construct a function that reduces multiple axes.

    The generated function reorders the loop axes according to the runtime strides of the input.

    For partial reductions, the function branches on which transposed
    positions correspond to reduction axes (C(ndim, n_reduced) branches), with
    each branch having a fixed loop nest and result indexing pattern. An
    un-transpose step restores the result to the original axis order.

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
    acc_dtype: dtype, optional
        The data type used during accumulation. Defaults to out_dtype if not provided
    out_dtype:
        The data type of the result.

    Returns
    =======
    A Python function that can be JITed.


    Examples
    --------

    The code generated for a full reduction sum(tensor3()) looks like:

    .. code-block:: python

        def careduce_add(x):
            # identity=np.float64(0.0)

            if x.flags.c_contiguous:
                x_t = x
                s = x.shape
            else:
                strides = np.empty(3, dtype=np.int64)
                for _i in range(3):
                    strides[_i] = abs(x.strides[_i])
                order = np.argsort(strides)[::-1]
                x_t = x.transpose((order[0], order[1], order[2]))
                s = x_t.shape

            res = identity
            for l0 in range(s[0]):
                for l1 in range(s[1]):
                    for l2 in range(s[2]):
                        res += x_t[(l0, l1, l2)]
            return np.array(res, dtype=np.float64)


    And for a partial reduction sum(tensor3(), axis=-1):

    .. code-block:: python

        def careduce_add(x):
            # identity=np.float64(0.0)

            if x.flags.c_contiguous:
                s = x.shape
                res = np.full((s[0], s[1]), identity, dtype=np.float64)
                for l0 in range(s[0]):
                    for l1 in range(s[1]):
                        for l2 in range(s[2]):
                            res[(l0, l1)] += x[(l0, l1, l2)]

            else:
                strides = np.empty(3, dtype=np.int64)
                for _i in range(3):
                    strides[_i] = abs(x.strides[_i])
                order = np.argsort(strides)[::-1]
                x_t = x.transpose((order[0], order[1], order[2]))
                s = x_t.shape

                order_reduced_axes = 0
                if order[0] == 2:
                    order_reduced_axes += 1
                if order[1] == 2:
                    order_reduced_axes += 2
                if order[2] == 2:
                    order_reduced_axes += 4

                if order_reduced_axes == 1:  # 001
                    res = np.full((s[1], s[2]), identity, dtype=np.float64)
                    for l0 in range(s[0]):
                        for l1 in range(s[1]):
                            for l2 in range(s[2]):
                                res[(l1, l2)] += x_t[(l0, l1, l2)]
                    kept_orig_0 = order[1]
                    kept_orig_1 = order[2]
                elif order_reduced_axes == 2:  # 010
                    res = np.full((s[0], s[2]), identity, dtype=np.float64)
                    for l0 in range(s[0]):
                        for l1 in range(s[1]):
                            for l2 in range(s[2]):
                                res[(l0, l2)] += x_t[(l0, l1, l2)]
                    kept_orig_0 = order[0]
                    kept_orig_1 = order[2]
                else:  # 100
                    res = np.full((s[0], s[1]), identity, dtype=np.float64)
                    for l0 in range(s[0]):
                        for l1 in range(s[1]):
                            for l2 in range(s[2]):
                                res[(l0, l1)] += x_t[(l0, l1, l2)]
                    kept_orig_0 = order[0]
                    kept_orig_1 = order[1]

                inv = np.argsort(np.array((kept_orig_0, kept_orig_1)))
                res = res.transpose((inv[0], inv[1]))

            return res

    """
    if axes is None:
        axes = tuple(range(ndim))
    else:
        axes = normalize_axis_tuple(axes, ndim)
    out_dtype = np.dtype(out_dtype)
    acc_dtype = out_dtype if acc_dtype is None else np.dtype(acc_dtype)
    # Numba doesn't allow converting complex to real with a simple `astype`
    complex_to_real = acc_dtype.kind == "c" and out_dtype.kind != "c"
    out_dtype_str = f"np.{out_dtype.name}"
    acc_dtype_str = f"np.{acc_dtype.name}"
    careduce_fn_name = f"careduce_{scalar_op}"

    if acc_dtype.kind in "ui" and not np.isfinite(identity):
        if np.isposinf(identity):
            identity = np.iinfo(acc_dtype).max
        else:
            identity = np.iinfo(acc_dtype).min

    # Make sure it has the correct dtype
    identity = getattr(np, acc_dtype.name)(identity)

    kept_axes = [i for i in range(ndim) if i not in axes]
    n_kept = len(kept_axes)

    def _emit_loop_nest(
        code: list[str | CODE_TOKEN], ndim: int, inplace_lines: list[str | CODE_TOKEN]
    ):
        """Append a loop nest over ``ndim`` dimensions to *code*.

        Generates ``for l0 … for l{ndim-1}`` with *inplace_lines* as the
        innermost body.
        """
        for i in range(ndim):
            code.append(f"for l{i} in range(s[{i}]):")
            code.append(CODE_TOKEN.INDENT)
        code.extend(inplace_lines)
        code.extend([CODE_TOKEN.DEDENT] * ndim)

    def tpl(x):
        # Helper to make code less verbose, and handle generators directly
        return create_tuple_string(tuple(x))

    code: list[str | CODE_TOKEN] = [
        f"def {careduce_fn_name}(x):",
        CODE_TOKEN.INDENT,
        f"# {identity=}",
        CODE_TOKEN.EMPTY_LINE,
    ]

    if n_kept == 0:
        # Full reduction: Use scalar accumulation
        if ndim <= 1:
            code.extend(["x_t = x", "s = x.shape"])
        else:
            # Sort strides (if not C-contiguous)
            code.extend(
                [
                    "if x.flags.c_contiguous:",
                    CODE_TOKEN.INDENT,
                    "x_t = x",
                    "s = x.shape",
                    CODE_TOKEN.DEDENT,
                    "else:",
                    CODE_TOKEN.INDENT,
                    f"strides = np.empty({ndim}, dtype=np.int64)",
                    f"for _i in range({ndim}):",
                    CODE_TOKEN.INDENT,
                    "strides[_i] = abs(x.strides[_i])",
                    CODE_TOKEN.DEDENT,
                    "order = np.argsort(strides)[::-1]",
                    f"x_t = x.transpose({tpl(f'order[{i}]' for i in range(ndim))})",
                    "s = x_t.shape",
                    CODE_TOKEN.DEDENT,
                ]
            )

        code.append(CODE_TOKEN.EMPTY_LINE)
        code.append("res = identity")
        arr_indices = tpl(f"l{i}" for i in range(ndim))
        inplace_lines = [
            l if isinstance(l, CODE_TOKEN) else l.replace("res[()]", "res")
            for l in scalar_in_place_fn(scalar_op, "()", "res", f"x_t[{arr_indices}]")
        ]
        _emit_loop_nest(code, ndim, inplace_lines)
        if complex_to_real:
            return_obj = f"np.array(res).real.astype({out_dtype_str})"
        else:
            return_obj = f"np.array(res, dtype={out_dtype_str})"
        code.append(f"return {return_obj}")

    else:
        # Partial reduction

        # C-contiguous fast path
        kept_shape_c = tpl(f"s[{i}]" for i in kept_axes)
        code.extend(
            [
                "if x.flags.c_contiguous:",
                CODE_TOKEN.INDENT,
                "s = x.shape",
                f"res = np.full({kept_shape_c}, identity, dtype={acc_dtype_str})",
            ]
        )
        c_res_idx = tpl(f"l{p}" for p in range(ndim) if p in kept_axes)
        c_arr_idx = tpl(f"l{i}" for i in range(ndim))
        c_inplace_lines = scalar_in_place_fn(
            scalar_op, c_res_idx, "res", f"x[{c_arr_idx}]"
        )
        _emit_loop_nest(code, ndim, c_inplace_lines)
        code.append(CODE_TOKEN.DEDENT)

        # Other layout path: Order strides, and transpose output at end
        code.extend(
            [
                CODE_TOKEN.EMPTY_LINE,
                "else:",
                CODE_TOKEN.INDENT,
                f"strides = np.empty({ndim}, dtype=np.int64)",
                f"for _i in range({ndim}):",
                CODE_TOKEN.INDENT,
                "strides[_i] = abs(x.strides[_i])",
                CODE_TOKEN.DEDENT,
                "order = np.argsort(strides)[::-1]",
                f"x_t = x.transpose({tpl(f'order[{i}]' for i in range(ndim))})",
                "s = x_t.shape",
                CODE_TOKEN.EMPTY_LINE,
            ]
        )

        # Branch on which transposed positions are reduction axes
        code.append("order_reduced_axes = 0")
        if axes:
            for i in range(ndim):
                code.extend(
                    [
                        f"if {' or '.join(f'order[{i}] == {a}' for a in sorted(axes))}:",
                        CODE_TOKEN.INDENT,
                        f"order_reduced_axes += {1 << i}",
                        CODE_TOKEN.DEDENT,
                    ]
                )
        code.append(CODE_TOKEN.EMPTY_LINE)

        # Generate C(ndim, n_reduced) branches
        reduced_position_combos = list(combinations(range(ndim), len(axes)))
        for branch_idx, reduced_pos in enumerate(reduced_position_combos):
            kept_pos = [p for p in range(ndim) if p not in reduced_pos]
            pattern_value = sum(1 << p for p in reduced_pos)
            pattern_comment = f"  # {pattern_value:0{ndim}b}"

            # Use 'else' for the last branch so numba knows all paths define res
            if branch_idx == 0:
                code.append(
                    f"if order_reduced_axes == {pattern_value}:{pattern_comment}"
                )
            elif branch_idx < len(reduced_position_combos) - 1:
                code.append(
                    f"elif order_reduced_axes == {pattern_value}:{pattern_comment}"
                )
            else:
                code.append(f"else:{pattern_comment}")
            code.append(CODE_TOKEN.INDENT)

            kept_shape = tpl(f"s[{p}]" for p in kept_pos)
            code.append(f"res = np.full({kept_shape}, identity, dtype={acc_dtype_str})")
            arr_idx = tpl(f"l{i}" for i in range(ndim))
            res_idx = tpl(f"l{p}" for p in kept_pos)
            inplace_lines = scalar_in_place_fn(
                scalar_op, res_idx, "res", f"x_t[{arr_idx}]"
            )
            _emit_loop_nest(code, ndim, inplace_lines)
            # kept_orig assignments (for un-transpose)
            if n_kept > 1:
                for k, kp in enumerate(kept_pos):
                    code.append(f"kept_orig_{k} = order[{kp}]")
            code.append(CODE_TOKEN.DEDENT)

        # Un-transpose result to original axis order
        if n_kept > 1:
            kept_orig_arr = tpl(f"kept_orig_{k}" for k in range(n_kept))
            inv_args = tpl(f"inv[{k}]" for k in range(n_kept))
            code.extend(
                [
                    CODE_TOKEN.EMPTY_LINE,
                    f"inv = np.argsort(np.array({kept_orig_arr}))",
                    f"res = res.transpose({inv_args})",
                ]
            )

        code.append(CODE_TOKEN.DEDENT)  # close else branch

        if complex_to_real:
            return_obj = f"res.real.astype({out_dtype_str})"
        elif acc_dtype != out_dtype:
            return_obj = f"res.astype({out_dtype_str})"
        else:
            return_obj = "res"
        code.append(f"return {return_obj}")

    src = build_source_code(code)
    careduce_fn = compile_numba_function_src(
        src, careduce_fn_name, globals() | {"np": np, "identity": identity}
    )
    return careduce_fn


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


@register_funcify_and_cache_key(Elemwise)
def numba_funcify_Elemwise(op, node, **kwargs):
    scalar_inputs = [get_scalar_type(dtype=input.dtype)() for input in node.inputs]
    scalar_node = op.scalar_op.make_node(*scalar_inputs)
    scalar_op_fn, scalar_cache_key = numba_funcify_and_cache_key(
        op.scalar_op,
        node=scalar_node,
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

    # Pure python implementation, that will be used in tests
    def elemwise(*inputs):
        Elemwise._check_runtime_broadcast(node, inputs)
        inputs_bc = np.broadcast_arrays(*inputs)
        shape = inputs_bc[0].shape

        if len(output_dtypes) == 1:
            output = np.empty(shape, dtype=output_dtypes[0])
            for idx in np.ndindex(shape):
                output[idx] = scalar_op_fn(*(inp[idx] for inp in inputs_bc))
            return output

        else:
            outputs = [np.empty(shape, dtype=dtype) for dtype in output_dtypes]
            for idx in np.ndindex(shape):
                outs_vals = scalar_op_fn(*(inp[idx] for inp in inputs_bc))
                for out, out_val in zip(outputs, outs_vals):
                    out[idx] = out_val
            return outputs

    @overload(elemwise)
    def ov_elemwise(*inputs):
        def impl(*inputs):
            return _vectorized(
                core_op_fn,
                input_bc_patterns_enc,
                output_bc_patterns_enc,
                output_dtypes_enc,
                inplace_pattern_enc,
                True,  # allow_core_scalar
                (),  # constant_inputs
                inputs,
                core_output_shapes,  # core_shapes
                None,  # size
            )

        return impl

    if scalar_cache_key is None:
        # If the scalar op cannot be cached, the Elemwise wrapper cannot be cached either
        elemwise_key = None
    else:
        elemwise_key = str(
            (
                type(op),
                tuple(op.inplace_pattern.items()),
                input_bc_patterns,
                scalar_cache_key,
            )
        )
        elemwise_key = sha256(elemwise_key.encode()).hexdigest()
    return elemwise, elemwise_key


@register_funcify_and_cache_key(CAReduce)
def numba_funcify_CAReduce(op, node, **kwargs):
    axes = op.axis
    if axes is None:
        axes = list(range(node.inputs[0].ndim))

    if hasattr(op, "acc_dtype") and op.acc_dtype is not None:
        acc_dtype = op.acc_dtype
    else:
        acc_dtype = node.outputs[0].type.dtype

    out_dtype = np.dtype(node.outputs[0].type.dtype)

    ndim = node.inputs[0].ndim
    careduce_py_fn = create_multiaxis_reducer(
        op.scalar_op,
        identity=op.scalar_op.identity,
        axes=axes,
        ndim=ndim,
        acc_dtype=acc_dtype,
        out_dtype=out_dtype,
    )
    careduce_fn = numba_basic.numba_njit(careduce_py_fn, boundscheck=False)

    cache_version = 2
    careduce_key = sha256(
        str(
            (
                type(op),
                type(op.scalar_op),
                axes,
                out_dtype,
                acc_dtype,
                op.scalar_op.identity,
                cache_version,
            )
        ).encode()
    ).hexdigest()
    return careduce_fn, careduce_key


@register_funcify_default_op_cache_key(DimShuffle)
def numba_funcify_DimShuffle(op: DimShuffle, node, **kwargs):
    # We use `as_strided` to achieve the DimShuffle behavior of transposing and expanding/squezing dimensions in one call
    # Numba doesn't currently support multiple expand/squeeze, and reshape is limited to contiguous arrays.
    new_order = tuple(op._new_order)
    drop = tuple(op.drop)
    shape_template = (1,) * node.outputs[0].ndim
    strides_template = (0,) * node.outputs[0].ndim

    if new_order == ():
        # Special case needed because of https://github.com/numba/numba/issues/9933

        @numba_basic.numba_njit
        def squeeze_to_0d(x):
            if not x.size == 1:
                raise ValueError(
                    "DimShuffle: Attempting to squeeze axes with size not equal to one"
                )
            assert x.size == 1
            return as_strided(x, shape=(), strides=())

        return squeeze_to_0d

    elif op.input_ndim == 0:
        # DimShuffle can only be an expand_dims or a no_op
        # This branch uses asarray in case we get a scalar due to https://github.com/numba/numba/issues/10358
        new_shape = shape_template
        new_strides = strides_template

        @numba_basic.numba_njit
        def dimshuffle(x):
            return as_strided(np.asarray(x), shape=new_shape, strides=new_strides)

    else:

        @numba_basic.numba_njit
        def dimshuffle(x):
            old_shape = x.shape
            old_strides = x.strides

            new_shape = shape_template
            new_strides = strides_template
            for i, o in enumerate(new_order):
                if o != -1:
                    new_shape = numba_basic.tuple_setitem(new_shape, i, old_shape[o])
                    new_strides = numba_basic.tuple_setitem(
                        new_strides, i, old_strides[o]
                    )
            if drop:
                for dropped_dim in drop:
                    if old_shape[dropped_dim] != 1:
                        raise ValueError(
                            "DimShuffle: Attempting to squeeze axes with size not equal to one"
                        )

            return as_strided(x, shape=new_shape, strides=new_strides)

    cache_version = 2
    return dimshuffle, cache_version


@register_funcify_default_op_cache_key(Softmax)
def numba_funcify_Softmax(op, node, **kwargs):
    ndim = node.inputs[0].type.ndim
    inp_dtype = node.inputs[0].type.numpy_dtype
    axis = op.axis

    reduce_max_py = create_multiaxis_reducer(
        maximum,
        identity=-np.inf,
        axes=axis,
        ndim=ndim,
        out_dtype=inp_dtype,
    )
    reduce_sum_py = create_multiaxis_reducer(
        add_as,
        identity=0.0,
        axes=axis,
        ndim=ndim,
        out_dtype=inp_dtype,
    )

    jit_fn = numba_basic.numba_njit(boundscheck=False)
    reduce_max = jit_fn(reduce_max_py)
    reduce_sum = jit_fn(reduce_sum_py)

    if ndim > 1 and axis is not None:

        @jit_fn
        def softmax(x):
            z = np.expand_dims(reduce_max(x), axis)
            e_x = np.exp(x - z)
            w = np.expand_dims(reduce_sum(e_x), axis)
            return e_x / w

    else:

        @jit_fn
        def softmax(x):
            z = reduce_max(x)
            e_x = np.exp(x - z)
            w = reduce_sum(e_x)
            return e_x / w

    cache_version = 2
    return softmax, cache_version


@register_funcify_default_op_cache_key(SoftmaxGrad)
def numba_funcify_SoftmaxGrad(op, node, **kwargs):
    ndim = node.inputs[0].type.ndim
    inp_dtype = node.inputs[0].type.numpy_dtype

    axis = op.axis
    reduce_sum_py = create_multiaxis_reducer(
        add_as,
        identity=0.0,
        axes=axis,
        ndim=ndim,
        out_dtype=inp_dtype,
    )

    jit_fn = numba_basic.numba_njit(boundscheck=False)
    reduce_sum = jit_fn(reduce_sum_py)
    if ndim > 1 and axis is not None:

        @jit_fn
        def softmax_grad(dy, sm):
            dy_times_sm = dy * sm
            sum_dy_times_sm = np.expand_dims(reduce_sum(dy_times_sm), axis)
            dx = dy_times_sm - sum_dy_times_sm * sm
            return dx
    else:

        @jit_fn
        def softmax_grad(dy, sm):
            dy_times_sm = dy * sm
            sum_dy_times_sm = reduce_sum(dy_times_sm)
            dx = dy_times_sm - sum_dy_times_sm * sm
            return dx

    cache_version = 2
    return softmax_grad, cache_version


@register_funcify_default_op_cache_key(LogSoftmax)
def numba_funcify_LogSoftmax(op, node, **kwargs):
    ndim = node.inputs[0].type.ndim
    inp_dtype = node.inputs[0].type.numpy_dtype
    axis = op.axis

    reduce_max_py = create_multiaxis_reducer(
        maximum,
        identity=-np.inf,
        axes=axis,
        ndim=ndim,
        out_dtype=inp_dtype,
    )
    reduce_sum_py = create_multiaxis_reducer(
        add_as,
        identity=0.0,
        axes=axis,
        ndim=ndim,
        out_dtype=inp_dtype,
    )

    jit_fn = numba_basic.numba_njit(boundscheck=False)
    reduce_max = jit_fn(reduce_max_py)
    reduce_sum = jit_fn(reduce_sum_py)

    if ndim > 1 and axis is not None:

        @jit_fn
        def log_softmax(x):
            xdev = x - np.expand_dims(reduce_max(x), axis)
            lsm = xdev - np.log(np.expand_dims(reduce_sum(np.exp(xdev)), axis))
            return lsm

    else:

        @jit_fn
        def log_softmax(x):
            xdev = x - reduce_max(x)
            lsm = xdev - np.log(reduce_sum(np.exp(xdev)))
            return lsm

    cache_version = 2
    return log_softmax, cache_version


@register_funcify_default_op_cache_key(Argmax)
def numba_funcify_Argmax(op, node, **kwargs):
    axis = op.axis
    x_pt = node.inputs[0]
    x_ndim = x_pt.ndim

    if x_ndim == 0:

        @numba_basic.numba_njit
        def argmax(x):
            return np.array(0, dtype="int64")

    else:
        if axis is None:
            axes = tuple(range(x_ndim))
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

    cache_version = 1
    return argmax, cache_version


@register_funcify_default_op_cache_key(Dot)
def numba_funcify_Dot(op, node, **kwargs):
    # Numba's `np.dot` does not support integer dtypes, so we need to cast to float.
    x, y = node.inputs
    [out] = node.outputs

    x_dtype = x.type.numpy_dtype
    y_dtype = y.type.numpy_dtype

    numba_dot_dtype = out_dtype = out.type.numpy_dtype
    if out_dtype.kind not in "fc":
        # Numba alawys returns non-integral outputs, we need to cast to float
        numba_dot_dtype = np.dtype(
            f"float{max((32, out.type.numpy_dtype.itemsize * 8))}"
        )

    if config.compiler_verbose and not (
        x_dtype == y_dtype == out_dtype == numba_dot_dtype
    ):
        print(  # noqa: T201
            "Numba Dot requires a type casting of inputs and/or output: "
            f"{x_dtype=}, {y_dtype=}, {out_dtype=}, {numba_dot_dtype=}"
        )

    if x_dtype == numba_dot_dtype and y_dtype == numba_dot_dtype:

        @numba_basic.numba_njit
        def dot(x, y):
            return np.asarray(np.dot(x, y))

    elif x_dtype == numba_dot_dtype and y_dtype != numba_dot_dtype:

        @numba_basic.numba_njit
        def dot(x, y):
            return np.asarray(np.dot(x, y.astype(numba_dot_dtype)))

    elif x_dtype != numba_dot_dtype and y_dtype == numba_dot_dtype:

        @numba_basic.numba_njit
        def dot(x, y):
            return np.asarray(np.dot(x.astype(numba_dot_dtype), y))

    else:

        @numba_basic.numba_njit
        def dot(x, y):
            return np.asarray(
                np.dot(x.astype(numba_dot_dtype), y.astype(numba_dot_dtype))
            )

    cache_version = 1

    if out_dtype == numba_dot_dtype:
        return dot, cache_version

    else:

        @numba_basic.numba_njit
        def dot_with_cast(x, y):
            return dot(x, y).astype(out_dtype)

        return dot_with_cast, cache_version


@register_funcify_default_op_cache_key(BatchedDot)
def numba_funcify_BatchedDot(op, node, **kwargs):
    dtype = node.outputs[0].type.numpy_dtype

    @numba_basic.numba_njit
    def batched_dot(x, y):
        # Numba does not support 3D matmul
        # https://github.com/numba/numba/issues/3804
        shape = x.shape[:-1] + y.shape[2:]
        z0 = np.empty(shape, dtype=dtype)
        for i in range(z0.shape[0]):
            z0[i] = np.dot(x[i], y[i])

        return z0

    return batched_dot
