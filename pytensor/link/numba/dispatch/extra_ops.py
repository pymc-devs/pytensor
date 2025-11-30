import warnings
from hashlib import sha256
from typing import cast

import numba
import numpy as np

from pytensor.graph import Apply
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
    get_numba_type,
    register_funcify_and_cache_key,
    register_funcify_default_op_cache_key,
)
from pytensor.tensor import TensorVariable
from pytensor.tensor.extra_ops import (
    Bartlett,
    CumOp,
    FillDiagonal,
    FillDiagonalOffset,
    RavelMultiIndex,
    Repeat,
    SearchsortedOp,
    Unique,
    UnravelIndex,
)


@register_funcify_default_op_cache_key(Bartlett)
def numba_funcify_Bartlett(op, **kwargs):
    @numba_basic.numba_njit
    def bartlett(x):
        return np.bartlett(x.item())

    return bartlett


@register_funcify_default_op_cache_key(CumOp)
def numba_funcify_CumOp(op: CumOp, node: Apply, **kwargs):
    axis = op.axis
    mode = op.mode
    ndim = cast(TensorVariable, node.outputs[0]).ndim

    reaxis_first = (axis, *(i for i in range(ndim) if i != axis))
    reaxis_first_inv = tuple(np.argsort(reaxis_first))

    if mode == "add":
        if ndim == 1:

            @numba_basic.numba_njit
            def cumop(x):
                return np.cumsum(x)

        else:

            @numba_basic.numba_njit(boundscheck=False)
            def cumop(x):
                out_dtype = x.dtype
                if x.shape[axis] < 2:
                    return x.astype(out_dtype)

                x_axis_first = x.transpose(reaxis_first)
                res = np.empty(x_axis_first.shape, dtype=out_dtype)

                res[0] = x_axis_first[0]
                for m in range(1, x.shape[axis]):
                    res[m] = res[m - 1] + x_axis_first[m]

                return res.transpose(reaxis_first_inv)

    else:
        if ndim == 1:

            @numba_basic.numba_njit
            def cumop(x):
                return np.cumprod(x)

        else:

            @numba_basic.numba_njit(boundscheck=False)
            def cumop(x):
                out_dtype = x.dtype
                if x.shape[axis] < 2:
                    return x.astype(out_dtype)

                x_axis_first = x.transpose(reaxis_first)
                res = np.empty(x_axis_first.shape, dtype=out_dtype)

                res[0] = x_axis_first[0]
                for m in range(1, x.shape[axis]):
                    res[m] = res[m - 1] * x_axis_first[m]

                return res.transpose(reaxis_first_inv)

    return cumop


@register_funcify_default_op_cache_key(FillDiagonal)
def numba_funcify_FillDiagonal(op, **kwargs):
    @numba_basic.numba_njit
    def filldiagonal(a, val):
        np.fill_diagonal(a, val)
        return a

    return filldiagonal


@register_funcify_default_op_cache_key(FillDiagonalOffset)
def numba_funcify_FillDiagonalOffset(op, node, **kwargs):
    @numba_basic.numba_njit
    def filldiagonaloffset(a, val, offset):
        height, width = a.shape
        offset_item = offset.item()
        if offset >= 0:
            start = offset_item
            num_of_step = min(min(width, height), width - offset)
        else:
            start = -offset_item * a.shape[1]
            num_of_step = min(min(width, height), height + offset)

        step = a.shape[1] + 1
        end = start + step * num_of_step
        b = a.ravel()
        b[start:end:step] = val
        # TODO: This isn't implemented in Numba
        # a.flat[start:end:step] = val
        # return a
        return b.reshape(a.shape)

    return filldiagonaloffset


@register_funcify_default_op_cache_key(RavelMultiIndex)
def numba_funcify_RavelMultiIndex(op, node, **kwargs):
    mode = op.mode
    order = op.order

    if order != "C":
        raise NotImplementedError(
            "Numba does not implement `order` in `numpy.ravel_multi_index`"
        )

    if mode == "raise":

        @numba_basic.numba_njit
        def mode_fn(*args):
            raise ValueError("invalid entry in coordinates array")

    elif mode == "wrap":

        @numba_basic.numba_njit(inline="always")
        def mode_fn(new_arr, i, j, v, d):
            new_arr[i, j] = v % d

    elif mode == "clip":

        @numba_basic.numba_njit(inline="always")
        def mode_fn(new_arr, i, j, v, d):
            new_arr[i, j] = min(max(v, 0), d - 1)

    if node.inputs[0].ndim == 0:

        @numba_basic.numba_njit
        def ravelmultiindex(*inp):
            shape = inp[-1]
            arr = np.stack(inp[:-1])

            new_arr = arr.T.astype(np.float64).copy()
            for i, b in enumerate(new_arr):
                if b < 0 or b >= shape[i]:
                    mode_fn(new_arr, i, 0, b, shape[i])

            a = np.ones(len(shape), dtype=np.float64)
            a[: len(shape) - 1] = np.cumprod(shape[-1:0:-1])[::-1]
            return np.array(a.dot(new_arr.T), dtype=np.int64)

    else:

        @numba_basic.numba_njit
        def ravelmultiindex(*inp):
            shape = inp[-1]
            arr = np.stack(inp[:-1])

            new_arr = arr.T.astype(np.float64).copy()
            for i, b in enumerate(new_arr):
                # no strict argument to this zip because numba doesn't support it
                for j, (d, v) in enumerate(zip(shape, b)):
                    if v < 0 or v >= d:
                        mode_fn(new_arr, i, j, v, d)

            a = np.ones(len(shape), dtype=np.float64)
            a[: len(shape) - 1] = np.cumprod(shape[-1:0:-1])[::-1]
            return a.dot(new_arr.T).astype(np.int64)

    return ravelmultiindex


@register_funcify_default_op_cache_key(Repeat)
def numba_funcify_Repeat(op, node, **kwargs):
    axis = op.axis
    a, _ = node.inputs

    # Numba only supports axis=None, which in our case is when axis is 0 and the input is a vector
    if axis == 0 and a.type.ndim == 1:

        @numba_basic.numba_njit
        def repeatop(x, repeats):
            return np.repeat(x, repeats)

        return repeatop

    else:
        return generate_fallback_impl(op, node)


@register_funcify_default_op_cache_key(Unique)
def numba_funcify_Unique(op, node, **kwargs):
    axis = op.axis

    use_python = False

    if axis is not None:
        use_python = True

    return_index = op.return_index
    return_inverse = op.return_inverse
    return_counts = op.return_counts

    returns_multi = return_index or return_inverse or return_counts
    use_python |= returns_multi

    if not use_python:

        @numba_basic.numba_njit
        def unique(x):
            return np.unique(x)

    else:
        warnings.warn(
            (
                "Numba will use object mode to allow the "
                "`axis` and/or `return_*` arguments to `numpy.unique`."
            ),
            UserWarning,
        )

        if returns_multi:
            ret_sig = numba.types.Tuple([get_numba_type(o.type) for o in node.outputs])
        else:
            ret_sig = get_numba_type(node.outputs[0].type)

        @numba_basic.numba_njit
        def unique(x):
            with numba.objmode(ret=ret_sig):
                ret = np.unique(x, return_index, return_inverse, return_counts, axis)
            return ret

    return unique


@register_funcify_and_cache_key(UnravelIndex)
def numba_funcify_UnravelIndex(op, node, **kwargs):
    out_ndim = node.outputs[0].type.ndim

    if out_ndim == 0:
        # Creating a tuple of 0d arrays in numba is basically impossible without codegen, so just go to obj_mode
        return generate_fallback_impl(op, node=node), None

    c_order = op.order == "C"
    inp_ndim = node.inputs[0].type.ndim
    transpose_axes = (inp_ndim, *range(inp_ndim))

    @numba_basic.numba_njit
    def unravelindex(indices, shape):
        a = np.ones(len(shape), dtype=np.int64)
        if c_order:
            # C-Order: Reverse shape (ignore dim0), cumulative product, then reverse back
            # Strides: [dim1*dim2, dim2, 1]
            a[1:] = shape[:0:-1]
            a = np.cumprod(a)[::-1]
        else:
            # F-Order: Standard shape, cumulative product
            # Strides: [1, dim0, dim0*dim1]
            a[1:] = shape[:-1]
            a = np.cumprod(a)

        # Broadcast with a and shape on the last axis
        unraveled_coords = (indices[..., None] // a) % shape

        # Then transpose it to the front
        # Numba doesn't have moveaxis (why would it), so we use transpose
        # res = np.moveaxis(res, -1, 0)
        unraveled_coords = unraveled_coords.transpose(transpose_axes)

        # This should be a tuple, but the array can be unpacked
        # into multiple variables with the same effect by the outer function
        # (special case for single entry is handled with an outer function below)
        return unraveled_coords

    cache_version = 1
    cache_key = sha256(
        str((type(op), op.order, len(node.outputs), cache_version)).encode()
    ).hexdigest()

    if len(node.outputs) == 1:

        @numba_basic.numba_njit
        def unravel_index_single_item(arr, shape):
            # Unpack single entry
            (res,) = unravelindex(arr, shape)
            return res

        return unravel_index_single_item, cache_key

    else:
        return unravelindex, cache_key


@register_funcify_default_op_cache_key(SearchsortedOp)
def numba_funcify_Searchsorted(op, node, **kwargs):
    side = op.side

    use_python = False
    if len(node.inputs) == 3:
        use_python = True

    if use_python:
        warnings.warn(
            (
                "Numba will use object mode to allow the "
                "`sorter` argument to `numpy.searchsorted`."
            ),
            UserWarning,
        )

        ret_sig = get_numba_type(node.outputs[0].type)

        @numba_basic.numba_njit
        def searchsorted(a, v, sorter):
            with numba.objmode(ret=ret_sig):
                ret = np.searchsorted(a, v, side, sorter)
            return ret

    else:

        @numba_basic.numba_njit
        def searchsorted(a, v):
            return np.searchsorted(a, v, side)

    return searchsorted
