from textwrap import dedent

import numpy as np
from numba.np.unsafe import ndarray as numba_ndarray

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    default_hash_key_from_props,
    register_funcify_and_cache_key,
    register_funcify_default_op_cache_key,
)
from pytensor.link.utils import compile_function_src
from pytensor.tensor.reshape import JoinDims, SplitDims
from pytensor.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape
from pytensor.tensor.type_other import NoneTypeT


@register_funcify_default_op_cache_key(Shape)
def numba_funcify_Shape(op, **kwargs):
    @numba_basic.numba_njit
    def shape(x):
        return np.asarray(np.shape(x))

    return shape


@register_funcify_default_op_cache_key(Shape_i)
def numba_funcify_Shape_i(op, **kwargs):
    i = op.i

    @numba_basic.numba_njit
    def shape_i(x):
        return np.asarray(np.shape(x)[i])

    return shape_i


@register_funcify_default_op_cache_key(SpecifyShape)
def numba_funcify_SpecifyShape(op, node, **kwargs):
    shape_inputs = node.inputs[1:]
    shape_input_names = ["shape_" + str(i) for i in range(len(shape_inputs))]

    func_conditions = [
        f"assert x.shape[{i}] == {eval_dim_name}, f'SpecifyShape: dim {{{i}}} of input has shape {{x.shape[{i}]}}, expected {{{eval_dim_name}.item()}}.'"
        for i, (node_dim_input, eval_dim_name) in enumerate(
            zip(shape_inputs, shape_input_names, strict=True)
        )
        if not isinstance(node_dim_input.type, NoneTypeT)
    ]

    func = dedent(
        f"""
        def specify_shape(x, {", ".join(shape_input_names)}):
            {"; ".join(func_conditions)}
            return x
        """
    )

    specify_shape = compile_function_src(func, "specify_shape", globals())
    return numba_basic.numba_njit(specify_shape)


@register_funcify_default_op_cache_key(Reshape)
def numba_funcify_Reshape(op, **kwargs):
    ndim = op.ndim

    if ndim == 0:

        @numba_basic.numba_njit
        def reshape(x, shape):
            return np.asarray(x.item())

    else:

        @numba_basic.numba_njit
        def reshape(x, shape):
            # TODO: Use this until https://github.com/numba/numba/issues/7353 is closed.
            return np.reshape(
                np.ascontiguousarray(np.asarray(x)),
                numba_ndarray.to_fixed_tuple(shape, ndim),
            )

    return reshape


@register_funcify_default_op_cache_key(JoinDims)
def numba_funcify_JoinDims(op, node, **kwargs):
    # The output rank is input_rank - n_axes + 1; input rank is already part of
    # numba's own cache key, so the default (props-based) key is safe here.
    start = op.start_axis
    n = op.n_axes
    out_ndim = node.outputs[0].type.ndim

    @numba_basic.numba_njit
    def join_dims(x):
        old_shape = x.shape
        new_shape = np.empty(out_ndim, dtype=np.int64)
        k = 0
        for i in range(start):
            new_shape[k] = old_shape[i]
            k += 1
        joined = 1
        for i in range(start, start + n):
            joined *= old_shape[i]
        new_shape[k] = joined
        k += 1
        for i in range(start + n, len(old_shape)):
            new_shape[k] = old_shape[i]
            k += 1
        # TODO: Use ascontiguousarray until https://github.com/numba/numba/issues/7353 is closed.
        return np.reshape(
            np.ascontiguousarray(np.asarray(x)),
            numba_ndarray.to_fixed_tuple(new_shape, out_ndim),
        )

    return join_dims


@register_funcify_and_cache_key(SplitDims)
def numba_funcify_SplitDims(op, node, **kwargs):
    axis = op.axis
    out_ndim = node.outputs[0].type.ndim
    n_split = out_ndim - node.inputs[0].type.ndim + 1

    @numba_basic.numba_njit
    def split_dims(x, shape):
        old_shape = x.shape
        new_shape = np.empty(out_ndim, dtype=np.int64)
        k = 0
        for i in range(axis):
            new_shape[k] = old_shape[i]
            k += 1
        for j in range(n_split):
            new_shape[k] = shape[j]
            k += 1
        for i in range(axis + 1, len(old_shape)):
            new_shape[k] = old_shape[i]
            k += 1
        # TODO: Use ascontiguousarray until https://github.com/numba/numba/issues/7353 is closed.
        return np.reshape(
            np.ascontiguousarray(np.asarray(x)),
            numba_ndarray.to_fixed_tuple(new_shape, out_ndim),
        )

    # out_ndim and n_split are baked into the closure but captured neither by the
    # op props nor by numba's input-rank key, so fold them into the cache key.
    key = default_hash_key_from_props(op, out_ndim=out_ndim, n_split=n_split)
    return split_dims, key
