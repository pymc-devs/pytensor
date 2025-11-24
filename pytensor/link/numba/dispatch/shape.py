from textwrap import dedent

import numpy as np
from numba.np.unsafe import ndarray as numba_ndarray

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
from pytensor.link.utils import compile_function_src
from pytensor.tensor import NoneConst
from pytensor.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape


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
        if node_dim_input is not NoneConst
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
