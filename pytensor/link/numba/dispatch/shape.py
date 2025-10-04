from textwrap import dedent

import numpy as np
from numba.np.unsafe.ndarray import to_fixed_tuple

from pytensor.link.numba.compile import (
    compile_and_cache_numba_function_src,
    create_arg_string,
    numba_njit,
)
from pytensor.link.numba.dispatch.basic import numba_funcify
from pytensor.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape
from pytensor.tensor.type_other import NoneTypeT


@numba_funcify.register(Shape)
def numba_funcify_Shape(op, **kwargs):
    @numba_njit
    def shape(x):
        return np.asarray(np.shape(x))

    return shape


@numba_funcify.register(Shape_i)
def numba_funcify_Shape_i(op, **kwargs):
    i = op.i

    @numba_njit
    def shape_i(x):
        return np.asarray(np.shape(x)[i])

    return shape_i


@numba_funcify.register(Reshape)
def numba_funcify_Reshape(op, **kwargs):
    ndim = op.ndim

    if ndim == 0:

        @numba_njit
        def reshape(x, shape):
            return np.asarray(x.item())

    else:

        @numba_njit
        def reshape(x, shape):
            # TODO: Use this until https://github.com/numba/numba/issues/7353 is closed.
            return np.reshape(
                np.ascontiguousarray(np.asarray(x)),
                to_fixed_tuple(shape, ndim),
            )

    return reshape


@numba_funcify.register(SpecifyShape)
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
        def specify_shape(x, {create_arg_string(shape_input_names)}):
            {"; ".join(func_conditions)}
            return x
        """
    )

    specify_shape = compile_and_cache_numba_function_src(
        func,
        "specify_shape",
        globals(),
    )
    return numba_njit(specify_shape)
