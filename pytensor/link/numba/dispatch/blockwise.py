from typing import cast

from numba.core.extending import overload
from numba.np.unsafe.ndarray import to_fixed_tuple

from pytensor.link.numba.dispatch.basic import numba_funcify
from pytensor.link.numba.dispatch.vectorize_codegen import (
    _jit_options,
    _vectorized,
    encode_literals,
    store_core_outputs,
)
from pytensor.tensor import TensorVariable, get_vector_length
from pytensor.tensor.blockwise import Blockwise, BlockwiseWithCoreShape


@numba_funcify.register
def numba_funcify_Blockwise(op: BlockwiseWithCoreShape, node, **kwargs):
    [blockwise_node] = op.fgraph.apply_nodes
    blockwise_op: Blockwise = blockwise_node.op
    core_op = blockwise_op.core_op
    nin = len(blockwise_node.inputs)
    nout = len(blockwise_node.outputs)
    if nout > 3:
        raise NotImplementedError(
            "Current implementation of BlockwiseWithCoreShape does not support more than 3 outputs."
        )

    core_shapes_len = [get_vector_length(sh) for sh in node.inputs[nin:]]
    core_shape_0 = core_shapes_len[0] if nout > 0 else None
    core_shape_1 = core_shapes_len[1] if nout > 1 else None
    core_shape_2 = core_shapes_len[2] if nout > 2 else None

    core_node = blockwise_op._create_dummy_core_node(
        cast(tuple[TensorVariable], blockwise_node.inputs)
    )
    core_op_fn = numba_funcify(
        core_op,
        node=core_node,
        parent_node=node,
        fastmath=_jit_options["fastmath"],
        **kwargs,
    )
    core_op_fn = store_core_outputs(core_op_fn, nin=nin, nout=nout)

    batch_ndim = blockwise_op.batch_ndim(node)

    # numba doesn't support nested literals right now...
    input_bc_patterns = encode_literals(
        tuple(inp.type.broadcastable[:batch_ndim] for inp in node.inputs)
    )
    output_bc_patterns = encode_literals(
        tuple(out.type.broadcastable[:batch_ndim] for out in node.outputs)
    )
    output_dtypes = encode_literals(tuple(out.type.dtype for out in node.outputs))
    inplace_pattern = encode_literals(())

    def blockwise_wrapper(*inputs_and_core_shapes):
        inputs, core_shapes = inputs_and_core_shapes[:nin], inputs_and_core_shapes[nin:]
        # Appease numba Gods :(
        # Secular solution welcomed
        if nout == 1:
            tuple_core_shapes = (to_fixed_tuple(core_shapes[0], core_shape_0),)
        elif nout == 2:
            tuple_core_shapes = (
                to_fixed_tuple(core_shapes[0], core_shape_0),
                to_fixed_tuple(core_shapes[1], core_shape_1),
            )
        else:
            tuple_core_shapes = (
                to_fixed_tuple(core_shapes[0], core_shape_0),
                to_fixed_tuple(core_shapes[1], core_shape_1),
                to_fixed_tuple(core_shapes[2], core_shape_2),
            )
        return _vectorized(
            core_op_fn,
            input_bc_patterns,
            output_bc_patterns,
            output_dtypes,
            inplace_pattern,
            (),  # constant_inputs
            inputs,
            tuple_core_shapes,
            None,  # size
        )

    def blockwise(*inputs_and_core_shapes):
        raise NotImplementedError("Non-jitted BlockwiseWithCoreShape not implemented")

    @overload(blockwise, jit_options=_jit_options)
    def ov_blockwise(*inputs_and_core_shapes):
        return blockwise_wrapper

    return blockwise
