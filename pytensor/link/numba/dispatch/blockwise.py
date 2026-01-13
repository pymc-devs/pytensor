from hashlib import sha256
from typing import cast

from numba.core.extending import overload
from numba.np.unsafe.ndarray import to_fixed_tuple

from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    numba_funcify_and_cache_key,
    register_funcify_and_cache_key,
)
from pytensor.link.numba.dispatch.vectorize_codegen import (
    _jit_options,
    _vectorized,
    encode_literals,
    store_core_outputs,
)
from pytensor.tensor import TensorVariable, get_vector_length
from pytensor.tensor.blockwise import Blockwise, BlockwiseWithCoreShape


@register_funcify_and_cache_key(BlockwiseWithCoreShape)
def numba_funcify_Blockwise(op: BlockwiseWithCoreShape, node, **kwargs):
    [blockwise_node] = op.fgraph.apply_nodes
    blockwise_op: Blockwise = blockwise_node.op
    core_op = blockwise_op.core_op
    nin = len(blockwise_node.inputs)
    nout = len(blockwise_node.outputs)
    core_shapes_len = tuple(get_vector_length(sh) for sh in node.inputs[nin:])

    core_node = blockwise_op._create_dummy_core_node(
        cast(tuple[TensorVariable], node.inputs[:nin]),
        propagate_unbatched_core_inputs=True,
    )
    core_op_fn, core_op_key = numba_funcify_and_cache_key(
        core_op,
        node=core_node,
        **kwargs,
    )
    core_op_fn = store_core_outputs(core_op_fn, nin=nin, nout=nout)

    batch_ndim = blockwise_op.batch_ndim(node)

    # numba doesn't support nested literals right now...
    input_bc_patterns = encode_literals(
        tuple(inp.type.broadcastable[:batch_ndim] for inp in node.inputs[:nin])
    )
    output_bc_patterns = encode_literals(
        tuple(out.type.broadcastable[:batch_ndim] for out in node.outputs)
    )
    output_dtypes = encode_literals(tuple(out.type.dtype for out in node.outputs))
    inplace_pattern = encode_literals(())

    # Numba does not allow a tuple generator in the Jitted function so we have to compile a helper to convert core_shapes into tuples
    # Alternatively, add an Op that converts shape vectors into tuples, like we did for JAX
    src = "def to_tuple(core_shapes): return ("
    for i in range(nout):
        src += f"to_fixed_tuple(core_shapes[{i}], {core_shapes_len[i]}),"
    src += ")"

    to_tuple = numba_basic.numba_njit(
        compile_numba_function_src(
            src,
            "to_tuple",
            global_env={"to_fixed_tuple": to_fixed_tuple},
        )
    )

    def blockwise(*inputs_and_core_shapes):
        raise NotImplementedError(
            "Numba implementation of Blockwise cannot be evaluated in Python (non-JIT) mode."
        )

    @overload(blockwise, jit_options=_jit_options)
    def ov_blockwise(*inputs_and_core_shapes):
        def impl(*inputs_and_core_shapes):
            inputs, core_shapes = (
                inputs_and_core_shapes[:nin],
                inputs_and_core_shapes[nin:],
            )
            tuple_core_shapes = to_tuple(core_shapes)
            return _vectorized(
                core_op_fn,
                input_bc_patterns,
                output_bc_patterns,
                output_dtypes,
                inplace_pattern,
                False,  # allow_core_scalar
                (),  # constant_inputs
                inputs,
                tuple_core_shapes,
                None,  # size
            )

        return impl

    if core_op_key is None:
        # If the core op cannot be cached, the Blockwise wrapper cannot be cached either
        blockwise_key = None
    else:
        blockwise_cache_version = 1
        blockwise_key = "_".join(
            map(
                str,
                (
                    type(op),
                    type(blockwise_op),
                    tuple(blockwise_op.destroy_map.items()),
                    blockwise_op.signature,
                    input_bc_patterns,
                    core_op_key,
                    blockwise_cache_version,
                ),
            )
        )
        blockwise_key = sha256(blockwise_key.encode()).hexdigest()
    return blockwise, blockwise_key
