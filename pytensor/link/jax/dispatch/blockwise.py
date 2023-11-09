import jax.numpy as jnp

from pytensor.graph import FunctionGraph
from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.blockwise import Blockwise


@jax_funcify.register(Blockwise)
def funcify_Blockwise(op: Blockwise, node, *args, **kwargs):
    signature = op.signature
    core_node = op._create_dummy_core_node(node.inputs)
    core_fgraph = FunctionGraph(inputs=core_node.inputs, outputs=core_node.outputs)
    tuple_core_fn = jax_funcify(core_fgraph)

    if len(node.outputs) == 1:

        def core_fn(*inputs):
            return tuple_core_fn(*inputs)[0]

    else:
        core_fn = tuple_core_fn

    vect_fn = jnp.vectorize(core_fn, signature=signature)

    def blockwise_fn(*inputs):
        op._check_runtime_broadcast(node, inputs)
        return vect_fn(*inputs)

    return blockwise_fn
