import jax.numpy as jnp

from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.blockwise import Blockwise


@jax_funcify.register(Blockwise)
def jax_funcify_Blockwise(op: Blockwise, node, **kwargs):
    signature = op.signature
    core_node = op._create_dummy_core_node(
        node.inputs, propagate_unbatched_core_inputs=True
    )
    core_fn = jax_funcify(core_node.op, node=core_node, **kwargs)

    vect_fn = jnp.vectorize(core_fn, signature=signature)

    def blockwise_fn(*inputs):
        op._check_runtime_broadcast(node, inputs)
        return vect_fn(*inputs)

    return blockwise_fn
