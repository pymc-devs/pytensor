import jax
import jax.numpy as jnp
import numpy as np

from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.blockwise import (
    Blockwise,
    JAXBlockwise,
    _check_runtime_broadcast_core,
)


@jax_funcify.register(Blockwise)
@jax_funcify.register(JAXBlockwise)
def jax_funcify_Blockwise(op, node, **kwargs):
    if isinstance(op, JAXBlockwise):
        blockwise_node = op.inner_outputs[0].owner
        op = blockwise_node.op
        batch_ndim = op.batch_ndim(blockwise_node)
        unbatched_inputs = frozenset(
            i for i, inp in enumerate(node.inputs) if inp.type.ndim == 0
        )
    else:
        unbatched_inputs = frozenset()
        batch_ndim = op.batch_ndim(node)

    core_node = op._create_dummy_core_node(
        node.inputs, propagate_unbatched_core_inputs=True
    )
    core_fn = jax_funcify(core_node.op, node=core_node, **kwargs)

    vectorized = [i for i in range(len(node.inputs)) if i not in unbatched_inputs]
    if vectorized:
        inputs_signature = ",".join(
            f"({','.join(op.inputs_sig[i])})" for i in vectorized
        )
        outputs_signature = ",".join(f"({','.join(sig)})" for sig in op.outputs_sig)
        signature = f"{inputs_signature}->{outputs_signature}"
        vect_fn = jnp.vectorize(core_fn, signature=signature, excluded=unbatched_inputs)
    else:
        batch_axes = tuple(range(batch_ndim))

        def vect_fn(*inputs):
            outputs = core_fn(*inputs)
            if len(node.outputs) == 1:
                return jnp.expand_dims(outputs, batch_axes)
            return tuple(jnp.expand_dims(out, batch_axes) for out in outputs)

    batch_bcast = [node.inputs[i].type.broadcastable[:batch_ndim] for i in vectorized]
    unbatched_dtypes = {i: node.inputs[i].type.dtype for i in unbatched_inputs}

    def blockwise_fn(*inputs):
        inputs = list(inputs)
        for i, dtype in unbatched_dtypes.items():
            # Preserve concrete values and their dtype without concretizing tracers.
            if not isinstance(inputs[i], jax.Array):
                inputs[i] = np.asarray(inputs[i], dtype=dtype)
        _check_runtime_broadcast_core(
            [inputs[i] for i in vectorized], batch_bcast, batch_ndim
        )
        return vect_fn(*inputs)

    return blockwise_fn
