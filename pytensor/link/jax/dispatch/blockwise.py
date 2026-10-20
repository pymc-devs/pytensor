import jax
import jax.numpy as jnp
import numpy as np

from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.basic import (
    as_tensor_variable,
    get_underlying_scalar_constant_value,
)
from pytensor.tensor.blockwise import Blockwise, _check_runtime_broadcast_core
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.type import tensor


class JAXUnbatchedScalar(Op):
    """Dummy Op that pads a scalar ``Blockwise`` input with broadcastable batch dims.

    Its JAX implementation returns the scalar as a NumPy scalar of the input dtype,
    which the ``Blockwise`` dispatch passes to the core function outside
    ``jnp.vectorize``. A ``Shape_i`` output stays static, so ``Scan``'s ``n_steps``
    can use it.
    """

    __props__ = ("ndim",)
    view_map = {0: [0]}

    def __init__(self, ndim: int):
        self.ndim = ndim

    def make_node(self, x):
        x = as_tensor_variable(x)
        if x.type.ndim != 0:
            raise ValueError(f"Expected a scalar, got {x.type}")
        out = tensor(dtype=x.type.dtype, shape=(1,) * self.ndim)
        return Apply(self, [x], [out])

    def perform(self, node, inputs, outputs):
        [x] = inputs
        outputs[0][0] = np.reshape(x, (1,) * self.ndim)


@jax_funcify.register(JAXUnbatchedScalar)
def jax_funcify_JAXUnbatchedScalar(op, node, **kwargs):
    dtype = node.inputs[0].type.dtype

    def unbatched_scalar(x):
        # JAX weakly types Python scalars. A NumPy scalar is static and keeps its dtype
        if isinstance(x, jax.Array):
            return x
        return np.asarray(x, dtype=dtype)

    return unbatched_scalar


def unbatched_scalar_constant(inp, sig):
    """Return the value of a constant scalar ``Blockwise`` input without batch dims.

    Return None if the input has core dims, batch dims or no constant value.
    """
    if sig or not all(inp.type.broadcastable):
        return None
    try:
        value = get_underlying_scalar_constant_value(inp)
    except NotScalarConstantError:
        return None
    return np.asarray(value, dtype=inp.type.dtype)


@jax_funcify.register(Blockwise)
def jax_funcify_Blockwise(op: Blockwise, node, **kwargs):
    core_node = op._create_dummy_core_node(
        node.inputs, propagate_unbatched_core_inputs=True
    )
    core_fn = jax_funcify(core_node.op, node=core_node, **kwargs)

    # Constants are passed as their static value, so they are never traced
    unbatched_constants = {
        i: value
        for i, (inp, sig) in enumerate(zip(node.inputs, op.inputs_sig, strict=True))
        if (value := unbatched_scalar_constant(inp, sig)) is not None
    }
    unbatched_scalars = frozenset(unbatched_constants).union(
        i
        for i, inp in enumerate(node.inputs)
        if isinstance(inp.owner_op, JAXUnbatchedScalar)
    )
    vectorized = [i for i in range(len(node.inputs)) if i not in unbatched_scalars]
    batch_ndim = op.batch_ndim(node)

    if not vectorized:
        batch_axes = tuple(range(batch_ndim))

        def unbatched_blockwise_fn(*inputs):
            inputs = [unbatched_constants.get(i, inp) for i, inp in enumerate(inputs)]
            outputs = core_fn(*inputs)
            if len(node.outputs) == 1:
                return jnp.expand_dims(outputs, batch_axes)
            return tuple(jnp.expand_dims(out, batch_axes) for out in outputs)

        return unbatched_blockwise_fn

    inputs_signature = ",".join(f"({','.join(op.inputs_sig[i])})" for i in vectorized)
    outputs_signature = ",".join(f"({','.join(sig)})" for sig in op.outputs_sig)
    signature = f"{inputs_signature}->{outputs_signature}"
    vect_fn = jnp.vectorize(core_fn, signature=signature, excluded=unbatched_scalars)

    batch_bcast = [node.inputs[i].type.broadcastable[:batch_ndim] for i in vectorized]

    def blockwise_fn(*inputs):
        inputs = [unbatched_constants.get(i, inp) for i, inp in enumerate(inputs)]
        _check_runtime_broadcast_core(
            [inputs[i] for i in vectorized], batch_bcast, batch_ndim
        )
        return vect_fn(*inputs)

    return blockwise_fn
