import numpy as np

from pytensor.link.python.dispatch.basic import python_funcify
from pytensor.tensor.blockwise import Blockwise, _check_runtime_broadcast_core


@python_funcify.register(Blockwise)
def python_funcify_Blockwise(op, node=None, **kwargs):
    core_node = op._create_dummy_core_node(
        node.inputs, propagate_unbatched_core_inputs=True
    )
    # Raises NotImplementedError when the core Op has no dispatch, which makes the
    # whole Blockwise fall back to its (vectorized) perform.
    core_fn = python_funcify(op.core_op, node=core_node)

    output_dtypes = [output.type.dtype for output in node.outputs]
    vectorized_fn = np.vectorize(core_fn, signature=op.signature, otypes=output_dtypes)

    batch_ndim = op.batch_ndim(node)
    if batch_ndim == 0 or len(node.inputs) < 2:
        # A lone input has nothing to broadcast against, matching the `single_in`
        # shortcut in `_vectorize_node_perform`.
        return vectorized_fn

    batch_bcast_patterns = [inp.type.broadcastable[:batch_ndim] for inp in node.inputs]

    def blockwise(*inputs):
        # `np.vectorize` broadcasts batch dimensions unconditionally, whereas
        # PyTensor forbids broadcasting a dimension the static type declared
        # non-broadcastable.
        _check_runtime_broadcast_core(inputs, batch_bcast_patterns, batch_ndim)
        return vectorized_fn(*inputs)

    return blockwise
