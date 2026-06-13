import numpy as np

from pytensor.link.python.dispatch.basic import python_funcify
from pytensor.tensor.blockwise import Blockwise


@python_funcify.register(Blockwise)
def python_funcify_Blockwise(op, node=None, **kwargs):
    core_node = op._create_dummy_core_node(
        node.inputs, propagate_unbatched_core_inputs=True
    )
    # Raises NotImplementedError when the core Op has no dispatch, which makes the
    # whole Blockwise fall back to its (vectorized) perform.
    core_fn = python_funcify(op.core_op, node=core_node)

    out_dtypes = [out.type.dtype for out in node.outputs]
    return np.vectorize(core_fn, signature=op.signature, otypes=out_dtypes)
