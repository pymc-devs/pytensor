"""Graph rewrites for ONNX backend compatibility.

These rewrites expand operations that don't have direct ONNX equivalents
into compositions of basic operations that do have ONNX support.
"""

import numpy as np

from pytensor import scalar as ps
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.math import add, exp, log, sub


@node_rewriter([Elemwise])
def expand_log1p_expm1_for_onnx(fgraph, node):
    """Expand log1p and expm1 into basic operations for ONNX export.

    ONNX doesn't have Log1p or Expm1 operators in the standard opset.
    We expand them as:
    - log1p(x) -> log(1 + x)
    - expm1(x) -> exp(x) - 1

    This rewrite is specific to the ONNX backend and should be applied
    before ONNX graph compilation.
    """
    if not isinstance(node.op, Elemwise):
        return None

    scalar_op = node.op.scalar_op

    # Expand log1p(x) -> log(1 + x)
    if isinstance(scalar_op, ps.Log1p):
        x = node.inputs[0]
        # Create log(1 + x)
        one = np.array(1, dtype=x.dtype)
        result = log(add(x, one))
        return [result]

    # Expand expm1(x) -> exp(x) - 1
    if isinstance(scalar_op, ps.Expm1):
        x = node.inputs[0]
        # Create exp(x) - 1
        one = np.array(1, dtype=x.dtype)
        result = sub(exp(x), one)
        return [result]

    return None
