from pytensor.graph import node_rewriter
from pytensor.tensor import tensordot
from pytensor.xtensor.basic import tensor_from_xtensor, xtensor_from_tensor
from pytensor.xtensor.math import XDot
from pytensor.xtensor.rewriting.utils import register_xcanonicalize


@register_xcanonicalize
@node_rewriter(tracks=[XDot])
def lower_dot(fgraph, node):
    """Rewrite XDot to tensor.dot.

    This rewrite converts an XDot operation to a tensor-based dot operation,
    handling dimension alignment and contraction.
    """
    [x, y] = node.inputs
    [out] = node.outputs

    # Convert inputs to tensors
    x_tensor = tensor_from_xtensor(x)
    y_tensor = tensor_from_xtensor(y)

    # Get dimensions to contract
    if node.op.dims is None:
        # Contract over all matching dimensions
        x_dims = set(x.type.dims)
        y_dims = set(y.type.dims)
        contract_dims = tuple(x_dims & y_dims)
    else:
        contract_dims = node.op.dims

    # Get axes to contract for each input
    x_axes = [x.type.dims.index(dim) for dim in contract_dims]
    y_axes = [y.type.dims.index(dim) for dim in contract_dims]

    # Perform dot product
    out_tensor = tensordot(x_tensor, y_tensor, axes=(x_axes, y_axes))

    # Convert back to xtensor
    return [xtensor_from_tensor(out_tensor, out.type.dims)]
