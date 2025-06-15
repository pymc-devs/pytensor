from pytensor.graph import node_rewriter
from pytensor.tensor import tensordot
from pytensor.xtensor.basic import tensor_from_xtensor, xtensor_from_tensor
from pytensor.xtensor.math import XDot
from pytensor.xtensor.rewriting.utils import register_lower_xtensor


@register_lower_xtensor
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

    # Get the axes for contraction
    x_axes = [x.type.dims.index(dim) for dim in node.op.dims]
    y_axes = [y.type.dims.index(dim) for dim in node.op.dims]

    # Check that shapes match along contracted dimensions
    for dim in node.op.dims:
        x_idx = x.type.dims.index(dim)
        y_idx = y.type.dims.index(dim)
        if x.type.shape[x_idx] != y.type.shape[y_idx]:
            raise ValueError(
                "Input arrays have inconsistent type shape along the axes "
                f"that are to be reduced with tensordot: {x.type.shape[x_idx]} != {y.type.shape[y_idx]}"
            )

    # Perform the tensordot operation
    out_tensor = tensordot(x_tensor, y_tensor, axes=(x_axes, y_axes))

    # Convert back to xtensor
    return [xtensor_from_tensor(out_tensor, out.type.dims)]
