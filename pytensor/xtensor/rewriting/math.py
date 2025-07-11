from string import ascii_lowercase

from pytensor.graph import node_rewriter
from pytensor.tensor import einsum
from pytensor.tensor.shape import specify_shape
from pytensor.xtensor.basic import tensor_from_xtensor, xtensor_from_tensor
from pytensor.xtensor.math import Dot
from pytensor.xtensor.rewriting.utils import register_lower_xtensor


@register_lower_xtensor
@node_rewriter(tracks=[Dot])
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

    # Collect all dimension names across inputs and output
    all_dims = list(
        dict.fromkeys(x.type.dims + y.type.dims + out.type.dims)
    )  # preserve order
    if len(all_dims) > len(ascii_lowercase):
        raise ValueError("Too many dimensions to map to einsum subscripts")

    dim_to_char = dict(zip(all_dims, ascii_lowercase))

    # Build einsum string
    x_subs = "".join(dim_to_char[d] for d in x.type.dims)
    y_subs = "".join(dim_to_char[d] for d in y.type.dims)
    out_subs = "".join(dim_to_char[d] for d in out.type.dims)
    einsum_str = f"{x_subs},{y_subs}->{out_subs}"

    # Perform the einsum operation
    out_tensor = einsum(einsum_str, x_tensor, y_tensor)

    # Reshape to match the output shape
    out_tensor = specify_shape(out_tensor, out.type.shape)

    return [xtensor_from_tensor(out_tensor, out.type.dims)]
