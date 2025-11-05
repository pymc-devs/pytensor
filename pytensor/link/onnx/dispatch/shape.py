"""ONNX conversion for shape operations."""

import numpy as np
from onnx import helper, numpy_helper

from pytensor.link.onnx.dispatch.basic import onnx_funcify


@onnx_funcify.register(type(None))
def onnx_funcify_None(op, **kwargs):
    """Handle None ops (used in some graph optimizations)."""
    return None


# Import DimShuffle after TensorVariable to avoid circular imports
try:
    from pytensor.tensor.elemwise import DimShuffle

    @onnx_funcify.register(DimShuffle)
    def onnx_funcify_DimShuffle(op, node, get_var_name, **kwargs):
        """Convert DimShuffle to ONNX operations.

        DimShuffle handles:
        - Adding dimensions (broadcasting): ('x',) -> Unsqueeze
        - Removing dimensions: drop -> Squeeze
        - Permuting dimensions: (1, 0) -> Transpose

        For now, we focus on the most common case: adding dimensions for broadcasting.
        """
        input_names = [get_var_name(inp) for inp in node.inputs]
        output_names = [get_var_name(out) for out in node.outputs]

        new_order = op.new_order
        input_ndim = op.input_ndim

        # Case 1: Adding dimensions (broadcasting a scalar or expanding dims)
        # Example: new_order = ('x',) means add a dimension at the start
        # Example: new_order = ('x', 0) means add dimension at start, keep original dim
        if "x" in new_order:
            # Find positions where 'x' appears - these are the axes to unsqueeze
            axes = [i for i, dim in enumerate(new_order) if dim == "x"]

            # In ONNX opset 13+, Unsqueeze requires axes as a separate input (not attribute)
            # Create a constant tensor for axes
            axes_tensor_name = f"{output_names[0]}_axes"
            axes_tensor = numpy_helper.from_array(
                np.array(axes, dtype=np.int64), name=axes_tensor_name
            )

            # Create the Unsqueeze node
            node = helper.make_node(
                "Unsqueeze",
                inputs=[input_names[0], axes_tensor_name],
                outputs=output_names,
                name=f"Unsqueeze_{output_names[0]}",
            )

            # Return (node, [initializers])
            return (node, [axes_tensor])

        # Case 2: Transpose (permuting dimensions)
        # new_order is a permutation of input dimensions
        elif len(new_order) == input_ndim and all(
            isinstance(d, int) for d in new_order
        ):
            return helper.make_node(
                "Transpose",
                inputs=input_names,
                outputs=output_names,
                name=f"Transpose_{output_names[0]}",
                perm=list(new_order),
            )

        # Case 3: Squeeze (removing dimensions)
        # This happens when new_order has fewer elements than input_ndim
        # and doesn't contain 'x'
        elif len(new_order) < input_ndim:
            # Find which dimensions to remove
            # The dimensions to squeeze are those not in new_order
            axes_to_keep = set(new_order)
            axes_to_squeeze = [i for i in range(input_ndim) if i not in axes_to_keep]

            return helper.make_node(
                "Squeeze",
                inputs=input_names,
                outputs=output_names,
                name=f"Squeeze_{output_names[0]}",
                axes=axes_to_squeeze,
            )

        else:
            raise NotImplementedError(
                f"DimShuffle with new_order={new_order} and input_ndim={input_ndim} "
                f"is not yet supported in ONNX backend."
            )


except ImportError:
    # DimShuffle not available
    pass
