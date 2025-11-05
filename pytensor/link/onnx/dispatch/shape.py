"""ONNX conversion for shape operations."""

import numpy as np
from onnx import helper, numpy_helper

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.shape import Shape, Shape_i, SpecifyShape, Reshape
from pytensor.tensor.basic import Join, Split
from pytensor.graph.basic import Constant
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.basic import get_scalar_constant_value


@onnx_funcify.register(type(None))
def onnx_funcify_None(op, **kwargs):
    """Handle None ops (used in some graph optimizations)."""
    return None


@onnx_funcify.register(Shape)
def onnx_funcify_Shape(op, node, get_var_name, **kwargs):
    """Convert Shape op to ONNX Shape node.

    Returns tensor containing shape of input.
    """
    input_name = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    onnx_node = helper.make_node(
        'Shape',
        inputs=[input_name],
        outputs=[output_name],
        name=f"Shape_{output_name}",
    )

    return onnx_node


@onnx_funcify.register(Shape_i)
def onnx_funcify_Shape_i(op, node, get_var_name, **kwargs):
    """Convert Shape_i op to ONNX Shape + Gather nodes.

    Shape_i extracts a specific dimension from a tensor's shape.
    This requires multiple ONNX nodes:
    1. Constant - create index constant
    2. Shape - get full shape tensor
    3. Gather - extract the specific dimension

    This operation demonstrates the multi-node return pattern.

    Example:
        x = pt.matrix('x')
        dim0 = x.shape[0]  # Shape_i with i=0

        ONNX graph:
            Constant(value=0) → idx
            Shape(x) → shape_tensor
            Gather(shape_tensor, idx, axis=0) → dim0
    """
    input_name = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    # Get dimension index from op
    axis_idx = op.i

    # Create intermediate names
    shape_name = f"{output_name}_shape"
    idx_name = f"{output_name}_idx"

    # Node 1: Create constant for index
    idx_constant = helper.make_node(
        'Constant',
        inputs=[],
        outputs=[idx_name],
        name=f"Constant_{idx_name}",
        value=helper.make_tensor(
            name=f"{idx_name}_value",
            data_type=helper.TensorProto.INT64,
            dims=[],
            vals=[axis_idx],
        )
    )

    # Node 2: Get full shape
    shape_node = helper.make_node(
        'Shape',
        inputs=[input_name],
        outputs=[shape_name],
        name=f"Shape_{shape_name}",
    )

    # Node 3: Gather specific dimension
    gather_node = helper.make_node(
        'Gather',
        inputs=[shape_name, idx_name],
        outputs=[output_name],
        name=f"Gather_{output_name}",
        axis=0,  # Gather from dimension 0 of shape tensor
    )

    # Return list of nodes - this is the key pattern!
    return [idx_constant, shape_node, gather_node]


@onnx_funcify.register(SpecifyShape)
def onnx_funcify_SpecifyShape(op, node, get_var_name, **kwargs):
    """SpecifyShape is just a hint - pass through input.

    SpecifyShape doesn't change the tensor data, it just provides
    shape information for optimization. In ONNX export, we can
    safely ignore it and just pass the input through.
    """
    # Return None - no ONNX node needed
    # The input will be directly connected to uses of the output
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


@onnx_funcify.register(Reshape)
def onnx_funcify_Reshape(op, node, get_var_name, **kwargs):
    """Convert Reshape op to ONNX Reshape node.

    Reshape changes tensor dimensions without changing data.
    ONNX Reshape takes two inputs:
    1. data - the tensor to reshape
    2. shape - target shape (as 1D int64 tensor)

    The shape can be constant or computed dynamically.
    """
    data_name = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    # The second input is the target shape
    # It may be a constant or computed from other tensors
    shape_input = node.inputs[1]

    if isinstance(shape_input, Constant):
        # Shape is constant - create ONNX Constant node
        shape_data = np.array(shape_input.data, dtype=np.int64)
        shape_name = f"{output_name}_shape"

        shape_constant = helper.make_node(
            'Constant',
            inputs=[],
            outputs=[shape_name],
            name=f"Constant_{shape_name}",
            value=helper.make_tensor(
                name=f"{shape_name}_value",
                data_type=helper.TensorProto.INT64,
                dims=[len(shape_data)],
                vals=shape_data.tolist(),
            )
        )

        reshape_node = helper.make_node(
            'Reshape',
            inputs=[data_name, shape_name],
            outputs=[output_name],
            name=f"Reshape_{output_name}",
        )

        return [shape_constant, reshape_node]
    else:
        # Shape is computed - use its name directly
        shape_name = get_var_name(shape_input)

        reshape_node = helper.make_node(
            'Reshape',
            inputs=[data_name, shape_name],
            outputs=[output_name],
            name=f"Reshape_{output_name}",
        )

        return reshape_node


@onnx_funcify.register(Join)
def onnx_funcify_Join(op, node, get_var_name, **kwargs):
    """Convert Join op to ONNX Concat node.

    Join concatenates tensors along a specified axis.
    The first input (node.inputs[0]) is the axis (as a scalar tensor).
    The remaining inputs (node.inputs[1:]) are the tensors to concatenate.

    ONNX Concat requires the axis as an attribute (not input), so we need
    to extract the constant axis value.
    """
    axis_input = node.inputs[0]
    tensor_inputs = node.inputs[1:]

    # Extract axis value - it must be constant
    try:
        axis = get_scalar_constant_value(axis_input)
        axis = int(axis)
    except NotScalarConstantError:
        raise NotImplementedError(
            "Join with non-constant axis is not supported for ONNX export. "
            "The axis must be a constant integer value."
        )

    # Get tensor input names
    input_names = [get_var_name(inp) for inp in tensor_inputs]
    output_name = get_var_name(node.outputs[0])

    # Create ONNX Concat node
    concat_node = helper.make_node(
        'Concat',
        inputs=input_names,
        outputs=[output_name],
        name=f"Concat_{output_name}",
        axis=axis,
    )

    return concat_node


@onnx_funcify.register(Split)
def onnx_funcify_Split(op, node, get_var_name, **kwargs):
    """Convert Split op to ONNX Split node.

    Split partitions a tensor along a specified axis.
    PyTensor Split takes: (tensor, axis, splits_size) as inputs
    where splits_size defines the size of each output chunk.

    ONNX Split takes the tensor as input and axis/split as attributes.
    """
    # Get input tensor
    input_tensor = node.inputs[0]
    axis_input = node.inputs[1]
    splits_input = node.inputs[2]

    input_name = get_var_name(input_tensor)
    output_names = [get_var_name(out) for out in node.outputs]

    # Extract axis - must be constant
    try:
        axis = get_scalar_constant_value(axis_input)
        axis = int(axis)
    except NotScalarConstantError:
        raise NotImplementedError(
            "Split with non-constant axis is not supported for ONNX export."
        )

    # Extract splits - must be constant
    # splits_input is typically a 1D array of split sizes
    # In ONNX opset 13+, split is provided as a second input tensor (not attribute)
    if isinstance(splits_input, Constant):
        splits_data = splits_input.data
        if np.isscalar(splits_data):
            # If it's a scalar, it means uniform split
            # Number of splits = number of outputs
            splits = np.array([int(splits_data)] * len(node.outputs), dtype=np.int64)
        else:
            # It's an array of split sizes
            splits = np.array([int(s) for s in splits_data], dtype=np.int64)
    else:
        raise NotImplementedError(
            "Split with non-constant split sizes is not supported for ONNX export. "
            "The split sizes must be constant values."
        )

    # Create constant node for split sizes (required in opset 13+)
    split_name = f"{output_names[0]}_split"
    split_constant = helper.make_node(
        'Constant',
        inputs=[],
        outputs=[split_name],
        name=f"Constant_{split_name}",
        value=helper.make_tensor(
            name=f"{split_name}_value",
            data_type=helper.TensorProto.INT64,
            dims=[len(splits)],
            vals=splits.tolist(),
        )
    )

    # Create ONNX Split node with split as an input
    split_node = helper.make_node(
        'Split',
        inputs=[input_name, split_name],
        outputs=output_names,
        name=f"Split_{output_names[0]}",
        axis=axis,
    )

    return [split_constant, split_node]
