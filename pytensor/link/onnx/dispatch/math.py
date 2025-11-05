"""ONNX conversion for math operations (reductions)."""

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.math import CAReduce, Argmax
from pytensor.scalar.basic import Add, Mul, Maximum, Minimum, AND, OR

try:
    from onnx import helper
    import numpy as np
except ImportError as e:
    raise ImportError("ONNX package required for export") from e


# Mapping from PyTensor scalar ops to ONNX reduction ops
REDUCE_OP_MAP = {
    Add: 'ReduceSum',
    Mul: 'ReduceProd',
    Maximum: 'ReduceMax',
    Minimum: 'ReduceMin',
    AND: 'ReduceMin',  # For boolean AND
    OR: 'ReduceMax',   # For boolean OR
}


@onnx_funcify.register(CAReduce)
def onnx_funcify_CAReduce(op, node, get_var_name, **kwargs):
    """Convert CAReduce op to ONNX reduction node.

    CAReduce performs reductions (sum, prod, max, min) along specified axes.

    For ONNX opset 18+, axes must be provided as an input tensor,
    not as an attribute.
    """
    scalar_op_type = type(op.scalar_op)

    if scalar_op_type not in REDUCE_OP_MAP:
        raise NotImplementedError(
            f"CAReduce with scalar op {scalar_op_type.__name__} not supported for ONNX export"
        )

    onnx_op_type = REDUCE_OP_MAP[scalar_op_type]

    input_name = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    # Get axis parameter
    axes = op.axis
    nodes = []

    if axes is not None:
        # Convert to list if needed
        if isinstance(axes, (tuple, list)):
            axes_list = list(axes)
        else:
            axes_list = [axes]

        # For opset 18+, axes must be an input tensor
        axes_name = f"{output_name}_axes"
        axes_constant = helper.make_node(
            'Constant',
            inputs=[],
            outputs=[axes_name],
            name=f"Constant_{axes_name}",
            value=helper.make_tensor(
                name=f"{axes_name}_value",
                data_type=helper.TensorProto.INT64,
                dims=[len(axes_list)],
                vals=axes_list,
            )
        )
        nodes.append(axes_constant)

        onnx_node = helper.make_node(
            onnx_op_type,
            inputs=[input_name, axes_name],
            outputs=[output_name],
            name=f"{onnx_op_type}_{output_name}",
            keepdims=0,  # PyTensor default is to not keep dims
        )
    else:
        # Reduce over all axes - don't provide axes input
        onnx_node = helper.make_node(
            onnx_op_type,
            inputs=[input_name],
            outputs=[output_name],
            name=f"{onnx_op_type}_{output_name}",
            keepdims=0,
        )

    nodes.append(onnx_node)
    return nodes if len(nodes) > 1 else onnx_node


@onnx_funcify.register(Argmax)
def onnx_funcify_Argmax(op, node, get_var_name, **kwargs):
    """Convert Argmax op to ONNX ArgMax node."""
    input_name = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    axis = op.axis
    if axis is None:
        # Argmax over all axes - need to flatten first
        flatten_name = f"{output_name}_flat"
        flatten_node = helper.make_node(
            'Flatten',
            inputs=[input_name],
            outputs=[flatten_name],
            name=f"Flatten_{flatten_name}",
            axis=0,
        )

        argmax_node = helper.make_node(
            'ArgMax',
            inputs=[flatten_name],
            outputs=[output_name],
            name=f"ArgMax_{output_name}",
            axis=0,
            keepdims=0,
        )

        return [flatten_node, argmax_node]
    else:
        # Argmax over specific axis
        # PyTensor stores axis as a tuple, ONNX ArgMax expects a single int
        if isinstance(axis, (tuple, list)):
            if len(axis) != 1:
                raise NotImplementedError(
                    f"ONNX ArgMax only supports single axis, got {axis}"
                )
            axis = axis[0]

        onnx_node = helper.make_node(
            'ArgMax',
            inputs=[input_name],
            outputs=[output_name],
            name=f"ArgMax_{output_name}",
            axis=int(axis),
            keepdims=0,
        )

        return onnx_node


