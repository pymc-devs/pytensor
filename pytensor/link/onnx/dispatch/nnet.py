"""ONNX conversion for neural network operations."""

from onnx import helper

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.special import LogSoftmax, Softmax


@onnx_funcify.register(Softmax)
def onnx_funcify_Softmax(op, node, get_var_name, **kwargs):
    """Convert Softmax op to ONNX Softmax node.

    PyTensor Softmax: Softmax(x, axis=axis)
    ONNX Softmax: Softmax operator with axis attribute

    Special case: When axis=None, PyTensor applies softmax to the entire
    flattened array. ONNX doesn't support this directly, so we need to:
    1. Flatten the input
    2. Apply softmax with axis=-1
    3. Reshape back to original shape

    Parameters
    ----------
    op : Softmax
        The Softmax operation
    node : Apply
        The Apply node
    get_var_name : callable
        Function to get variable names
    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    onnx.NodeProto or list[onnx.NodeProto]
        ONNX node(s) for the operation
    """
    input_x = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    if op.axis is None:
        # axis=None means apply to flattened array
        # Need to: Flatten -> Softmax(axis=-1) -> Reshape

        # Get input shape for reshaping back
        shape_name = f"{output_name}_orig_shape"
        flatten_name = f"{output_name}_flat"
        softmax_name = f"{output_name}_softmax"

        # Get original shape
        shape_node = helper.make_node(
            "Shape",
            inputs=[input_x],
            outputs=[shape_name],
            name=f"Shape_{output_name}",
        )

        # Flatten to 1D
        flatten_node = helper.make_node(
            "Flatten",
            inputs=[input_x],
            outputs=[flatten_name],
            name=f"Flatten_{output_name}",
            axis=0,  # Flatten to 1D
        )

        # Apply softmax to flattened array (axis=-1)
        softmax_node = helper.make_node(
            "Softmax",
            inputs=[flatten_name],
            outputs=[softmax_name],
            name=f"Softmax_{output_name}",
            axis=-1,
        )

        # Reshape back to original shape
        reshape_node = helper.make_node(
            "Reshape",
            inputs=[softmax_name, shape_name],
            outputs=[output_name],
            name=f"Reshape_{output_name}",
        )

        return [shape_node, flatten_node, softmax_node, reshape_node]
    else:
        # Normal case: axis is specified
        softmax_node = helper.make_node(
            "Softmax",
            inputs=[input_x],
            outputs=[output_name],
            name=f"Softmax_{output_name}",
            axis=op.axis,
        )

        return softmax_node


@onnx_funcify.register(LogSoftmax)
def onnx_funcify_LogSoftmax(op, node, get_var_name, **kwargs):
    """Convert LogSoftmax op to ONNX LogSoftmax node.

    PyTensor LogSoftmax: LogSoftmax(x, axis=axis)
    ONNX LogSoftmax: LogSoftmax operator with axis attribute

    Special case: When axis=None, PyTensor applies logsoftmax to the entire
    flattened array. ONNX doesn't support this directly, so we need to:
    1. Flatten the input
    2. Apply logsoftmax with axis=-1
    3. Reshape back to original shape

    Parameters
    ----------
    op : LogSoftmax
        The LogSoftmax operation
    node : Apply
        The Apply node
    get_var_name : callable
        Function to get variable names
    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    onnx.NodeProto or list[onnx.NodeProto]
        ONNX node(s) for the operation
    """
    input_x = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    if op.axis is None:
        # axis=None means apply to flattened array
        # Need to: Flatten -> LogSoftmax(axis=-1) -> Reshape

        # Get input shape for reshaping back
        shape_name = f"{output_name}_orig_shape"
        flatten_name = f"{output_name}_flat"
        logsoftmax_name = f"{output_name}_logsoftmax"

        # Get original shape
        shape_node = helper.make_node(
            "Shape",
            inputs=[input_x],
            outputs=[shape_name],
            name=f"Shape_{output_name}",
        )

        # Flatten to 1D
        flatten_node = helper.make_node(
            "Flatten",
            inputs=[input_x],
            outputs=[flatten_name],
            name=f"Flatten_{output_name}",
            axis=0,  # Flatten to 1D
        )

        # Apply logsoftmax to flattened array (axis=-1)
        logsoftmax_node = helper.make_node(
            "LogSoftmax",
            inputs=[flatten_name],
            outputs=[logsoftmax_name],
            name=f"LogSoftmax_{output_name}",
            axis=-1,
        )

        # Reshape back to original shape
        reshape_node = helper.make_node(
            "Reshape",
            inputs=[logsoftmax_name, shape_name],
            outputs=[output_name],
            name=f"Reshape_{output_name}",
        )

        return [shape_node, flatten_node, logsoftmax_node, reshape_node]
    else:
        # Normal case: axis is specified
        logsoftmax_node = helper.make_node(
            "LogSoftmax",
            inputs=[input_x],
            outputs=[output_name],
            name=f"LogSoftmax_{output_name}",
            axis=op.axis,
        )

        return logsoftmax_node
