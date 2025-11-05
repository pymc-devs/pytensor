"""Core ONNX dispatch functions for converting PyTensor graphs to ONNX."""

from functools import singledispatch

import numpy as np
import onnx
from onnx import helper, numpy_helper

from pytensor.compile.ops import DeepCopyOp
from pytensor.graph import Constant
from pytensor.graph.fg import FunctionGraph


# Mapping from PyTensor dtypes to ONNX TensorProto dtypes
PYTENSOR_DTYPE_TO_ONNX = {
    "float32": onnx.TensorProto.FLOAT,
    "float64": onnx.TensorProto.DOUBLE,
    "int32": onnx.TensorProto.INT32,
    "int64": onnx.TensorProto.INT64,
    "uint8": onnx.TensorProto.UINT8,
    "int8": onnx.TensorProto.INT8,
    "uint16": onnx.TensorProto.UINT16,
    "int16": onnx.TensorProto.INT16,
    "bool": onnx.TensorProto.BOOL,
}


@singledispatch
def onnx_typify(data, dtype=None, name=None, **kwargs):
    """Convert Python/NumPy data to ONNX TensorProto.

    Parameters
    ----------
    data : array-like
        Data to convert
    dtype : str, optional
        Data type
    name : str, optional
        Name for the tensor

    Returns
    -------
    onnx.TensorProto
        ONNX tensor representation
    """
    # Default: try to convert to numpy array first
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=dtype)
    return numpy_helper.from_array(data, name=name)


@onnx_typify.register(np.ndarray)
def onnx_typify_ndarray(data, dtype=None, name=None, **kwargs):
    """Convert NumPy array to ONNX TensorProto."""
    if dtype is not None:
        data = data.astype(dtype)
    return numpy_helper.from_array(data, name=name)


@singledispatch
def onnx_funcify(op, node=None, **kwargs):
    """Convert a PyTensor Op to an ONNX node.

    This is the core dispatch function that converts PyTensor operations
    to their ONNX equivalents.

    Parameters
    ----------
    op : Op or FunctionGraph
        The operation or graph to convert
    node : Apply, optional
        The Apply node containing this operation
    **kwargs : dict
        Additional arguments passed through the conversion

    Returns
    -------
    onnx.NodeProto or onnx.ModelProto
        ONNX representation of the operation

    Raises
    ------
    NotImplementedError
        If no ONNX conversion is available for this operation
    """
    op_type = type(op).__name__
    raise NotImplementedError(
        f"No ONNX conversion available for: {op_type}. "
        f"The operation {op} is not yet supported in the ONNX backend."
    )


def make_value_info(var, name):
    """Create ONNX ValueInfoProto from PyTensor Variable.

    Parameters
    ----------
    var : Variable
        PyTensor variable
    name : str
        Name for the ONNX value

    Returns
    -------
    onnx.ValueInfoProto
        ONNX value info with shape and dtype
    """
    # Get dtype
    dtype_str = var.type.dtype
    if dtype_str not in PYTENSOR_DTYPE_TO_ONNX:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. "
            f"Supported dtypes: {list(PYTENSOR_DTYPE_TO_ONNX.keys())}"
        )
    onnx_dtype = PYTENSOR_DTYPE_TO_ONNX[dtype_str]

    # Get shape - handle both static and symbolic shapes
    # For now, we'll use None for unknown dimensions
    ndim = var.type.ndim
    shape = [None] * ndim  # Unknown dimensions

    # Create tensor type
    return helper.make_tensor_value_info(name, onnx_dtype, shape)


@onnx_funcify.register(FunctionGraph)
def onnx_funcify_FunctionGraph(
    fgraph,
    opset_version=18,
    **kwargs,
):
    """Convert a PyTensor FunctionGraph to an ONNX ModelProto.

    This function:
    1. Does topological sort of nodes
    2. Converts each node to ONNX via onnx_funcify
    3. Collects constants as initializers
    4. Creates ONNX ModelProto with inputs, outputs, and nodes

    Parameters
    ----------
    fgraph : FunctionGraph
        The function graph to convert
    opset_version : int
        ONNX opset version to use

    Returns
    -------
    onnx.ModelProto
        Complete ONNX model
    """
    # Track variable names to ensure uniqueness
    var_names = {}
    var_counter = 0

    def get_var_name(var):
        """Get or create unique name for a variable."""
        nonlocal var_counter
        if var not in var_names:
            if hasattr(var, "name") and var.name:
                base_name = var.name
            else:
                base_name = "var"
            # Ensure uniqueness
            name = f"{base_name}_{var_counter}"
            var_counter += 1
            var_names[var] = name
        return var_names[var]

    # Collect all nodes in topological order
    nodes = []
    initializers = []
    value_infos = []

    # Process constants first
    for var in fgraph.variables:
        if isinstance(var, Constant):
            name = get_var_name(var)
            # Convert constant to ONNX initializer
            tensor_proto = onnx_typify(var.data, name=name)
            initializers.append(tensor_proto)

    # Process each node in topological order
    for node in fgraph.toposort():
        # Convert node via dispatch
        result = onnx_funcify(
            node.op,
            node=node,
            var_names=var_names,
            get_var_name=get_var_name,
            **kwargs,
        )

        # Handle both single node and (node, initializers) tuple returns
        if result is not None:
            if isinstance(result, tuple):
                # Returned (node, additional_initializers)
                onnx_node, node_initializers = result
                if onnx_node is not None:
                    nodes.append(onnx_node)
                if node_initializers:
                    initializers.extend(node_initializers)
            else:
                # Returned single node
                nodes.append(result)

    # Create input ValueInfos
    inputs = []
    for inp in fgraph.inputs:
        if not isinstance(inp, Constant):
            name = get_var_name(inp)
            value_info = make_value_info(inp, name)
            inputs.append(value_info)

    # Create output ValueInfos
    outputs = []
    for out in fgraph.outputs:
        name = get_var_name(out)
        value_info = make_value_info(out, name)
        outputs.append(value_info)

    # Create the graph
    graph_def = helper.make_graph(
        nodes=nodes,
        name="pytensor_graph",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )

    # Create the model with IR version 9 for compatibility with ONNX Runtime
    model_def = helper.make_model(
        graph_def,
        opset_imports=[helper.make_opsetid("", opset_version)],
        producer_name="PyTensor",
        ir_version=9,  # Use IR version 9 for ONNX Runtime compatibility
    )

    # Check the model
    onnx.checker.check_model(model_def)

    return model_def


@onnx_funcify.register(Constant)
def onnx_funcify_Constant(op, **kwargs):
    """Constants are handled as initializers, not nodes."""
    # Constants don't produce nodes - they're added as initializers
    # in the FunctionGraph converter
    return None


@onnx_funcify.register(DeepCopyOp)
def onnx_funcify_DeepCopyOp(op, node, get_var_name, **kwargs):
    """Convert DeepCopyOp to ONNX Identity node.

    DeepCopyOp is equivalent to Identity in ONNX.
    """
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    return helper.make_node(
        "Identity",
        inputs=input_names,
        outputs=output_names,
        name=f"Identity_{output_names[0]}",
    )
