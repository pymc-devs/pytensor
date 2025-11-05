"""ONNX conversion for tensor basic operations (allocation, etc.)."""

import numpy as np
from onnx import helper

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.basic import Alloc, AllocEmpty, MakeVector, ARange
from pytensor.graph.basic import Constant


@onnx_funcify.register(Alloc)
def onnx_funcify_Alloc(op, node, get_var_name, **kwargs):
    """Convert Alloc op to ONNX Expand node.

    Alloc broadcasts a value to a specified shape.
    ONNX Expand does the same thing.

    Example:
        x = pt.alloc(5.0, 3, 4)  # Create 3x4 array filled with 5.0

        ONNX: Expand(value=5.0, shape=[3, 4]) -> result
    """
    value_input = node.inputs[0]
    shape_inputs = node.inputs[1:]

    value_name = get_var_name(value_input)
    output_name = get_var_name(node.outputs[0])

    # Create shape tensor from shape inputs
    # Shape inputs are scalars that specify each dimension
    shape_name = f"{output_name}_shape"
    nodes = []

    if all(isinstance(inp, Constant) for inp in shape_inputs):
        # All shape dimensions are constants
        shape_data = np.array([inp.data for inp in shape_inputs], dtype=np.int64)

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
        nodes.append(shape_constant)

        expand_node = helper.make_node(
            'Expand',
            inputs=[value_name, shape_name],
            outputs=[output_name],
            name=f"Expand_{output_name}",
        )
        nodes.append(expand_node)

        return nodes
    else:
        # Some shape dimensions are dynamic - need to use Concat
        # First, unsqueeze each scalar shape dimension to make it 1D
        unsqueezed_names = []
        for i, inp in enumerate(shape_inputs):
            if isinstance(inp, Constant):
                # Create constant for this dimension
                dim_name = f"{shape_name}_dim{i}"
                dim_constant = helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=[dim_name],
                    name=f"Constant_{dim_name}",
                    value=helper.make_tensor(
                        name=f"{dim_name}_value",
                        data_type=helper.TensorProto.INT64,
                        dims=[1],
                        vals=[inp.data],
                    )
                )
                nodes.append(dim_constant)
                unsqueezed_names.append(dim_name)
            else:
                # Dynamic dimension - need to unsqueeze it
                inp_name = get_var_name(inp)
                unsqueezed_name = f"{shape_name}_unsqueezed{i}"

                # Create axes constant for Unsqueeze
                axes_name = f"{unsqueezed_name}_axes"
                axes_constant = helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=[axes_name],
                    name=f"Constant_{axes_name}",
                    value=helper.make_tensor(
                        name=f"{axes_name}_value",
                        data_type=helper.TensorProto.INT64,
                        dims=[1],
                        vals=[0],
                    )
                )
                nodes.append(axes_constant)

                unsqueeze_node = helper.make_node(
                    'Unsqueeze',
                    inputs=[inp_name, axes_name],
                    outputs=[unsqueezed_name],
                    name=f"Unsqueeze_{unsqueezed_name}",
                )
                nodes.append(unsqueeze_node)
                unsqueezed_names.append(unsqueezed_name)

        # Concatenate shape elements into shape vector
        concat_node = helper.make_node(
            'Concat',
            inputs=unsqueezed_names,
            outputs=[shape_name],
            name=f"Concat_{shape_name}",
            axis=0,
        )
        nodes.append(concat_node)

        expand_node = helper.make_node(
            'Expand',
            inputs=[value_name, shape_name],
            outputs=[output_name],
            name=f"Expand_{output_name}",
        )
        nodes.append(expand_node)

        return nodes


@onnx_funcify.register(AllocEmpty)
def onnx_funcify_AllocEmpty(op, node, get_var_name, **kwargs):
    """Convert AllocEmpty to ONNX ConstantOfShape.

    AllocEmpty creates uninitialized array. In ONNX, we use
    ConstantOfShape with value 0 (values don't matter, just shape/dtype).

    Example:
        x = pt.AllocEmpty('float32')(3, 4)  # Create uninitialized 3x4 array

        ONNX: ConstantOfShape(shape=[3, 4], value=0.0) -> result
    """
    shape_inputs = node.inputs
    output_name = get_var_name(node.outputs[0])

    # Create shape tensor
    shape_name = f"{output_name}_shape"
    nodes = []

    if all(isinstance(inp, Constant) for inp in shape_inputs):
        # Constant shape
        shape_data = np.array([inp.data for inp in shape_inputs], dtype=np.int64)

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
        nodes.append(shape_constant)
    else:
        # Dynamic shape - similar to Alloc
        unsqueezed_names = []
        for i, inp in enumerate(shape_inputs):
            if isinstance(inp, Constant):
                dim_name = f"{shape_name}_dim{i}"
                dim_constant = helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=[dim_name],
                    name=f"Constant_{dim_name}",
                    value=helper.make_tensor(
                        name=f"{dim_name}_value",
                        data_type=helper.TensorProto.INT64,
                        dims=[1],
                        vals=[inp.data],
                    )
                )
                nodes.append(dim_constant)
                unsqueezed_names.append(dim_name)
            else:
                inp_name = get_var_name(inp)
                unsqueezed_name = f"{shape_name}_unsqueezed{i}"

                axes_name = f"{unsqueezed_name}_axes"
                axes_constant = helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=[axes_name],
                    name=f"Constant_{axes_name}",
                    value=helper.make_tensor(
                        name=f"{axes_name}_value",
                        data_type=helper.TensorProto.INT64,
                        dims=[1],
                        vals=[0],
                    )
                )
                nodes.append(axes_constant)

                unsqueeze_node = helper.make_node(
                    'Unsqueeze',
                    inputs=[inp_name, axes_name],
                    outputs=[unsqueezed_name],
                    name=f"Unsqueeze_{unsqueezed_name}",
                )
                nodes.append(unsqueeze_node)
                unsqueezed_names.append(unsqueezed_name)

        concat_node = helper.make_node(
            'Concat',
            inputs=unsqueezed_names,
            outputs=[shape_name],
            name=f"Concat_{shape_name}",
            axis=0,
        )
        nodes.append(concat_node)

    # ConstantOfShape with value 0
    dtype = op.dtype
    dtype_map = {
        'float32': helper.TensorProto.FLOAT,
        'float64': helper.TensorProto.DOUBLE,
        'int32': helper.TensorProto.INT32,
        'int64': helper.TensorProto.INT64,
    }
    onnx_dtype = dtype_map.get(dtype, helper.TensorProto.FLOAT)

    constant_of_shape_node = helper.make_node(
        'ConstantOfShape',
        inputs=[shape_name],
        outputs=[output_name],
        name=f"ConstantOfShape_{output_name}",
        value=helper.make_tensor(
            name=f"{output_name}_value",
            data_type=onnx_dtype,
            dims=[1],
            vals=[0],
        )
    )
    nodes.append(constant_of_shape_node)

    return nodes


@onnx_funcify.register(MakeVector)
def onnx_funcify_MakeVector(op, node, get_var_name, **kwargs):
    """Convert MakeVector to ONNX Concat of Unsqueezed scalars.

    MakeVector creates a 1D vector from scalars.

    Example:
        x = pt.make_vector(1.0, 2.0, 3.0)  # Create [1.0, 2.0, 3.0]

        ONNX:
            Unsqueeze(1.0, axes=[0]) -> [1.0]
            Unsqueeze(2.0, axes=[0]) -> [2.0]
            Unsqueeze(3.0, axes=[0]) -> [3.0]
            Concat([1.0], [2.0], [3.0], axis=0) -> [1.0, 2.0, 3.0]
    """
    output_name = get_var_name(node.outputs[0])

    if len(node.inputs) == 0:
        # Empty vector
        dtype = op.dtype
        dtype_map = {
            'float32': helper.TensorProto.FLOAT,
            'float64': helper.TensorProto.DOUBLE,
            'int32': helper.TensorProto.INT32,
            'int64': helper.TensorProto.INT64,
        }
        onnx_dtype = dtype_map.get(dtype, helper.TensorProto.FLOAT)

        empty_constant = helper.make_node(
            'Constant',
            inputs=[],
            outputs=[output_name],
            name=f"Constant_{output_name}",
            value=helper.make_tensor(
                name=f"{output_name}_value",
                data_type=onnx_dtype,
                dims=[0],
                vals=[],
            )
        )

        return empty_constant

    # Unsqueeze each scalar to shape (1,), then concatenate
    nodes = []
    unsqueezed_names = []

    for i, inp in enumerate(node.inputs):
        input_name = get_var_name(inp)
        unsqueezed_name = f"{output_name}_elem_{i}"

        # Create axes constant for Unsqueeze
        axes_name = f"{unsqueezed_name}_axes"
        axes_constant = helper.make_node(
            'Constant',
            inputs=[],
            outputs=[axes_name],
            name=f"Constant_{axes_name}",
            value=helper.make_tensor(
                name=f"{axes_name}_value",
                data_type=helper.TensorProto.INT64,
                dims=[1],
                vals=[0],
            )
        )
        nodes.append(axes_constant)

        unsqueeze_node = helper.make_node(
            'Unsqueeze',
            inputs=[input_name, axes_name],
            outputs=[unsqueezed_name],
            name=f"Unsqueeze_{unsqueezed_name}",
        )
        nodes.append(unsqueeze_node)
        unsqueezed_names.append(unsqueezed_name)

    # Concatenate all elements
    concat_node = helper.make_node(
        'Concat',
        inputs=unsqueezed_names,
        outputs=[output_name],
        name=f"Concat_{output_name}",
        axis=0,
    )
    nodes.append(concat_node)

    return nodes


@onnx_funcify.register(ARange)
def onnx_funcify_ARange(op, node, get_var_name, **kwargs):
    """Convert ARange to ONNX Range node.

    IMPORTANT: ONNX Range requires constant inputs (start, limit, delta).
    Dynamic ranges are not supported in ONNX standard.

    Example:
        x = pt.arange(0, 10, 2, dtype='int64')  # Create [0, 2, 4, 6, 8]

        ONNX:
            Constant(0) -> start
            Constant(10) -> stop
            Constant(2) -> step
            Range(start, stop, step) -> [0, 2, 4, 6, 8]
    """
    start_input = node.inputs[0]
    stop_input = node.inputs[1]
    step_input = node.inputs[2]

    # Verify all inputs are constants
    if not all(isinstance(inp, Constant) for inp in [start_input, stop_input, step_input]):
        raise NotImplementedError(
            "ARange with dynamic (non-constant) inputs is not supported in ONNX. "
            "All start, stop, step values must be constants."
        )

    output_name = get_var_name(node.outputs[0])

    # Create constant nodes for start, limit, delta
    start_name = f"{output_name}_start"
    stop_name = f"{output_name}_stop"
    step_name = f"{output_name}_step"

    dtype = op.dtype
    dtype_map = {
        'int32': helper.TensorProto.INT32,
        'int64': helper.TensorProto.INT64,
        'float32': helper.TensorProto.FLOAT,
        'float64': helper.TensorProto.DOUBLE,
    }
    onnx_dtype = dtype_map.get(dtype, helper.TensorProto.INT64)

    start_constant = helper.make_node(
        'Constant',
        inputs=[],
        outputs=[start_name],
        name=f"Constant_{start_name}",
        value=helper.make_tensor(
            name=f"{start_name}_value",
            data_type=onnx_dtype,
            dims=[],
            vals=[int(start_input.data) if 'int' in dtype else float(start_input.data)],
        )
    )

    stop_constant = helper.make_node(
        'Constant',
        inputs=[],
        outputs=[stop_name],
        name=f"Constant_{stop_name}",
        value=helper.make_tensor(
            name=f"{stop_name}_value",
            data_type=onnx_dtype,
            dims=[],
            vals=[int(stop_input.data) if 'int' in dtype else float(stop_input.data)],
        )
    )

    step_constant = helper.make_node(
        'Constant',
        inputs=[],
        outputs=[step_name],
        name=f"Constant_{step_name}",
        value=helper.make_tensor(
            name=f"{step_name}_value",
            data_type=onnx_dtype,
            dims=[],
            vals=[int(step_input.data) if 'int' in dtype else float(step_input.data)],
        )
    )

    # Range node
    range_node = helper.make_node(
        'Range',
        inputs=[start_name, stop_name, step_name],
        outputs=[output_name],
        name=f"Range_{output_name}",
    )

    return [start_constant, stop_constant, step_constant, range_node]
