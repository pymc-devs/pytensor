"""ONNX conversion for subtensor (slicing) operations."""

import sys
import numpy as np
from onnx import helper, numpy_helper

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.subtensor import Subtensor, AdvancedSubtensor, AdvancedSubtensor1, IncSubtensor
from pytensor.graph.basic import Constant


@onnx_funcify.register(Subtensor)
def onnx_funcify_Subtensor(op, node, get_var_name, **kwargs):
    """Convert Subtensor (slicing) to ONNX Slice node.

    Subtensor performs array slicing like x[start:stop:step].

    ONNX Slice (opset 11+) takes inputs:
    - data: the tensor to slice
    - starts: starting indices for each axis (1D tensor)
    - ends: ending indices for each axis (1D tensor)
    - axes: which axes to slice (optional, 1D tensor)
    - steps: step size for each axis (optional, 1D tensor)

    Key challenges:
    1. PyTensor idx_list contains Type objects (placeholders) and slice objects
    2. Actual slice bounds are in node.inputs[1:] as Constants or Variables
    3. Scalar indices reduce dimensionality (not supported by Slice alone)
    4. Negative indices must be converted using Shape operations

    For now, we focus on basic slicing with constant bounds.
    """
    from pytensor.tensor.subtensor import indices_from_subtensor

    input_name = get_var_name(node.inputs[0])
    output_name = get_var_name(node.outputs[0])

    # Reconstruct the actual slice objects from op.idx_list and node.inputs
    # This gives us slice objects with actual Constant values
    actual_indices = indices_from_subtensor(node.inputs[1:], op.idx_list)

    # For now, we only handle pure slice objects (not scalar indices)
    # Scalar indices would reduce dimensionality and require Gather + Squeeze
    if not all(isinstance(idx, slice) for idx in actual_indices):
        raise NotImplementedError(
            f"Subtensor with scalar indices not yet supported. "
            f"Got indices: {actual_indices}. "
            f"Only slice objects (e.g., x[1:3]) are supported."
        )

    # Extract starts, ends, steps, axes from slice objects
    starts = []
    ends = []
    steps = []
    axes = []

    has_negative_indices = False
    has_non_constant_bounds = False

    for axis, idx in enumerate(actual_indices):
        if isinstance(idx, slice):
            # Get start, stop, step from the slice
            # These might be None, int, or Constant Variables
            start = idx.start
            stop = idx.stop
            step = idx.step

            # Convert None to appropriate defaults
            if start is None:
                start_val = 0
            elif isinstance(start, Constant):
                start_val = int(start.data)
            elif isinstance(start, int):
                start_val = start
            else:
                # Dynamic/non-constant start - not yet supported
                has_non_constant_bounds = True
                start_val = 0  # placeholder

            if stop is None:
                stop_val = sys.maxsize
            elif isinstance(stop, Constant):
                stop_val = int(stop.data)
            elif isinstance(stop, int):
                stop_val = stop
            else:
                # Dynamic/non-constant stop
                has_non_constant_bounds = True
                stop_val = sys.maxsize  # placeholder

            if step is None:
                step_val = 1
            elif isinstance(step, Constant):
                step_val = int(step.data)
            elif isinstance(step, int):
                step_val = step
            else:
                # Dynamic/non-constant step
                has_non_constant_bounds = True
                step_val = 1  # placeholder

            # Check for negative indices
            if start_val < 0 or stop_val < 0:
                has_negative_indices = True

            starts.append(start_val)
            ends.append(stop_val)
            steps.append(step_val)
            axes.append(axis)

    # Check for unsupported cases
    if has_non_constant_bounds:
        raise NotImplementedError(
            "Subtensor with dynamic (non-constant) slice bounds not yet supported. "
            "All start, stop, step values must be constants at export time."
        )

    # If no slicing needed (all slices are [:]), pass through
    if not starts:
        return None

    if has_negative_indices:
        raise NotImplementedError(
            f"Subtensor with negative indices not yet implemented. "
            f"Please use non-negative indices for now. "
            f"Got starts={starts}, ends={ends}"
        )

    # Simple case: all indices are non-negative constants
    # Create constant tensors for starts, ends, axes, steps
    starts_name = f"{output_name}_starts"
    ends_name = f"{output_name}_ends"
    axes_name = f"{output_name}_axes"
    steps_name = f"{output_name}_steps"

    # Create constants as initializers
    starts_tensor = numpy_helper.from_array(
        np.array(starts, dtype=np.int64), name=starts_name
    )
    ends_tensor = numpy_helper.from_array(
        np.array(ends, dtype=np.int64), name=ends_name
    )
    axes_tensor = numpy_helper.from_array(
        np.array(axes, dtype=np.int64), name=axes_name
    )
    steps_tensor = numpy_helper.from_array(
        np.array(steps, dtype=np.int64), name=steps_name
    )

    # Create Slice node with input tensors
    slice_node = helper.make_node(
        'Slice',
        inputs=[input_name, starts_name, ends_name, axes_name, steps_name],
        outputs=[output_name],
        name=f"Slice_{output_name}",
    )

    # Return (node, initializers)
    return (slice_node, [starts_tensor, ends_tensor, axes_tensor, steps_tensor])


@onnx_funcify.register(AdvancedSubtensor1)
def onnx_funcify_AdvancedSubtensor1(op, node, get_var_name, **kwargs):
    """Convert AdvancedSubtensor1 to ONNX Gather node.

    AdvancedSubtensor1 performs integer array indexing like x[[0, 2, 5]].
    This maps directly to ONNX Gather operation.

    Example:
        x = pt.vector('x')
        indices = pt.vector('indices', dtype='int64')
        y = x[indices]  # AdvancedSubtensor1

        ONNX: Gather(x, indices, axis=0)
    """
    data_name = get_var_name(node.inputs[0])
    indices_name = get_var_name(node.inputs[1])
    output_name = get_var_name(node.outputs[0])

    gather_node = helper.make_node(
        'Gather',
        inputs=[data_name, indices_name],
        outputs=[output_name],
        name=f"Gather_{output_name}",
        axis=0,  # AdvancedSubtensor1 operates on axis 0
    )

    return gather_node


@onnx_funcify.register(AdvancedSubtensor)
def onnx_funcify_AdvancedSubtensor(op, node, get_var_name, **kwargs):
    """Convert AdvancedSubtensor to ONNX Gather or GatherND node.

    AdvancedSubtensor implements NumPy's advanced indexing.

    For simple cases (single integer array on axis 0), this maps to Gather.
    For complex multi-dimensional indexing, this would require GatherND.

    For now, we handle the simple case: x[indices] where indices is a vector.
    This is the most common case and matches AdvancedSubtensor1 behavior.

    Example:
        x = pt.vector('x')
        indices = pt.vector('indices', dtype='int64')
        y = x[indices]  # AdvancedSubtensor (gets optimized to AdvancedSubtensor1 in normal mode)

        ONNX: Gather(x, indices, axis=0)
    """
    # For now, we only handle the simple case that matches AdvancedSubtensor1
    # More complex cases would need GatherND or multiple operations

    if len(node.inputs) != 2:
        raise NotImplementedError(
            f"AdvancedSubtensor with {len(node.inputs)} inputs not supported. "
            f"Only simple integer array indexing (2 inputs) is currently supported."
        )

    data_name = get_var_name(node.inputs[0])
    indices_name = get_var_name(node.inputs[1])
    output_name = get_var_name(node.outputs[0])

    # Use Gather for simple indexing on axis 0
    gather_node = helper.make_node(
        'Gather',
        inputs=[data_name, indices_name],
        outputs=[output_name],
        name=f"Gather_{output_name}",
        axis=0,  # Simple indexing operates on axis 0
    )

    return gather_node


@onnx_funcify.register(IncSubtensor)
def onnx_funcify_IncSubtensor(op, node, get_var_name, **kwargs):
    """Convert IncSubtensor to ONNX Scatter operations.

    IncSubtensor has two modes:
    1. set_subtensor: x[indices] = values (op.set_instead_of_inc=True)
    2. inc_subtensor: x[indices] += values (op.set_instead_of_inc=False)

    ONNX doesn't have in-place ops, so we use ScatterElements or ScatterND.

    For basic slicing (e.g., x[2:5] = values), we implement this as:
    1. Extract the slice range as indices using ONNX Range
    2. Use ScatterElements to scatter the values at those indices
    3. For inc_subtensor, first extract current values, add, then scatter

    This implementation handles the basic slicing case with constant bounds.
    Advanced cases (negative indices, dynamic bounds, multi-dim) are not yet supported.
    """
    from pytensor.tensor.subtensor import indices_from_subtensor

    # Inputs: [data, values, ...slice_bounds...]
    # Output: modified data
    data_name = get_var_name(node.inputs[0])
    values_name = get_var_name(node.inputs[1])
    output_name = get_var_name(node.outputs[0])

    # Reconstruct the actual slice objects from op.idx_list and node.inputs[2:]
    actual_indices = indices_from_subtensor(node.inputs[2:], op.idx_list)

    # For now, only handle simple 1D slicing on the first axis
    # x[start:stop] = values
    if len(actual_indices) != 1 or not isinstance(actual_indices[0], slice):
        raise NotImplementedError(
            f"IncSubtensor only supports basic 1D slicing for ONNX export. "
            f"Got indices: {actual_indices}. "
            f"Only single-axis slice objects (e.g., x[2:5]) are supported."
        )

    slice_obj = actual_indices[0]
    start = slice_obj.start
    stop = slice_obj.stop
    step = slice_obj.step

    # Extract constant values
    if start is None:
        start_val = 0
    elif isinstance(start, Constant):
        start_val = int(start.data)
    elif isinstance(start, int):
        start_val = start
    else:
        raise NotImplementedError(
            "IncSubtensor with dynamic start index not yet supported"
        )

    if stop is None:
        raise NotImplementedError(
            "IncSubtensor with unbounded stop not yet supported"
        )
    elif isinstance(stop, Constant):
        stop_val = int(stop.data)
    elif isinstance(stop, int):
        stop_val = stop
    else:
        raise NotImplementedError(
            "IncSubtensor with dynamic stop index not yet supported"
        )

    if step is None:
        step_val = 1
    elif isinstance(step, Constant):
        step_val = int(step.data)
    elif isinstance(step, int):
        step_val = step
    else:
        raise NotImplementedError(
            "IncSubtensor with dynamic step not yet supported"
        )

    if step_val != 1:
        raise NotImplementedError(
            "IncSubtensor with step != 1 not yet supported"
        )

    if start_val < 0 or stop_val < 0:
        raise NotImplementedError(
            "IncSubtensor with negative indices not yet supported"
        )

    # Build ONNX graph:
    # 1. Create indices tensor: [start, start+1, ..., stop-1]
    # 2. For set_subtensor: ScatterElements(data, indices, values, axis=0)
    # 3. For inc_subtensor: current = Gather(data, indices),
    #                       new_values = Add(current, values),
    #                       ScatterElements(data, indices, new_values, axis=0)

    nodes = []

    # Create Range node to generate indices [start, start+1, ..., stop-1]
    indices_name = f"{output_name}_indices"
    start_name = f"{output_name}_start"
    stop_name = f"{output_name}_stop"
    step_name = f"{output_name}_step"

    # Create Constant nodes for start, stop, step
    start_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=[start_name],
        name=f"Constant_{start_name}",
        value=helper.make_tensor(
            name=f"{start_name}_value",
            data_type=helper.TensorProto.INT64,
            dims=[],
            vals=[start_val],
        )
    )
    nodes.append(start_const)

    stop_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=[stop_name],
        name=f"Constant_{stop_name}",
        value=helper.make_tensor(
            name=f"{stop_name}_value",
            data_type=helper.TensorProto.INT64,
            dims=[],
            vals=[stop_val],
        )
    )
    nodes.append(stop_const)

    step_const = helper.make_node(
        'Constant',
        inputs=[],
        outputs=[step_name],
        name=f"Constant_{step_name}",
        value=helper.make_tensor(
            name=f"{step_name}_value",
            data_type=helper.TensorProto.INT64,
            dims=[],
            vals=[step_val],
        )
    )
    nodes.append(step_const)

    # Range node: creates [start, start+1, ..., stop-1]
    range_node = helper.make_node(
        'Range',
        inputs=[start_name, stop_name, step_name],
        outputs=[indices_name],
        name=f"Range_{indices_name}",
    )
    nodes.append(range_node)

    # Handle set_subtensor vs inc_subtensor
    if op.set_instead_of_inc:
        # set_subtensor: directly scatter the new values
        scatter_node = helper.make_node(
            'ScatterElements',
            inputs=[data_name, indices_name, values_name],
            outputs=[output_name],
            name=f"ScatterElements_{output_name}",
            axis=0,
        )
        nodes.append(scatter_node)
    else:
        # inc_subtensor: gather current, add, then scatter
        # 1. Gather current values
        current_values_name = f"{output_name}_current"
        gather_node = helper.make_node(
            'Gather',
            inputs=[data_name, indices_name],
            outputs=[current_values_name],
            name=f"Gather_{current_values_name}",
            axis=0,
        )
        nodes.append(gather_node)

        # 2. Add current + new values
        sum_values_name = f"{output_name}_sum"
        add_node = helper.make_node(
            'Add',
            inputs=[current_values_name, values_name],
            outputs=[sum_values_name],
            name=f"Add_{sum_values_name}",
        )
        nodes.append(add_node)

        # 3. Scatter the summed values
        scatter_node = helper.make_node(
            'ScatterElements',
            inputs=[data_name, indices_name, sum_values_name],
            outputs=[output_name],
            name=f"ScatterElements_{output_name}",
            axis=0,
        )
        nodes.append(scatter_node)

    # Return list of nodes
    return nodes
