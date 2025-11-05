"""ONNX conversion for subtensor (slicing) operations."""

import sys
import numpy as np
from onnx import helper, numpy_helper

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.subtensor import Subtensor, AdvancedSubtensor1, IncSubtensor
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


@onnx_funcify.register(IncSubtensor)
def onnx_funcify_IncSubtensor(op, node, get_var_name, **kwargs):
    """Convert IncSubtensor to ONNX Scatter operations.

    IncSubtensor has two modes:
    1. set_subtensor: x[indices] = values (op.set_instead_of_inc=True)
    2. inc_subtensor: x[indices] += values (op.set_instead_of_inc=False)

    ONNX doesn't have in-place ops, so we use ScatterElements or ScatterND.

    This is complex and not yet implemented.
    """
    raise NotImplementedError(
        "IncSubtensor (set_subtensor/inc_subtensor) not yet implemented for ONNX export. "
        "This operation requires ScatterElements or ScatterND which is complex to implement."
    )
