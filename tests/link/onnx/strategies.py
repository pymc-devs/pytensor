"""Hypothesis strategies and operation registries for ONNX backend testing."""

from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, array_shapes
import numpy as np
import pytensor.tensor as pt
from typing import Dict, Callable, Any


# ============================================================================
# HYPOTHESIS STRATEGIES (Custom Helpers) - Define first!
# ============================================================================

def factorize(n):
    """Simple factorization for shape generation."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors if factors else [n]


def compatible_shape_for_size(total_size):
    """Generate shapes compatible with given total size."""
    # Simple factorizations
    factors = factorize(total_size)
    shapes = [
        (total_size,),
        (1, total_size),
        (total_size, 1),
    ]
    if len(factors) >= 2:
        shapes.append(tuple(factors[:2]))
    return st.sampled_from(shapes)


def reshape_strategy():
    """Generate tensor and compatible reshape target."""
    @st.composite
    def strategy(draw):
        # Original shape
        shape = draw(array_shapes(min_dims=2, max_dims=3, min_side=2, max_side=6))
        total_size = int(np.prod(shape))

        # Generate tensor
        x = np.random.randn(*shape).astype('float32')

        # Generate compatible new shape (same total size)
        new_shape = draw(compatible_shape_for_size(total_size))

        return x, new_shape

    return strategy()


def concatenate_strategy():
    """Generate tensors and axis for concatenation."""
    @st.composite
    def strategy(draw):
        # Generate base shape
        shape = draw(array_shapes(min_dims=2, max_dims=3, min_side=2, max_side=8))
        axis = draw(st.integers(0, len(shape) - 1))

        # Generate two tensors with same shape except along axis
        a = np.random.randn(*shape).astype('float32')

        b_shape = list(shape)
        b_shape[axis] = draw(st.integers(2, 8))  # Different size along axis
        b = np.random.randn(*b_shape).astype('float32')

        return a, b, axis

    return strategy()


def tensor_with_axis_strategy(dtype='float32', allow_none=True):
    """Generate tensor and valid axis for reduction operations."""
    @st.composite
    def strategy(draw):
        # Generate shape
        shape = draw(array_shapes(min_dims=2, max_dims=4, min_side=2, max_side=10))

        # Generate tensor
        if dtype == 'bool':
            x = draw(arrays(dtype=np.bool_, shape=shape))
        else:
            x = draw(arrays(dtype=getattr(np, dtype), shape=shape, elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)))

        # Generate axis
        if allow_none:
            axis = draw(st.one_of(
                st.none(),
                st.integers(0, len(shape) - 1),
                st.lists(st.integers(0, len(shape) - 1), min_size=1, max_size=len(shape), unique=True)
            ))
        else:
            axis = draw(st.integers(0, len(shape) - 1))

        return x, axis

    return strategy()


def alloc_strategy():
    """Generate scalar value and shape for Alloc."""
    return st.builds(
        lambda val, s1, s2: (val, s1, s2),
        val=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        s1=st.integers(2, 10),
        s2=st.integers(2, 10)
    )


def arange_strategy():
    """Generate valid start, stop, step for arange (constant only)."""
    @st.composite
    def strategy(draw):
        start = draw(st.integers(0, 5))
        stop = draw(st.integers(start + 2, start + 20))
        step = draw(st.integers(1, 3))
        return start, stop, step

    return strategy()


def set_subtensor_strategy():
    """Generate tensor and values for set_subtensor."""
    @st.composite
    def strategy(draw):
        size = draw(st.integers(10, 20))
        x = np.arange(size, dtype='float32')
        values = draw(arrays(dtype=np.float32, shape=(3,), elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)))
        return x, values

    return strategy()


def advanced_index_strategy():
    """Generate tensor and integer indices for advanced indexing."""
    @st.composite
    def strategy(draw):
        size = draw(st.integers(10, 20))
        x = np.arange(size, dtype='float32')
        indices = draw(st.lists(st.integers(0, size - 1), min_size=1, max_size=5))
        return x, np.array(indices, dtype='int64')

    return strategy()


# ============================================================================
# SHAPE OPERATIONS REGISTRY (Tier 2)
# ============================================================================

SHAPE_OPERATIONS: Dict[str, Dict[str, Any]] = {
    # Shape inspection (already implemented in Phase 0)
    "shape": {
        "build_graph": lambda x: ([x], x.shape),
        "strategy": st.builds(
            lambda shape: np.random.randn(*shape).astype('float32'),
            shape=array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=10)
        ),
        "expected_onnx_ops": ['Shape'],
        "description": "Get tensor shape"
    },

    "shape_i": {
        "build_graph": lambda x, i: ([x], x.shape[i]),
        "strategy": st.builds(
            lambda shape, i: (np.random.randn(*shape).astype('float32'), min(i, len(shape)-1)),
            shape=array_shapes(min_dims=2, max_dims=4, min_side=1, max_side=10),
            i=st.integers(0, 3)
        ),
        "expected_onnx_ops": ['Shape', 'Gather'],
        "description": "Get specific dimension"
    },

    # Reshape operations
    "reshape": {
        "build_graph": lambda x, new_shape: ([x], x.reshape(new_shape)),
        "strategy": reshape_strategy(),
        "expected_onnx_ops": ['Reshape'],
        "description": "Reshape tensor"
    },

    "transpose": {
        "build_graph": lambda x: ([x], x.T),
        "strategy": st.builds(
            lambda shape: np.random.randn(*shape).astype('float32'),
            shape=st.tuples(st.integers(2, 10), st.integers(2, 10))
        ),
        "expected_onnx_ops": ['Transpose'],
        "description": "Transpose matrix"
    },

    "dimshuffle_add_dim": {
        "build_graph": lambda x: ([x], x.dimshuffle('x', 0)),
        "strategy": st.builds(
            lambda size: np.random.randn(size).astype('float32'),
            size=st.integers(2, 20)
        ),
        "expected_onnx_ops": ['Unsqueeze'],
        "description": "Add dimension via dimshuffle"
    },

    "dimshuffle_squeeze": {
        "build_graph": lambda x: ([x], x.dimshuffle(0, 2)),
        "strategy": st.builds(
            lambda s1, s2: np.random.randn(s1, 1, s2).astype('float32'),
            s1=st.integers(2, 10),
            s2=st.integers(2, 10)
        ),
        "expected_onnx_ops": ['Squeeze'],
        "description": "Remove dimension via dimshuffle"
    },

    # Join/Split operations
    "concatenate": {
        "build_graph": lambda a, b, axis: ([a, b], pt.concatenate([a, b], axis=axis)),
        "strategy": concatenate_strategy(),
        "expected_onnx_ops": ['Concat'],
        "description": "Concatenate tensors"
    },

    "stack": {
        "build_graph": lambda a, b: ([a, b], pt.stack([a, b], axis=0)),
        "strategy": st.builds(
            lambda shape: (
                np.random.randn(*shape).astype('float32'),
                np.random.randn(*shape).astype('float32')
            ),
            shape=array_shapes(min_dims=1, max_dims=3, min_side=2, max_side=10)
        ),
        "expected_onnx_ops": ['Concat', 'Unsqueeze'],
        "description": "Stack tensors"
    },
}


# ============================================================================
# REDUCTION OPERATIONS REGISTRY (Tier 3)
# ============================================================================

REDUCTION_OPERATIONS: Dict[str, Dict[str, Any]] = {
    "sum": {
        "build_graph": lambda x_data, axis: (
            lambda x_var: ([x_var], pt.sum(x_var, axis=axis))
        )(pt.tensor('x', dtype=x_data.dtype, shape=(None,) * x_data.ndim)),
        "strategy": tensor_with_axis_strategy(),
        "expected_onnx_ops": ['ReduceSum'],
        "description": "Sum reduction"
    },

    "prod": {
        "build_graph": lambda x_data, axis: (
            lambda x_var: ([x_var], pt.prod(x_var, axis=axis))
        )(pt.tensor('x', dtype=x_data.dtype, shape=(None,) * x_data.ndim)),
        "strategy": tensor_with_axis_strategy(),
        "expected_onnx_ops": ['ReduceProd'],
        "description": "Product reduction"
    },

    "max": {
        "build_graph": lambda x_data, axis: (
            lambda x_var: ([x_var], pt.max(x_var, axis=axis))
        )(pt.tensor('x', dtype=x_data.dtype, shape=(None,) * x_data.ndim)),
        "strategy": tensor_with_axis_strategy(),
        "expected_onnx_ops": ['ReduceMax'],
        "description": "Max reduction"
    },

    "min": {
        "build_graph": lambda x_data, axis: (
            lambda x_var: ([x_var], pt.min(x_var, axis=axis))
        )(pt.tensor('x', dtype=x_data.dtype, shape=(None,) * x_data.ndim)),
        "strategy": tensor_with_axis_strategy(),
        "expected_onnx_ops": ['Neg', 'ReduceMax'],  # Min is implemented as -max(-x)
        "description": "Min reduction"
    },

    "argmax": {
        "build_graph": lambda x_data, axis: (
            lambda x_var: ([x_var], pt.argmax(x_var, axis=axis))
        )(pt.tensor('x', dtype=x_data.dtype, shape=(None,) * x_data.ndim)),
        "strategy": tensor_with_axis_strategy(allow_none=False),
        "expected_onnx_ops": ['ArgMax'],
        "description": "Argmax reduction"
    },

    "argmin": {
        "build_graph": lambda x_data, axis: (
            lambda x_var: ([x_var], pt.argmin(x_var, axis=axis))
        )(pt.tensor('x', dtype=x_data.dtype, shape=(None,) * x_data.ndim)),
        "strategy": tensor_with_axis_strategy(allow_none=False),
        "expected_onnx_ops": ['Neg', 'ArgMax'],  # Argmin is implemented as argmax(-x)
        "description": "Argmin reduction"
    },

    # Skip all/any for now - they have issues with boolean types in ONNX
}


# ============================================================================
# ALLOCATION OPERATIONS REGISTRY (Tier 3)
# ============================================================================

ALLOCATION_OPERATIONS: Dict[str, Dict[str, Any]] = {
    "alloc_scalar": {
        "build_graph": lambda val, s1, s2: ([], pt.alloc(val, s1, s2)),
        "strategy": alloc_strategy(),
        "expected_onnx_ops": ['Expand'],
        "description": "Allocate tensor from scalar"
    },

    "alloc_empty": {
        "build_graph": lambda s1, s2: ([], pt.empty((s1, s2), dtype='float32')),
        "strategy": st.tuples(st.integers(2, 10), st.integers(2, 10)),
        "expected_onnx_ops": ['ConstantOfShape'],
        "description": "Allocate uninitialized tensor"
    },

    "make_vector": {
        "build_graph": lambda v1, v2, v3: ([], pt.stack([v1, v2, v3])),
        "strategy": st.builds(
            lambda: tuple(float(x) for x in np.random.randn(3)),
        ),
        "expected_onnx_ops": ['Concat', 'Unsqueeze'],
        "description": "Create vector from scalars"
    },

    "arange": {
        "build_graph": lambda start, stop, step: ([], pt.arange(start, stop, step, dtype='int64')),
        "strategy": arange_strategy(),
        "expected_onnx_ops": ['Range'],
        "description": "Create range tensor"
    },
}


# ============================================================================
# SUBTENSOR OPERATIONS REGISTRY
# ============================================================================

SUBTENSOR_OPERATIONS: Dict[str, Dict[str, Any]] = {
    "slice_basic": {
        "build_graph": lambda x: ([x], x[2:5]),
        "strategy": st.builds(
            lambda size: np.arange(size, dtype='float32'),
            size=st.integers(10, 20)
        ),
        "expected_onnx_ops": ['Slice'],
        "description": "Basic slicing"
    },

    "slice_multidim": {
        "build_graph": lambda x: ([x], x[1:3, 2:4]),
        "strategy": st.builds(
            lambda s1, s2: np.arange(s1 * s2).reshape(s1, s2).astype('float32'),
            s1=st.integers(5, 10),
            s2=st.integers(5, 10)
        ),
        "expected_onnx_ops": ['Slice'],
        "description": "Multi-dimensional slicing"
    },

    "slice_with_step": {
        "build_graph": lambda x: ([x], x[::2]),
        "strategy": st.builds(
            lambda size: np.arange(size, dtype='float32'),
            size=st.integers(10, 20)
        ),
        "expected_onnx_ops": ['Slice'],
        "description": "Slicing with step"
    },

    "advanced_index": {
        "build_graph": lambda x, indices: ([x], x[indices]),
        "strategy": advanced_index_strategy(),
        "expected_onnx_ops": ['Gather'],
        "description": "Advanced indexing with integer array"
    },
}


# ============================================================================
# INCSUBTENSOR OPERATIONS REGISTRY
# ============================================================================

INCSUBTENSOR_OPERATIONS: Dict[str, Dict[str, Any]] = {
    "set_subtensor": {
        "build_graph": lambda x, values: ([x], pt.set_subtensor(x[2:5], values)),
        "strategy": set_subtensor_strategy(),
        "expected_onnx_ops": ['ScatterND', 'ScatterElements'],
        "description": "Set subtensor values"
    },

    "inc_subtensor": {
        "build_graph": lambda x, values: ([x], pt.inc_subtensor(x[2:5], values)),
        "strategy": set_subtensor_strategy(),
        "expected_onnx_ops": ['ScatterND', 'ScatterElements', 'Add'],
        "description": "Increment subtensor values"
    },
}
