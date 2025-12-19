"""Hypothesis strategies and operation registries for ONNX backend testing."""

from typing import Any

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

import pytensor.tensor as pt


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
    # Generate valid shapes from factors
    # For 2-factor shapes, use pairs that multiply to total_size
    if len(factors) >= 2:
        # Use first factor and product of remaining factors
        factor1 = factors[0]
        remaining_product = total_size // factor1
        shapes.append((factor1, remaining_product))

        # Also try middle split if we have at least 2 factors
        if len(factors) >= 2:
            mid = len(factors) // 2
            left_product = int(np.prod(factors[:mid]))
            right_product = int(np.prod(factors[mid:]))
            shapes.append((left_product, right_product))

    return st.sampled_from(shapes)


def reshape_strategy():
    """Generate tensor and compatible reshape target."""

    @st.composite
    def strategy(draw):
        # Original shape
        shape = draw(array_shapes(min_dims=2, max_dims=3, min_side=2, max_side=6))
        total_size = int(np.prod(shape))

        # Generate tensor
        x = np.random.randn(*shape).astype("float32")

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
        a = np.random.randn(*shape).astype("float32")

        b_shape = list(shape)
        b_shape[axis] = draw(st.integers(2, 8))  # Different size along axis
        b = np.random.randn(*b_shape).astype("float32")

        return a, b, axis

    return strategy()


def tensor_with_axis_strategy(dtype="float32", allow_none=True):
    """Generate tensor and valid axis for reduction operations."""

    @st.composite
    def strategy(draw):
        # Generate shape
        shape = draw(array_shapes(min_dims=2, max_dims=4, min_side=2, max_side=10))

        # Generate tensor
        if dtype == "bool":
            x = draw(arrays(dtype=np.bool_, shape=shape))
        else:
            x = draw(
                arrays(
                    dtype=getattr(np, dtype),
                    shape=shape,
                    elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
                )
            )

        # Generate axis
        if allow_none:
            axis = draw(
                st.one_of(
                    st.none(),
                    st.integers(0, len(shape) - 1),
                    st.lists(
                        st.integers(0, len(shape) - 1),
                        min_size=1,
                        max_size=len(shape),
                        unique=True,
                    ),
                )
            )
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
        s2=st.integers(2, 10),
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
        x = np.arange(size, dtype="float32")
        values = draw(
            arrays(
                dtype=np.float32,
                shape=(3,),
                elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
            )
        )
        return x, values

    return strategy()


def advanced_index_strategy():
    """Generate tensor and integer indices for advanced indexing."""

    @st.composite
    def strategy(draw):
        size = draw(st.integers(10, 20))
        x = np.arange(size, dtype="float32")
        indices = draw(st.lists(st.integers(0, size - 1), min_size=1, max_size=5))
        return x, np.array(indices, dtype="int64")

    return strategy()


def binary_float32_arrays_strategy():
    """
    Generate two float32 arrays for binary operations.

    Returns a Hypothesis strategy (lazy evaluation) that generates pairs of
    arrays with identical shapes. Arrays are compatible for element-wise
    operations but not tested for broadcasting in this phase.

    Shape range: 1-3 dimensions, 2-10 elements per dimension
    Value range: [-10, 10] (finite values only)

    Note: Broadcasting validation is deferred to Phase 2.
    """

    @st.composite
    def strategy(draw):
        # Generate compatible shapes for broadcasting
        shape = draw(array_shapes(min_dims=1, max_dims=3, min_side=2, max_side=10))

        # Generate two arrays with same shape
        x = draw(
            arrays(
                dtype=np.float32,
                shape=shape,
                elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
            )
        )
        y = draw(
            arrays(
                dtype=np.float32,
                shape=shape,
                elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
            )
        )

        return x, y

    return strategy()


def unary_float32_array_strategy():
    """
    Generate one float32 array for unary operations.

    Returns a Hypothesis strategy for single array generation.

    Shape range: 1-3 dimensions, 2-10 elements per dimension
    Value range: [-10, 10] (finite values only)
    """
    return arrays(
        dtype=np.float32,
        shape=array_shapes(min_dims=1, max_dims=3, min_side=2, max_side=10),
        elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
    )


def positive_float32_array_strategy():
    """
    Generate positive float32 arrays for operations requiring x > 0.

    Used for: log (requires positive inputs)

    Constraint rationale:
    - Lower bound 1e-3 (not 0) for numerical stability
    - Avoids values too close to zero where log becomes unstable
    - Upper bound 10 keeps values in reasonable range

    Shape range: 1-3 dimensions, 2-10 elements per dimension
    Value range: [1e-3, 10] (strictly positive, finite values only)
    """
    return arrays(
        dtype=np.float32,
        shape=array_shapes(min_dims=1, max_dims=3, min_side=2, max_side=10),
        elements=st.floats(1e-3, 10, allow_nan=False, allow_infinity=False),
    )


def non_negative_float32_array_strategy():
    """
    Generate non-negative float32 arrays for operations requiring x >= 0.

    Used for: sqrt (requires non-negative inputs)

    Constraint rationale:
    - Lower bound 0 (inclusive) is mathematically valid for sqrt
    - No numerical stability issues at zero for sqrt
    - Upper bound 10 keeps values in reasonable range

    Shape range: 1-3 dimensions, 2-10 elements per dimension
    Value range: [0, 10] (non-negative, finite values only)
    """
    return arrays(
        dtype=np.float32,
        shape=array_shapes(min_dims=1, max_dims=3, min_side=2, max_side=10),
        elements=st.floats(0, 10, allow_nan=False, allow_infinity=False),
    )


# ============================================================================
# SHAPE OPERATIONS REGISTRY (Tier 2)
# ============================================================================

SHAPE_OPERATIONS: dict[str, dict[str, Any]] = {
    # Shape inspection (already implemented in Phase 0)
    "shape": {
        "build_graph": lambda x: ([x], x.shape),
        "strategy": st.builds(
            lambda shape: np.random.randn(*shape).astype("float32"),
            shape=array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=10),
        ),
        "expected_onnx_ops": ["Shape"],
        "description": "Get tensor shape",
    },
    "shape_i": {
        "build_graph": lambda x, i: (
            [x],
            # Use Shape_i directly instead of x.shape[i] to avoid Subtensor
            # Shape_i is imported from pytensor.tensor.shape
            __import__("pytensor.tensor.shape", fromlist=["Shape_i"]).Shape_i(i)(x),
        ),
        "strategy": st.builds(
            lambda shape, i: (
                np.random.randn(*shape).astype("float32"),
                min(i, len(shape) - 1),
            ),
            shape=array_shapes(min_dims=2, max_dims=4, min_side=1, max_side=10),
            i=st.integers(0, 3),
        ),
        "expected_onnx_ops": ["Shape", "Gather"],
        "description": "Get specific dimension",
    },
    # Reshape operations
    "reshape": {
        "build_graph": lambda x, new_shape: ([x], x.reshape(new_shape)),
        "strategy": reshape_strategy(),
        "expected_onnx_ops": ["Reshape"],
        "description": "Reshape tensor",
    },
    "transpose": {
        "build_graph": lambda x: ([x], x.T),
        "strategy": st.builds(
            lambda shape: np.random.randn(*shape).astype("float32"),
            shape=st.tuples(st.integers(2, 10), st.integers(2, 10)),
        ),
        "expected_onnx_ops": ["Transpose"],
        "description": "Transpose matrix",
    },
    "dimshuffle_add_dim": {
        "build_graph": lambda x: ([x], x.dimshuffle("x", 0)),
        "strategy": st.builds(
            lambda size: np.random.randn(size).astype("float32"),
            size=st.integers(2, 20),
        ),
        "expected_onnx_ops": ["Unsqueeze"],
        "description": "Add dimension via dimshuffle",
    },
    "dimshuffle_squeeze": {
        "build_graph": lambda x: ([x], x.dimshuffle(0, 2)),
        "strategy": st.builds(
            lambda s1, s2: np.random.randn(s1, 1, s2).astype("float32"),
            s1=st.integers(2, 10),
            s2=st.integers(2, 10),
        ),
        "expected_onnx_ops": ["Squeeze"],
        "description": "Remove dimension via dimshuffle",
    },
    # Join/Split operations
    "concatenate": {
        "build_graph": lambda a, b, axis: ([a, b], pt.concatenate([a, b], axis=axis)),
        "strategy": concatenate_strategy(),
        "expected_onnx_ops": ["Concat"],
        "description": "Concatenate tensors",
    },
    "stack": {
        "build_graph": lambda a, b: ([a, b], pt.stack([a, b], axis=0)),
        "strategy": st.builds(
            lambda shape: (
                np.random.randn(*shape).astype("float32"),
                np.random.randn(*shape).astype("float32"),
            ),
            shape=array_shapes(min_dims=1, max_dims=3, min_side=2, max_side=10),
        ),
        "expected_onnx_ops": ["Concat", "Unsqueeze"],
        "description": "Stack tensors",
    },
}


# ============================================================================
# REDUCTION OPERATIONS REGISTRY (Tier 3)
# ============================================================================

REDUCTION_OPERATIONS: dict[str, dict[str, Any]] = {
    "sum": {
        "build_graph": lambda x_data, axis: (
            lambda x_var: ([x_var], pt.sum(x_var, axis=axis))
        )(pt.tensor("x", dtype=x_data.dtype, shape=(None,) * x_data.ndim)),
        "strategy": tensor_with_axis_strategy(),
        "expected_onnx_ops": ["ReduceSum"],
        "description": "Sum reduction",
    },
    "prod": {
        "build_graph": lambda x_data, axis: (
            lambda x_var: ([x_var], pt.prod(x_var, axis=axis))
        )(pt.tensor("x", dtype=x_data.dtype, shape=(None,) * x_data.ndim)),
        "strategy": tensor_with_axis_strategy(),
        "expected_onnx_ops": ["ReduceProd"],
        "description": "Product reduction",
    },
    "max": {
        "build_graph": lambda x_data, axis: (
            lambda x_var: ([x_var], pt.max(x_var, axis=axis))
        )(pt.tensor("x", dtype=x_data.dtype, shape=(None,) * x_data.ndim)),
        "strategy": tensor_with_axis_strategy(),
        "expected_onnx_ops": ["ReduceMax"],
        "description": "Max reduction",
    },
    "min": {
        "build_graph": lambda x_data, axis: (
            lambda x_var: ([x_var], pt.min(x_var, axis=axis))
        )(pt.tensor("x", dtype=x_data.dtype, shape=(None,) * x_data.ndim)),
        "strategy": tensor_with_axis_strategy(),
        "expected_onnx_ops": ["Neg", "ReduceMax"],  # Min is implemented as -max(-x)
        "description": "Min reduction",
    },
    "argmax": {
        "build_graph": lambda x_data, axis: (
            lambda x_var: ([x_var], pt.argmax(x_var, axis=axis))
        )(pt.tensor("x", dtype=x_data.dtype, shape=(None,) * x_data.ndim)),
        "strategy": tensor_with_axis_strategy(allow_none=False),
        "expected_onnx_ops": ["ArgMax"],
        "description": "Argmax reduction",
    },
    "argmin": {
        "build_graph": lambda x_data, axis: (
            lambda x_var: ([x_var], pt.argmin(x_var, axis=axis))
        )(pt.tensor("x", dtype=x_data.dtype, shape=(None,) * x_data.ndim)),
        "strategy": tensor_with_axis_strategy(allow_none=False),
        "expected_onnx_ops": ["Neg", "ArgMax"],  # Argmin is implemented as argmax(-x)
        "description": "Argmin reduction",
    },
    # Skip all/any for now - they have issues with boolean types in ONNX
}


# ============================================================================
# ALLOCATION OPERATIONS REGISTRY (Tier 3)
# ============================================================================

ALLOCATION_OPERATIONS: dict[str, dict[str, Any]] = {
    "alloc_scalar": {
        "build_graph": lambda val, s1, s2: ([], pt.alloc(val, s1, s2)),
        "strategy": alloc_strategy(),
        "expected_onnx_ops": ["Expand"],
        "description": "Allocate tensor from scalar",
    },
    "alloc_empty": {
        "build_graph": lambda s1, s2: ([], pt.empty((s1, s2), dtype="float32")),
        "strategy": st.tuples(st.integers(2, 10), st.integers(2, 10)),
        "expected_onnx_ops": ["ConstantOfShape"],
        "description": "Allocate uninitialized tensor",
    },
    "make_vector": {
        "build_graph": lambda v1, v2, v3: ([], pt.stack([v1, v2, v3])),
        "strategy": st.builds(
            lambda: tuple(float(x) for x in np.random.randn(3)),
        ),
        "expected_onnx_ops": ["Concat", "Unsqueeze"],
        "description": "Create vector from scalars",
    },
    "arange": {
        "build_graph": lambda start, stop, step: (
            [],
            pt.arange(start, stop, step, dtype="int64"),
        ),
        "strategy": arange_strategy(),
        "expected_onnx_ops": ["Range"],
        "description": "Create range tensor",
    },
}


# ============================================================================
# SUBTENSOR OPERATIONS REGISTRY
# ============================================================================

SUBTENSOR_OPERATIONS: dict[str, dict[str, Any]] = {
    "slice_basic": {
        "build_graph": lambda x_val: (lambda x: ([x], x[2:5]))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim)
        ),
        "strategy": st.builds(
            lambda size: np.arange(size, dtype="float32"), size=st.integers(10, 20)
        ),
        "expected_onnx_ops": ["Slice"],
        "description": "Basic slicing",
    },
    "slice_multidim": {
        "build_graph": lambda x_val: (lambda x: ([x], x[1:3, 2:4]))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim)
        ),
        "strategy": st.builds(
            lambda s1, s2: np.arange(s1 * s2).reshape(s1, s2).astype("float32"),
            s1=st.integers(5, 10),
            s2=st.integers(5, 10),
        ),
        "expected_onnx_ops": ["Slice"],
        "description": "Multi-dimensional slicing",
    },
    "slice_with_step": {
        "build_graph": lambda x_val: (lambda x: ([x], x[::2]))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim)
        ),
        "strategy": st.builds(
            lambda size: np.arange(size, dtype="float32"), size=st.integers(10, 20)
        ),
        "expected_onnx_ops": ["Slice"],
        "description": "Slicing with step",
    },
    "advanced_index": {
        "build_graph": lambda x_val, indices_val: (
            lambda x, indices: ([x, indices], x[indices])
        )(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim),
            pt.tensor("indices", dtype="int64", shape=(None,)),
        ),
        "strategy": advanced_index_strategy(),
        "expected_onnx_ops": ["Gather"],
        "description": "Advanced indexing with integer array",
    },
}


# ============================================================================
# INCSUBTENSOR OPERATIONS REGISTRY
# ============================================================================

INCSUBTENSOR_OPERATIONS: dict[str, dict[str, Any]] = {
    "set_subtensor": {
        "build_graph": lambda x_val, values_val: (
            lambda x, values: ([x, values], pt.set_subtensor(x[2:5], values))
        )(
            pt.tensor("x", dtype="float32", shape=(None,)),
            pt.tensor("values", dtype="float32", shape=(None,)),
        ),
        "strategy": set_subtensor_strategy(),
        "expected_onnx_ops": ["ScatterND", "ScatterElements"],
        "description": "Set subtensor values",
    },
    "inc_subtensor": {
        "build_graph": lambda x_val, values_val: (
            lambda x, values: ([x, values], pt.inc_subtensor(x[2:5], values))
        )(
            pt.tensor("x", dtype="float32", shape=(None,)),
            pt.tensor("values", dtype="float32", shape=(None,)),
        ),
        "strategy": set_subtensor_strategy(),
        "expected_onnx_ops": ["ScatterND", "ScatterElements", "Add"],
        "description": "Increment subtensor values",
    },
}


# ============================================================================
# ELEMWISE OPERATIONS REGISTRY (Tier 1)
# ============================================================================

ELEMWISE_OPERATIONS: dict[str, dict[str, Any]] = {
    # =================================================================
    # BINARY ARITHMETIC OPERATIONS
    # =================================================================
    "add": {
        "build_graph": lambda x_val, y_val: (lambda x, y: ([x, y], x + y))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim),
            pt.tensor("y", dtype="float32", shape=(None,) * y_val.ndim),
        ),
        "strategy": binary_float32_arrays_strategy(),
        "expected_onnx_ops": ["Add"],
        "description": "Element-wise addition",
    },
    "mul": {
        "build_graph": lambda x_val, y_val: (lambda x, y: ([x, y], x * y))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim),
            pt.tensor("y", dtype="float32", shape=(None,) * y_val.ndim),
        ),
        "strategy": binary_float32_arrays_strategy(),
        "expected_onnx_ops": ["Mul"],
        "description": "Element-wise multiplication",
    },
    "sub": {
        "build_graph": lambda x_val, y_val: (lambda x, y: ([x, y], x - y))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim),
            pt.tensor("y", dtype="float32", shape=(None,) * y_val.ndim),
        ),
        "strategy": binary_float32_arrays_strategy(),
        "expected_onnx_ops": ["Sub"],
        "description": "Element-wise subtraction",
    },
    "div": {
        "build_graph": lambda x_val, y_val: (lambda x, y: ([x, y], x / y))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim),
            pt.tensor("y", dtype="float32", shape=(None,) * y_val.ndim),
        ),
        "strategy": binary_float32_arrays_strategy(),
        "expected_onnx_ops": ["Div"],
        "description": "Element-wise division",
    },
    "int_div": {
        "build_graph": lambda x_val, y_val: (lambda x, y: ([x, y], x // y))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim),
            pt.tensor("y", dtype="float32", shape=(None,) * y_val.ndim),
        ),
        "strategy": binary_float32_arrays_strategy(),
        # NOTE: expected_onnx_ops couples test to implementation details
        # This specifies HOW int_div is implemented (div + floor) rather than
        # just testing correctness. This is intentional for ONNX backend validation
        # but makes tests brittle if implementation changes.
        "expected_onnx_ops": ["Div", "Floor"],  # Integer division is div + floor
        "description": "Element-wise integer division",
    },
    "pow": {
        "build_graph": lambda x_val, y_val: (lambda x, y: ([x, y], x**y))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim),
            pt.tensor("y", dtype="float32", shape=(None,) * y_val.ndim),
        ),
        "strategy": binary_float32_arrays_strategy(),
        "expected_onnx_ops": ["Pow"],
        "description": "Element-wise power",
    },
    # =================================================================
    # ELEMENT-WISE MIN/MAX OPERATIONS
    # =================================================================
    "maximum": {
        "build_graph": lambda x_val, y_val: (lambda x, y: ([x, y], pt.maximum(x, y)))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim),
            pt.tensor("y", dtype="float32", shape=(None,) * y_val.ndim),
        ),
        "strategy": binary_float32_arrays_strategy(),
        "expected_onnx_ops": ["Max"],
        "description": "Element-wise maximum",
    },
    "minimum": {
        "build_graph": lambda x_val, y_val: (lambda x, y: ([x, y], pt.minimum(x, y)))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim),
            pt.tensor("y", dtype="float32", shape=(None,) * y_val.ndim),
        ),
        "strategy": binary_float32_arrays_strategy(),
        "expected_onnx_ops": ["Min"],
        "description": "Element-wise minimum",
    },
    # =================================================================
    # UNARY OPERATIONS
    # =================================================================
    "neg": {
        "build_graph": lambda x_val: (lambda x: ([x], -x))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim)
        ),
        "strategy": unary_float32_array_strategy(),
        "expected_onnx_ops": ["Neg"],
        "description": "Element-wise negation",
    },
    "abs": {
        "build_graph": lambda x_val: (lambda x: ([x], pt.abs(x)))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim)
        ),
        "strategy": unary_float32_array_strategy(),
        "expected_onnx_ops": ["Abs"],
        "description": "Element-wise absolute value",
    },
    "exp": {
        "build_graph": lambda x_val: (lambda x: ([x], pt.exp(x)))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim)
        ),
        "strategy": unary_float32_array_strategy(),
        "expected_onnx_ops": ["Exp"],
        "description": "Element-wise exponential",
    },
    # =================================================================
    # CONSTRAINED UNARY OPERATIONS
    # =================================================================
    "log": {
        "build_graph": lambda x_val: (lambda x: ([x], pt.log(x)))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim)
        ),
        "strategy": positive_float32_array_strategy(),
        "expected_onnx_ops": ["Log"],
        "description": "Element-wise natural logarithm",
    },
    "sqrt": {
        "build_graph": lambda x_val: (lambda x: ([x], pt.sqrt(x)))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim)
        ),
        "strategy": non_negative_float32_array_strategy(),
        "expected_onnx_ops": ["Sqrt"],
        "description": "Element-wise square root",
    },
    # =================================================================
    # ROUNDING OPERATIONS
    # =================================================================
    "floor": {
        "build_graph": lambda x_val: (lambda x: ([x], pt.floor(x)))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim)
        ),
        "strategy": unary_float32_array_strategy(),
        "expected_onnx_ops": ["Floor"],
        "description": "Element-wise floor",
    },
    "ceil": {
        "build_graph": lambda x_val: (lambda x: ([x], pt.ceil(x)))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim)
        ),
        "strategy": unary_float32_array_strategy(),
        "expected_onnx_ops": ["Ceil"],
        "description": "Element-wise ceiling",
    },
    "round": {
        "build_graph": lambda x_val: (lambda x: ([x], pt.round(x)))(
            pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim)
        ),
        "strategy": unary_float32_array_strategy(),
        "expected_onnx_ops": ["Round"],
        "description": "Element-wise rounding (half to even)",
    },
    "round_away": {
        "build_graph": lambda x_val: (
            lambda x: ([x], pt.round(x, mode="half_away_from_zero"))
        )(pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim)),
        "strategy": unary_float32_array_strategy(),
        "expected_onnx_ops": ["Round"],
        "description": "Element-wise rounding (half away from zero)",
    },
    # =================================================================
    # SPECIAL OPERATIONS
    # =================================================================
    "clip": {
        "build_graph": lambda x_val, min_val, max_val: (
            lambda x: ([x], pt.clip(x, min_val, max_val))
        )(pt.tensor("x", dtype="float32", shape=(None,) * x_val.ndim)),
        # Strategy ensures min_v < max_v by construction:
        # min_v from [-5, 0] and max_v from [0, 5] guarantees min_v <= 0 <= max_v
        # Edge case: min_v == max_v == 0 is possible but rare
        # This edge case (all values clipped to same value) is worth testing
        # separately in Phase 2 manual tests if needed
        "strategy": st.builds(
            lambda x, min_v, max_v: (x, float(min_v), float(max_v)),
            x=unary_float32_array_strategy(),
            min_v=st.floats(-5, 0),
            max_v=st.floats(0, 5),
        ),
        "expected_onnx_ops": ["Clip"],
        "description": "Element-wise clipping",
    },
}
