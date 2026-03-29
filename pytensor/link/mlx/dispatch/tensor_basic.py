import mlx.core as mx
import numpy as np

from pytensor.link.mlx.dispatch.basic import convert_dtype_to_mlx, mlx_funcify
from pytensor.tensor import get_vector_length
from pytensor.tensor.basic import (
    Alloc,
    AllocEmpty,
    ARange,
    ExtractDiag,
    Eye,
    Join,
    MakeVector,
    ScalarFromTensor,
    Split,
    TensorFromScalar,
    get_scalar_constant_value,
)
from pytensor.tensor.exceptions import NotScalarConstantError


MLX_DYNAMIC_SHAPE_ERROR = (
    "MLX compilation limitation: Alloc operations with dynamic shapes "
    "cannot be used inside compiled functions. This is because MLX "
    "compilation forbids evaluating arrays to extract shape values. "
    "\n\nWorkarounds:"
    "\n1. Avoid using Alloc with dynamic shapes in compiled contexts"
    "\n2. Use static shapes when possible"
    "\n3. Move Alloc operations outside compiled functions"
)


@mlx_funcify.register(Join)
def mlx_funcify_Join(op, **kwargs):
    def join(axis, *tensors):
        return mx.concatenate(tensors, axis=axis)

    return join


ARANGE_CONCRETE_VALUE_ERROR = (
    "MLX's arange requires all arguments (start, stop, step) to be concrete "
    "Python int/float values, not symbolic variables. Unlike NumPy and JAX, "
    "MLX does not accept array inputs for arange at all."
    "\n\nAn example of a valid graph:"
    "\n>>> import pytensor.tensor as pt"
    "\n>>> pt.arange(1, 10, 2)"
)


@mlx_funcify.register(ARange)
def mlx_funcify_ARange(op, node, **kwargs):
    # MLX's arange only accepts Python int/float, not arrays,
    # so all arguments must be known at graph-construction time.
    try:
        start, stop, step = [
            get_scalar_constant_value(arg).item() for arg in node.inputs
        ]
    except NotScalarConstantError:
        raise NotImplementedError(ARANGE_CONCRETE_VALUE_ERROR)
    dtype = convert_dtype_to_mlx(op.dtype)

    def arange(*_args):
        return mx.arange(start, stop, step, dtype=dtype)

    return arange


@mlx_funcify.register(Split)
def mlx_funcify_Split(op: Split, node, **kwargs):
    _, axis_sym, splits_sym = node.inputs

    try:
        constant_axis = get_scalar_constant_value(axis_sym)
    except NotScalarConstantError:
        constant_axis = None

    try:
        constant_splits = np.array(
            [
                get_scalar_constant_value(splits_sym[i])
                for i in range(get_vector_length(splits_sym))
            ]
        )
    except (ValueError, NotScalarConstantError):
        constant_splits = None

    def split(x, axis, splits):
        # Resolve constants for significant performance improvement (14x speedup)
        if constant_axis is not None:
            axis = int(constant_axis)
        else:
            raise ValueError(
                "Symbolic axis is not supported in MLX Split implementation."
            )

        if constant_splits is not None:
            splits_arr = mx.array(constant_splits)
        else:
            splits_arr = mx.array(splits)

        cumsum_splits = mx.cumsum(splits_arr[:-1]).tolist()

        # Validation checks
        if len(splits) != op.len_splits:
            raise ValueError("Length of 'splits' is not equal to n_splits")
        if np.sum(np.asarray(splits)) != x.shape[axis]:
            raise ValueError(
                "Split sizes do not sum to the input length on the chosen axis."
            )
        if np.any(np.asarray(splits) < 0):
            raise ValueError("Split sizes cannot be negative.")

        return mx.split(x, cumsum_splits, axis=axis)

    return split


@mlx_funcify.register(ExtractDiag)
def mlx_funcify_ExtractDiag(op, **kwargs):
    offset, axis1, axis2 = op.offset, op.axis1, op.axis2

    def extract_diag(x, offset=offset, axis1=axis1, axis2=axis2):
        return mx.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)

    return extract_diag


@mlx_funcify.register(Eye)
def mlx_funcify_Eye(op, node, **kwargs):
    # Extract constants for performance optimization
    const_args = [getattr(inp, "data", None) for inp in node.inputs]
    dtype = convert_dtype_to_mlx(op.dtype)

    def eye(*args):
        # Replace args with compile-time constants when available for better performance
        args = [
            arg if const_a is None else const_a
            for arg, const_a in zip(args, const_args, strict=True)
        ]
        N, M, k = args
        return mx.eye(int(N), int(M), int(k), dtype=dtype)

    return eye


@mlx_funcify.register(MakeVector)
def mlx_funcify_MakeVector(op, **kwargs):
    dtype = convert_dtype_to_mlx(op.dtype)

    def makevector(*x):
        return mx.array(x, dtype=dtype)

    return makevector


@mlx_funcify.register(TensorFromScalar)
def mlx_funcify_TensorFromScalar(op, **kwargs):
    def tensor_from_scalar(x):
        return x  # already an MLX array / scalar

    return tensor_from_scalar


@mlx_funcify.register(ScalarFromTensor)
def mlx_funcify_ScalarFromTensor(op, **kwargs):
    def scalar_from_tensor(x):
        "We can't not return a scalar in MLX without trigger evaluation"
        return x

    return scalar_from_tensor


@mlx_funcify.register(AllocEmpty)
def mlx_funcify_AllocEmpty(op, node, **kwargs):
    dtype = convert_dtype_to_mlx(op.dtype)
    node_inputs = node.inputs
    static_dims = (
        _extract_static_dims(node_inputs)
        if node_inputs and len(node_inputs) > 1
        else None
    )

    def allocempty(*shape):
        resolved_shape = (
            _resolve_shape(static_dims, shape)
            if static_dims is not None
            else tuple(_coerce_to_int(dim) for dim in shape)
        )
        return mx.zeros(resolved_shape, dtype=dtype)

    return allocempty


@mlx_funcify.register(Alloc)
def mlx_funcify_Alloc(op, node, **kwargs):
    node_inputs = node.inputs
    static_dims = (
        _extract_static_dims(node_inputs[1:])
        if node_inputs and len(node_inputs) > 1
        else None
    )

    def alloc(x, *shape):
        resolved_shape = (
            _resolve_shape(static_dims, shape)
            if static_dims is not None
            else tuple(_coerce_to_int(dim) for dim in shape)
        )
        result = mx.broadcast_to(x, resolved_shape)
        if node_inputs is not None:
            value_for_check = x if hasattr(x, "shape") else np.asarray(x)
            Alloc._check_runtime_broadcast(node, value_for_check, resolved_shape)
        return result

    return alloc


def _extract_static_dims(shape_inputs):
    static_dims = []
    for dim in shape_inputs:
        try:
            static_dims.append(int(get_scalar_constant_value(dim)))
        except NotScalarConstantError:
            static_dims.append(None)
    return tuple(static_dims)


def _resolve_shape(static_dims, runtime_shape):
    if len(static_dims) != len(runtime_shape):
        raise ValueError("Alloc received unexpected number of shape dimensions")

    resolved = []
    for const_dim, dim in zip(static_dims, runtime_shape, strict=True):
        resolved.append(const_dim if const_dim is not None else _coerce_to_int(dim))

    return tuple(resolved)


def _coerce_to_int(value):
    if isinstance(value, np.integer | int):
        return int(value)
    try:
        if hasattr(value, "item"):
            return int(value.item())
        return int(value)
    except (ValueError, TypeError) as exc:
        _rethrow_dynamic_shape_error(exc)
        raise


def _rethrow_dynamic_shape_error(exc):
    msg = str(exc)
    if "[eval] Attempting to eval an array during function transformations" in msg:
        raise ValueError(f"{MLX_DYNAMIC_SHAPE_ERROR}\n\nOriginal error: {msg}") from exc
