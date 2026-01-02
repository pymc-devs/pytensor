import mlx.core as mx
import numpy as np

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor import get_vector_length
from pytensor.tensor.basic import (
    Alloc,
    AllocEmpty,
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


def convert_dtype_to_mlx(dtype_str, auto_cast_unsupported=True):
    """Convert PyTensor dtype strings to MLX dtype objects.

    MLX expects dtype objects rather than string literals for type conversion.
    This function maps common dtype strings to their MLX equivalents.

    Parameters
    ----------
    dtype_str : str or MLX dtype
        The dtype to convert
    auto_cast_unsupported : bool
        If True, automatically cast unsupported dtypes to supported ones with warnings

    Returns
    -------
    MLX dtype object
    """
    import warnings

    if isinstance(dtype_str, str):
        if dtype_str == "bool":
            return mx.bool_
        elif dtype_str == "int8":
            return mx.int8
        elif dtype_str == "int16":
            return mx.int16
        elif dtype_str == "int32":
            return mx.int32
        elif dtype_str == "int64":
            return mx.int64
        elif dtype_str == "uint8":
            return mx.uint8
        elif dtype_str == "uint16":
            return mx.uint16
        elif dtype_str == "uint32":
            return mx.uint32
        elif dtype_str == "uint64":
            return mx.uint64
        elif dtype_str == "float16":
            return mx.float16
        elif dtype_str == "float32":
            return mx.float32
        elif dtype_str == "float64":
            if auto_cast_unsupported:
                warnings.warn(
                    "MLX does not support float64 on GPU. Automatically casting to float32. "
                    "This may result in reduced precision. To avoid this warning, "
                    "explicitly use float32 in your code or set floatX='float32' in PyTensor config.",
                    UserWarning,
                    stacklevel=3,
                )
                return mx.float32
            else:
                return mx.float64
        elif dtype_str == "bfloat16":
            return mx.bfloat16
        elif dtype_str == "complex64":
            return mx.complex64
        elif dtype_str == "complex128":
            if auto_cast_unsupported:
                warnings.warn(
                    "MLX does not support complex128. Automatically casting to complex64. "
                    "This may result in reduced precision. To avoid this warning, "
                    "explicitly use complex64 in your code.",
                    UserWarning,
                    stacklevel=3,
                )
                return mx.complex64
            else:
                # Return the original even though it might fail
                # This allows users to opt out of auto-casting if needed
                return mx.complex64  # MLX doesn't have complex128, so fallback
    # Return as is if it's already an MLX dtype or not a recognized string
    return dtype_str


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
    raise TypeError(
        "MLX Alloc expects integer shape components; got value of type "
        f"{type(value).__name__}."
    )


def _rethrow_dynamic_shape_error(exc):
    msg = str(exc)
    if "[eval] Attempting to eval an array during function transformations" in msg:
        raise ValueError(f"{MLX_DYNAMIC_SHAPE_ERROR}\n\nOriginal error: {msg}") from exc
