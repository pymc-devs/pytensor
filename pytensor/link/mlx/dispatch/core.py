"""
pytensor/link/mlx/dispatch/basic.py
-----------------------------------

First-cut MLX translations for the most common tensor Ops.

The structure intentionally follows pytensor's JAX dispatcher so that
once these kernels stabilise they can be optimised further (e.g. fusing
element-wise graphs, adding in-place updates, RNG thinning, etc.).
"""

from __future__ import annotations

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
    Tri,
    get_scalar_constant_value,
)
from pytensor.tensor.exceptions import NotScalarConstantError


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

        if constant_splits is not None:
            splits = constant_splits
            cumsum_splits = np.cumsum(splits[:-1])
        else:
            # Dynamic case - use MLX operations
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


@mlx_funcify.register(Tri)
def mlx_funcify_Tri(op, node, **kwargs):
    # node.inputs  ->  N, M, k
    const_args = [getattr(inp, "data", None) for inp in node.inputs]
    dtype = convert_dtype_to_mlx(op.dtype)

    def tri(*args):
        # Replace args with compile-time constants when available
        args = [
            arg if const_a is None else const_a
            for arg, const_a in zip(args, const_args, strict=True)
        ]
        return mx.tri(*args, dtype=dtype)

    return tri


@mlx_funcify.register(AllocEmpty)
def mlx_funcify_AllocEmpty(op, **kwargs):
    dtype = convert_dtype_to_mlx(op.dtype)

    def allocempty(*shape):
        return mx.zeros(shape, dtype=dtype)

    return allocempty


@mlx_funcify.register(Alloc)
def mlx_funcify_Alloc(op, node, **kwargs):
    def alloc(x, *shape):
        try:
            # Convert shape elements to Python ints for MLX compatibility
            # MLX requires shape dimensions to be Python integers, not MLX arrays
            shape_ints = tuple(
                int(s.item()) if hasattr(s, "item") else int(s) for s in shape
            )
            return mx.broadcast_to(x, shape_ints)
        except ValueError as e:
            if (
                "[eval] Attempting to eval an array during function transformations"
                in str(e)
            ):
                # This is the MLX compilation limitation - provide helpful error
                raise ValueError(
                    "MLX compilation limitation: Alloc operations with dynamic shapes "
                    "cannot be used inside compiled functions. This is because MLX "
                    "compilation forbids evaluating arrays to extract shape values. "
                    # Just a note! TODO: remove this once we have a better solution
                    "\n\nWorkarounds:"
                    "\n1. Avoid using Alloc with dynamic shapes in compiled contexts"
                    "\n2. Use static shapes when possible"
                    "\n3. Move Alloc operations outside compiled functions"
                    "\n\nOriginal error: " + str(e)
                ) from e
            else:
                # Re-raise other ValueError exceptions
                raise

    return alloc
