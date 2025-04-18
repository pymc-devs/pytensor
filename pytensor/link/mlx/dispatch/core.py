"""
pytensor/link/mlx/dispatch/basic.py
-----------------------------------

First-cut MLX translations for the most common tensor Ops.

The structure intentionally follows pytensor's JAX dispatcher so that
once these kernels stabilise they can be optimised further (e.g. fusing
element-wise graphs, adding in-place updates, RNG thinning, etc.).
"""

from __future__ import annotations

import warnings

import mlx.core as mx  # MLX
import numpy as np

from pytensor.link.mlx.dispatch.basic import mlx_funcify  # MLX
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


# ------------------------------------------------------------------
# Join
# ------------------------------------------------------------------
@mlx_funcify.register(Join)  # MLX
def mlx_funcify_Join(op, **kwargs):
    def join(axis, *tensors):
        view = op.view
        if (view != -1) and all(
            tensors[i].shape[axis] == 0  # MLX
            for i in list(range(view)) + list(range(view + 1, len(tensors)))
        ):
            return tensors[view]

        return mx.concatenate(tensors, axis=axis)  # MLX

    return join


# ------------------------------------------------------------------
# Split
# ------------------------------------------------------------------
@mlx_funcify.register(Split)  # MLX
def mlx_funcify_Split(op: Split, node, **kwargs):
    _, axis_sym, splits_sym = node.inputs

    try:
        constant_axis = get_scalar_constant_value(axis_sym)
    except NotScalarConstantError:
        constant_axis = None
        warnings.warn(
            "Split node does not have a constant axis. MLX implementation may fail."
        )

    try:
        constant_splits = np.array(
            [
                get_scalar_constant_value(splits_sym[i])
                for i in range(get_vector_length(splits_sym))
            ]
        )
    except (ValueError, NotScalarConstantError):
        constant_splits = None
        warnings.warn(
            "Split node does not have constant split positions. MLX implementation may fail."
        )

    def split(x, axis, splits):
        # Resolve constants (avoids tracing extra ops)
        if constant_axis is not None:
            axis = int(constant_axis)

        if constant_splits is not None:
            splits = constant_splits
            cumsum_splits = np.cumsum(splits[:-1])
        else:
            # dynamic - keep in graph
            splits_arr = mx.array(splits)  # MLX
            cumsum_splits = mx.cumsum(
                splits_arr[:-1]
            ).tolist()  # python list for mx.split

        if len(splits) != op.len_splits:
            raise ValueError("Length of 'splits' is not equal to n_splits")
        if np.sum(np.asarray(splits)) != x.shape[axis]:
            raise ValueError(
                "Split sizes do not sum to the input length on the chosen axis."
            )
        if np.any(np.asarray(splits) < 0):
            raise ValueError("Split sizes cannot be negative.")

        return mx.split(x, cumsum_splits, axis=axis)  # MLX

    return split


# ------------------------------------------------------------------
# ExtractDiag
# ------------------------------------------------------------------
@mlx_funcify.register(ExtractDiag)  # MLX
def mlx_funcify_ExtractDiag(op, **kwargs):
    offset, axis1, axis2 = op.offset, op.axis1, op.axis2

    def extract_diag(x, offset=offset, axis1=axis1, axis2=axis2):
        return mx.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)  # MLX

    return extract_diag


# ------------------------------------------------------------------
# Eye
# ------------------------------------------------------------------
@mlx_funcify.register(Eye)  # MLX
def mlx_funcify_Eye(op, **kwargs):
    dtype = op.dtype

    def eye(N, M, k):
        return mx.eye(int(N), int(M), int(k), dtype=dtype)  # MLX

    return eye


# ------------------------------------------------------------------
# MakeVector
# ------------------------------------------------------------------
@mlx_funcify.register(MakeVector)  # MLX
def mlx_funcify_MakeVector(op, **kwargs):
    def makevector(*x):
        return mx.array(x, dtype=op.dtype)  # MLX

    return makevector


# ------------------------------------------------------------------
# TensorFromScalar  (identity for MLX)
# ------------------------------------------------------------------
@mlx_funcify.register(TensorFromScalar)  # MLX
def mlx_funcify_TensorFromScalar(op, **kwargs):
    def tensor_from_scalar(x):
        return x  # already an MLX array / scalar

    return tensor_from_scalar


# ------------------------------------------------------------------
# ScalarFromTensor
# ------------------------------------------------------------------
@mlx_funcify.register(ScalarFromTensor)  # MLX
def mlx_funcify_ScalarFromTensor(op, **kwargs):
    def scalar_from_tensor(x):
        return mx.array(x).reshape(-1)[0]  # MLX

    return scalar_from_tensor


# ------------------------------------------------------------------
# Tri
# ------------------------------------------------------------------
@mlx_funcify.register(Tri)  # MLX
def mlx_funcify_Tri(op, node, **kwargs):
    # node.inputs  ->  N, M, k
    const_args = [getattr(inp, "data", None) for inp in node.inputs]

    def tri(*args):
        # Replace args with compile-time constants when available
        args = [
            arg if const_a is None else const_a
            for arg, const_a in zip(args, const_args, strict=True)
        ]
        return mx.tri(*args, dtype=op.dtype)  # MLX

    return tri


@mlx_funcify.register(AllocEmpty)
def mlx_funcify_AllocEmpty(op, **kwargs):
    def allocempty(*shape):
        return mx.zeros(shape, dtype=op.dtype)

    return allocempty


@mlx_funcify.register(Alloc)
def mlx_funcify_Alloc(op, node, **kwargs):
    def alloc(x, *shape):
        res = mx.broadcast_to(x, shape)
        Alloc._check_runtime_broadcast(node, mx.array(x), res.shape)
        return res

    return alloc
