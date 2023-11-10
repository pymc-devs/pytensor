import warnings

import torch
import numpy as np

from pytensor.graph.basic import Constant
from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
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
    Tri,
    get_underlying_scalar_constant_value,
)
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.shape import Shape_i


ARANGE_CONCRETE_VALUE_ERROR = """PyTorch requires the arguments of `torch.arange`
to be constants. The graph that you defined thus cannot be JIT-compiled
by PyTorch. An example of a graph that can be compiled to PyTorch:
>>> import pytensor.tensor basic
>>> at.arange(1, 10, 2)
"""


@pytorch_funcify.register(AllocEmpty)
def pytorch_funcify_AllocEmpty(op, **kwargs):
    def allocempty(*shape):
        return torch.empty(shape, dtype=op.dtype)

    return allocempty


@pytorch_funcify.register(Alloc)
def pytorch_funcify_Alloc(op, node, **kwargs):
    def alloc(x, *shape):
        res = torch.broadcast_to(x, shape)
        Alloc._check_runtime_broadcast(node, torch.as_tensor(x), res.shape)
        return res

    return alloc


@pytorch_funcify.register(ARange)
def pytorch_funcify_ARange(op, node, **kwargs):
    """Register a PyTorch implementation for `ARange`.

    `torch.arange` requires concrete values for its arguments. Here we check
    that the arguments are constant, and raise otherwise.

    TODO: Handle other situations in which values are concrete (shape of an array).

    """
    arange_args = node.inputs
    constant_args = []
    for arg in arange_args:
        if arg.owner and isinstance(arg.owner.op, Shape_i):
            constant_args.append(None)
        elif isinstance(arg, Constant):
            constant_args.append(arg.value)
        else:
            # TODO: This might be failing without need (e.g., if arg = shape(x)[-1] + 1)!
            raise NotImplementedError(ARANGE_CONCRETE_VALUE_ERROR)

    constant_start, constant_stop, constant_step = constant_args

    def arange(start, stop, step):
        start = start if constant_start is None else constant_start
        stop = stop if constant_stop is None else constant_stop
        step = step if constant_step is None else constant_step
        return torch.arange(start, stop, step, dtype=op.dtype)

    return arange


@pytorch_funcify.register(Join)
def pytorch_funcify_Join(op, **kwargs):
    def join(axis, *tensors):
        # tensors could also be tuples, and in this case they don't have a ndim
        tensors = [torch.as_tensor(tensor) for tensor in tensors]
        view = op.view
        if (view != -1) and all(
            tensor.shape[axis] == 0 for tensor in tensors[0:view] + tensors[view + 1 :]
        ):
            return tensors[view]

        else:
            return torch.cat(tensors, dim=axis)

    return join


@pytorch_funcify.register(Split)
def pytorch_funcify_Split(op: Split, node, **kwargs):
    _, axis, splits = node.inputs
    try:
        constant_axis = get_underlying_scalar_constant_value(axis)
    except NotScalarConstantError:
        constant_axis = None
        warnings.warn(
            "Split node does not have constant axis. PyTorch implementation will likely fail"
        )

    try:
        constant_splits = np.array(
            [
                get_underlying_scalar_constant_value(splits[i])
                for i in range(get_vector_length(splits))
            ]
        )
    except (ValueError, NotScalarConstantError):
        constant_splits = None
        warnings.warn(
            "Split node does not have constant split positions. PyTorch implementation will likely fail"
        )

    def split(x, axis, splits):
        if constant_axis is not None:
            axis = constant_axis
        if constant_splits is not None:
            splits = constant_splits
            cumsum_splits = np.cumsum(splits[:-1])
        else:
            cumsum_splits = torch.cumsum(splits[:-1])

        if len(splits) != op.len_splits:
            raise ValueError("Length of splits is not equal to n_splits")
        if np.sum(splits) != x.shape[axis]:
            raise ValueError(
                f"Split sizes do not sum up to input length along axis: {x.shape[axis]}"
            )
        if np.any(splits < 0):
            raise ValueError("Split sizes cannot be negative")

        return torch.split(x, cumsum_splits, axis=axis)

    return split


@pytorch_funcify.register(ExtractDiag)
def pytorch_funcify_ExtractDiag(op, **kwargs):
    offset = op.offset
    axis1 = op.axis1
    axis2 = op.axis2

    def extract_diag(x, offset=offset, axis1=axis1, axis2=axis2):
        return torch.diagonal(x, offset=offset, dim1=axis1, dim2=axis2)

    return extract_diag


@pytorch_funcify.register(Eye)
def pytorch_funcify_Eye(op, **kwargs):
    dtype = op.dtype

    def eye(N, M, k):
        return torch.eye(N, M, k, dtype=dtype)

    return eye


@pytorch_funcify.register(MakeVector)
def pytorch_funcify_MakeVector(op, **kwargs):
    def makevector(*x):
        return torch.tensor(x, dtype=op.dtype)

    return makevector


@pytorch_funcify.register(TensorFromScalar)
def pytorch_funcify_TensorFromScalar(op, **kwargs):
    def tensor_from_scalar(x):
        return x

    return tensor_from_scalar


@pytorch_funcify.register(ScalarFromTensor)
def pytorch_funcify_ScalarFromTensor(op, **kwargs):
    def scalar_from_tensor(x):
        return torch.tensor(x).flatten()[0]

    return scalar_from_tensor


@pytorch_funcify.register(Tri)
def pytorch_funcify_Tri(op, node, **kwargs):
    # node.inputs is N, M, k
    const_args = [getattr(x, "data", None) for x in node.inputs]

    def tri(*args):
        # args is N, M, k
        args = [
            x if const_x is None else const_x for x, const_x in zip(args, const_args)
        ]
        return torch.tri(*args, dtype=op.dtype)

    return tri