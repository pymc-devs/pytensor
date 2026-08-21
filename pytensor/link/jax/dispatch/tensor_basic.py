import warnings

import jax.numpy as jnp
import numpy as np

from pytensor.graph.basic import Constant
from pytensor.link.jax.dispatch.basic import jax_funcify
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
from pytensor.tensor.shape import Shape, Shape_i
from pytensor.tensor.subtensor import Subtensor


ARANGE_CONCRETE_VALUE_ERROR = """JAX requires the arguments of `jax.numpy.arange` to be constants.
The graph that you defined thus cannot be JIT-compiled by JAX.
An example of a graph that can be compiled to JAX:
>>> import pytensor.tensor as pt
>>> pt.arange(1, 10, 2)
"""


@jax_funcify.register(AllocEmpty)
def jax_funcify_AllocEmpty(op, **kwargs):
    def allocempty(*shape):
        return jnp.empty(shape, dtype=op.dtype)

    return allocempty


@jax_funcify.register(Alloc)
def jax_funcify_Alloc(op, node, **kwargs):
    static_shapes = []
    for shape_input in node.inputs[1:]:
        try:
            static_shapes.append(int(get_scalar_constant_value(shape_input)))
        except NotScalarConstantError:
            concrete_shape = None
            break
    else:
        concrete_shape = tuple(static_shapes)

    def alloc(x, *shape):
        res = jnp.broadcast_to(
            x, concrete_shape if concrete_shape is not None else shape
        )
        Alloc._check_runtime_broadcast(node, jnp.asarray(x), res.shape)
        return res

    return alloc


@jax_funcify.register(ARange)
def jax_funcify_ARange(op, node, **kwargs):
    """Register a JAX implementation for `ARange`.

    `jax.numpy.arange` requires concrete values for its arguments. Here we check
    that the arguments are constant, and raise otherwise.

    TODO: Handle other situations in which values are concrete (shape of an array).

    """
    arange_args = node.inputs
    constant_args = []
    for arg in arange_args:
        # Under JAX tracing an array's shape is concrete, so any element of it is a
        # valid ``arange`` bound
        match arg.owner_op_and_inputs:
            case (Shape_i(), *_):
                constant_args.append(None)
            case (Subtensor(), shape_var, *_) if isinstance(shape_var.owner_op, Shape):
                constant_args.append(None)
            case _ if isinstance(arg, Constant):
                # Cast to the Op's dtype: PyTensor types integer literals (e.g. 0/1 arange
                # start/step) as int8, and jnp.arange bounds-checks stop against the argument dtype,
                # overflowing for stop > 127.
                constant_args.append(np.asarray(arg.value, op.dtype))
            case _:
                # TODO: This might be failing without need (e.g., if arg = shape(x)[-1] + 1)!
                raise NotImplementedError(ARANGE_CONCRETE_VALUE_ERROR)

    constant_start, constant_stop, constant_step = constant_args

    def arange(start, stop, step):
        start = start if constant_start is None else constant_start
        stop = stop if constant_stop is None else constant_stop
        step = step if constant_step is None else constant_step
        return jnp.arange(start, stop, step, dtype=op.dtype)

    return arange


@jax_funcify.register(Join)
def jax_funcify_Join(op, **kwargs):
    axis = op.axis

    def join(*tensors):
        # tensors could also be tuples, and in this case they don't have a ndim
        tensors = [jnp.asarray(tensor) for tensor in tensors]
        return jnp.concatenate(tensors, axis=axis)

    return join


@jax_funcify.register(Split)
def jax_funcify_Split(op: Split, node, **kwargs):
    _x, splits = node.inputs
    axis = op.axis

    try:
        constant_splits = np.array(
            [
                get_scalar_constant_value(splits[i])
                for i in range(get_vector_length(splits))
            ]
        )
    except (ValueError, NotScalarConstantError):
        constant_splits = None
        warnings.warn(
            "Split node does not have constant split positions. Jax implementation will likely fail"
        )

    def split(x, splits):
        if len(splits) != op.len_splits:
            raise ValueError("Length of splits is not equal to n_splits")

        if constant_splits is not None:
            splits = constant_splits
            cumsum_splits = np.cumsum(splits[:-1])
            if (splits < 0).any():
                raise ValueError("Split sizes cannot be negative")
            if splits.sum() != x.shape[axis]:
                raise ValueError(
                    f"Split sizes do not sum up to input length along axis: {x.shape[axis]}"
                )
        else:
            cumsum_splits = jnp.cumsum(splits[:-1])

        return jnp.split(x, cumsum_splits, axis=axis)

    return split


@jax_funcify.register(ExtractDiag)
def jax_funcify_ExtractDiag(op, **kwargs):
    offset = op.offset
    axis1 = op.axis1
    axis2 = op.axis2

    def extract_diag(x, offset=offset, axis1=axis1, axis2=axis2):
        return jnp.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)

    return extract_diag


@jax_funcify.register(Eye)
def jax_funcify_Eye(op, **kwargs):
    dtype = op.dtype

    def eye(N, M, k):
        return jnp.eye(N, M, k, dtype=dtype)

    return eye


@jax_funcify.register(MakeVector)
def jax_funcify_MakeVector(op, **kwargs):
    def makevector(*x):
        return jnp.array(x, dtype=op.dtype)

    return makevector


@jax_funcify.register(TensorFromScalar)
def jax_funcify_TensorFromScalar(op, **kwargs):
    def tensor_from_scalar(x):
        return x

    return tensor_from_scalar


@jax_funcify.register(ScalarFromTensor)
def jax_funcify_ScalarFromTensor(op, **kwargs):
    def scalar_from_tensor(x):
        return jnp.array(x).flatten()[0]

    return scalar_from_tensor
