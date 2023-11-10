import functools
import typing
from typing import Callable, Optional

import torch

from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.scalar import Softplus
from pytensor.scalar.basic import (
    Add,
    Cast,
    Clip,
    Composite,
    Identity,
    IntDiv,
    Mod,
    Mul,
    ScalarOp,
    Second,
    Sub,
)
from pytensor.scalar.math import Erf, Erfc, Erfcinv, Erfcx, Erfinv, Iv, Log1mexp, Psi


def try_import_torch_op(op: ScalarOp, torch_op_name: Optional[str] = None) -> Callable:
    try:
        import torch
    except ModuleNotFoundError:
        raise NotImplementedError(
            f"No PyTorch implementation for Op {op.name}. "
            "Implementation is available if PyTorch is installed"
        )

    if torch_op_name is None:
        torch_op_name = op.name
    return typing.cast(Callable, getattr(torch, torch_op_name))


def check_if_inputs_scalars(node):
    """Check whether all the inputs of an `Elemwise` are scalar values.

    `torch` functions systematically return `Tensors`,
    while the corresponding Python operators return concrete values when passed
    concrete values. In order to be able to compile the largest number of graphs
    possible we need to preserve concrete values whenever we can. We thus need
    to dispatch differently the PyTensor operators depending on whether the inputs
    are scalars.

    """
    ndims_input = [inp.type.ndim for inp in node.inputs]
    are_inputs_scalars = True
    for ndim in ndims_input:
        try:
            if ndim > 0:
                are_inputs_scalars = False
        except TypeError:
            are_inputs_scalars = False

    return are_inputs_scalars


@pytorch_funcify.register(ScalarOp)
def pytorch_funcify_ScalarOp(op, node, **kwargs):
    func_name = op.nfunc_spec[0]

    # We dispatch some PyTensor operators to Python operators
    # whenever the inputs are all scalars.
    are_inputs_scalars = check_if_inputs_scalars(node)
    if are_inputs_scalars:
        elemwise = elemwise_scalar(op)
        if elemwise is not None:
            return elemwise

    if "." in func_name:
        torch_func = functools.reduce(getattr, [torch] + func_name.split("."))
    else:
        torch_func = getattr(torch, func_name)

    if hasattr(op, "nfunc_variadic"):
        # These are special cases that handle invalid arities due to the broken
        # PyTensor `Op` type contract (e.g. binary `Op`s that also function as
        # their own variadic counterparts--even when those counterparts already
        # exist as independent `Op`s).
        torch_variadic_func = getattr(torch, op.nfunc_variadic)

        def elemwise(*args):
            if len(args) > op.nfunc_spec[1]:
                return torch_variadic_func(
                    torch.stack(torch.broadcast_tensors(*args), dim=0), dim=0
                )
            else:
                return torch_func(*args)

        return elemwise
    else:
        return torch_func


@functools.singledispatch
def elemwise_scalar(op):
    return None


@elemwise_scalar.register(Add)
def elemwise_scalar_add(op):
    def elemwise(*inputs):
        return sum(inputs)

    return elemwise


@elemwise_scalar.register(Mul)
def elemwise_scalar_mul(op):
    import operator
    from functools import reduce

    def elemwise(*inputs):
        return reduce(operator.mul, inputs, 1)

    return elemwise


@elemwise_scalar.register(Sub)
def elemwise_scalar_sub(op):
    def elemwise(x, y):
        return x - y

    return elemwise


@elemwise_scalar.register(IntDiv)
def elemwise_scalar_intdiv(op):
    def elemwise(x, y):
        return x // y

    return elemwise


@elemwise_scalar.register(Mod)
def elemwise_scalar_mod(op):
    def elemwise(x, y):
        return x % y

    return elemwise


@pytorch_funcify.register(Cast)
def pytorch_funcify_Cast(op, **kwargs):
    def cast(x):
        return torch.tensor(x).type(op.o_type.dtype)

    return cast


@pytorch_funcify.register(Identity)
def pytorch_funcify_Identity(op, **kwargs):
    def identity(x):
        return x

    return identity


@pytorch_funcify.register(Clip)
def pytorch_funcify_Clip(op, **kwargs):
    """Register the translation for the `Clip` `Op`.

    PyTensor's `Clip` operator operates differently from PyTorch's when the
    specified `min` is larger than the `max` so we cannot reuse `torch.clip`
    to maintain consistency with PyTensor.

    """

    def clip(x, min, max):
        return torch.where(x < min, min, torch.where(x > max, max, x))

    return clip


@pytorch_funcify.register(Composite)
def pytorch_funcify_Composite(op, node, vectorize=True, **kwargs):
    pytorch_impl = pytorch_funcify(op.fgraph)

    if len(node.outputs) == 1:

        def composite(*args):
            return pytorch_impl(*args)[0]

    else:

        def composite(*args):
            return pytorch_impl(*args)

    return torch.vectorize(composite)


@pytorch_funcify.register(Second)
def pytorch_funcify_Second(op, **kwargs):
    def second(x, y):
        _, y = torch.broadcast_tensors(x, y)
        return y

    return second


@pytorch_funcify.register(Erf)
def pytorch_funcify_Erf(op, node, **kwargs):
    def erf(x):
        return torch.erf(x)

    return erf


@pytorch_funcify.register(Erfc)
def pytorch_funcify_Erfc(op, **kwargs):
    def erfc(x):
        return torch.erfc(x)

    return erfc


@pytorch_funcify.register(Erfinv)
def pytorch_funcify_Erfinv(op, **kwargs):
    def erfinv(x):
        return torch.erfinv(x)

    return erfinv


@pytorch_funcify.register(Erfcx)
@pytorch_funcify.register(Erfcinv)
def pytorch_funcify_from_tfp(op, **kwargs):
    torch_op = try_import_torch_op(op)

    return torch_op


@pytorch_funcify.register(Iv)
def pytorch_funcify_Iv(op, **kwargs):
    ive = try_import_torch_op(op, torch_op_name="bessel_ive")

    def iv(v, x):
        return ive(v, x) / torch.exp(-torch.abs(torch.real(x)))

    return iv


@pytorch_funcify.register(Log1mexp)
def pytorch_funcify_Log1mexp(op, node, **kwargs):
    def log1mexp(x):
        return torch.where(
            x < torch.log(0.5), torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x))
        )

    return log1mexp


@pytorch_funcify.register(Psi)
def pytorch_funcify_Psi(op, node, **kwargs):
    def psi(x):
        return torch.digamma(x)

    return psi


@pytorch_funcify.register(Softplus)
def pytorch_funcify_Softplus(op, **kwargs):
    def softplus(x):
        return torch.where(
            x < -37.0,
            torch.exp(x),
            torch.where(
                x < 18.0,
                torch.log1p(torch.exp(x)),
                torch.where(
                    x < 33.3,
                    x + torch.exp(-x),
                    x,
                ),
            ),
        )

    return softplus