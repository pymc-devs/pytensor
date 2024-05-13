import functools
import typing
from collections.abc import Callable

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
from pytensor.scalar.math import (
    BetaIncInv,
    Erf,
    Erfc,
    Erfcinv,
    Erfcx,
    Erfinv,
    GammaIncCInv,
    GammaIncInv,
    Iv,
    Ive,
    Log1mexp,
    Psi,
    TriGamma,
)


def try_import_tfp_jax_op(op: ScalarOp, jax_op_name: str | None = None) -> Callable:
    try:
        import tensorflow_probability.substrates.jax.math as tfp_jax_math
    except ModuleNotFoundError:
        raise NotImplementedError(
            f"No JAX implementation for Op {op.name}. "
            "Implementation is available if TensorFlow Probability is installed"
        )

    if jax_op_name is None:
        jax_op_name = op.name
    return typing.cast(Callable, getattr(tfp_jax_math, jax_op_name))


def all_inputs_are_scalar(node):
    """Check whether all the inputs of an `Elemwise` are scalar values.

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
    """Return pytorch function that implements the same computation as the Scalar Op.

    This dispatch is expected to return a pytorch function that works on Array inputs as Elemwise does,
    even though it's dispatched on the Scalar Op.
    """

    # We dispatch some PyTensor operators to Python operators
    # whenever the inputs are all scalars.
    if all_inputs_are_scalar(node):
        pytorch_func = pytorch_funcify_scalar_op_via_py_operators(op)
        if pytorch_func is not None:
            return pytorch_func

    nfunc_spec = getattr(op, "nfunc_spec", None)
    if nfunc_spec is None:
        raise NotImplementedError(f"Dispatch not implemented for Scalar Op {op}")

    func_name = nfunc_spec[0]
    print
    # if "." in func_name:
    #     pytorch_func = functools.reduce(getattr, [jax, *func_name.split(".")])
    # else:
    pytorch_func = getattr(torch, func_name)

    if len(node.inputs) > op.nfunc_spec[1]:
        # Some Scalar Ops accept multiple number of inputs, behaving as a variadic function,
        # even though the base Op from `func_name` is specified as a binary Op.
        # This happens with `Add`, which can work as a `Sum` for multiple scalars.
        pytorch_variadic_func = getattr(torch, op.nfunc_variadic, None)
        if not pytorch_variadic_func:
            raise NotImplementedError(
                f"Dispatch not implemented for Scalar Op {op} with {len(node.inputs)} inputs"
            )

        def pytorch_func(*args):
            return pytorch_variadic_func(
                torch.stack(torch.broadcast_tensors(*args), axis=0), axis=0
            )

    return pytorch_func


@functools.singledispatch
def pytorch_funcify_scalar_op_via_py_operators(op):
    """Specialized JAX dispatch for Elemwise operations where all inputs are Scalar arrays.

    Scalar (constant) arrays in the JAX backend get lowered to the native types (int, floats),
    which can perform better with Python operators, and more importantly, avoid upcasting to array types
    not supported by some JAX functions.
    """
    return None


@pytorch_funcify_scalar_op_via_py_operators.register(Add)
def pytorch_funcify_scalar_Add(op):
    def elemwise(*inputs):
        return sum(inputs)

    return elemwise


@pytorch_funcify_scalar_op_via_py_operators.register(Mul)
def pytorch_funcify_scalar_Mul(op):
    import operator
    from functools import reduce

    def elemwise(*inputs):
        return reduce(operator.mul, inputs, 1)

    return elemwise


@pytorch_funcify_scalar_op_via_py_operators.register(Sub)
def pytorch_funcify_scalar_Sub(op):
    def elemwise(x, y):
        return x - y

    return elemwise


@pytorch_funcify_scalar_op_via_py_operators.register(IntDiv)
def pytorch_funcify_scalar_IntDiv(op):
    def elemwise(x, y):
        return x // y

    return elemwise


@pytorch_funcify_scalar_op_via_py_operators.register(Mod)
def pytorch_funcify_scalar_Mod(op):
    def elemwise(x, y):
        return x % y

    return elemwise


@pytorch_funcify.register(Cast)
def pytorch_funcify_Cast(op, **kwargs):
    def cast(x):
        return torch.tensor(x).astype(op.o_type.dtype)

    return cast


@pytorch_funcify.register(Identity)
def pytorch_funcify_Identity(op, **kwargs):
    def identity(x):
        return x

    return identity


@pytorch_funcify.register(Clip)
def pytorch_funcify_Clip(op, **kwargs):
    """Register the translation for the `Clip` `Op`.

    """

    def clip(x, min, max):
        return torch.where(x < min, min, torch.where(x > max, max, x))

    return clip


# @pytorch_funcify.register(Composite)
# def pytorch_funcify_Composite(op, node, vectorize=True, **kwargs):
#     jax_impl = pytorch_funcify(op.fgraph)

#     if len(node.outputs) == 1:

#         def composite(*args):
#             return jax_impl(*args)[0]

#     else:

#         def composite(*args):
#             return jax_impl(*args)

#     return jnp.vectorize(composite)


@pytorch_funcify.register(Second)
def pytorch_funcify_Second(op, **kwargs):
    def second(x, y):
        _, y = torch.broadcast_tensors(x, y)
        return y

    return second


@pytorch_funcify.register(GammaIncInv)
def pytorch_funcify_GammaIncInv(op, **kwargs):
    gammaincinv = try_import_tfp_jax_op(op, jax_op_name="igammainv")

    return gammaincinv


@pytorch_funcify.register(GammaIncCInv)
def pytorch_funcify_GammaIncCInv(op, **kwargs):
    gammainccinv = try_import_tfp_jax_op(op, jax_op_name="igammacinv")

    return gammainccinv


@pytorch_funcify.register(Erf)
def pytorch_funcify_Erf(op, node, **kwargs):
    def erf(x):
        return torch.special.erf(x)

    return erf


@pytorch_funcify.register(Erfc)
def pytorch_funcify_Erfc(op, **kwargs):
    def erfc(x):
        return torch.special.erfc(x)

    return erfc


@pytorch_funcify.register(Erfinv)
def pytorch_funcify_Erfinv(op, **kwargs):
    def erfinv(x):
        return torch.special.erfinv(x)

    return erfinv


@pytorch_funcify.register(BetaIncInv)
@pytorch_funcify.register(Erfcx)
@pytorch_funcify.register(Erfcinv)
def pytorch_funcify_from_tfp(op, **kwargs):
    tfp_jax_op = try_import_tfp_jax_op(op)

    return tfp_jax_op


@pytorch_funcify.register(Iv)
def pytorch_funcify_Iv(op, **kwargs):
    ive = try_import_tfp_jax_op(op, jax_op_name="bessel_ive")

    def iv(v, x):
        return ive(v, x) / torch.exp(-torch.abs(torch.real(x)))

    return iv


@pytorch_funcify.register(Ive)
def pytorch_funcify_Ive(op, **kwargs):
    ive = try_import_tfp_jax_op(op, jax_op_name="bessel_ive")

    return ive


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
        return torch.special.digamma(x)

    return psi


@pytorch_funcify.register(TriGamma)
def pytorch_funcify_TriGamma(op, node, **kwargs):
    def tri_gamma(x):
        return torch.special.polygamma(1, x)

    return tri_gamma


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
