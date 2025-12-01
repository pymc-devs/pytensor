import functools
import typing
from collections.abc import Callable

import jax
import jax.numpy as jnp

from pytensor.link.jax.dispatch.basic import jax_funcify
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
    Kve,
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
            "Implementation is available if tfp-nightly is installed"
        )

    if jax_op_name is None:
        jax_op_name = op.name
    return typing.cast(Callable, getattr(tfp_jax_math, jax_op_name))


def all_inputs_are_scalar(node):
    """Check whether all the inputs of an `Elemwise` are scalar values.

    `jax.lax` or `jax.numpy` functions systematically return `TracedArrays`,
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


@jax_funcify.register(ScalarOp)
def jax_funcify_ScalarOp(op, node, **kwargs):
    """Return JAX function that implements the same computation as the Scalar Op.

    This dispatch is expected to return a JAX function that works on Array inputs as Elemwise does,
    even though it's dispatched on the Scalar Op.
    """

    # We dispatch some PyTensor operators to Python operators
    # whenever the inputs are all scalars.
    if all_inputs_are_scalar(node):
        jax_func = jax_funcify_scalar_op_via_py_operators(op)
        if jax_func is not None:
            return jax_func

    nfunc_spec = getattr(op, "nfunc_spec", None)
    if nfunc_spec is None:
        raise NotImplementedError(f"Dispatch not implemented for Scalar Op {op}")

    func_name = nfunc_spec[0]
    if "." in func_name:
        jax_func = functools.reduce(getattr, [jax, *func_name.split(".")])
    else:
        jax_func = getattr(jnp, func_name)

    if len(node.inputs) > op.nfunc_spec[1]:
        # Some Scalar Ops accept multiple number of inputs, behaving as a variadic function,
        # even though the base Op from `func_name` is specified as a binary Op.
        # This happens with `Add`, which can work as a `Sum` for multiple scalars.
        jax_variadic_func = getattr(jnp, op.nfunc_variadic, None)
        if not jax_variadic_func:
            raise NotImplementedError(
                f"Dispatch not implemented for Scalar Op {op} with {len(node.inputs)} inputs"
            )

        def jax_func(*args):
            return jax_variadic_func(
                jnp.stack(jnp.broadcast_arrays(*args), axis=0), axis=0
            )

    return jax_func


@functools.singledispatch
def jax_funcify_scalar_op_via_py_operators(op):
    """Specialized JAX dispatch for Elemwise operations where all inputs are Scalar arrays.

    Scalar (constant) arrays in the JAX backend get lowered to the native types (int, floats),
    which can perform better with Python operators, and more importantly, avoid upcasting to array types
    not supported by some JAX functions.
    """
    return None


@jax_funcify_scalar_op_via_py_operators.register(Add)
def jax_funcify_scalar_Add(op):
    def elemwise(*inputs):
        return sum(inputs)

    return elemwise


@jax_funcify_scalar_op_via_py_operators.register(Mul)
def jax_funcify_scalar_Mul(op):
    import operator
    from functools import reduce

    def elemwise(*inputs):
        return reduce(operator.mul, inputs, 1)

    return elemwise


@jax_funcify_scalar_op_via_py_operators.register(Sub)
def jax_funcify_scalar_Sub(op):
    def elemwise(x, y):
        return x - y

    return elemwise


@jax_funcify_scalar_op_via_py_operators.register(IntDiv)
def jax_funcify_scalar_IntDiv(op):
    def elemwise(x, y):
        return x // y

    return elemwise


@jax_funcify_scalar_op_via_py_operators.register(Mod)
def jax_funcify_scalar_Mod(op):
    def elemwise(x, y):
        return x % y

    return elemwise


@jax_funcify.register(Cast)
def jax_funcify_Cast(op, **kwargs):
    def cast(x):
        return jnp.array(x).astype(op.o_type.dtype)

    return cast


@jax_funcify.register(Identity)
def jax_funcify_Identity(op, **kwargs):
    def identity(x):
        return x

    return identity


@jax_funcify.register(Clip)
def jax_funcify_Clip(op, **kwargs):
    """Register the translation for the `Clip` `Op`.

    PyTensor's `Clip` operator operates differently from NumPy's when the
    specified `min` is larger than the `max` so we cannot reuse `jax.numpy.clip`
    to maintain consistency with PyTensor.

    """

    def clip(x, min, max):
        return jnp.where(x < min, min, jnp.where(x > max, max, x))

    return clip


@jax_funcify.register(Composite)
def jax_funcify_Composite(op, node, vectorize=True, **kwargs):
    jax_impl = jax_funcify(op.fgraph)

    if len(node.outputs) == 1:

        def composite(*args):
            return jax_impl(*args)[0]

    else:

        def composite(*args):
            return jax_impl(*args)

    return jnp.vectorize(composite)


@jax_funcify.register(Second)
def jax_funcify_Second(op, **kwargs):
    def second(x, y):
        _, y = jnp.broadcast_arrays(x, y)
        return y

    return second


@jax_funcify.register(GammaIncInv)
def jax_funcify_GammaIncInv(op, **kwargs):
    gammaincinv = try_import_tfp_jax_op(op, jax_op_name="igammainv")

    return gammaincinv


@jax_funcify.register(GammaIncCInv)
def jax_funcify_GammaIncCInv(op, **kwargs):
    gammainccinv = try_import_tfp_jax_op(op, jax_op_name="igammacinv")

    return gammainccinv


@jax_funcify.register(Erf)
def jax_funcify_Erf(op, node, **kwargs):
    def erf(x):
        return jax.scipy.special.erf(x)

    return erf


@jax_funcify.register(Erfc)
def jax_funcify_Erfc(op, **kwargs):
    def erfc(x):
        return jax.scipy.special.erfc(x)

    return erfc


@jax_funcify.register(Erfinv)
def jax_funcify_Erfinv(op, **kwargs):
    def erfinv(x):
        return jax.scipy.special.erfinv(x)

    return erfinv


@jax_funcify.register(BetaIncInv)
@jax_funcify.register(Erfcx)
@jax_funcify.register(Erfcinv)
def jax_funcify_from_tfp(op, **kwargs):
    tfp_jax_op = try_import_tfp_jax_op(op)

    return tfp_jax_op


@jax_funcify.register(Iv)
def jax_funcify_Iv(op, **kwargs):
    ive = try_import_tfp_jax_op(op, jax_op_name="bessel_ive")

    def iv(v, x):
        return ive(v, x) / jnp.exp(-jnp.abs(jnp.real(x)))

    return iv


@jax_funcify.register(Ive)
def jax_funcify_Ive(op, **kwargs):
    return try_import_tfp_jax_op(op, jax_op_name="bessel_ive")


@jax_funcify.register(Kve)
def jax_funcify_Kve(op, **kwargs):
    return try_import_tfp_jax_op(op, jax_op_name="bessel_kve")


@jax_funcify.register(Log1mexp)
def jax_funcify_Log1mexp(op, node, **kwargs):
    def log1mexp(x):
        return jnp.where(
            x < jnp.log(0.5), jnp.log1p(-jnp.exp(x)), jnp.log(-jnp.expm1(x))
        )

    return log1mexp


@jax_funcify.register(Psi)
def jax_funcify_Psi(op, node, **kwargs):
    def psi(x):
        return jax.scipy.special.digamma(x)

    return psi


@jax_funcify.register(TriGamma)
def jax_funcify_TriGamma(op, node, **kwargs):
    def tri_gamma(x):
        return jax.scipy.special.polygamma(1, x)

    return tri_gamma


@jax_funcify.register(Softplus)
def jax_funcify_Softplus(op, **kwargs):
    def softplus(x):
        return jnp.where(
            x < -37.0,
            jnp.exp(x),
            jnp.where(
                x < 18.0,
                jnp.log1p(jnp.exp(x)),
                jnp.where(
                    x < 33.3,
                    x + jnp.exp(-x),
                    x,
                ),
            ),
        )

    return softplus
