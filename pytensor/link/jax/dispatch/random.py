from functools import singledispatch

import jax
import numpy as np
from numpy.random import Generator
from numpy.random.bit_generator import (  # type: ignore[attr-defined]
    _coerce_to_uint32_array,
)

import pytensor.tensor.random.basic as ptr
from pytensor.graph import Constant
from pytensor.link.jax.dispatch.basic import jax_funcify, jax_typify
from pytensor.link.jax.dispatch.shape import JAXShapeTuple
from pytensor.tensor.shape import Shape, Shape_i
from pytensor.tensor.type_other import NoneTypeT


try:
    import numpyro  # noqa: F401

    numpyro_available = True
except ImportError:
    numpyro_available = False

numpy_bit_gens = {"MT19937": 0, "PCG64": 1, "Philox": 2, "SFC64": 3}


SIZE_NOT_COMPATIBLE = """JAX random variables require concrete values for the `size` parameter of the distributions.
Concrete values are either constants:

>>> import pytensor.tensor as pt
>>> x_rv = pt.random.normal(0, 1, size=(3, 2))

or the shape of an array:

>>> m = pt.matrix()
>>> x_rv = pt.random.normal(0, 1, size=m.shape)
"""


def assert_size_argument_jax_compatible(node):
    """Assert whether the current node can be JIT-compiled by JAX.

    JAX can JIT-compile `jax.random` functions when the `size` argument
    is a concrete value, i.e. either a constant or the shape of any
    traced value.

    """
    size = node.inputs[1]
    size_node = size.owner
    if (size_node is not None) and (
        not isinstance(size_node.op, Shape | Shape_i | JAXShapeTuple)
    ):
        raise NotImplementedError(SIZE_NOT_COMPATIBLE)


@jax_typify.register(Generator)
def jax_typify_Generator(rng, **kwargs):
    state = rng.__getstate__()
    state["bit_generator"] = numpy_bit_gens[state["bit_generator"]]

    # XXX: Is this a reasonable approach?
    state["jax_state"] = _coerce_to_uint32_array(state["state"]["state"])[0:2]

    # The "state" and "inc" values in a NumPy `Generator` are 128 bits, which
    # JAX can't handle, so we split these values into arrays of 32 bit integers
    # and then combine the first two into a single 64 bit integers.
    #
    # XXX: Depending on how we expect these values to be used, is this approach
    # reasonable?
    #
    # TODO: We might as well remove these altogether, since this conversion
    # should only occur once (e.g. when the graph is converted/JAX-compiled),
    # and, from then on, we use the custom "jax_state" value.
    inc_32 = _coerce_to_uint32_array(state["state"]["inc"])
    state_32 = _coerce_to_uint32_array(state["state"]["state"])
    state["state"]["inc"] = inc_32[0] << 32 | inc_32[1]
    state["state"]["state"] = state_32[0] << 32 | state_32[1]
    return state


@jax_funcify.register(ptr.RandomVariable)
def jax_funcify_RandomVariable(op: ptr.RandomVariable, node, **kwargs):
    """JAX implementation of random variables."""
    rv = node.outputs[1]
    out_dtype = rv.type.dtype
    static_shape = rv.type.shape
    batch_ndim = op.batch_ndim(node)

    # Try to pass static size directly to JAX
    static_size = static_shape[:batch_ndim]
    if None in static_size:
        # Sometimes size can be constant folded during rewrites,
        # without the RandomVariable node being updated with new static types
        size_param = op.size_param(node)
        if isinstance(size_param, Constant) and not isinstance(
            size_param.type, NoneTypeT
        ):
            static_size = tuple(size_param.data)

    # If one dimension has unknown size, either the size is determined
    # by a `Shape` operator in which case JAX will compile, or it is
    # not and we fail gracefully.
    if None in static_size:
        assert_size_argument_jax_compatible(node)

        def sample_fn(rng, size, *parameters):
            return jax_sample_fn(op, node=node)(rng, size, out_dtype, *parameters)

    else:

        def sample_fn(rng, size, *parameters):
            return jax_sample_fn(op, node=node)(
                rng, static_size, out_dtype, *parameters
            )

    return sample_fn


@singledispatch
def jax_sample_fn(op, node):
    name = op.name
    raise NotImplementedError(
        f"No JAX implementation for the given distribution: {name}"
    )


@jax_sample_fn.register(ptr.BetaRV)
@jax_sample_fn.register(ptr.DirichletRV)
@jax_sample_fn.register(ptr.PoissonRV)
@jax_sample_fn.register(ptr.MvNormalRV)
def jax_sample_fn_generic(op, node):
    """Generic JAX implementation of random variables."""
    name = op.name
    jax_op = getattr(jax.random, name)

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        sample = jax_op(sampling_key, *parameters, shape=size, dtype=dtype)
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(ptr.CauchyRV)
@jax_sample_fn.register(ptr.GumbelRV)
@jax_sample_fn.register(ptr.LaplaceRV)
@jax_sample_fn.register(ptr.LogisticRV)
@jax_sample_fn.register(ptr.NormalRV)
def jax_sample_fn_loc_scale(op, node):
    """JAX implementation of random variables in the loc-scale families.

    JAX only implements the standard version of random variables in the
    loc-scale family. We thus need to translate and rescale the results
    manually.

    """
    name = op.name
    jax_op = getattr(jax.random, name)

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        loc, scale = parameters
        if size is None:
            size = jax.numpy.broadcast_arrays(loc, scale)[0].shape
        sample = loc + jax_op(sampling_key, size, dtype) * scale
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(ptr.BernoulliRV)
def jax_sample_fn_bernoulli(op, node):
    """JAX implementation of `BernoulliRV`."""

    # We need a separate dispatch, because there is no dtype argument for Bernoulli in JAX
    def sample_fn(rng, size, dtype, p):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        sample = jax.random.bernoulli(sampling_key, p, shape=size)
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(ptr.CategoricalRV)
def jax_sample_fn_categorical(op, node):
    """JAX implementation of `CategoricalRV`."""

    # We need a separate dispatch because Categorical expects logits in JAX
    def sample_fn(rng, size, dtype, p):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)

        logits = jax.scipy.special.logit(p)
        sample = jax.random.categorical(sampling_key, logits=logits, shape=size)
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(ptr.IntegersRV)
@jax_sample_fn.register(ptr.UniformRV)
def jax_sample_fn_uniform(op, node):
    """JAX implementation of random variables with uniform density.

    We need to pass the arguments as keyword arguments since the order
    of arguments is not the same.

    """
    name = op.name
    # IntegersRV is equivalent to RandintRV
    if isinstance(op, ptr.IntegersRV):
        name = "randint"
    jax_op = getattr(jax.random, name)

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        minval, maxval = parameters
        sample = jax_op(
            sampling_key, shape=size, dtype=dtype, minval=minval, maxval=maxval
        )
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(ptr.ParetoRV)
@jax_sample_fn.register(ptr.GammaRV)
def jax_sample_fn_shape_scale(op, node):
    """JAX implementation of random variables in the shape-scale family.

    JAX only implements the standard version of random variables in the
    shape-scale family. We thus need to rescale the results manually.

    """
    name = op.name
    jax_op = getattr(jax.random, name)

    def sample_fn(rng, size, dtype, shape, scale):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        if size is None:
            size = jax.numpy.broadcast_arrays(shape, scale)[0].shape
        sample = jax_op(sampling_key, shape, size, dtype) * scale
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(ptr.ExponentialRV)
def jax_sample_fn_exponential(op, node):
    """JAX implementation of `ExponentialRV`."""

    def sample_fn(rng, size, dtype, scale):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        if size is None:
            size = jax.numpy.asarray(scale).shape
        sample = jax.random.exponential(sampling_key, size, dtype) * scale
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(ptr.StudentTRV)
def jax_sample_fn_t(op, node):
    """JAX implementation of `StudentTRV`."""

    def sample_fn(rng, size, dtype, df, loc, scale):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        if size is None:
            size = jax.numpy.broadcast_arrays(df, loc, scale)[0].shape
        sample = loc + jax.random.t(sampling_key, df, size, dtype) * scale
        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(ptr.ChoiceWithoutReplacement)
def jax_funcify_choice(op: ptr.ChoiceWithoutReplacement, node):
    """JAX implementation of `ChoiceRV`."""

    batch_ndim = op.batch_ndim(node)
    a_core_ndim, *p_core_ndim, _ = op.ndims_params

    if batch_ndim and a_core_ndim == 0:
        raise NotImplementedError(
            "Batch dimensions are not supported for 0d arrays. "
            "A default JAX rewrite should have materialized the implicit arange"
        )

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)

        if op.has_p_param:
            a, p, core_shape = parameters
        else:
            a, core_shape = parameters
            p = None
        core_shape = tuple(np.asarray(core_shape)[(0,) * batch_ndim])

        if batch_ndim == 0:
            sample = jax.random.choice(
                sampling_key, a, shape=core_shape, replace=False, p=p
            )

        else:
            if size is None:
                if p is None:
                    size = a.shape[:batch_ndim]
                else:
                    size = jax.numpy.broadcast_shapes(
                        a.shape[:batch_ndim],
                        p.shape[:batch_ndim],
                    )

            a = jax.numpy.broadcast_to(a, size + a.shape[batch_ndim:])
            if p is not None:
                p = jax.numpy.broadcast_to(p, size + p.shape[batch_ndim:])

            batch_sampling_keys = jax.random.split(sampling_key, np.prod(size))

            # Ravel the batch dimensions because vmap only works along a single axis
            raveled_batch_a = a.reshape((-1,) + a.shape[batch_ndim:])
            if p is None:
                raveled_sample = jax.vmap(
                    lambda key, a: jax.random.choice(
                        key, a, shape=core_shape, replace=False, p=None
                    )
                )(batch_sampling_keys, raveled_batch_a)
            else:
                raveled_batch_p = p.reshape((-1,) + p.shape[batch_ndim:])
                raveled_sample = jax.vmap(
                    lambda key, a, p: jax.random.choice(
                        key, a, shape=core_shape, replace=False, p=p
                    )
                )(batch_sampling_keys, raveled_batch_a, raveled_batch_p)

            # Reshape the batch dimensions
            sample = raveled_sample.reshape(size + raveled_sample.shape[1:])

        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(ptr.PermutationRV)
def jax_sample_fn_permutation(op, node):
    """JAX implementation of `PermutationRV`."""

    batch_ndim = op.batch_ndim(node)

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        (x,) = parameters
        if batch_ndim:
            # jax.random.permutation has no concept of batch dims
            if size is None:
                size = x.shape[:batch_ndim]
            else:
                x = jax.numpy.broadcast_to(x, size + x.shape[batch_ndim:])

            batch_sampling_keys = jax.random.split(sampling_key, np.prod(size))
            raveled_batch_x = x.reshape((-1,) + x.shape[batch_ndim:])
            raveled_sample = jax.vmap(lambda key, x: jax.random.permutation(key, x))(
                batch_sampling_keys, raveled_batch_x
            )
            sample = raveled_sample.reshape(size + raveled_sample.shape[1:])
        else:
            sample = jax.random.permutation(sampling_key, x)

        rng["jax_state"] = rng_key
        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(ptr.BinomialRV)
def jax_sample_fn_binomial(op, node):
    if not numpyro_available:
        raise NotImplementedError(
            f"No JAX implementation for the given distribution: {op.name}. "
            "Implementation is available if NumPyro is installed."
        )

    from numpyro.distributions.util import binomial

    def sample_fn(rng, size, dtype, n, p):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)

        sample = binomial(key=sampling_key, n=n, p=p, shape=size)

        rng["jax_state"] = rng_key

        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(ptr.MultinomialRV)
def jax_sample_fn_multinomial(op, node):
    if not numpyro_available:
        raise NotImplementedError(
            f"No JAX implementation for the given distribution: {op.name}. "
            "Implementation is available if NumPyro is installed."
        )

    from numpyro.distributions.util import multinomial

    def sample_fn(rng, size, dtype, n, p):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)

        sample = multinomial(key=sampling_key, n=n, p=p, shape=size)

        rng["jax_state"] = rng_key

        return (rng, sample)

    return sample_fn


@jax_sample_fn.register(ptr.VonMisesRV)
def jax_sample_fn_vonmises(op, node):
    if not numpyro_available:
        raise NotImplementedError(
            f"No JAX implementation for the given distribution: {op.name}. "
            "Implementation is available if NumPyro is installed."
        )

    from numpyro.distributions.util import von_mises_centered

    def sample_fn(rng, size, dtype, mu, kappa):
        rng_key = rng["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)

        sample = von_mises_centered(
            key=sampling_key, concentration=kappa, shape=size, dtype=dtype
        )
        sample = (sample + mu + np.pi) % (2.0 * np.pi) - np.pi

        rng["jax_state"] = rng_key

        return (rng, sample)

    return sample_fn
