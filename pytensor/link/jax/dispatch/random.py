from functools import singledispatch

import jax
import jax.numpy as jnp
import numpy as np
from numpy.random import Generator

import pytensor.tensor.random.basic as ptr
from pytensor.graph import Constant
from pytensor.link.backend_conversion import (
    BackendConversion,
    register_backend_conversion,
)
from pytensor.link.jax.dispatch.basic import jax_funcify, jax_typify
from pytensor.link.jax.dispatch.shape import JAXShapeTuple
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.shape import Shape, Shape_i
from pytensor.tensor.type_other import NoneTypeT


try:
    import numpyro  # noqa: F401

    numpyro_available = True
except ImportError:
    numpyro_available = False

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
    """Derive a JAX threefry key from a host numpy ``Generator``.

    The forward half of the lossy map completed by `jax_detypify_Generator`.
    Hashing the ~256-bit PCG64 state into 2 words is not reversible, so this
    seeds the JAX key deterministically but does not preserve the host stream.
    """
    # A JAX threefry key holds only 2 uint32 words, far fewer than the ~256 bits
    # of a NumPy Generator's state. Rather than truncate to the low
    # (worst-quality) bits of the LCG state and drop the stream `inc`, fold the
    # whole state through SeedSequence's mixer so all the entropy is hashed into
    # the 2 words we keep.
    state = rng.bit_generator.state["state"]
    seed = np.random.SeedSequence([state["state"], state["inc"]])
    return seed.generate_state(2, dtype=np.uint32)


def jax_detypify_Generator(key):
    """Reconcile a JAX-advanced threefry key back to a host numpy ``Generator``.

    The reverse half of the lossy map started by `jax_typify_Generator`. The
    host cannot continue the JAX stream, so this reseeds a ``Generator`` from the
    key bits deterministically, starting a new, unrelated stream.
    """
    key = np.asarray(key).astype(np.uint32).ravel()
    seed = np.random.SeedSequence(int.from_bytes(key.tobytes(), "little"))
    return np.random.Generator(np.random.PCG64(seed))


register_backend_conversion(
    BackendConversion(
        tag="jax",
        handles=lambda type_: isinstance(type_, RandomType),
        to_native=jax_typify,
        from_native=jax_detypify_Generator,
        lossy=True,
    )
)


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
            next_rng, sampling_key = jax.random.split(rng, 2)
            sample = jax_sample_fn(op, node=node)(
                sampling_key, size, out_dtype, *parameters
            )
            return (next_rng, sample)

    else:

        def sample_fn(rng, size, *parameters):
            next_rng, sampling_key = jax.random.split(rng, 2)
            sample = jax_sample_fn(op, node=node)(
                sampling_key, static_size, out_dtype, *parameters
            )
            return (next_rng, sample)

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
def jax_sample_fn_generic(op, node):
    """Generic JAX implementation of random variables."""
    name = op.name
    jax_op = getattr(jax.random, name)

    def sample_fn(rng_key, size, dtype, *parameters):
        sample = jax_op(rng_key, *parameters, shape=size, dtype=dtype)
        return sample

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

    def sample_fn(rng_key, size, dtype, *parameters):
        loc, scale = parameters
        if size is None:
            size = jax.numpy.broadcast_arrays(loc, scale)[0].shape
        sample = loc + jax_op(rng_key, size, dtype) * scale
        return sample

    return sample_fn


@jax_sample_fn.register(ptr.MvNormalRV)
def jax_sample_mvnormal(op, node):
    def sample_fn(rng_key, size, dtype, mean, cov):
        sample = jax.random.multivariate_normal(
            rng_key, mean, cov, shape=size, dtype=dtype, method=op.method
        )
        return sample

    return sample_fn


@jax_sample_fn.register(ptr.BernoulliRV)
def jax_sample_fn_bernoulli(op, node):
    """JAX implementation of `BernoulliRV`."""

    # We need a separate dispatch, because there is no dtype argument for Bernoulli in JAX
    def sample_fn(rng_key, size, dtype, p):
        sample = jax.random.bernoulli(rng_key, p, shape=size)
        return sample

    return sample_fn


@jax_sample_fn.register(ptr.CategoricalRV)
def jax_sample_fn_categorical(op, node):
    """JAX implementation of `CategoricalRV`."""

    # We need a separate dispatch because Categorical expects logits in JAX
    def sample_fn(rng_key, size, dtype, p):
        logits = jax.scipy.special.logit(p)
        sample = jax.random.categorical(rng_key, logits=logits, shape=size)
        return sample

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

    def sample_fn(rng_key, size, dtype, *parameters):
        minval, maxval = parameters
        sample = jax_op(rng_key, shape=size, dtype=dtype, minval=minval, maxval=maxval)
        return sample

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

    def sample_fn(rng_key, size, dtype, shape, scale):
        if size is None:
            size = jax.numpy.broadcast_arrays(shape, scale)[0].shape
        sample = jax_op(rng_key, shape, size, dtype) * scale
        return sample

    return sample_fn


@jax_sample_fn.register(ptr.ExponentialRV)
def jax_sample_fn_exponential(op, node):
    """JAX implementation of `ExponentialRV`."""

    def sample_fn(rng_key, size, dtype, scale):
        if size is None:
            size = jax.numpy.asarray(scale).shape
        sample = jax.random.exponential(rng_key, size, dtype) * scale
        return sample

    return sample_fn


@jax_sample_fn.register(ptr.StudentTRV)
def jax_sample_fn_t(op, node):
    """JAX implementation of `StudentTRV`."""

    def sample_fn(rng_key, size, dtype, df, loc, scale):
        if size is None:
            size = jax.numpy.broadcast_arrays(df, loc, scale)[0].shape
        sample = loc + jax.random.t(rng_key, df, size, dtype) * scale
        return sample

    return sample_fn


@jax_sample_fn.register(ptr.ChoiceWithoutReplacement)
def jax_funcify_choice(op: ptr.ChoiceWithoutReplacement, node):
    """JAX implementation of `ChoiceRV`."""

    batch_ndim = op.batch_ndim(node)
    a_core_ndim, *_p_core_ndim, _ = op.ndims_params

    if batch_ndim and a_core_ndim == 0:
        raise NotImplementedError(
            "Batch dimensions are not supported for 0d arrays. "
            "A default JAX rewrite should have materialized the implicit arange"
        )

    def sample_fn(rng_key, size, dtype, *parameters):
        if op.has_p_param:
            a, p, core_shape = parameters
        else:
            a, core_shape = parameters
            p = None
        core_shape = tuple(np.asarray(core_shape)[(0,) * batch_ndim])

        if batch_ndim == 0:
            sample = jax.random.choice(rng_key, a, shape=core_shape, replace=False, p=p)

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

            batch_sampling_keys = jax.random.split(rng_key, np.prod(size))

            # Ravel the batch dimensions because vmap only works along a single axis
            raveled_batch_a = a.reshape((-1, *a.shape[batch_ndim:]))
            if p is None:
                raveled_sample = jax.vmap(
                    lambda key, a: jax.random.choice(
                        key, a, shape=core_shape, replace=False, p=None
                    )
                )(batch_sampling_keys, raveled_batch_a)
            else:
                raveled_batch_p = p.reshape((-1, *p.shape[batch_ndim:]))
                raveled_sample = jax.vmap(
                    lambda key, a, p: jax.random.choice(
                        key, a, shape=core_shape, replace=False, p=p
                    )
                )(batch_sampling_keys, raveled_batch_a, raveled_batch_p)

            # Reshape the batch dimensions
            sample = raveled_sample.reshape(size + raveled_sample.shape[1:])

        return sample

    return sample_fn


@jax_sample_fn.register(ptr.PermutationRV)
def jax_sample_fn_permutation(op, node):
    """JAX implementation of `PermutationRV`."""

    batch_ndim = op.batch_ndim(node)

    def sample_fn(rng_key, size, dtype, *parameters):
        (x,) = parameters
        if batch_ndim:
            # jax.random.permutation has no concept of batch dims
            if size is None:
                size = x.shape[:batch_ndim]
            else:
                x = jax.numpy.broadcast_to(x, size + x.shape[batch_ndim:])

            batch_sampling_keys = jax.random.split(rng_key, np.prod(size))
            raveled_batch_x = x.reshape((-1, *x.shape[batch_ndim:]))
            raveled_sample = jax.vmap(lambda key, x: jax.random.permutation(key, x))(
                batch_sampling_keys, raveled_batch_x
            )
            sample = raveled_sample.reshape(size + raveled_sample.shape[1:])
        else:
            sample = jax.random.permutation(rng_key, x)

        return sample

    return sample_fn


@jax_sample_fn.register(ptr.BinomialRV)
def jax_sample_fn_binomial(op, node):
    if not numpyro_available:
        raise NotImplementedError(
            f"No JAX implementation for the given distribution: {op.name}. "
            "Implementation is available if NumPyro is installed."
        )

    from numpyro.distributions.util import binomial

    def sample_fn(rng_key, size, dtype, n, p):
        sample = binomial(key=rng_key, n=n, p=p, shape=size)
        return sample

    return sample_fn


@jax_sample_fn.register(ptr.MultinomialRV)
def jax_sample_fn_multinomial(op, node):
    def sample_fn(rng_key, size, dtype, n, p):
        if size is not None:
            n = jnp.broadcast_to(n, size)
            p = jnp.broadcast_to(p, size + jnp.shape(p)[-1:])

        else:
            broadcast_shape = jax.lax.broadcast_shapes(jnp.shape(n), jnp.shape(p)[:-1])
            n = jnp.broadcast_to(n, broadcast_shape)
            p = jnp.broadcast_to(p, broadcast_shape + jnp.shape(p)[-1:])

        binom_p = jnp.moveaxis(p, -1, 0)[:-1, ...]
        sampling_rng = jax.random.split(rng_key, binom_p.shape[0])

        def _binomial_sample_fn(carry, p_rng):
            remaining_n, remaining_p = carry
            p, rng = p_rng
            samples = jnp.where(
                remaining_n == 0,
                0,
                jax.random.binomial(rng, remaining_n, p / remaining_p),
            )
            remaining_n -= samples
            remaining_p -= p
            return ((remaining_n, remaining_p), samples)

        (remain, _), samples = jax.lax.scan(
            _binomial_sample_fn,
            (n.astype(np.float64), jnp.ones(binom_p.shape[1:])),
            (binom_p, sampling_rng),
        )
        sample = jnp.concatenate(
            [jnp.moveaxis(samples, 0, -1), jnp.expand_dims(remain, -1)], axis=-1
        )
        return sample

    return sample_fn


@jax_sample_fn.register(ptr.VonMisesRV)
def jax_sample_fn_vonmises(op, node):
    if not numpyro_available:
        raise NotImplementedError(
            f"No JAX implementation for the given distribution: {op.name}. "
            "Implementation is available if NumPyro is installed."
        )

    from numpyro.distributions.util import von_mises_centered

    def sample_fn(rng_key, size, dtype, mu, kappa):
        sample = von_mises_centered(
            key=rng_key, concentration=kappa, shape=size, dtype=dtype
        )
        sample = (sample + mu + np.pi) % (2.0 * np.pi) - np.pi

        return sample

    return sample_fn
