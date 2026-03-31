from functools import singledispatch

import mlx.core as mx
from numpy.random import Generator

import pytensor.tensor.random.basic as ptr
from pytensor.link.mlx.dispatch.basic import mlx_funcify, mlx_typify
from pytensor.link.mlx.dispatch.core import convert_dtype_to_mlx, mlx_to_list_shape


def numpy_generator_to_mlx_key(rng: Generator) -> mx.array:
    """Convert a NumPy Generator to an MLX random key.

    MLX uses a functional RNG model where each random call takes an explicit
    key rather than mutating shared state. The PCG64 state is 128 bits, which
    MLX cannot accept directly. We fold both 64-bit halves together via XOR
    to use all 128 bits of entropy in a single 64-bit seed.
    """
    state_128 = int(rng.bit_generator.state["state"]["state"])
    upper = (state_128 >> 64) & 0xFFFFFFFFFFFFFFFF
    lower = state_128 & 0xFFFFFFFFFFFFFFFF
    return mx.random.key(upper ^ lower)


@mlx_typify.register(Generator)
def mlx_typify_Generator(rng, **kwargs):
    return numpy_generator_to_mlx_key(rng)


@mlx_funcify.register(ptr.RandomVariable)
def mlx_funcify_RandomVariable(op, node, **kwargs):
    rv = node.outputs[1]
    out_dtype = rv.type.dtype

    sample_fn_inner = mlx_sample_fn(op, node)

    def sample_fn(rng, size, *parameters):
        new_keys = mx.random.split(rng, num=2)
        new_rng = new_keys[0]
        sampling_key = new_keys[1]
        sample = sample_fn_inner(sampling_key, size, out_dtype, *parameters)
        return (new_rng, sample)

    return sample_fn


@singledispatch
def mlx_sample_fn(op, node):
    raise NotImplementedError(
        f"No MLX implementation for the given distribution: {op.name}"
    )


@mlx_sample_fn.register(ptr.NormalRV)
def mlx_sample_fn_normal(op, node):
    def sample_fn(rng_key, size, dtype, mu, sigma):
        mlx_dtype = convert_dtype_to_mlx(dtype)
        mu = mx.array(mu, dtype=mlx_dtype)
        sigma = mx.array(sigma, dtype=mlx_dtype)
        if size is None:
            shape = mx.broadcast_arrays(mu, sigma)[0].shape
        else:
            shape = mlx_to_list_shape(size)
        s = mx.random.normal(shape=shape, dtype=mlx_dtype, key=rng_key)
        return mu + sigma * s

    return sample_fn


@mlx_sample_fn.register(ptr.UniformRV)
def mlx_sample_fn_uniform(op, node):
    def sample_fn(rng_key, size, dtype, low, high):
        mlx_dtype = convert_dtype_to_mlx(dtype)
        low = mx.array(low, dtype=mlx_dtype)
        high = mx.array(high, dtype=mlx_dtype)
        if size is None:
            shape = mx.broadcast_arrays(low, high)[0].shape
        else:
            shape = mlx_to_list_shape(size)
        return mx.random.uniform(
            low=low, high=high, shape=shape, dtype=mlx_dtype, key=rng_key
        )

    return sample_fn


@mlx_sample_fn.register(ptr.BernoulliRV)
def mlx_sample_fn_bernoulli(op, node):
    def sample_fn(rng_key, size, dtype, p):
        p = mx.array(p)
        if size is None:
            shape = p.shape
        else:
            shape = mlx_to_list_shape(size)
        return mx.random.bernoulli(p=p, shape=shape, key=rng_key)

    return sample_fn


@mlx_sample_fn.register(ptr.CategoricalRV)
def mlx_sample_fn_categorical(op, node):
    def sample_fn(rng_key, size, dtype, p):
        logits = mx.log(mx.array(p))
        shape = mlx_to_list_shape(size) if size is not None else None
        return mx.random.categorical(logits=logits, axis=-1, shape=shape, key=rng_key)

    return sample_fn


@mlx_sample_fn.register(ptr.MvNormalRV)
def mlx_sample_fn_mvnormal(op, node):
    def sample_fn(rng_key, size, dtype, mean, cov):
        mlx_dtype = convert_dtype_to_mlx(dtype)
        shape = mlx_to_list_shape(size) if size is not None else []
        # multivariate_normal uses SVD internally, which requires mx.cpu in MLX.
        return mx.random.multivariate_normal(
            mean=mean,
            cov=cov,
            shape=shape,
            dtype=mlx_dtype,
            key=rng_key,
            stream=mx.cpu,
        )

    return sample_fn


@mlx_sample_fn.register(ptr.LaplaceRV)
def mlx_sample_fn_laplace(op, node):
    def sample_fn(rng_key, size, dtype, loc, scale):
        mlx_dtype = convert_dtype_to_mlx(dtype)
        loc = mx.array(loc, dtype=mlx_dtype)
        scale = mx.array(scale, dtype=mlx_dtype)
        if size is None:
            shape = mx.broadcast_arrays(loc, scale)[0].shape
        else:
            shape = mlx_to_list_shape(size)
        s = mx.random.laplace(shape=shape, dtype=mlx_dtype, key=rng_key)
        return loc + scale * s

    return sample_fn


@mlx_sample_fn.register(ptr.GumbelRV)
def mlx_sample_fn_gumbel(op, node):
    def sample_fn(rng_key, size, dtype, loc, scale):
        mlx_dtype = convert_dtype_to_mlx(dtype)
        loc = mx.array(loc, dtype=mlx_dtype)
        scale = mx.array(scale, dtype=mlx_dtype)
        if size is None:
            shape = mx.broadcast_arrays(loc, scale)[0].shape
        else:
            shape = mlx_to_list_shape(size)
        s = mx.random.gumbel(shape=shape, dtype=mlx_dtype, key=rng_key)
        return loc + scale * s

    return sample_fn


@mlx_sample_fn.register(ptr.PermutationRV)
def mlx_sample_fn_permutation(op, node):
    batch_ndim = op.batch_ndim(node)

    def sample_fn(rng_key, size, dtype, x):
        if batch_ndim:
            raise NotImplementedError(
                "MLX random.permutation does not support batch dimensions."
            )
        return mx.random.permutation(x, key=rng_key)

    return sample_fn


@mlx_sample_fn.register(ptr.IntegersRV)
def mlx_sample_fn_integers(op, node):
    def sample_fn(rng_key, size, dtype, low, high):
        mlx_dtype = convert_dtype_to_mlx(dtype)
        low = mx.array(low, dtype=mlx_dtype)
        high = mx.array(high, dtype=mlx_dtype)
        if size is None:
            shape = mx.broadcast_arrays(low, high)[0].shape
        else:
            shape = mlx_to_list_shape(size)
        return mx.random.randint(
            low=low, high=high, shape=shape, dtype=mlx_dtype, key=rng_key
        )

    return sample_fn
