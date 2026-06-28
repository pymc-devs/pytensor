from functools import singledispatch

import mlx.core as mx
from numpy.random import Generator

import pytensor.tensor.random.basic as ptr
from pytensor.link.mlx.dispatch.basic import mlx_funcify, mlx_typify
from pytensor.link.mlx.dispatch.tensor_basic import (
    convert_dtype_to_mlx,
    mlx_to_list_shape,
)


def numpy_generator_to_mlx_key(rng: Generator) -> mx.array:
    """Convert a NumPy Generator to an MLX random key.

    MLX uses a functional RNG model where each random call takes an explicit
    key rather than mutating shared state. The PCG64 state is 128 bits, which
    MLX cannot accept directly. We fold both 64-bit halves together via XOR
    to use all 128 bits of entropy in a single 64-bit seed.
    """
    state = rng.bit_generator.state
    if state["bit_generator"] not in ("PCG64", "PCG64DXSM"):
        raise NotImplementedError(
            "MLX RNG conversion only supports the PCG64 bit generator, got "
            f"{state['bit_generator']}."
        )
    state_128 = int(state["state"]["state"])
    upper = (state_128 >> 64) & 0xFFFFFFFFFFFFFFFF
    lower = state_128 & 0xFFFFFFFFFFFFFFFF
    return mx.random.key(upper ^ lower)


def _shape_from_size(size, *parameters) -> list[int] | tuple[int, ...]:
    """Resolve the sampling shape for a draw.

    When ``size`` is given PyTensor has already computed the full output shape;
    otherwise the shape is the broadcast of the distribution parameters.
    """
    if size is not None:
        return mlx_to_list_shape(size)
    return mx.broadcast_shapes(*(p.shape for p in parameters))


@mlx_typify.register(Generator)
def mlx_typify_Generator(rng, **kwargs):
    return numpy_generator_to_mlx_key(rng)


@mlx_funcify.register(ptr.RandomVariable)
def mlx_funcify_RandomVariable(op, node, **kwargs):
    rv = node.outputs[1]
    out_dtype = rv.type.dtype

    sample_fn_inner = mlx_sample_fn(op, node)

    def sample_fn(rng, size, *parameters):
        new_rng, sampling_key = mx.random.split(rng, num=2)
        sample = sample_fn_inner(sampling_key, size, out_dtype, *parameters)
        return (new_rng, sample)

    return sample_fn


@singledispatch
def mlx_sample_fn(op, node):
    raise NotImplementedError(
        f"No MLX implementation for the given distribution: {op.name}"
    )


@mlx_sample_fn.register(ptr.NormalRV)
@mlx_sample_fn.register(ptr.LaplaceRV)
@mlx_sample_fn.register(ptr.GumbelRV)
def mlx_sample_fn_loc_scale(op, node):
    """Loc-scale families whose standardized variate MLX exposes directly.

    MLX names the standardized sampler the same as the PyTensor Op, so we can
    draw it and apply the affine ``loc + scale * z`` shift, mirroring the JAX
    dispatch.
    """
    mlx_op = getattr(mx.random, op.name)

    def sample_fn(rng_key, size, dtype, loc, scale):
        mlx_dtype = convert_dtype_to_mlx(dtype)
        loc = mx.array(loc, dtype=mlx_dtype)
        scale = mx.array(scale, dtype=mlx_dtype)
        shape = _shape_from_size(size, loc, scale)
        return loc + scale * mlx_op(shape=shape, dtype=mlx_dtype, key=rng_key)

    return sample_fn


@mlx_sample_fn.register(ptr.UniformRV)
def mlx_sample_fn_uniform(op, node):
    def sample_fn(rng_key, size, dtype, low, high):
        mlx_dtype = convert_dtype_to_mlx(dtype)
        low = mx.array(low, dtype=mlx_dtype)
        high = mx.array(high, dtype=mlx_dtype)
        shape = _shape_from_size(size, low, high)
        return mx.random.uniform(
            low=low, high=high, shape=shape, dtype=mlx_dtype, key=rng_key
        )

    return sample_fn


@mlx_sample_fn.register(ptr.IntegersRV)
def mlx_sample_fn_integers(op, node):
    def sample_fn(rng_key, size, dtype, low, high):
        low = mx.array(low)
        high = mx.array(high)
        shape = _shape_from_size(size, low, high)
        # Sample within the original bounds at full int64 width, then cast the
        # result. PyTensor casts the output (not the bounds), so a narrow dtype
        # wraps rather than collapsing the half-open interval, and wide bounds
        # are not clipped to MLX's default int32 sampling dtype.
        return mx.random.randint(
            low=low, high=high, shape=shape, dtype=mx.int64, key=rng_key
        ).astype(convert_dtype_to_mlx(dtype))

    return sample_fn


@mlx_sample_fn.register(ptr.BernoulliRV)
def mlx_sample_fn_bernoulli(op, node):
    def sample_fn(rng_key, size, dtype, p):
        p = mx.array(p)
        shape = mlx_to_list_shape(size) if size is not None else None
        # MLX returns a boolean draw; cast to the integer dtype PyTensor declares.
        return mx.random.bernoulli(p=p, shape=shape, key=rng_key).astype(
            convert_dtype_to_mlx(dtype)
        )

    return sample_fn


@mlx_sample_fn.register(ptr.CategoricalRV)
def mlx_sample_fn_categorical(op, node):
    def sample_fn(rng_key, size, dtype, p):
        logits = mx.log(mx.array(p))
        shape = mlx_to_list_shape(size) if size is not None else None
        # MLX returns ``uint32``; cast to the integer dtype PyTensor declares.
        return mx.random.categorical(
            logits=logits, axis=-1, shape=shape, key=rng_key
        ).astype(convert_dtype_to_mlx(dtype))

    return sample_fn


@mlx_sample_fn.register(ptr.MvNormalRV)
def mlx_sample_fn_mvnormal(op, node):
    method = op.method

    def sample_fn(rng_key, size, dtype, mean, cov):
        mlx_dtype = convert_dtype_to_mlx(dtype)
        mean = mx.array(mean, dtype=mlx_dtype)
        cov = mx.array(cov, dtype=mlx_dtype)

        n = cov.shape[-1]
        if size is not None:
            batch_shape = mlx_to_list_shape(size)
        else:
            batch_shape = mx.broadcast_shapes(mean.shape[:-1], cov.shape[:-2])

        if 0 in tuple(batch_shape):
            # An empty batch dim crashes the MLX compiled matmul path; the draw
            # is trivially an empty array of the broadcast output shape.
            return mx.broadcast_to(mean, [*batch_shape, n])

        # Factor ``cov = A @ A.T`` so that ``mean + A @ z`` has covariance ``cov``.
        if method == "cholesky":
            A = mx.linalg.cholesky(cov, stream=mx.cpu)
        elif method == "svd":
            U, s, _ = mx.linalg.svd(cov, stream=mx.cpu)
            A = U * mx.sqrt(s)[..., None, :]
        else:  # eigh
            w, vecs = mx.linalg.eigh(cov, stream=mx.cpu)
            A = vecs * mx.sqrt(w)[..., None, :]

        z = mx.random.normal(shape=[*batch_shape, n], dtype=mlx_dtype, key=rng_key)
        # Broadcasting matmul handles both batched params and leading size dims.
        return mean + (A @ z[..., None])[..., 0]

    return sample_fn


@mlx_sample_fn.register(ptr.PermutationRV)
def mlx_sample_fn_permutation(op, node):
    if op.batch_ndim(node):
        raise NotImplementedError(
            "MLX random.permutation does not support batch dimensions."
        )

    def sample_fn(rng_key, size, dtype, x):
        return mx.random.permutation(x, key=rng_key)

    return sample_fn


@mlx_sample_fn.register(ptr.LogNormalRV)
def mlx_sample_fn_lognormal(op, node):
    def sample_fn(rng_key, size, dtype, mu, sigma):
        mlx_dtype = convert_dtype_to_mlx(dtype)
        mu = mx.array(mu, dtype=mlx_dtype)
        sigma = mx.array(sigma, dtype=mlx_dtype)
        shape = _shape_from_size(size, mu, sigma)
        z = mx.random.normal(shape=shape, dtype=mlx_dtype, key=rng_key)
        return mx.exp(mu + sigma * z)

    return sample_fn


@mlx_sample_fn.register(ptr.HalfNormalRV)
def mlx_sample_fn_halfnormal(op, node):
    def sample_fn(rng_key, size, dtype, loc, scale):
        mlx_dtype = convert_dtype_to_mlx(dtype)
        loc = mx.array(loc, dtype=mlx_dtype)
        scale = mx.array(scale, dtype=mlx_dtype)
        shape = _shape_from_size(size, loc, scale)
        z = mx.random.normal(shape=shape, dtype=mlx_dtype, key=rng_key)
        return loc + scale * mx.abs(z)

    return sample_fn


@mlx_sample_fn.register(ptr.ExponentialRV)
def mlx_sample_fn_exponential(op, node):
    def sample_fn(rng_key, size, dtype, scale):
        mlx_dtype = convert_dtype_to_mlx(dtype)
        scale = mx.array(scale, dtype=mlx_dtype)
        shape = _shape_from_size(size, scale)
        u = mx.random.uniform(shape=shape, dtype=mlx_dtype, key=rng_key)
        return -scale * mx.log(u)

    return sample_fn


@mlx_sample_fn.register(ptr.LogisticRV)
def mlx_sample_fn_logistic(op, node):
    def sample_fn(rng_key, size, dtype, loc, scale):
        mlx_dtype = convert_dtype_to_mlx(dtype)
        loc = mx.array(loc, dtype=mlx_dtype)
        scale = mx.array(scale, dtype=mlx_dtype)
        shape = _shape_from_size(size, loc, scale)
        u = mx.random.uniform(shape=shape, dtype=mlx_dtype, key=rng_key)
        return loc + scale * mx.log(u / (1 - u))

    return sample_fn


@mlx_sample_fn.register(ptr.CauchyRV)
def mlx_sample_fn_cauchy(op, node):
    def sample_fn(rng_key, size, dtype, loc, scale):
        mlx_dtype = convert_dtype_to_mlx(dtype)
        loc = mx.array(loc, dtype=mlx_dtype)
        scale = mx.array(scale, dtype=mlx_dtype)
        shape = _shape_from_size(size, loc, scale)
        u = mx.random.uniform(shape=shape, dtype=mlx_dtype, key=rng_key)
        return loc + scale * mx.tan(mx.pi * (u - 0.5))

    return sample_fn
