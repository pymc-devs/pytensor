from functools import singledispatch

import mlx.core as mx

import pytensor.tensor.random.basic as ptr
from pytensor.graph import Constant
from pytensor.link.mlx.dispatch.basic import convert_dtype_to_mlx, mlx_funcify
from pytensor.tensor.type_other import NoneTypeT


SIZE_NOT_COMPATIBLE = """MLX random variables require a statically known `size`, since shapes must be
concrete when tracing with `mx.compile`. Provide a constant size:

>>> import pytensor.tensor as pt
>>> x_rv = pt.random.normal(0, 1, size=(3, 2))

or ensure the sampled variable has a static shape.
"""


@mlx_funcify.register(ptr.RandomVariable)
def mlx_funcify_RandomVariable(op: ptr.RandomVariable, node, **kwargs):
    """MLX implementation of random variables."""
    rv = node.outputs[1]
    out_dtype = convert_dtype_to_mlx(rv.type.dtype)
    static_shape = rv.type.shape
    batch_ndim = op.batch_ndim(node)

    static_size = static_shape[:batch_ndim]
    if None in static_size:
        # Sometimes size can be constant folded during rewrites,
        # without the RandomVariable node being updated with new static types
        size_param = op.size_param(node)
        if isinstance(size_param, Constant) and not isinstance(
            size_param.type, NoneTypeT
        ):
            static_size = tuple(size_param.data)

    if None in static_size:
        raise NotImplementedError(SIZE_NOT_COMPATIBLE)

    static_size = tuple(int(s) for s in static_size)
    sample_fn = mlx_sample_fn(op, node=node)

    def random_variable(rng, size, *parameters):
        new_rng, sampling_key = mx.random.split(rng, 2)
        sample = sample_fn(sampling_key, static_size, out_dtype, *parameters)
        return (new_rng, sample)

    return random_variable


@singledispatch
def mlx_sample_fn(op, node):
    name = op.name
    raise NotImplementedError(
        f"No MLX implementation for the given distribution: {name}"
    )


@mlx_sample_fn.register(ptr.NormalRV)
def mlx_sample_fn_normal(op, node):
    def sample_fn(key, size, dtype, loc, scale):
        return loc + mx.random.normal(shape=size, dtype=dtype, key=key) * scale

    return sample_fn


@mlx_sample_fn.register(ptr.LaplaceRV)
def mlx_sample_fn_laplace(op, node):
    def sample_fn(key, size, dtype, loc, scale):
        return loc + mx.random.laplace(shape=size, dtype=dtype, key=key) * scale

    return sample_fn


@mlx_sample_fn.register(ptr.GumbelRV)
def mlx_sample_fn_gumbel(op, node):
    def sample_fn(key, size, dtype, loc, scale):
        return loc + mx.random.gumbel(shape=size, dtype=dtype, key=key) * scale

    return sample_fn


@mlx_sample_fn.register(ptr.UniformRV)
def mlx_sample_fn_uniform(op, node):
    def sample_fn(key, size, dtype, low, high):
        return mx.random.uniform(low=low, high=high, shape=size, dtype=dtype, key=key)

    return sample_fn


@mlx_sample_fn.register(ptr.IntegersRV)
def mlx_sample_fn_integers(op, node):
    def sample_fn(key, size, dtype, low, high):
        # `mx.random.randint` truncates towards zero, so a negative `low` would
        # never be sampled. Draw from `[0, high - low)` and shift instead. The
        # shift can upcast the result, so cast back to the requested dtype.
        draw = low + mx.random.randint(
            low=0, high=high - low, shape=size, dtype=dtype, key=key
        )
        return draw.astype(dtype)

    return sample_fn


@mlx_sample_fn.register(ptr.BernoulliRV)
def mlx_sample_fn_bernoulli(op, node):
    def sample_fn(key, size, dtype, p):
        return mx.random.bernoulli(p=p, shape=size, key=key).astype(dtype)

    return sample_fn


@mlx_sample_fn.register(ptr.CategoricalRV)
def mlx_sample_fn_categorical(op, node):
    def sample_fn(key, size, dtype, p):
        # MLX `categorical` applies a softmax over `logits`, so pass `log(p)`.
        logits = mx.log(p)
        return mx.random.categorical(logits, shape=size, key=key).astype(dtype)

    return sample_fn
