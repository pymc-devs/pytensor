from functools import singledispatch

import numpy as np
import torch
from numpy.random import Generator

import pytensor.tensor.random.basic as ptr
from pytensor.link.pytorch.dispatch.basic import pytorch_funcify, pytorch_typify


@pytorch_typify.register(Generator)
def pytorch_typify_Generator(rng, **kwargs):
    # XXX: Check if there is a better way.
    # Numpy uses PCG64 while Torch uses Mersenne-Twister (https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/CPUGeneratorImpl.cpp)
    state = rng.__getstate__()
    rng_copy = np.random.default_rng()
    rng_copy.bit_generator.state = rng.bit_generator.state
    seed = torch.from_numpy(rng_copy.integers([2**32]))
    state["pytorch_gen"] = torch.manual_seed(seed)
    return state


@pytorch_funcify.register(ptr.RandomVariable)
def torch_funcify_RandomVariable(op: ptr.RandomVariable, node, **kwargs):
    rv = node.outputs[1]
    out_dtype = rv.type.dtype
    shape = rv.type.shape

    def sample_fn(rng, size, *parameters):
        return pytorch_sample_fn(op, node=node)(rng, shape, out_dtype, *parameters)

    return sample_fn


@singledispatch
def pytorch_sample_fn(op, node):
    name = op.name
    raise NotImplementedError(
        f"No PyTorch implementation for the given distribution: {name}"
    )


@pytorch_sample_fn.register(ptr.BernoulliRV)
def pytorch_sample_fn_bernoulli(op, node):
    def sample_fn(rng, size, dtype, p):
        gen = rng["pytorch_gen"]
        if not size:
            size = (1,)

        sample = torch.bernoulli(torch.broadcast_to(p, size), generator=gen)
        rng["pytorch_gen"] = gen
        return (rng, sample)

    return sample_fn


@pytorch_sample_fn.register(ptr.BinomialRV)
def pytorch_sample_fn_binomial(op, node):
    def sample_fn(rng, size, dtype, n, p):
        gen = rng["pytorch_gen"]
        if not size:
            size = (1,)

        sample = torch.binomial(
            torch.broadcast_to(n.to(p.dtype), size),
            torch.broadcast_to(p, size),
            generator=gen,
        )
        rng["pytorch_gen"] = gen
        return (rng, sample)

    return sample_fn
