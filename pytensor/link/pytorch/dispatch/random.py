from functools import singledispatch

import numpy.random
import torch

import pytensor.tensor.random.basic as ptr
from pytensor.link.pytorch.dispatch.basic import pytorch_funcify, pytorch_typify


@pytorch_typify.register(numpy.random.Generator)
def pytorch_typify_Generator(rng, **kwargs):
    # XXX: Check if there is a better way.
    # Numpy uses PCG64 while Torch uses Mersenne-Twister (https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/CPUGeneratorImpl.cpp)
    seed = torch.from_numpy(rng.integers([2**32]))
    return torch.manual_seed(seed)


@pytorch_typify.register(torch._C.Generator)
def pytorch_typify_pass_generator(rng, **kwargs):
    return rng


@pytorch_funcify.register(ptr.RandomVariable)
def torch_funcify_RandomVariable(op: ptr.RandomVariable, node, **kwargs):
    rv = node.outputs[1]
    out_dtype = rv.type.dtype
    shape = rv.type.shape
    rv_sample = pytorch_sample_fn(op, node=node)

    def sample_fn(rng, size, *args):
        new_rng = torch.Generator(device="cpu")
        new_rng.set_state(rng.get_state().clone())
        return rv_sample(new_rng, shape, out_dtype, *args)

    return sample_fn


@singledispatch
def pytorch_sample_fn(op, node):
    name = op.name
    raise NotImplementedError(
        f"No PyTorch implementation for the given distribution: {name}"
    )


@pytorch_sample_fn.register(ptr.BernoulliRV)
def pytorch_sample_fn_bernoulli(op, node):
    def sample_fn(gen, size, dtype, p):
        sample = torch.bernoulli(torch.broadcast_to(p, size), generator=gen)
        return (gen, sample)

    return sample_fn


@pytorch_sample_fn.register(ptr.BinomialRV)
def pytorch_sample_fn_binomial(op, node):
    def sample_fn(gen, size, dtype, n, p):
        sample = torch.binomial(
            torch.broadcast_to(n, size).to(torch.float32),
            torch.broadcast_to(p, size).to(torch.float32),
            generator=gen,
        )
        return (gen, sample)

    return sample_fn


@pytorch_sample_fn.register(ptr.UniformRV)
def pytorch_sample_fn_uniform(op, node):
    def sample_fn(gen, size, dtype, low, high):
        sample = torch.FloatTensor(size)
        sample.uniform_(low.item(), high.item(), generator=gen)
        return (gen, sample)

    return sample_fn
