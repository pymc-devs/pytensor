from functools import singledispatch

import torch
from numpy.random import Generator

import pytensor.tensor.random.basic as ptr
from pytensor.graph import Constant
from pytensor.link.pytorch.dispatch.basic import pytorch_funcify, pytorch_typify
from pytensor.tensor.type_other import NoneTypeT


@pytorch_typify.register(Generator)
def pytorch_typify_Generator(rng, **kwargs):
    state = rng.__getstate__()
    state["pytorch_state"] = torch.manual_seed(123).get_state()  # XXX: replace
    return state


@pytorch_funcify.register(ptr.RandomVariable)
def torch_funcify_RandomVariable(op: ptr.RandomVariable, node, **kwargs):
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

    def sample_fn(rng, size, *parameters):
        return pytorch_sample_fn(op, node=node)(
            rng, static_size, out_dtype, *parameters
        )

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
        # XXX replace
        state_ = rng["pytorch_state"]
        gen = torch.Generator().set_state(state_)
        sample = torch.bernoulli(torch.expand_copy(p, size), generator=gen)
        return (rng, sample)

    return sample_fn
