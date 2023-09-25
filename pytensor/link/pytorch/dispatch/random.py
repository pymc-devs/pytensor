import torch
from functools import singledispatch

import numpy as np
from numpy.random import Generator, RandomState
from numpy.random.bit_generator import (  # type: ignore[attr-defined]
    _coerce_to_uint32_array,
)

import pytensor.tensor.random.basic as aer
from pytensor.link.pytorch.dispatch.basic import pytorch_funcify, pytorch_typify
from pytensor.link.pytorch.dispatch.shape import PyTorchShapeTuple
from pytensor.tensor.shape import Shape, Shape_i

numpy_bit_gens = {"MT19937": 0, "PCG64": 1, "Philox": 2, "SFC64": 3}

SIZE_NOT_COMPATIBLE = """PyTorch random variables require concrete values for the `size` parameter of the distributions.
Concrete values are either constants:

>>> import pytensor.tensor as at
>>> x_rv = at.random.normal(0, 1, size=(3, 2))

or the shape of an array:

>>> m = at.matrix()
>>> x_rv = at.random.normal(0, 1, size=m.shape)
"""

def assert_size_argument_pytorch_compatible(node):
    """Assert whether the current node can be JIT-compiled by PyTorch.

    PyTorch can JIT-compile `torch.random` functions when the `size` argument
    is a concrete value, i.e. either a constant or the shape of any
    traced value.

    """
    size = node.inputs[1]
    size_node = size.owner
    if (size_node is not None) and (
        not isinstance(size_node.op, (Shape, Shape_i, PyTorchShapeTuple))
    ):
        raise NotImplementedError(SIZE_NOT_COMPATIBLE)

@pytorch_typify.register(RandomState)
def pytorch_typify_RandomState(state, **kwargs):
    state = state.get_state(legacy=False)
    state["bit_generator"] = numpy_bit_gens[state["bit_generator"]]
    # XXX: Is this a reasonable approach?
    state["pytorch_state"] = state["state"]["key"][0:2]
    return state

@pytorch_typify.register(Generator)
def pytorch_typify_Generator(rng, **kwargs):
    state = rng.__getstate__()
    state["bit_generator"] = numpy_bit_gens[state["bit_generator"]]

    # XXX: Is this a reasonable approach?
    state["pytorch_state"] = _coerce_to_uint32_array(state["state"]["state"])[0:2]

    # The "state" and "inc" values in a NumPy `Generator` are 128 bits, which
    # PyTorch can't handle, so we split these values into arrays of 32 bit integers
    # and then combine the first two into a single 64 bit integers.
    #
    # XXX: Depending on how we expect these values to be used, is this approach
    # reasonable?
    #
    # TODO: We might as well remove these altogether, since this conversion
    # should only occur once (e.g. when the graph is converted/PyTorch-compiled),
    # and, from then on, we use the custom "pytorch_state" value.
    inc_32 = _coerce_to_uint32_array(state["state"]["inc"])
    state_32 = _coerce_to_uint32_array(state["state"]["state"])
    state["state"]["inc"] = inc_32[0] << 32 | inc_32[1]
    state["state"]["state"] = state_32[0] << 32 | state_32[1]
    return state

@pytorch_funcify.register(aer.RandomVariable)
def pytorch_funcify_RandomVariable(op, node, **kwargs):
    """PyTorch implementation of random variables."""
    rv = node.outputs[1]
    out_dtype = rv.type.dtype
    out_size = rv.type.shape

    if op.ndim_supp > 0:
        out_size = node.outputs[1].type.shape[: -op.ndim_supp]

    # If one dimension has unknown size, either the size is determined
    # by a `Shape` operator in which case PyTorch will compile, or it is
    # not and we fail gracefully.
    if None in out_size:
        assert_size_argument_pytorch_compatible(node)

        def sample_fn(rng, size, dtype, *parameters):
            return pytorch_sample_fn(op)(rng, size, out_dtype, *parameters)

    else:

        def sample_fn(rng, size, dtype, *parameters):
            return pytorch_sample_fn(op)(rng, out_size, out_dtype, *parameters)

    return sample_fn

@singledispatch
def pytorch_sample_fn(op):
    name = op.name
    raise NotImplementedError(
        f"No PyTorch implementation for the given distribution: {name}"
    )

@pytorch_sample_fn.register(aer.BetaRV)
@pytorch_sample_fn.register(aer.DirichletRV)
@pytorch_sample_fn.register(aer.PoissonRV)
@pytorch_sample_fn.register(aer.MvNormalRV)
def pytorch_sample_fn_generic(op):
    """Generic PyTorch implementation of random variables."""
    name = op.name
    pytorch_op = getattr(torch.distributions, name)

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["pytorch_state"]
        sample = pytorch_op(*parameters).sample(size)
        rng["pytorch_state"] = rng_key
        return (rng, sample)

    return sample_fn

@pytorch_sample_fn.register(aer.CauchyRV)
@pytorch_sample_fn.register(aer.GumbelRV)
@pytorch_sample_fn.register(aer.LaplaceRV)
@pytorch_sample_fn.register(aer.LogisticRV)
@pytorch_sample_fn.register(aer.NormalRV)
@pytorch_sample_fn.register(aer.StandardNormalRV)
def pytorch_sample_fn_loc_scale(op):
    """PyTorch implementation of random variables in the loc-scale families.

    PyTorch only implements the standard version of random variables in the
    loc-scale family. We thus need to translate and rescale the results
    manually.

    """
    name = op.name
    pytorch_op = getattr(torch.distributions, name)

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["pytorch_state"]
        loc, scale = parameters
        sample = loc + pytorch_op(*parameters).sample(size) * scale
        rng["pytorch_state"] = rng_key
        return (rng, sample)

    return sample_fn

@pytorch_sample_fn.register(aer.BernoulliRV)
@pytorch_sample_fn.register(aer.CategoricalRV)
def pytorch_sample_fn_no_dtype(op):
    """Generic PyTorch implementation of random variables."""
    name = op.name
    pytorch_op = getattr(torch.distributions, name)

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["pytorch_state"]
        sample = pytorch_op(*parameters).sample(size)
        rng["pytorch_state"] = rng_key
        return (rng, sample)

    return sample_fn

@pytorch_sample_fn.register(aer.RandIntRV)
@pytorch_sample_fn.register(aer.IntegersRV)
@pytorch_sample_fn.register(aer.UniformRV)
def pytorch_sample_fn_uniform(op):
    """PyTorch implementation of random variables with uniform density.

    We need to pass the arguments as keyword arguments since the order
    of arguments is not the same.

    """
    name = op.name
    # IntegersRV is equivalent to RandintRV
    if isinstance(op, aer.IntegersRV):
        name = "randint"
    pytorch_op = getattr(torch.distributions, name)

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["pytorch_state"]
        minval, maxval = parameters
        sample = pytorch_op(*parameters).sample(size)
        rng["pytorch_state"] = rng_key
        return (rng, sample)

    return sample_fn

@pytorch_sample_fn.register(aer.ParetoRV)
@pytorch_sample_fn.register(aer.GammaRV)
def pytorch_sample_fn_shape_rate(op):
    """PyTorch implementation of random variables in the shape-rate family.

    PyTorch only implements the standard version of random variables in the
    shape-rate family. We thus need to rescale the results manually.

    """
    name = op.name
    pytorch_op = getattr(torch.distributions, name)

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["pytorch_state"]
        (shape, rate) = parameters
        sample = pytorch_op(*parameters).sample(size) / rate
        rng["pytorch_state"] = rng_key
        return (rng, sample)

    return sample_fn

@pytorch_sample_fn.register(aer.ExponentialRV)
def pytorch_sample_fn_exponential(op):
    """PyTorch implementation of `ExponentialRV`."""

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["pytorch_state"]
        (scale,) = parameters
        sample = torch.distributions.Exponential(scale).sample(size)
        rng["pytorch_state"] = rng_key
        return (rng, sample)

    return sample_fn

@pytorch_sample_fn.register(aer.StudentTRV)
def pytorch_sample_fn_t(op):
    """PyTorch implementation of `StudentTRV`."""

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["pytorch_state"]
        (
            df,
            loc,
            scale,
        ) = parameters
        sample = loc + torch.distributions.StudentT(df).sample(size) * scale
        rng["pytorch_state"] = rng_key
        return (rng, sample)

    return sample_fn

@pytorch_sample_fn.register(aer.ChoiceRV)
def pytorch_funcify_choice(op):
    """PyTorch implementation of `ChoiceRV`."""

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["pytorch_state"]
        (a, p, replace) = parameters
        smpl_value = torch.multinomial(p, size, replacement=replace)
        rng["pytorch_state"] = rng_key
        return (rng, smpl_value)

    return sample_fn

@pytorch_sample_fn.register(aer.PermutationRV)
def pytorch_sample_fn_permutation(op):
    """PyTorch implementation of `PermutationRV`."""

    def sample_fn(rng, size, dtype, *parameters):
        rng_key = rng["pytorch_state"]
        (x,) = parameters
        sample = torch.randperm(x)
        rng["pytorch_state"] = rng_key
        return (rng, sample)

    return sample_fn

@pytorch_sample_fn.register(aer.BinomialRV)
def pytorch_sample_fn_binomial(op):
    def sample_fn(rng, size, dtype, n, p):
        rng_key = rng["pytorch_state"]
        sample = torch.distributions.Binomial(n, p).sample(size)
        rng["pytorch_state"] = rng_key
        return (rng, sample)

    return sample_fn

@pytorch_sample_fn.register(aer.MultinomialRV)
def pytorch_sample_fn_multinomial(op):
    def sample_fn(rng, size, dtype, n, p):
        rng_key = rng["pytorch_state"]
        sample = torch.distributions.Multinomial(n, p).sample(size)
        rng["pytorch_state"] = rng_key
        return (rng, sample)

    return sample_fn

@pytorch_sample_fn.register(aer.VonMisesRV)
def pytorch_sample_fn_vonmises(op):
    def sample_fn(rng, size, dtype, mu, kappa):
        rng_key = rng["pytorch_state"]
        sample = torch.distributions.VonMises(mu, kappa).sample(size)
        rng["pytorch_state"] = rng_key
        return (rng, sample)

    return sample_fn