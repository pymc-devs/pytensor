from collections.abc import Callable
from copy import copy
from functools import singledispatch
from textwrap import dedent

import numba
import numba.np.unsafe.ndarray as numba_ndarray
import numpy as np
from numba import types
from numba.core.extending import overload

import pytensor.tensor.random.basic as ptr
from pytensor.graph import Apply
from pytensor.graph.op import Op
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import direct_cast, numba_funcify
from pytensor.link.numba.dispatch.vectorize_codegen import (
    _jit_options,
    _vectorized,
    encode_literals,
    store_core_outputs,
)
from pytensor.link.utils import (
    compile_function_src,
)
from pytensor.tensor import get_vector_length
from pytensor.tensor.random.op import RandomVariable, RandomVariableWithCoreShape
from pytensor.tensor.type_other import NoneTypeT
from pytensor.tensor.utils import _parse_gufunc_signature


@overload(copy)
def copy_NumPyRandomGenerator(rng):
    def impl(rng):
        # TODO: Open issue on Numba?
        with numba.objmode(new_rng=types.npy_rng):
            new_rng = copy(rng)

        return new_rng

    return impl


@singledispatch
def numba_core_rv_funcify(op: Op, node: Apply) -> Callable:
    """Return the core function for a random variable operation."""
    raise NotImplementedError(f"Core implementation of {op} not implemented.")


@numba_core_rv_funcify.register(ptr.UniformRV)
@numba_core_rv_funcify.register(ptr.TriangularRV)
@numba_core_rv_funcify.register(ptr.BetaRV)
@numba_core_rv_funcify.register(ptr.NormalRV)
@numba_core_rv_funcify.register(ptr.LogNormalRV)
@numba_core_rv_funcify.register(ptr.GammaRV)
@numba_core_rv_funcify.register(ptr.ExponentialRV)
@numba_core_rv_funcify.register(ptr.WeibullRV)
@numba_core_rv_funcify.register(ptr.LogisticRV)
@numba_core_rv_funcify.register(ptr.VonMisesRV)
@numba_core_rv_funcify.register(ptr.PoissonRV)
@numba_core_rv_funcify.register(ptr.GeometricRV)
# @numba_core_rv_funcify.register(ptr.HyperGeometricRV)  # Not implemented in numba
@numba_core_rv_funcify.register(ptr.WaldRV)
@numba_core_rv_funcify.register(ptr.LaplaceRV)
@numba_core_rv_funcify.register(ptr.BinomialRV)
@numba_core_rv_funcify.register(ptr.NegBinomialRV)
@numba_core_rv_funcify.register(ptr.MultinomialRV)
@numba_core_rv_funcify.register(ptr.PermutationRV)
@numba_core_rv_funcify.register(ptr.IntegersRV)
def numba_core_rv_default(op, node):
    """Create a default RV core numba function.

    @njit
    def random(rng, i0, i1, ..., in):
        return rng.name(i0, i1, ..., in)
    """
    name = op.name

    inputs = [f"i{i}" for i in range(len(op.ndims_params))]
    input_signature = ",".join(inputs)

    func_src = dedent(f"""
    def {name}(rng, {input_signature}):
        return rng.{name}({input_signature})
    """)

    func = compile_function_src(func_src, name, {**globals()})
    return numba_basic.numba_njit(func)


@numba_core_rv_funcify.register(ptr.BernoulliRV)
def numba_core_BernoulliRV(op, node):
    out_dtype = node.outputs[1].type.numpy_dtype

    @numba_basic.numba_njit()
    def random(rng, p):
        return (
            direct_cast(0, out_dtype)
            if p < rng.uniform()
            else direct_cast(1, out_dtype)
        )

    return random


@numba_core_rv_funcify.register(ptr.HalfNormalRV)
def numba_core_HalfNormalRV(op, node):
    @numba_basic.numba_njit
    def random_fn(rng, loc, scale):
        return loc + scale * np.abs(rng.standard_normal())

    return random_fn


@numba_core_rv_funcify.register(ptr.CauchyRV)
def numba_core_CauchyRV(op, node):
    @numba_basic.numba_njit
    def random(rng, loc, scale):
        return (loc + rng.standard_cauchy()) / scale

    return random


@numba_core_rv_funcify.register(ptr.ParetoRV)
def numba_core_ParetoRV(op, node):
    @numba_basic.numba_njit
    def random(rng, b, scale):
        # Follows scipy implementation
        U = rng.random()
        return np.power(1 - U, -1 / b) * scale

    return random


@numba_core_rv_funcify.register(ptr.CategoricalRV)
def core_CategoricalRV(op, node):
    @numba_basic.numba_njit
    def random_fn(rng, p):
        unif_sample = rng.uniform(0, 1)
        return np.searchsorted(np.cumsum(p), unif_sample)

    return random_fn


@numba_core_rv_funcify.register(ptr.MvNormalRV)
def core_MvNormalRV(op, node):
    @numba_basic.numba_njit
    def random_fn(rng, mean, cov):
        chol = np.linalg.cholesky(cov)
        stdnorm = rng.normal(size=cov.shape[-1])
        return np.dot(chol, stdnorm) + mean

    random_fn.handles_out = True
    return random_fn


@numba_core_rv_funcify.register(ptr.DirichletRV)
def core_DirichletRV(op, node):
    @numba_basic.numba_njit
    def random_fn(rng, alpha):
        y = np.empty_like(alpha)
        for i in range(len(alpha)):
            y[i] = rng.gamma(alpha[i], 1.0)
        return y / y.sum()

    return random_fn


@numba_core_rv_funcify.register(ptr.GumbelRV)
def core_GumbelRV(op, node):
    """Code adapted from Numpy Implementation

    https://github.com/numpy/numpy/blob/6f6be042c6208815b15b90ba87d04159bfa25fd3/numpy/random/src/distributions/distributions.c#L502-L511
    """

    @numba_basic.numba_njit
    def random_fn(rng, loc, scale):
        U = 1.0 - rng.random()
        if U < 1.0:
            return loc - scale * np.log(-np.log(U))
        else:
            return random_fn(rng, loc, scale)

    return random_fn


@numba_core_rv_funcify.register(ptr.VonMisesRV)
def core_VonMisesRV(op, node):
    """Code adapted from Numpy Implementation

    https://github.com/numpy/numpy/blob/6f6be042c6208815b15b90ba87d04159bfa25fd3/numpy/random/src/distributions/distributions.c#L855-L925
    """

    @numba_basic.numba_njit
    def random_fn(rng, mu, kappa):
        if np.isnan(kappa):
            return np.nan
        if kappa < 1e-8:
            # Use a uniform for very small values of kappa
            return np.pi * (2 * rng.random() - 1)
        else:
            # with double precision rho is zero until 1.4e-8
            if kappa < 1e-5:
                # second order taylor expansion around kappa = 0
                # precise until relatively large kappas as second order is 0
                s = 1.0 / kappa + kappa
            else:
                if kappa <= 1e6:
                    # Path for 1e-5 <= kappa <= 1e6
                    r = 1 + np.sqrt(1 + 4 * kappa * kappa)
                    rho = (r - np.sqrt(2 * r)) / (2 * kappa)
                    s = (1 + rho * rho) / (2 * rho)
                else:
                    # Fallback to wrapped normal distribution for kappa > 1e6
                    result = mu + np.sqrt(1.0 / kappa) * rng.standard_normal()
                    # Ensure result is within bounds
                    if result < -np.pi:
                        result += 2 * np.pi
                    if result > np.pi:
                        result -= 2 * np.pi
                    return result

            while True:
                U = rng.random()
                Z = np.cos(np.pi * U)
                W = (1 + s * Z) / (s + Z)
                Y = kappa * (s - W)
                V = rng.random()
                # V == 0.0 is ok here since Y >= 0 always leads
                # to accept, while Y < 0 always rejects
                if (Y * (2 - Y) - V >= 0) or (np.log(Y / V) + 1 - Y >= 0):
                    break

            U = rng.random()

            result = np.arccos(W)
            if U < 0.5:
                result = -result
            result += mu
            neg = result < 0
            mod = np.abs(result)
            mod = np.mod(mod + np.pi, 2 * np.pi) - np.pi
            if neg:
                mod *= -1

            return mod

    return random_fn


@numba_core_rv_funcify.register(ptr.ChoiceWithoutReplacement)
def core_ChoiceWithoutReplacement(op: ptr.ChoiceWithoutReplacement, node):
    [core_shape_len_sig] = _parse_gufunc_signature(op.signature)[0][-1]
    core_shape_len = int(core_shape_len_sig)
    implicit_arange = op.ndims_params[0] == 0

    if op.has_p_param:

        @numba_basic.numba_njit
        def random_fn(rng, a, p, core_shape):
            # Adapted from Numpy: https://github.com/numpy/numpy/blob/2a9b9134270371b43223fc848b753fceab96b4a5/numpy/random/_generator.pyx#L922-L941
            size = np.prod(core_shape)
            core_shape = numba_ndarray.to_fixed_tuple(core_shape, core_shape_len)
            if implicit_arange:
                pop_size = a
            else:
                pop_size = a.shape[0]

            if size > pop_size:
                raise ValueError(
                    "Cannot take a larger sample than population without replacement"
                )
            if np.count_nonzero(p > 0) < size:
                raise ValueError("Fewer non-zero entries in p than size")

            p = p.copy()
            n_uniq = 0
            idx = np.zeros(core_shape, dtype=np.int64)
            flat_idx = idx.ravel()
            while n_uniq < size:
                x = rng.random((size - n_uniq,))
                # Set the probabilities of items that have already been found to 0
                p[flat_idx[:n_uniq]] = 0
                # Take new (unique) categorical draws from the remaining probabilities
                cdf = np.cumsum(p)
                cdf /= cdf[-1]
                new = np.searchsorted(cdf, x, side="right")

                # Numba doesn't support return_index in np.unique
                # _, unique_indices = np.unique(new, return_index=True)
                # unique_indices.sort()
                new.sort()
                unique_indices = [
                    idx
                    for idx, prev_item in enumerate(new[:-1], 1)
                    if new[idx] != prev_item
                ]
                unique_indices = np.array([0] + unique_indices)  # noqa: RUF005

                new = new[unique_indices]
                flat_idx[n_uniq : n_uniq + new.size] = new
                n_uniq += new.size

            if implicit_arange:
                return idx
            else:
                # Numba doesn't support advanced indexing, so we ravel index and reshape
                return a[idx.ravel()].reshape(core_shape + a.shape[1:])

    else:

        @numba_basic.numba_njit
        def random_fn(rng, a, core_shape):
            # Until Numba supports generator.choice we use a poor implementation
            # that permutates the whole arange array and takes the first `size` elements
            # This is widely inefficient when size << a.shape[0]
            size = np.prod(core_shape)
            core_shape = numba_ndarray.to_fixed_tuple(core_shape, core_shape_len)
            idx = rng.permutation(size)[:size]

            # Numba doesn't support advanced indexing so index on the flat dimension and reshape
            # idx = idx.reshape(core_shape)
            # if implicit_arange:
            #     return idx
            # else:
            #     return a[idx]

            if implicit_arange:
                return idx.reshape(core_shape)
            else:
                return a[idx].reshape(core_shape + a.shape[1:])

    return random_fn


@numba_funcify.register
def numba_funcify_RandomVariable_core(op: RandomVariable, **kwargs):
    raise RuntimeError(
        "It is necessary to replace RandomVariable with RandomVariableWithCoreShape. "
        "This is done by the default rewrites during compilation."
    )


@numba_funcify.register
def numba_funcify_RandomVariable(op: RandomVariableWithCoreShape, node, **kwargs):
    core_shape = node.inputs[0]

    [rv_node] = op.fgraph.apply_nodes
    rv_op: RandomVariable = rv_node.op
    size = rv_op.size_param(rv_node)
    dist_params = rv_op.dist_params(rv_node)
    size_len = None if isinstance(size.type, NoneTypeT) else get_vector_length(size)
    core_shape_len = get_vector_length(core_shape)
    inplace = rv_op.inplace

    core_rv_fn = numba_core_rv_funcify(rv_op, rv_node)
    nin = 1 + len(dist_params)  # rng + params
    core_op_fn = store_core_outputs(core_rv_fn, nin=nin, nout=1)

    batch_ndim = rv_op.batch_ndim(rv_node)

    # numba doesn't support nested literals right now...
    input_bc_patterns = encode_literals(
        tuple(input_var.type.broadcastable[:batch_ndim] for input_var in dist_params)
    )
    output_bc_patterns = encode_literals(
        (rv_node.outputs[1].type.broadcastable[:batch_ndim],)
    )
    output_dtypes = encode_literals((rv_node.default_output().type.dtype,))
    inplace_pattern = encode_literals(())

    def random_wrapper(core_shape, rng, size, *dist_params):
        if not inplace:
            rng = copy(rng)

        draws = _vectorized(
            core_op_fn,
            input_bc_patterns,
            output_bc_patterns,
            output_dtypes,
            inplace_pattern,
            (rng,),
            dist_params,
            (numba_ndarray.to_fixed_tuple(core_shape, core_shape_len),),
            None if size_len is None else numba_ndarray.to_fixed_tuple(size, size_len),
        )
        return rng, draws

    def random(core_shape, rng, size, *dist_params):
        pass

    @overload(random, jit_options=_jit_options)
    def ov_random(core_shape, rng, size, *dist_params):
        return random_wrapper

    return random
