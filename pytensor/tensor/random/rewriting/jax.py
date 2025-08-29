import re

from pytensor.compile import optdb
from pytensor.graph import Constant
from pytensor.graph.rewriting.basic import dfs_rewriter, in2out, node_rewriter
from pytensor.tensor import abs as abs_t
from pytensor.tensor import broadcast_arrays, exp, floor, log, log1p, reciprocal, sqrt
from pytensor.tensor.basic import (
    MakeVector,
    arange,
    cast,
    ones_like,
    switch,
    zeros_like,
)
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.random.basic import (
    BetaBinomialRV,
    ChoiceWithoutReplacement,
    GenGammaRV,
    GeometricRV,
    HalfNormalRV,
    InvGammaRV,
    LogNormalRV,
    NegBinomialRV,
    WaldRV,
    _gamma,
    beta,
    binomial,
    normal,
    poisson,
    uniform,
)
from pytensor.tensor.random.op import RandomVariable


@node_rewriter([RandomVariable])
def size_parameter_as_tuple(fgraph, node):
    """Replace `MakeVector` and `DimShuffle` (when used to transform a scalar
    into a 1d vector) when they are found as the input of a `size` or `shape`
    parameter by `JAXShapeTuple` during transpilation.

    The JAX implementations of `MakeVector` and `DimShuffle` always return JAX
    `TracedArrays`, but JAX only accepts concrete values as inputs for the `size`
    or `shape` parameter. When these `Op`s are used to convert scalar or tuple
    inputs, however, we can avoid tracing by making them return a tuple of their
    inputs instead.

    Note that JAX does not accept scalar inputs for the `size` or `shape`
    parameters, and this rewrite also ensures that scalar inputs are turned into
    tuples during transpilation.

    """
    from pytensor.link.jax.dispatch.shape import JAXShapeTuple

    size_arg = node.inputs[1]
    size_node = size_arg.owner

    if size_node is None:
        return

    if isinstance(size_node.op, JAXShapeTuple):
        return

    if isinstance(size_node.op, MakeVector) or (
        isinstance(size_node.op, DimShuffle)
        and size_node.op.input_ndim == 0
        and size_node.op.new_order == ("x",)
    ):
        # Here PyTensor converted a tuple or list to a tensor
        new_size_args = JAXShapeTuple()(*size_node.inputs)
        new_inputs = list(node.inputs)
        new_inputs[1] = new_size_args

        new_node = node.clone_with_new_inputs(new_inputs)
        return new_node.outputs


@node_rewriter([LogNormalRV])
def lognormal_from_normal(fgraph, node):
    next_rng, n = normal.make_node(*node.inputs).outputs
    return [next_rng, exp(n)]


@node_rewriter([HalfNormalRV])
def halfnormal_from_normal(fgraph, node):
    *other_inputs, loc, scale = node.inputs
    next_rng, n = normal.make_node(*other_inputs, zeros_like(loc), scale).outputs
    h = abs_t(n) + loc
    return [next_rng, cast(h, dtype=node.default_output().dtype)]


@node_rewriter([GeometricRV])
def geometric_from_uniform(fgraph, node):
    *other_inputs, p = node.inputs
    next_rng, u = uniform.make_node(*other_inputs, zeros_like(p), 1).outputs
    g = floor(log(u) / log1p(-p)) + 1
    return [next_rng, cast(g, dtype=node.default_output().dtype)]


@node_rewriter([NegBinomialRV])
def negative_binomial_from_gamma_poisson(fgraph, node):
    rng, *other_inputs, n, p = node.inputs
    next_rng, g = _gamma.make_node(rng, *other_inputs, n, (1 - p) / p).outputs
    next_rng, p = poisson.make_node(next_rng, *other_inputs, g).outputs
    return [next_rng, p]


@node_rewriter([InvGammaRV])
def inverse_gamma_from_gamma(fgraph, node):
    *other_inputs, shape, scale = node.inputs
    next_rng, g = _gamma.make_node(*other_inputs, shape, 1 / scale).outputs
    return [next_rng, reciprocal(g)]


@node_rewriter([GenGammaRV])
def generalized_gamma_from_gamma(fgraph, node):
    *other_inputs, alpha, p, lambd = node.inputs
    next_rng, g = _gamma.make_node(*other_inputs, alpha / p, ones_like(lambd)).outputs
    g = (g ** reciprocal(p)) * lambd
    return [next_rng, cast(g, dtype=node.default_output().dtype)]


@node_rewriter([WaldRV])
def wald_from_normal_uniform(fgraph, node):
    rng, *other_inputs, mean, scale = node.inputs
    next_rng, n = normal.make_node(
        rng, *other_inputs, zeros_like(mean), ones_like(scale)
    ).outputs
    next_rng, u = uniform.make_node(
        next_rng, *other_inputs, zeros_like(mean), ones_like(scale)
    ).outputs

    mu_2l = mean / (2 * scale)
    y = mean * n * n
    x = mean + mu_2l * (y - sqrt(4 * scale * y + y * y))
    w = switch(u <= mean / (mean + x), x, mean * mean / x)
    return [next_rng, cast(w, dtype=node.default_output().dtype)]


@node_rewriter([BetaBinomialRV])
def beta_binomial_from_beta_binomial(fgraph, node):
    rng, *other_inputs, n, a, b = node.inputs
    n, a, b = broadcast_arrays(n, a, b)
    next_rng, b = beta.make_node(rng, *other_inputs, a, b).outputs
    next_rng, b = binomial.make_node(next_rng, *other_inputs, n, b).outputs
    return [next_rng, b]


@node_rewriter([ChoiceWithoutReplacement])
def materialize_implicit_arange_choice_without_replacement(fgraph, node):
    """JAX random.choice does not support 0d arrays but when we have batch_ndim we need to vmap through batched `a`.

    This rewrite materializes the implicit `a`
    """
    op = node.op
    if op.batch_ndim(node) == 0 or op.ndims_params[0] > 0:
        # No need to materialize arange
        return None

    rng, size, a_scalar_param, *other_params = node.inputs
    if not all(a_scalar_param.type.broadcastable):
        # Automatic vectorization could have made this parameter batched,
        # there is no nice way to materialize a batched arange
        return None

    # We need to try and do an eager squeeze here because arange will fail in jax
    # if there is an array leading to it, even if it's constant
    if isinstance(a_scalar_param, Constant):
        a_scalar_param = a_scalar_param.data
    a_vector_param = arange(a_scalar_param.squeeze())

    new_props_dict = op._props_dict().copy()
    # Signature changes from something like "(),(a),(2)->(s0, s1)" to "(a),(a),(2)->(s0, s1)"
    # I.e., we substitute the first `()` by `(a)`
    new_props_dict["signature"] = re.sub(r"\(\)", "(a)", op.signature, count=1)
    new_op = type(op)(**new_props_dict)
    return new_op.make_node(rng, size, a_vector_param, *other_params).outputs


random_vars_opt = dfs_rewriter(
    lognormal_from_normal,
    halfnormal_from_normal,
    geometric_from_uniform,
    negative_binomial_from_gamma_poisson,
    inverse_gamma_from_gamma,
    generalized_gamma_from_gamma,
    wald_from_normal_uniform,
    beta_binomial_from_beta_binomial,
    materialize_implicit_arange_choice_without_replacement,
)
optdb.register("jax_random_vars_rewrites", random_vars_opt, "jax", position=110)

optdb.register(
    "jax_size_parameter_as_tuple", in2out(size_parameter_as_tuple), "jax", position=100
)
