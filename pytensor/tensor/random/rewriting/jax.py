from pytensor.compile import optdb
from pytensor.graph.rewriting.basic import in2out, node_rewriter
from pytensor.graph.rewriting.db import SequenceDB
from pytensor.tensor import abs as abs_t
from pytensor.tensor import broadcast_arrays, exp, floor, log, log1p, reciprocal, sqrt
from pytensor.tensor.basic import MakeVector, cast, ones_like, switch, zeros_like
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.random.basic import (
    BetaBinomialRV,
    ChiSquareRV,
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
        and size_node.op.input_broadcastable == ()
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


@node_rewriter([ChiSquareRV])
def chi_square_from_gamma(fgraph, node):
    *other_inputs, df = node.inputs
    next_rng, g = _gamma.make_node(*other_inputs, df / 2, 2).outputs
    return [next_rng, g]


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


random_vars_opt = SequenceDB()
random_vars_opt.register(
    "lognormal_from_normal",
    in2out(lognormal_from_normal),
    "jax",
)
random_vars_opt.register(
    "halfnormal_from_normal",
    in2out(halfnormal_from_normal),
    "jax",
)
random_vars_opt.register(
    "geometric_from_uniform",
    in2out(geometric_from_uniform),
    "jax",
)
random_vars_opt.register(
    "negative_binomial_from_gamma_poisson",
    in2out(negative_binomial_from_gamma_poisson),
    "jax",
)
random_vars_opt.register(
    "inverse_gamma_from_gamma",
    in2out(inverse_gamma_from_gamma),
    "jax",
)
random_vars_opt.register(
    "chi_square_from_gamma",
    in2out(chi_square_from_gamma),
    "jax",
)
random_vars_opt.register(
    "generalized_gamma_from_gamma",
    in2out(generalized_gamma_from_gamma),
    "jax",
)
random_vars_opt.register(
    "wald_from_normal_uniform",
    in2out(wald_from_normal_uniform),
    "jax",
)
random_vars_opt.register(
    "beta_binomial_from_beta_binomial",
    in2out(beta_binomial_from_beta_binomial),
    "jax",
)
optdb.register("jax_random_vars_rewrites", random_vars_opt, "jax", position=110)

optdb.register(
    "jax_size_parameter_as_tuple", in2out(size_parameter_as_tuple), "jax", position=100
)
