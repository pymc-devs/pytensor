"""XRV constructor helpers for building xtensor random variable expressions."""

import warnings
from collections.abc import Sequence
from typing import Literal

import pytensor.tensor.random.basic as ptrb
from pytensor.graph.basic import Variable
from pytensor.tensor.random.op import RandomVariable
from pytensor.xtensor.type import as_xtensor
from pytensor.xtensor.vectorization import XRV


__all__ = [
    "as_xrv",
    "bernoulli",
    "beta",
    "betabinom",
    "binomial",
    "categorical",
    "cauchy",
    "chisquare",
    "dirichlet",
    "exponential",
    "gamma",
    "gengamma",
    "geometric",
    "gumbel",
    "halfcauchy",
    "halfnormal",
    "hypergeometric",
    "integers",
    "invgamma",
    "laplace",
    "logistic",
    "lognormal",
    "multinomial",
    "multivariate_normal",
    "nbinom",
    "negative_binomial",
    "normal",
    "pareto",
    "poisson",
    "rayleigh",
    "standard_normal",
    "t",
    "triangular",
    "truncexpon",
    "uniform",
    "vonmises",
    "wald",
    "weibull",
]


def as_xrv(
    core_op: RandomVariable,
    core_inps_dims_map: Sequence[Sequence[int]] | None = None,
    core_out_dims_map: Sequence[int] | None = None,
    name: str | None = None,
):
    """Create an XRV constructor function for a given core RandomVariable op.

    Parameters
    ----------
    core_op : RandomVariable
        The core random variable operation to wrap.
    core_inps_dims_map : Sequence[Sequence[int]] | None
        Mapping of core dimensions for each input parameter.
        If None, assumes positional left-to-right.
    core_out_dims_map : Sequence[int] | None
        Mapping of core dimensions for the output.
        If None, assumes positional left-to-right.
    name : str | None
        Display name for the XRV op.
    """
    if core_inps_dims_map is None:
        core_inps_dims_map = [tuple(range(ndim)) for ndim in core_op.ndims_params]
    if core_out_dims_map is None:
        core_out_dims_map = tuple(range(core_op.ndim_supp))

    core_dims_needed = max(
        max(
            (
                max((entry + 1 for entry in dims_map), default=0)
                for dims_map in core_inps_dims_map
            ),
            default=0,
        ),
        max((entry + 1 for entry in core_out_dims_map), default=0),
    )

    def xrv_constructor(
        *params,
        core_dims: Sequence[str] | str | None = None,
        extra_dims: dict[str, Variable] | None = None,
        rng: Variable | None = None,
        return_next_rng: bool = False,
    ):
        if core_dims is None:
            core_dims_tuple: tuple[str, ...] = ()
            if core_dims_needed:
                raise ValueError(
                    f"{core_op.name} needs {core_dims_needed} core_dims to be specified"
                )
        elif isinstance(core_dims, str):
            core_dims_tuple = (core_dims,)
        else:
            core_dims_tuple = tuple(core_dims)

        if len(core_dims_tuple) != core_dims_needed:
            raise ValueError(
                f"{core_op.name} needs {core_dims_needed} core_dims, but got {len(core_dims_tuple)}"
            )

        full_input_core_dims = tuple(
            tuple(core_dims_tuple[i] for i in inp_dims_map)
            for inp_dims_map in core_inps_dims_map
        )
        full_output_core_dims = tuple(core_dims_tuple[i] for i in core_out_dims_map)
        full_core_dims = (full_input_core_dims, full_output_core_dims)

        if extra_dims is None:
            extra_dims = {}

        if not return_next_rng:
            warnings.warn(
                "XRV Ops will stop hiding the rng output in a future version. "
                "Set return_next_rng=True to suppress this warning.",
                DeprecationWarning,
                stacklevel=2,
            )

        node = XRV(
            core_op,
            core_dims=full_core_dims,
            extra_dims=tuple(extra_dims.keys()),
            name=name,
        ).make_node(rng, *extra_dims.values(), *params)
        out = node.default_output()
        if return_next_rng:
            next_rng = node.op.update(node)[node.op.rng_param(node)]
            return next_rng, out
        return out

    return xrv_constructor


def multivariate_normal(
    mean,
    cov,
    *,
    core_dims: Sequence[str],
    extra_dims=None,
    rng=None,
    method: Literal["cholesky", "svd", "eigh"] = "cholesky",
    return_next_rng: bool = False,
):
    """Multivariate normal random variable for xtensors."""
    mean = as_xtensor(mean)
    if len(core_dims) != 2:
        raise ValueError(
            f"multivariate_normal requires 2 core_dims, got {len(core_dims)}"
        )

    # Align core_dims so the dim in mean comes first (output core dim)
    if core_dims[0] not in mean.type.dims:
        core_dims = core_dims[::-1]

    xop = as_xrv(ptrb.MvNormalRV(method=method))
    return xop(
        mean,
        cov,
        core_dims=core_dims,
        extra_dims=extra_dims,
        rng=rng,
        return_next_rng=return_next_rng,
    )


# Named distribution constructors (functional API)
bernoulli = as_xrv(ptrb.bernoulli)
beta = as_xrv(ptrb.beta)
betabinom = as_xrv(ptrb.betabinom)
binomial = as_xrv(ptrb.binomial)
categorical = as_xrv(ptrb.categorical)
cauchy = as_xrv(ptrb.cauchy)
dirichlet = as_xrv(ptrb.dirichlet)
exponential = as_xrv(ptrb.exponential)
gamma = as_xrv(ptrb._gamma)
gengamma = as_xrv(ptrb.gengamma)
geometric = as_xrv(ptrb.geometric)
gumbel = as_xrv(ptrb.gumbel)
halfcauchy = as_xrv(ptrb.halfcauchy)
halfnormal = as_xrv(ptrb.halfnormal)
hypergeometric = as_xrv(ptrb.hypergeometric)
integers = as_xrv(ptrb.integers)
invgamma = as_xrv(ptrb.invgamma)
laplace = as_xrv(ptrb.laplace)
logistic = as_xrv(ptrb.logistic)
lognormal = as_xrv(ptrb.lognormal)
multinomial = as_xrv(ptrb.multinomial)
negative_binomial = as_xrv(ptrb.negative_binomial)
nbinom = negative_binomial
normal = as_xrv(ptrb.normal)
pareto = as_xrv(ptrb.pareto)
poisson = as_xrv(ptrb.poisson)
t = as_xrv(ptrb.t)
triangular = as_xrv(ptrb.triangular)
truncexpon = as_xrv(ptrb.truncexpon)
uniform = as_xrv(ptrb.uniform)
vonmises = as_xrv(ptrb.vonmises)
wald = as_xrv(ptrb.wald)
weibull = as_xrv(ptrb.weibull)


def standard_normal(
    extra_dims=None,
    rng=None,
    return_next_rng=False,
):
    return normal(0, 1, extra_dims=extra_dims, rng=rng, return_next_rng=return_next_rng)


def chisquare(
    df,
    extra_dims=None,
    rng=None,
    return_next_rng=False,
):
    return gamma(
        df / 2.0, 2.0, extra_dims=extra_dims, rng=rng, return_next_rng=return_next_rng
    )


def rayleigh(
    scale,
    extra_dims=None,
    rng=None,
    return_next_rng=False,
):
    from pytensor.xtensor.math import sqrt

    df = scale * 0 + 2
    next_rng, chisquare_draws = chisquare(
        df, extra_dims=extra_dims, rng=rng, return_next_rng=True
    )
    rayleigh_draws = sqrt(chisquare_draws) * scale
    if return_next_rng:
        return next_rng, rayleigh_draws
    return rayleigh_draws
