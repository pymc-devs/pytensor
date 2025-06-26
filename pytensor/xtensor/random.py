from collections.abc import Sequence
from functools import wraps
from typing import Literal

import pytensor.tensor.random.basic as ptr
from pytensor.graph.basic import Variable
from pytensor.tensor.random.op import RandomVariable
from pytensor.xtensor.math import sqrt
from pytensor.xtensor.type import as_xtensor
from pytensor.xtensor.vectorization import XRV


def _as_xrv(
    core_op: RandomVariable,
    core_inps_dims_map: Sequence[Sequence[int]] | None = None,
    core_out_dims_map: Sequence[int] | None = None,
    name: str | None = None,
):
    """Helper function to define an XRV constructor.

    Parameters
    ----------
    core_op : RandomVariable
        The core random variable operation to wrap.
    core_inps_dims_map : Sequence[Sequence[int]] | None, optional
        A sequence of sequences mapping the core dimensions (specified by the user)
        for each input parameter. This is used when lowering to a RandomVariable operation,
        to decide the ordering of the core dimensions for each input.
        If None, it assumes the core dimensions are positional from left to right.
    core_out_dims_map : Sequence[int] | None, optional
        A sequence mapping the core dimensions (specified by the user) for the output variable.
        This is used when lowering to a RandomVariable operation,
        to decide the ordering of the core dimensions for the output.
        If None, it assumes the core dimensions are positional from left to right.

    """
    if core_inps_dims_map is None:
        # Assume core_dims map positionally from left to right
        core_inps_dims_map = [tuple(range(ndim)) for ndim in core_op.ndims_params]
    if core_out_dims_map is None:
        # Assume core_dims map positionally from left to right
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

    @wraps(core_op)
    def xrv_constructor(
        *params,
        core_dims: Sequence[str] | str | None = None,
        extra_dims: dict[str, Variable] | None = None,
        rng: Variable | None = None,
    ):
        if core_dims is None:
            core_dims = ()
            if core_dims_needed:
                raise ValueError(
                    f"{core_op.name} needs {core_dims_needed} core_dims to be specified"
                )
        elif isinstance(core_dims, str):
            core_dims = (core_dims,)

        if len(core_dims) != core_dims_needed:
            raise ValueError(
                f"{core_op.name} needs {core_dims_needed} core_dims, but got {len(core_dims)}"
            )

        full_input_core_dims = tuple(
            tuple(core_dims[i] for i in inp_dims_map)
            for inp_dims_map in core_inps_dims_map
        )
        full_output_core_dims = tuple(core_dims[i] for i in core_out_dims_map)
        full_core_dims = (full_input_core_dims, full_output_core_dims)

        if extra_dims is None:
            extra_dims = {}

        return XRV(
            core_op,
            core_dims=full_core_dims,
            extra_dims=tuple(extra_dims.keys()),
            name=name,
        )(rng, *extra_dims.values(), *params)

    return xrv_constructor


bernoulli = _as_xrv(ptr.bernoulli)
beta = _as_xrv(ptr.beta)
betabinom = _as_xrv(ptr.betabinom)
binomial = _as_xrv(ptr.binomial)
categorical = _as_xrv(ptr.categorical)
cauchy = _as_xrv(ptr.cauchy)
dirichlet = _as_xrv(ptr.dirichlet)
exponential = _as_xrv(ptr.exponential)
gamma = _as_xrv(ptr._gamma)
gengamma = _as_xrv(ptr.gengamma)
geometric = _as_xrv(ptr.geometric)
gumbel = _as_xrv(ptr.gumbel)
halfcauchy = _as_xrv(ptr.halfcauchy)
halfnormal = _as_xrv(ptr.halfnormal)
hypergeometric = _as_xrv(ptr.hypergeometric)
integers = _as_xrv(ptr.integers)
invgamma = _as_xrv(ptr.invgamma)
laplace = _as_xrv(ptr.laplace)
logistic = _as_xrv(ptr.logistic)
lognormal = _as_xrv(ptr.lognormal)
multinomial = _as_xrv(ptr.multinomial)
nbinom = negative_binomial = _as_xrv(ptr.negative_binomial)
normal = _as_xrv(ptr.normal)
pareto = _as_xrv(ptr.pareto)
poisson = _as_xrv(ptr.poisson)
t = _as_xrv(ptr.t)
triangular = _as_xrv(ptr.triangular)
truncexpon = _as_xrv(ptr.truncexpon)
uniform = _as_xrv(ptr.uniform)
vonmises = _as_xrv(ptr.vonmises)
wald = _as_xrv(ptr.wald)
weibull = _as_xrv(ptr.weibull)


def multivariate_normal(
    mean,
    cov,
    *,
    core_dims: Sequence[str],
    extra_dims=None,
    rng=None,
    method: Literal["cholesky", "svd", "eigh"] = "cholesky",
):
    mean = as_xtensor(mean)
    if len(core_dims) != 2:
        raise ValueError(
            f"multivariate_normal requires 2 core_dims, got {len(core_dims)}"
        )

    # Align core_dims, so that the dim that exists in mean comes before the one that only exists in cov
    # This will be the core dimension of the output
    if core_dims[0] not in mean.type.dims:
        core_dims = core_dims[::-1]

    xop = _as_xrv(ptr.MvNormalRV(method=method))
    return xop(mean, cov, core_dims=core_dims, extra_dims=extra_dims, rng=rng)


def standard_normal(
    extra_dims: dict[str, Variable] | None = None,
    rng: Variable | None = None,
):
    """Standard normal random variable."""
    return normal(0, 1, extra_dims=extra_dims, rng=rng)


def chisquare(
    df,
    extra_dims: dict[str, Variable] | None = None,
    rng: Variable | None = None,
):
    """Chi-square random variable."""
    return gamma(df / 2.0, 2.0, extra_dims=extra_dims, rng=rng)


def rayleigh(
    scale,
    extra_dims: dict[str, Variable] | None = None,
    rng: Variable | None = None,
):
    """Rayleigh random variable."""

    df = scale * 0 + 2  # Poor man's broadcasting, to pass dimensions of scale to the RV
    return sqrt(chisquare(df, extra_dims=extra_dims, rng=rng)) * scale
