from collections.abc import Sequence
from typing import Literal

from pytensor.tensor.slinalg import Cholesky, Solve
from pytensor.xtensor.type import as_xtensor
from pytensor.xtensor.vectorization import XBlockwise


def cholesky(
    x,
    lower: bool = True,
    *,
    check_finite: bool = False,
    overwrite_a: bool = False,
    on_error: Literal["raise", "nan"] = "raise",
    dims: Sequence[str],
):
    if len(dims) != 2:
        raise ValueError(f"Cholesky needs two dims, got {len(dims)}")

    core_op = Cholesky(
        lower=lower,
        check_finite=check_finite,
        overwrite_a=overwrite_a,
        on_error=on_error,
    )
    core_dims = (
        ((dims[0], dims[1]),),
        ((dims[0], dims[1]),),
    )
    x_op = XBlockwise(core_op, core_dims=core_dims)
    return x_op(x)


def solve(
    a,
    b,
    dims: Sequence[str],
    assume_a="gen",
    lower: bool = False,
    check_finite: bool = False,
):
    a, b = as_xtensor(a), as_xtensor(b)
    input_core_dims: tuple[tuple[str, str], tuple[str] | tuple[str, str]]
    output_core_dims: tuple[tuple[str] | tuple[str, str]]
    if len(dims) == 2:
        b_ndim = 1
        [m1_dim] = [dim for dim in dims if dim not in b.type.dims]
        m2_dim = dims[0] if dims[0] != m1_dim else dims[1]
        input_core_dims = ((m1_dim, m2_dim), (m2_dim,))
        # The shared dim disappears in the output
        output_core_dims = ((m1_dim,),)
    elif len(dims) == 3:
        b_ndim = 2
        [n_dim] = [dim for dim in dims if dim not in a.type.dims]
        [m1_dim, m2_dim] = [dim for dim in dims if dim != n_dim]
        input_core_dims = ((m1_dim, m2_dim), (m2_dim, n_dim))
        # The shared dim disappears in the output
        output_core_dims = ((m1_dim, n_dim),)
    else:
        raise ValueError("Solve dims must have length 2 or 3")

    core_op = Solve(
        b_ndim=b_ndim, assume_a=assume_a, lower=lower, check_finite=check_finite
    )
    x_op = XBlockwise(
        core_op,
        core_dims=(input_core_dims, output_core_dims),
    )
    return x_op(a, b)
