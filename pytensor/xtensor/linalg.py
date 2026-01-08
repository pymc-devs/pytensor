from collections.abc import Sequence

from pytensor.tensor.slinalg import Cholesky, Solve
from pytensor.xtensor.type import as_xtensor
from pytensor.xtensor.vectorization import XBlockwise


def cholesky(
    x,
    lower: bool = True,
    *,
    check_finite: bool = True,
    dims: Sequence[str],
):
    """Compute the Cholesky decomposition of an XTensorVariable.

    Parameters
    ----------
    x : XTensorVariable
        The input variable to decompose.
    lower : bool, optional
        Whether to return the lower triangular matrix. Default is True.
    check_finite : bool
        Unused by PyTensor. PyTensor will return nan if the operation fails.
    dims : Sequence[str]
        The two core dimensions of the input variable, over which the Cholesky decomposition is computed.
    """
    if len(dims) != 2:
        raise ValueError(f"Cholesky needs two dims, got {len(dims)}")

    core_op = Cholesky(lower=lower)
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
    check_finite: bool = True,
):
    """Solve a system of linear equations using XTensorVariables.

    Parameters
    ----------
    a : XTensorVariable
        The left hand-side xtensor.
    b : XTensorVariable
        The right-hand side xtensor.
    dims : Sequence[str]
        The core dimensions over which to solve the linear equations.
        If length is 2, we are solving a matrix-vector equation,
        and the two dimensions should be present in `a`, but only one in `b`.
        If length is 3, we are solving a matrix-matrix equation,
        and two dimensions should be present in `a`, two in `b`, and only one should be shared.
        In both cases the shared dimension will not appear in the output.
    assume_a : str, optional
        The type of matrix `a` is assumed to be. Default is 'gen' (general).
        Options are ["gen", "sym", "her", "pos", "tridiagonal", "banded"].
        Long form options can also be used ["general", "symmetric", "hermitian", "positive_definite"].
    lower : bool, optional
        Whether `a` is lower triangular. Default is False. Only relevant if `assume_a` is "sym", "her", or "pos".
    check_finite : bool
        Unused by PyTensor. PyTensor will return nan if the operation fails.
    """
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

    core_op = Solve(b_ndim=b_ndim, assume_a=assume_a, lower=lower)
    x_op = XBlockwise(
        core_op,
        core_dims=(input_core_dims, output_core_dims),
    )
    return x_op(a, b)
