import warnings
from functools import partial

import numpy as np
import scipy.linalg as scipy_linalg

from pytensor import tensor as pt
from pytensor.graph.op import Op
from pytensor.tensor.basic import diagonal
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg.decomposition.lu import pivot_to_permutation
from pytensor.tensor.linalg.solvers.core import SolveBase, _default_b_ndim
from pytensor.tensor.linalg.solvers.triangular import solve_triangular
from pytensor.tensor.variable import TensorVariable


class Solve(SolveBase):
    """
    Solve a system of linear equations.
    """

    __props__ = (
        "assume_a",
        "lower",
        "b_ndim",
        "overwrite_a",
        "overwrite_b",
    )

    def __init__(self, *, assume_a="gen", **kwargs):
        # Triangular and diagonal are handled outside of Solve
        valid_options = ["gen", "sym", "her", "pos", "tridiagonal", "banded"]

        assume_a = assume_a.lower()
        # We use the old names as the different dispatches are more likely to support them
        long_to_short = {
            "general": "gen",
            "symmetric": "sym",
            "hermitian": "her",
            "positive definite": "pos",
        }
        assume_a = long_to_short.get(assume_a, assume_a)

        if assume_a not in valid_options:
            raise ValueError(
                f"Invalid assume_a: {assume_a}. It must be one of {valid_options} or {list(long_to_short.keys())}"
            )

        if assume_a in ("tridiagonal", "banded"):
            from scipy import __version__ as sp_version

            if tuple(map(int, sp_version.split(".")[:-1])) < (1, 15):
                warnings.warn(
                    f"assume_a={assume_a} requires scipy>=1.5.0. Defaulting to assume_a='gen'.",
                    UserWarning,
                )
                assume_a = "gen"

        super().__init__(**kwargs)
        self.assume_a = assume_a

    def perform(self, node, inputs, outputs):
        a, b = inputs
        try:
            outputs[0][0] = scipy_linalg.solve(
                a=a,
                b=b,
                lower=self.lower,
                check_finite=False,
                assume_a=self.assume_a,
                overwrite_a=self.overwrite_a,
                overwrite_b=self.overwrite_b,
            )
        except np.linalg.LinAlgError:
            outputs[0][0] = np.full(a.shape, np.nan, dtype=a.dtype)

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        if not allowed_inplace_inputs:
            return self
        new_props = self._props_dict()  # type: ignore
        # PyTensor doesn't allow an output to destroy two inputs yet
        # new_props["overwrite_a"] = 0 in allowed_inplace_inputs
        # new_props["overwrite_b"] = 1 in allowed_inplace_inputs
        if 1 in allowed_inplace_inputs:
            # Give preference to overwrite_b
            new_props["overwrite_b"] = True
        # We can't overwrite_a if we're assuming tridiagonal
        elif not self.assume_a == "tridiagonal":  # allowed inputs == [0]
            new_props["overwrite_a"] = True
        return type(self)(**new_props)


def solve(
    a,
    b,
    *,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: str = "gen",
    transposed: bool = False,
    b_ndim: int | None = None,
):
    """Solves the linear equation set ``a * x = b`` for the unknown ``x`` for square ``a`` matrix.

    If the data matrix is known to be a particular type then supplying the
    corresponding string to ``assume_a`` key chooses the dedicated solver.
    The available options are

    ===================  ================================
     diagonal             'diagonal'
     tridiagonal          'tridiagonal'
     banded               'banded'
     upper triangular     'upper triangular'
     lower triangular     'lower triangular'
     symmetric            'symmetric' (or 'sym')
     hermitian            'hermitian' (or 'her')
     positive definite    'positive definite' (or 'pos')
     general              'general' (or 'gen')
    ===================  ================================

    If omitted, ``'general'`` is the default structure.

    The datatype of the arrays define which solver is called regardless
    of the values. In other words, even when the complex array entries have
    precisely zero imaginary parts, the complex solver will be called based
    on the data type of the array.

    Parameters
    ----------
    a : (..., N, N) array_like
        Square input data
    b : (..., N, NRHS) array_like
        Input data for the right hand side.
    lower : bool, default False
        Ignored unless ``assume_a`` is one of ``'sym'``, ``'her'``, or ``'pos'``.
        If True, the calculation uses only the data in the lower triangle of `a`;
        entries above the diagonal are ignored. If False (default), the
        calculation uses only the data in the upper triangle of `a`; entries
        below the diagonal are ignored.
    overwrite_a : bool
        Unused by PyTensor. PyTensor will always perform the operation in-place if possible.
    overwrite_b : bool
        Unused by PyTensor. PyTensor will always perform the operation in-place if possible.
    check_finite : bool
        Unused by PyTensor. PyTensor returns nan if the operation fails.
    assume_a : str, optional
        Valid entries are explained above.
    transposed: bool, default False
        If True, solves the system A^T x = b. Default is False.
    b_ndim : int
        Whether the core case of b is a vector (1) or matrix (2).
        This will influence how batched dimensions are interpreted.
        By default, we assume b_ndim = b.ndim is 2 if b.ndim > 1, else 1.
    """
    assume_a = assume_a.lower()

    if assume_a in ("lower triangular", "upper triangular"):
        lower = "lower" in assume_a
        return solve_triangular(
            a,
            b,
            lower=lower,
            trans=transposed,
            b_ndim=b_ndim,
        )

    b_ndim = _default_b_ndim(b, b_ndim)

    if assume_a == "diagonal":
        a_diagonal = diagonal(a, axis1=-2, axis2=-1)
        b_transposed = b[None, :] if b_ndim == 1 else b.mT
        x = (b_transposed / pt.expand_dims(a_diagonal, -2)).mT
        if b_ndim == 1:
            x = x.squeeze(-1)
        return x

    if transposed:
        a = a.mT
        lower = not lower

    return Blockwise(
        Solve(
            lower=lower,
            assume_a=assume_a,
            b_ndim=b_ndim,
        )
    )(a, b)


def _lu_solve(
    lu_and_piv: "TensorVariable",
    pivots: "TensorVariable",
    b: "TensorVariable",
    trans: bool = False,
    b_ndim: int | None = None,
):
    b_ndim = _default_b_ndim(b, b_ndim)

    lu_and_piv, pivots, b = map(pt.as_tensor_variable, [lu_and_piv, pivots, b])

    inv_permutation = pivot_to_permutation(pivots, inverse=True)
    x = b[inv_permutation] if not trans else b
    # TODO: Use PermuteRows on b
    # x = permute_rows(b, pivots) if not trans else b

    x = solve_triangular(
        lu_and_piv,
        x,
        lower=not trans,
        unit_diagonal=not trans,
        trans=trans,
        b_ndim=b_ndim,
    )

    x = solve_triangular(
        lu_and_piv,
        x,
        lower=trans,
        unit_diagonal=trans,
        trans=trans,
        b_ndim=b_ndim,
    )

    # TODO: Use PermuteRows(inverse=True) on x
    # if trans:
    #     x = permute_rows(x, pivots, inverse=True)
    x = x[pt.argsort(inv_permutation)] if trans else x
    return x


def lu_solve(
    LU_and_pivots: tuple["TensorVariable", "TensorVariable"],
    b: "TensorVariable",
    trans: bool = False,
    b_ndim: int | None = None,
    check_finite: bool = True,
    overwrite_b: bool = False,
):
    """
    Solve a system of linear equations given the LU decomposition of the matrix.

    Parameters
    ----------
    LU_and_pivots: tuple[TensorLike, TensorLike]
        LU decomposition of the matrix, as returned by `lu_factor`
    b: TensorLike
        Right-hand side of the equation
    trans: bool
        If True, solve A^T x = b, instead of Ax = b. Default is False
    b_ndim: int, optional
        The number of core dimensions in b. Used to distinguish between a batch of vectors (b_ndim=1) and a matrix
        of vectors (b_ndim=2). Default is None, which will infer the number of core dimensions from the input.
    check_finite: bool
        Unused by PyTensor. PyTensor will return nan if the operation fails.
    overwrite_b: bool
        Ignored by Pytensor. Pytensor will always compute inplace when possible.
    """
    b_ndim = _default_b_ndim(b, b_ndim)
    if b_ndim == 1:
        signature = "(m,m),(m),(m)->(m)"
    else:
        signature = "(m,m),(m),(m,n)->(m,n)"
    partialled_func = partial(_lu_solve, trans=trans, b_ndim=b_ndim)
    return pt.vectorize(partialled_func, signature=signature)(*LU_and_pivots, b)
