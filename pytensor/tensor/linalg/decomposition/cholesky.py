import warnings
from typing import Literal

import numpy as np
from scipy import linalg as scipy_linalg

from pytensor.graph import Apply, Op
from pytensor.raise_op import CheckAndRaise
from pytensor.tensor import TensorLike
from pytensor.tensor import basic as ptb
from pytensor.tensor import math as ptm
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.type import tensor


class Cholesky(Op):
    # TODO: LAPACK wrapper with in-place behavior, for solve also

    __props__ = ("lower", "overwrite_a")
    gufunc_signature = "(m,m)->(m,m)"

    def __init__(
        self,
        *,
        lower: bool = True,
        overwrite_a: bool = False,
    ):
        self.lower = lower
        self.overwrite_a = overwrite_a

        if self.overwrite_a:
            self.destroy_map = {0: [0]}

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]

    def make_node(self, x):
        x = as_tensor_variable(x)
        if x.type.ndim != 2:
            raise TypeError(
                f"Cholesky only allowed on matrix (2-D) inputs, got {x.type.ndim}-D input"
            )
        # Call scipy to find output dtype
        dtype = scipy_linalg.cholesky(np.eye(1, dtype=x.type.dtype)).dtype
        return Apply(self, [x], [tensor(shape=x.type.shape, dtype=dtype)])

    def perform(self, node, inputs, outputs):
        [x] = inputs
        [out] = outputs

        (potrf,) = scipy_linalg.get_lapack_funcs(("potrf",), (x,))

        # Quick return for square empty array
        if x.size == 0:
            out[0] = np.empty_like(x, dtype=potrf.dtype)
            return

        # Squareness check
        if x.shape[0] != x.shape[1]:
            raise ValueError(
                f"Input array is expected to be square but has the shape: {x.shape}."
            )

        # Scipy cholesky only makes use of overwrite_a when it is F_CONTIGUOUS
        # If we have a `C_CONTIGUOUS` array we transpose to benefit from it
        c_contiguous_input = self.overwrite_a and x.flags["C_CONTIGUOUS"]
        if c_contiguous_input:
            x = x.T
            lower = not self.lower
            overwrite_a = True
        else:
            lower = self.lower
            overwrite_a = self.overwrite_a

        c, info = potrf(x, lower=lower, overwrite_a=overwrite_a, clean=True)

        if info != 0:
            c[...] = np.nan
            out[0] = c
        else:
            # Transpose result if input was transposed
            out[0] = c.T if c_contiguous_input else c

    def pullback(self, inputs, outputs, gradients):
        """
        Cholesky decomposition reverse-mode gradient update.

        Symbolic expression for reverse-mode Cholesky gradient taken from [#]_

        References
        ----------
        .. [#] I. Murray, "Differentiation of the Cholesky decomposition",
           http://arxiv.org/abs/1602.07527

        """

        dz = gradients[0]
        chol_x = outputs[0]

        # deal with upper triangular by converting to lower triangular
        if not self.lower:
            chol_x = chol_x.T
            dz = dz.T

        def tril_and_halve_diagonal(mtx):
            """Extracts lower triangle of square matrix and halves diagonal."""
            return ptb.tril(mtx) - ptb.diag(ptb.diagonal(mtx) / 2.0)

        def conjugate_solve_triangular(outer, inner):
            """Computes L^{-T} P L^{-1} for lower-triangular L."""
            from pytensor.tensor.linalg.solvers.triangular import SolveTriangular

            solve_upper = SolveTriangular(lower=False, b_ndim=2)
            return solve_upper(outer.T, solve_upper(outer.T, inner.T).T)

        s = conjugate_solve_triangular(
            chol_x, tril_and_halve_diagonal(chol_x.T.dot(dz))
        )

        if self.lower:
            grad = ptb.tril(s + s.T) - ptb.diag(ptb.diagonal(s))
        else:
            grad = ptb.triu(s + s.T) - ptb.diag(ptb.diagonal(s))

        return [grad]

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        if not allowed_inplace_inputs:
            return self
        new_props = self._props_dict()  # type: ignore
        new_props["overwrite_a"] = True
        return type(self)(**new_props)


def cholesky(
    x: "TensorLike",
    lower: bool = True,
    *,
    check_finite: bool = True,
    overwrite_a: bool = False,
    on_error: Literal["raise", "nan"] = "nan",
):
    """
    Return a triangular matrix square root of positive semi-definite `x`.

    L = cholesky(X, lower=True) implies dot(L, L.T) == X.

    Parameters
    ----------
    x: tensor_like
    lower : bool, default=True
        Whether to return the lower or upper cholesky factor
    check_finite : bool
        Unused by PyTensor. PyTensor will return nan if the operation fails.
    overwrite_a: bool, ignored
        Whether to use the same memory for the output as `a`. This argument is ignored, and is present here only
        for consistency with scipy.linalg.cholesky.
    on_error : ['raise', 'nan']
        If on_error is set to 'raise', this Op will raise a `scipy.linalg.LinAlgError` if the matrix is not positive definite.
        If on_error is set to 'nan', it will return a matrix containing nans instead.

    Returns
    -------
    TensorVariable
        Lower or upper triangular Cholesky factor of `x`

    Example
    -------
    .. testcode::

        import pytensor
        import pytensor.tensor as pt
        import numpy as np

        x = pt.tensor('x', shape=(5, 5), dtype='float64')
        L = pt.linalg.cholesky(x)

        f = pytensor.function([x], L)
        x_value = np.random.normal(size=(5, 5))
        x_value = x_value @ x_value.T # Ensures x is positive definite
        L_value = f(x_value)
        assert np.allclose(L_value @ L_value.T, x_value)

    """
    res = Blockwise(Cholesky(lower=lower))(x)

    if on_error == "raise":
        # For back-compatibility
        warnings.warn(
            'Cholesky on_raise == "raise" is deprecated. The operation will return nan when in fails. Setting this argument will fail in the future',
            FutureWarning,
        )
        res = CheckAndRaise(np.linalg.LinAlgError, "Matrix is not positive definite")(
            res, ~ptm.isnan(res).any()
        )

    return res
