import logging
import warnings
from collections.abc import Sequence
from functools import partial, reduce
from typing import Literal, cast

import numpy as np
import scipy.linalg as scipy_linalg
from scipy.linalg import get_lapack_funcs

import pytensor
from pytensor import ifelse
from pytensor import tensor as pt
from pytensor.gradient import DisconnectedType, disconnected_type
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.raise_op import Assert, CheckAndRaise
from pytensor.tensor import TensorLike
from pytensor.tensor import basic as ptb
from pytensor.tensor import math as ptm
from pytensor.tensor.basic import as_tensor_variable, diagonal
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.type import matrix, tensor, vector
from pytensor.tensor.variable import TensorVariable


logger = logging.getLogger(__name__)


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

    def L_op(self, inputs, outputs, gradients):
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


class SolveBase(Op):
    """Base class for `scipy.linalg` matrix equation solvers."""

    __props__: tuple[str, ...] = (
        "lower",
        "b_ndim",
        "overwrite_a",
        "overwrite_b",
    )

    def __init__(
        self,
        *,
        lower=False,
        b_ndim,
        overwrite_a=False,
        overwrite_b=False,
    ):
        self.lower = lower

        assert b_ndim in (1, 2)
        self.b_ndim = b_ndim
        if b_ndim == 1:
            self.gufunc_signature = "(m,m),(m)->(m)"
        else:
            self.gufunc_signature = "(m,m),(m,n)->(m,n)"
        self.overwrite_a = overwrite_a
        self.overwrite_b = overwrite_b
        destroy_map = {}
        if self.overwrite_a and self.overwrite_b:
            # An output destroying two inputs is not yet supported
            # destroy_map[0] = [0, 1]
            raise NotImplementedError(
                "It's not yet possible to overwrite_a and overwrite_b simultaneously"
            )
        elif self.overwrite_a:
            destroy_map[0] = [0]
        elif self.overwrite_b:
            destroy_map[0] = [1]
        self.destroy_map = destroy_map

    def perform(self, node, inputs, outputs):
        raise NotImplementedError(
            "SolveBase should be subclassed with an perform method"
        )

    def make_node(self, A, b):
        A = as_tensor_variable(A)
        b = as_tensor_variable(b)

        if A.ndim != 2:
            raise ValueError(f"`A` must be a matrix; got {A.type} instead.")
        if b.ndim != self.b_ndim:
            raise ValueError(f"`b` must have {self.b_ndim} dims; got {b.type} instead.")

        # Infer dtype by solving the most simple case with 1x1 matrices
        o_dtype = scipy_linalg.solve(
            np.ones((1, 1), dtype=A.dtype),
            np.ones((1,), dtype=b.dtype),
        ).dtype
        x = tensor(dtype=o_dtype, shape=b.type.shape)
        return Apply(self, [A, b], [x])

    def infer_shape(self, fgraph, node, shapes):
        Ashape, Bshape = shapes
        rows = Ashape[1]
        if len(Bshape) == 1:
            return [(rows,)]
        else:
            cols = Bshape[1]
            return [(rows, cols)]

    def L_op(self, inputs, outputs, output_gradients):
        r"""Reverse-mode gradient updates for matrix solve operation :math:`c = A^{-1} b`.

        Symbolic expression for updates taken from [#]_.

        References
        ----------
        .. [#] M. B. Giles, "An extended collection of matrix derivative results
          for forward and reverse mode automatic differentiation",
          http://eprints.maths.ox.ac.uk/1079/

        """
        A, _b = inputs

        c = outputs[0]
        # C is a scalar representing the entire graph
        # `output_gradients` is (dC/dc,)
        # We need to return (dC/d[inv(A)], dC/db)
        c_bar = output_gradients[0]

        props_dict = self._props_dict()
        props_dict["lower"] = not self.lower

        solve_op = type(self)(**props_dict)

        b_bar = solve_op(A.mT, c_bar)
        # force outer product if vector second input
        A_bar = -ptm.outer(b_bar, c) if c.ndim == 1 else -b_bar.dot(c.T)

        if props_dict.get("unit_diagonal", False):
            n = A_bar.shape[-1]
            A_bar = A_bar[pt.arange(n), pt.arange(n)].set(pt.zeros(n))

        return [A_bar, b_bar]


def _default_b_ndim(b, b_ndim):
    if b_ndim is not None:
        assert b_ndim in (1, 2)
        return b_ndim

    b = as_tensor_variable(b)
    if b_ndim is None:
        return min(b.ndim, 2)  # By default, assume the core case is a matrix


class CholeskySolve(SolveBase):
    __props__ = (
        "lower",
        "b_ndim",
        "overwrite_b",
    )

    def __init__(self, **kwargs):
        if kwargs.get("overwrite_a", False):
            raise ValueError("overwrite_a is not supported for CholeskySolve")
        super().__init__(**kwargs)

    def make_node(self, *inputs):
        # Allow base class to do input validation
        super_apply = super().make_node(*inputs)
        A, b = super_apply.inputs
        [super_out] = super_apply.outputs
        # The dtype of chol_solve does not match solve, which the base class checks
        dtype = scipy_linalg.cho_solve(
            (np.ones((1, 1), dtype=A.dtype), False),
            np.ones((1,), dtype=b.dtype),
        ).dtype
        out = tensor(dtype=dtype, shape=super_out.type.shape)
        return Apply(self, [A, b], [out])

    def perform(self, node, inputs, output_storage):
        c, b = inputs

        (potrs,) = get_lapack_funcs(("potrs",), (c, b))

        if c.shape[0] != c.shape[1]:
            raise ValueError("The factored matrix c is not square.")
        if c.shape[1] != b.shape[0]:
            raise ValueError(f"incompatible dimensions ({c.shape} and {b.shape})")

        # Quick return for empty arrays
        if b.size == 0:
            output_storage[0][0] = np.empty_like(b, dtype=potrs.dtype)
            return

        x, info = potrs(c, b, lower=self.lower, overwrite_b=self.overwrite_b)
        if info != 0:
            x[...] = np.nan

        output_storage[0][0] = x

    def L_op(self, *args, **kwargs):
        # TODO: Base impl should work, let's try it
        raise NotImplementedError()

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        if 1 in allowed_inplace_inputs:
            new_props = self._props_dict()  # type: ignore
            new_props["overwrite_b"] = True
            return type(self)(**new_props)
        else:
            return self


def cho_solve(
    c_and_lower: tuple[TensorLike, bool],
    b: TensorLike,
    *,
    b_ndim: int | None = None,
):
    """Solve the linear equations A x = b, given the Cholesky factorization of A.

    Parameters
    ----------
    c_and_lower : tuple of (TensorLike, bool)
        Cholesky factorization of a, as given by cho_factor
    b : TensorLike
        Right-hand side
    check_finite : bool
        Unused by PyTensor. PyTensor will return nan if the operation fails.
    b_ndim : int
        Whether the core case of b is a vector (1) or matrix (2).
        This will influence how batched dimensions are interpreted.
    """
    A, lower = c_and_lower
    b_ndim = _default_b_ndim(b, b_ndim)
    return Blockwise(CholeskySolve(lower=lower, b_ndim=b_ndim))(A, b)


class LU(Op):
    """Decompose a matrix into lower and upper triangular matrices."""

    __props__ = ("permute_l", "overwrite_a", "p_indices")

    def __init__(self, *, permute_l=False, overwrite_a=False, p_indices=False):
        if permute_l and p_indices:
            raise ValueError("Only one of permute_l and p_indices can be True")
        self.permute_l = permute_l
        self.p_indices = p_indices
        self.overwrite_a = overwrite_a

        if self.permute_l:
            # permute_l overrides p_indices in the scipy function. We can copy that behavior
            self.gufunc_signature = "(m,m)->(m,m),(m,m)"
        elif self.p_indices:
            self.gufunc_signature = "(m,m)->(m),(m,m),(m,m)"
        else:
            self.gufunc_signature = "(m,m)->(m,m),(m,m),(m,m)"

        if self.overwrite_a:
            self.destroy_map = {0: [0]} if self.permute_l else {1: [0]}

    def infer_shape(self, fgraph, node, shapes):
        n = shapes[0][0]
        if self.permute_l:
            return [(n, n), (n, n)]
        elif self.p_indices:
            return [(n,), (n, n), (n, n)]
        else:
            return [(n, n), (n, n), (n, n)]

    def make_node(self, x):
        x = as_tensor_variable(x)
        if x.type.ndim != 2:
            raise TypeError(
                f"LU only allowed on matrix (2-D) inputs, got {x.type.ndim}-D input"
            )

        if x.type.numpy_dtype.kind in "ibu":
            if x.type.numpy_dtype.itemsize <= 2:
                out_dtype = "float32"
            else:
                out_dtype = "float64"
        else:
            out_dtype = x.type.dtype
        L = tensor(shape=x.type.shape, dtype=out_dtype)
        U = tensor(shape=x.type.shape, dtype=out_dtype)

        if self.permute_l:
            # In this case, L is actually P @ L
            return Apply(self, inputs=[x], outputs=[L, U])
        if self.p_indices:
            p_indices = tensor(shape=(x.type.shape[0],), dtype="int32")
            return Apply(self, inputs=[x], outputs=[p_indices, L, U])

        if out_dtype.startswith("complex"):
            P_dtype = "float64" if out_dtype == "complex128" else "float32"
        else:
            P_dtype = out_dtype

        P = tensor(shape=x.type.shape, dtype=P_dtype)
        return Apply(self, inputs=[x], outputs=[P, L, U])

    def perform(self, node, inputs, outputs):
        [A] = inputs

        out = scipy_linalg.lu(
            A,
            permute_l=self.permute_l,
            overwrite_a=self.overwrite_a,
            p_indices=self.p_indices,
        )

        outputs[0][0] = out[0]
        outputs[1][0] = out[1]

        if not self.permute_l:
            # In all cases except permute_l, there are three returns
            outputs[2][0] = out[2]

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        if 0 in allowed_inplace_inputs:
            new_props = self._props_dict()  # type: ignore
            new_props["overwrite_a"] = True
            return type(self)(**new_props)

        else:
            return self

    def L_op(
        self,
        inputs: Sequence[ptb.Variable],
        outputs: Sequence[ptb.Variable],
        output_grads: Sequence[ptb.Variable],
    ) -> list[ptb.Variable]:
        r"""
        Derivation is due to Differentiation of Matrix Functionals Using Triangular Factorization
        F. R. De Hoog, R.S. Anderssen, M. A. Lukas
        """
        [A] = inputs
        A = cast(TensorVariable, A)

        if self.permute_l:
            # P has no gradient contribution (by assumption...), so PL_bar is the same as L_bar
            L_bar, U_bar = output_grads

            # TODO: Rewrite into permute_l = False for graphs where we need to compute the gradient
            # We need L, not PL. It's not possible to recover it from PL, though. So we need to do a new forward pass
            P_or_indices, L, U = lu(  # type: ignore
                A, permute_l=False, p_indices=False
            )

        else:
            # In both other cases, there are 3 outputs. The first output will either be the permutation index itself,
            # or indices that can be used to reconstruct the permutation matrix.
            P_or_indices, L, U = outputs
            _, L_bar, U_bar = output_grads

        L_bar = (
            L_bar if not isinstance(L_bar.type, DisconnectedType) else pt.zeros_like(A)
        )
        U_bar = (
            U_bar if not isinstance(U_bar.type, DisconnectedType) else pt.zeros_like(A)
        )

        x1 = ptb.tril(L.T @ L_bar, k=-1)
        x2 = ptb.triu(U_bar @ U.T)

        LT_inv_x = solve_triangular(L.T, x1 + x2, lower=False, unit_diagonal=True)

        # Where B = P.T @ A is a change of variable to avoid the permutation matrix in the gradient derivation
        B_bar = solve_triangular(U, LT_inv_x.T, lower=False).T

        if not self.p_indices:
            A_bar = P_or_indices @ B_bar
        else:
            A_bar = B_bar[P_or_indices]

        return [A_bar]


def lu(
    a: TensorLike,
    permute_l=False,
    check_finite=True,
    p_indices=False,
    overwrite_a: bool = False,
) -> (
    tuple[TensorVariable, TensorVariable, TensorVariable]
    | tuple[TensorVariable, TensorVariable]
):
    """
    Factorize a matrix as the product of a unit lower triangular matrix and an upper triangular matrix:

    ... math::

        A = P L U

    Where P is a permutation matrix, L is lower triangular with unit diagonal elements, and U is upper triangular.

    Parameters
    ----------
    a: TensorLike
        Matrix to be factorized
    permute_l: bool
        If True, L is a product of permutation and unit lower triangular matrices. Only two values, PL and U, will
        be returned in this case, and PL will not be lower triangular.
    check_finite : bool
        Unused by PyTensor. PyTensor will return nan if the operation fails.
    p_indices: bool
        If True, return integer matrix indices for the permutation matrix. Otherwise, return the permutation matrix
        itself.
    overwrite_a: bool
        Ignored by Pytensor. Pytensor will always perform computation inplace if possible.
    Returns
    -------
    P: TensorVariable
        Permutation matrix, or array of integer indices for permutation matrix. Not returned if permute_l is True.
    L: TensorVariable
        Lower triangular matrix, or product of permutation and unit lower triangular matrices if permute_l is True.
    U: TensorVariable
        Upper triangular matrix
    """
    return cast(
        tuple[TensorVariable, TensorVariable, TensorVariable]
        | tuple[TensorVariable, TensorVariable],
        Blockwise(LU(permute_l=permute_l, p_indices=p_indices))(a),
    )


class PivotToPermutations(Op):
    gufunc_signature = "(x)->(x)"
    __props__ = ("inverse",)

    def __init__(self, inverse=True):
        self.inverse = inverse

    def make_node(self, pivots):
        pivots = as_tensor_variable(pivots)
        if pivots.ndim != 1:
            raise ValueError("PivotToPermutations only works on 1-D inputs")

        permutations = pivots.type.clone(dtype="int64")()
        return Apply(self, [pivots], [permutations])

    def perform(self, node, inputs, outputs):
        [pivots] = inputs
        p_inv = np.arange(len(pivots), dtype="int64")

        for i in range(len(pivots)):
            p_inv[i], p_inv[pivots[i]] = p_inv[pivots[i]], p_inv[i]

        if self.inverse:
            outputs[0][0] = p_inv
        else:
            outputs[0][0] = np.argsort(p_inv)


def pivot_to_permutation(p: TensorLike, inverse=False):
    p = pt.as_tensor_variable(p)
    return PivotToPermutations(inverse=inverse)(p)


class LUFactor(Op):
    __props__ = ("overwrite_a",)
    gufunc_signature = "(m,m)->(m,m),(m)"

    def __init__(self, *, overwrite_a=False):
        self.overwrite_a = overwrite_a

        if self.overwrite_a:
            self.destroy_map = {1: [0]}

    def make_node(self, A):
        A = as_tensor_variable(A)
        if A.type.ndim != 2:
            raise TypeError(
                f"LU only allowed on matrix (2-D) inputs, got {A.type.ndim}-D input"
            )

        LU = matrix(shape=A.type.shape, dtype=A.type.dtype)
        pivots = vector(shape=(A.type.shape[0],), dtype="int32")

        return Apply(self, [A], [LU, pivots])

    def infer_shape(self, fgraph, node, shapes):
        n = shapes[0][0]
        return [(n, n), (n,)]

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        if 0 in allowed_inplace_inputs:
            new_props = self._props_dict()  # type: ignore
            new_props["overwrite_a"] = True
            return type(self)(**new_props)
        else:
            return self

    def perform(self, node, inputs, outputs):
        A = inputs[0]

        # Quick return for empty arrays
        if A.size == 0:
            outputs[0][0] = np.empty_like(A)
            outputs[1][0] = np.array([], dtype=np.int32)
            return

        (getrf,) = get_lapack_funcs(("getrf",), (A,))
        LU, p, info = getrf(A, overwrite_a=self.overwrite_a)
        if info != 0:
            LU[...] = np.nan

        outputs[0][0] = LU
        outputs[1][0] = p

    def L_op(self, inputs, outputs, output_gradients):
        [A] = inputs
        LU_bar, _ = output_gradients
        LU, p_indices = outputs

        eye = ptb.identity_like(A)
        L = cast(TensorVariable, ptb.tril(LU, k=-1) + eye)
        U = cast(TensorVariable, ptb.triu(LU))

        p_indices = pivot_to_permutation(p_indices, inverse=False)

        # Split LU_bar into L_bar and U_bar. This is valid because of the triangular structure of L and U
        L_bar = ptb.tril(LU_bar, k=-1)
        U_bar = ptb.triu(LU_bar)

        # From here we're in the same situation as the LU gradient derivation
        x1 = ptb.tril(L.T @ L_bar, k=-1)
        x2 = ptb.triu(U_bar @ U.T)

        LT_inv_x = solve_triangular(L.T, x1 + x2, lower=False, unit_diagonal=True)
        B_bar = solve_triangular(U, LT_inv_x.T, lower=False).T
        A_bar = B_bar[p_indices]

        return [A_bar]


def lu_factor(
    a: TensorLike,
    *,
    check_finite: bool = True,
    overwrite_a: bool = False,
) -> tuple[TensorVariable, TensorVariable]:
    """
    LU factorization with partial pivoting.

    Parameters
    ----------
    a: TensorLike
        Matrix to be factorized
    check_finite: bool
        Unused by PyTensor. PyTensor will return nan if the operation fails.
    overwrite_a: bool
        Unused by PyTensor. PyTensor will always perform the operation in-place if possible.

    Returns
    -------
    LU: TensorVariable
        LU decomposition of `a`
    pivots: TensorVariable
        An array of integers representin the pivot indices
    """

    return cast(
        tuple[TensorVariable, TensorVariable],
        Blockwise(LUFactor())(a),
    )


def _lu_solve(
    LU: TensorLike,
    pivots: TensorLike,
    b: TensorLike,
    trans: bool = False,
    b_ndim: int | None = None,
):
    b_ndim = _default_b_ndim(b, b_ndim)

    LU, pivots, b = map(pt.as_tensor_variable, [LU, pivots, b])

    inv_permutation = pivot_to_permutation(pivots, inverse=True)
    x = b[inv_permutation] if not trans else b
    # TODO: Use PermuteRows on b
    # x = permute_rows(b, pivots) if not trans else b

    x = solve_triangular(
        LU,
        x,
        lower=not trans,
        unit_diagonal=not trans,
        trans=trans,
        b_ndim=b_ndim,
    )

    x = solve_triangular(
        LU,
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
    LU_and_pivots: tuple[TensorLike, TensorLike],
    b: TensorLike,
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


class SolveTriangular(SolveBase):
    """Solve a system of linear equations."""

    __props__ = (
        "unit_diagonal",
        "lower",
        "b_ndim",
        "overwrite_b",
    )

    def __init__(self, *, unit_diagonal=False, **kwargs):
        if kwargs.get("overwrite_a", False):
            raise ValueError("overwrite_a is not supported for SolverTriangulare")

        # There's a naming inconsistency between solve_triangular (trans) and solve (transposed). Internally, we can use
        # transpose everywhere, but expose the same API as scipy.linalg.solve_triangular
        super().__init__(**kwargs)
        self.unit_diagonal = unit_diagonal

    def perform(self, node, inputs, outputs):
        A, b = inputs

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("expected square matrix")

        if A.shape[0] != b.shape[0]:
            raise ValueError(f"shapes of a {A.shape} and b {b.shape} are incompatible")

        (trtrs,) = get_lapack_funcs(("trtrs",), (A, b))

        # Quick return for empty arrays
        if b.size == 0:
            outputs[0][0] = np.empty_like(b, dtype=trtrs.dtype)
            return

        if A.flags["F_CONTIGUOUS"]:
            x, info = trtrs(
                A,
                b,
                overwrite_b=self.overwrite_b,
                lower=self.lower,
                trans=0,
                unitdiag=self.unit_diagonal,
            )
        else:
            # transposed system is solved since trtrs expects Fortran ordering
            x, info = trtrs(
                A.T,
                b,
                overwrite_b=self.overwrite_b,
                lower=not self.lower,
                trans=1,
                unitdiag=self.unit_diagonal,
            )

        if info != 0:
            x[...] = np.nan

        outputs[0][0] = x

    def L_op(self, inputs, outputs, output_gradients):
        res = super().L_op(inputs, outputs, output_gradients)

        if self.lower:
            res[0] = ptb.tril(res[0])
        else:
            res[0] = ptb.triu(res[0])

        return res

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        if 1 in allowed_inplace_inputs:
            new_props = self._props_dict()  # type: ignore
            new_props["overwrite_b"] = True
            return type(self)(**new_props)
        else:
            return self


def solve_triangular(
    a: TensorVariable,
    b: TensorVariable,
    *,
    trans: int | str = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    check_finite: bool = True,
    b_ndim: int | None = None,
) -> TensorVariable:
    """Solve the equation `a x = b` for `x`, assuming `a` is a triangular matrix.

    Parameters
    ----------
    a: TensorVariable
        Square input data
    b: TensorVariable
        Input data for the right hand side.
    lower : bool, optional
        Use only data contained in the lower triangle of `a`. Default is to use upper triangle.
    trans: {0, 1, 2, 'N', 'T', 'C'}, optional
        Type of system to solve:
        trans       system
        0 or 'N'    a x = b
        1 or 'T'    a^T x = b
        2 or 'C'    a^H x = b
    unit_diagonal: bool, optional
        If True, diagonal elements of `a` are assumed to be 1 and will not be referenced.
    check_finite : bool, optional
        Unused by PyTensor. PyTensor will return nan if the operation fails.
    b_ndim : int
        Whether the core case of b is a vector (1) or matrix (2).
        This will influence how batched dimensions are interpreted.
    """
    b_ndim = _default_b_ndim(b, b_ndim)

    if trans in [1, "T", True]:
        a = a.mT
        lower = not lower
    if trans in [2, "C"]:
        a = a.conj().mT
        lower = not lower

    ret = Blockwise(
        SolveTriangular(
            lower=lower,
            unit_diagonal=unit_diagonal,
            b_ndim=b_ndim,
        )
    )(a, b)
    return cast(TensorVariable, ret)


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


class Eigvalsh(Op):
    """
    Generalized eigenvalues of a Hermitian positive definite eigensystem.

    """

    __props__ = ("lower",)

    def __init__(self, lower=True):
        assert lower in [True, False]
        self.lower = lower

    def make_node(self, a, b):
        if b == pytensor.tensor.type_other.NoneConst:
            a = as_tensor_variable(a)
            assert a.ndim == 2

            out_dtype = pytensor.scalar.upcast(a.dtype)
            w = vector(dtype=out_dtype)
            return Apply(self, [a], [w])
        else:
            a = as_tensor_variable(a)
            b = as_tensor_variable(b)
            assert a.ndim == 2
            assert b.ndim == 2

            out_dtype = pytensor.scalar.upcast(a.dtype, b.dtype)
            w = vector(dtype=out_dtype)
            return Apply(self, [a, b], [w])

    def perform(self, node, inputs, outputs):
        (w,) = outputs
        if len(inputs) == 2:
            w[0] = scipy_linalg.eigvalsh(a=inputs[0], b=inputs[1], lower=self.lower)
        else:
            w[0] = scipy_linalg.eigvalsh(a=inputs[0], b=None, lower=self.lower)

    def grad(self, inputs, g_outputs):
        a, b = inputs
        (gw,) = g_outputs
        return EigvalshGrad(self.lower)(a, b, gw)

    def infer_shape(self, fgraph, node, shapes):
        n = shapes[0][0]
        return [(n,)]


class EigvalshGrad(Op):
    """
    Gradient of generalized eigenvalues of a Hermitian positive definite
    eigensystem.

    """

    # Note: This Op (EigvalshGrad), should be removed and replaced with a graph
    # of pytensor ops that is constructed directly in Eigvalsh.grad.
    # But this can only be done once scipy.linalg.eigh is available as an Op
    # (currently the Eigh uses numpy.linalg.eigh, which doesn't let you
    # pass the right-hand-side matrix for a generalized eigenproblem.) See the
    # discussion on GitHub at
    # https://github.com/Theano/Theano/pull/1846#discussion-diff-12486764

    __props__ = ("lower",)

    def __init__(self, lower=True):
        assert lower in [True, False]
        self.lower = lower
        if lower:
            self.tri0 = np.tril
            self.tri1 = lambda a: np.triu(a, 1)
        else:
            self.tri0 = np.triu
            self.tri1 = lambda a: np.tril(a, -1)

    def make_node(self, a, b, gw):
        a = as_tensor_variable(a)
        b = as_tensor_variable(b)
        gw = as_tensor_variable(gw)
        assert a.ndim == 2
        assert b.ndim == 2
        assert gw.ndim == 1

        out_dtype = pytensor.scalar.upcast(a.dtype, b.dtype, gw.dtype)
        out1 = matrix(dtype=out_dtype)
        out2 = matrix(dtype=out_dtype)
        return Apply(self, [a, b, gw], [out1, out2])

    def perform(self, node, inputs, outputs):
        (a, b, gw) = inputs
        w, v = scipy_linalg.eigh(a, b, lower=self.lower)
        gA = v.dot(np.diag(gw).dot(v.T))
        gB = -v.dot(np.diag(gw * w).dot(v.T))

        # See EighGrad comments for an explanation of these lines
        out1 = self.tri0(gA) + self.tri1(gA).T
        out2 = self.tri0(gB) + self.tri1(gB).T
        outputs[0][0] = np.asarray(out1, dtype=node.outputs[0].dtype)
        outputs[1][0] = np.asarray(out2, dtype=node.outputs[1].dtype)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0], shapes[1]]


def eigvalsh(a, b, lower=True):
    return Eigvalsh(lower)(a, b)


class Expm(Op):
    """
    Compute the matrix exponential of a square array.
    """

    __props__ = ()
    gufunc_signature = "(m,m)->(m,m)"

    def make_node(self, A):
        A = as_tensor_variable(A)
        assert A.ndim == 2

        expm = matrix(dtype=A.dtype, shape=A.type.shape)

        return Apply(self, [A], [expm])

    def perform(self, node, inputs, outputs):
        (A,) = inputs
        (expm,) = outputs
        expm[0] = scipy_linalg.expm(A)

    def L_op(self, inputs, outputs, output_grads):
        # Kalbfleisch and Lawless, J. Am. Stat. Assoc. 80 (1985) Equation 3.4
        # Kind of... You need to do some algebra from there to arrive at
        # this expression.
        (A,) = inputs
        (_,) = outputs  # Outputs not used; included for signature consistency only
        (A_bar,) = output_grads

        w, V = pt.linalg.eig(A)

        exp_w = pt.exp(w)
        numer = pt.sub.outer(exp_w, exp_w)
        denom = pt.sub.outer(w, w)

        # When w_i ≈ w_j, we have a removable singularity in the expression for X, because
        # lim b->a (e^a - e^b) / (a - b) = e^a (derivation left for the motivated reader)
        X = pt.where(pt.abs(denom) < 1e-8, exp_w, numer / denom)

        diag_idx = pt.arange(w.shape[0])
        X = X[..., diag_idx, diag_idx].set(exp_w)

        inner = solve(V, A_bar.T @ V).T
        result = solve(V.T, inner * X) @ V.T

        # At this point, result is always a complex dtype. If the input was real, the output should be
        # real as well (and all the imaginary parts are numerical noise)
        if A.dtype not in ("complex64", "complex128"):
            return [result.real]

        return [result]

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]


expm = Blockwise(Expm())


def _largest_common_dtype(tensors: Sequence[TensorVariable]) -> np.dtype:
    return reduce(lambda l, r: np.promote_types(l, r), [x.dtype for x in tensors])


class BaseBlockDiagonal(Op):
    __props__: tuple[str, ...] = ("n_inputs",)

    def __init__(self, n_inputs):
        input_sig = ",".join(f"(m{i},n{i})" for i in range(n_inputs))
        self.gufunc_signature = f"{input_sig}->(m,n)"

        if n_inputs == 0:
            raise ValueError("n_inputs must be greater than 0")
        self.n_inputs = n_inputs

    def grad(self, inputs, gout):
        shapes = pt.stack([i.shape for i in inputs])
        index_end = shapes.cumsum(0)
        index_begin = index_end - shapes
        slices = [
            ptb.ix_(
                pt.arange(index_begin[i, 0], index_end[i, 0]),
                pt.arange(index_begin[i, 1], index_end[i, 1]),
            )
            for i in range(len(inputs))
        ]
        return [gout[0][slc] for slc in slices]

    def infer_shape(self, fgraph, nodes, shapes):
        first, second = zip(*shapes, strict=True)
        return [(pt.add(*first), pt.add(*second))]

    def _validate_and_prepare_inputs(self, matrices, as_tensor_func):
        if len(matrices) != self.n_inputs:
            raise ValueError(
                f"Expected {self.n_inputs} matri{'ces' if self.n_inputs > 1 else 'x'}, got {len(matrices)}"
            )
        matrices = list(map(as_tensor_func, matrices))
        if any(mat.type.ndim != 2 for mat in matrices):
            raise TypeError("All inputs must have dimension 2")
        return matrices


class BlockDiagonal(BaseBlockDiagonal):
    __props__ = ("n_inputs",)

    def make_node(self, *matrices):
        matrices = self._validate_and_prepare_inputs(matrices, pt.as_tensor)
        dtype = _largest_common_dtype(matrices)

        shapes_by_dim = tuple(zip(*(m.type.shape for m in matrices)))
        out_shape = tuple(
            [
                sum(dim_shapes)
                if not any(shape is None for shape in dim_shapes)
                else None
                for dim_shapes in shapes_by_dim
            ]
        )

        out_type = pytensor.tensor.matrix(shape=out_shape, dtype=dtype)
        return Apply(self, matrices, [out_type])

    def perform(self, node, inputs, output_storage, params=None):
        dtype = node.outputs[0].type.dtype
        output_storage[0][0] = scipy_linalg.block_diag(*inputs).astype(dtype)


def block_diag(*matrices: TensorVariable):
    """
    Construct a block diagonal matrix from a sequence of input tensors.

    Given the inputs `A`, `B` and `C`, the output will have these arrays arranged on the diagonal:

    [[A, 0, 0],
     [0, B, 0],
     [0, 0, C]]

    Parameters
    ----------
    A, B, C ... : tensors
        Input tensors to form the block diagonal matrix. last two dimensions of the inputs will be used, and all
        inputs should have at least 2 dimensins.

    Returns
    -------
    out: tensor
        The block diagonal matrix formed from the input matrices.

    Examples
    --------
    Create a block diagonal matrix from two 2x2 matrices:

    ..code-block:: python

        import numpy as np
        from pytensor.tensor.linalg import block_diag

        A = pt.as_tensor_variable(np.array([[1, 2], [3, 4]]))
        B = pt.as_tensor_variable(np.array([[5, 6], [7, 8]]))

        result = block_diagonal(A, B, name='X')
        print(result.eval())
        Out: array([[1, 2, 0, 0],
                     [3, 4, 0, 0],
                     [0, 0, 5, 6],
                     [0, 0, 7, 8]])
    """
    _block_diagonal_matrix = Blockwise(BlockDiagonal(n_inputs=len(matrices)))
    return _block_diagonal_matrix(*matrices)


class QR(Op):
    """
    QR Decomposition
    """

    __props__ = (
        "overwrite_a",
        "mode",
        "pivoting",
    )

    def __init__(
        self,
        mode: Literal["full", "r", "economic", "raw"] = "full",
        overwrite_a: bool = False,
        pivoting: bool = False,
    ):
        self.mode = mode
        self.overwrite_a = overwrite_a
        self.pivoting = pivoting

        self.destroy_map = {}

        if overwrite_a:
            self.destroy_map = {0: [0]}

        match self.mode:
            case "economic":
                self.gufunc_signature = "(m,n)->(m,k),(k,n)"
            case "full":
                self.gufunc_signature = "(m,n)->(m,m),(m,n)"
            case "r":
                self.gufunc_signature = "(m,n)->(m,n)"
            case "raw":
                self.gufunc_signature = "(m,n)->(n,m),(k),(m,n)"
            case _:
                raise ValueError(
                    f"Invalid mode '{mode}'. Supported modes are 'full', 'economic', 'r', and 'raw'."
                )

        if pivoting:
            self.gufunc_signature += ",(n)"

    def make_node(self, x):
        x = as_tensor_variable(x)

        assert x.ndim == 2, "The input of qr function should be a matrix."

        # Preserve static shape information if possible
        M, N = x.type.shape
        if M is not None and N is not None:
            K = min(M, N)
        else:
            K = None

        in_dtype = x.type.numpy_dtype
        if in_dtype.kind in "ibu":
            out_dtype = "float64" if in_dtype.itemsize > 2 else "float32"
        else:
            out_dtype = "float64" if in_dtype.itemsize > 4 else "float32"

        match self.mode:
            case "full":
                outputs = [
                    tensor(shape=(M, M), dtype=out_dtype),
                    tensor(shape=(M, N), dtype=out_dtype),
                ]
            case "economic":
                outputs = [
                    tensor(shape=(M, K), dtype=out_dtype),
                    tensor(shape=(K, N), dtype=out_dtype),
                ]
            case "r":
                outputs = [
                    tensor(shape=(M, N), dtype=out_dtype),
                ]
            case "raw":
                outputs = [
                    tensor(shape=(M, M), dtype=out_dtype),
                    tensor(shape=(K,), dtype=out_dtype),
                    tensor(shape=(M, N), dtype=out_dtype),
                ]
            case _:
                raise NotImplementedError

        if self.pivoting:
            outputs = [*outputs, tensor(shape=(N,), dtype="int32")]

        return Apply(self, [x], outputs)

    def infer_shape(self, fgraph, node, shapes):
        (x_shape,) = shapes

        M, N = x_shape
        K = ptm.minimum(M, N)

        Q_shape = None
        R_shape = None
        tau_shape = None
        P_shape = None

        match self.mode:
            case "full":
                Q_shape = (M, M)
                R_shape = (M, N)
            case "economic":
                Q_shape = (M, K)
                R_shape = (K, N)
            case "r":
                R_shape = (M, N)
            case "raw":
                Q_shape = (M, M)  # Actually this is H in this case
                tau_shape = (K,)
                R_shape = (M, N)

        if self.pivoting:
            P_shape = (N,)

        return [
            shape
            for shape in (Q_shape, tau_shape, R_shape, P_shape)
            if shape is not None
        ]

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        if not allowed_inplace_inputs:
            return self
        new_props = self._props_dict()  # type: ignore
        new_props["overwrite_a"] = True
        return type(self)(**new_props)

    def _call_and_get_lwork(self, fn, *args, lwork, **kwargs):
        if lwork in [-1, None]:
            *_, work, _info = fn(*args, lwork=-1, **kwargs)
            lwork = work.item()

        return fn(*args, lwork=lwork, **kwargs)

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        M, N = x.shape

        if self.pivoting:
            (geqp3,) = get_lapack_funcs(("geqp3",), (x,))
            qr, jpvt, tau, *_work_info = self._call_and_get_lwork(
                geqp3, x, lwork=-1, overwrite_a=self.overwrite_a
            )
            jpvt -= 1  # geqp3 returns a 1-based index array, so subtract 1
        else:
            (geqrf,) = get_lapack_funcs(("geqrf",), (x,))
            qr, tau, *_work_info = self._call_and_get_lwork(
                geqrf, x, lwork=-1, overwrite_a=self.overwrite_a
            )

        if self.mode not in ["economic", "raw"] or M < N:
            R = np.triu(qr)
        else:
            R = np.triu(qr[:N, :])

        if self.mode == "r" and self.pivoting:
            outputs[0][0] = R
            outputs[1][0] = jpvt
            return

        elif self.mode == "r":
            outputs[0][0] = R
            return

        elif self.mode == "raw" and self.pivoting:
            outputs[0][0] = qr
            outputs[1][0] = tau
            outputs[2][0] = R
            outputs[3][0] = jpvt
            return

        elif self.mode == "raw":
            outputs[0][0] = qr
            outputs[1][0] = tau
            outputs[2][0] = R
            return

        (gor_un_gqr,) = get_lapack_funcs(("orgqr",), (qr,))

        if M < N:
            Q, _work, _info = self._call_and_get_lwork(
                gor_un_gqr, qr[:, :M], tau, lwork=-1, overwrite_a=1
            )
        elif self.mode == "economic":
            Q, _work, _info = self._call_and_get_lwork(
                gor_un_gqr, qr, tau, lwork=-1, overwrite_a=1
            )
        else:
            t = qr.dtype.char
            qqr = np.empty((M, M), dtype=t)
            qqr[:, :N] = qr

            # Always overwite qqr -- it's a meaningless intermediate value
            Q, _work, _info = self._call_and_get_lwork(
                gor_un_gqr, qqr, tau, lwork=-1, overwrite_a=1
            )

        outputs[0][0] = Q
        outputs[1][0] = R

        if self.pivoting:
            outputs[2][0] = jpvt

    def L_op(self, inputs, outputs, output_grads):
        """
        Reverse-mode gradient of the QR function.

        References
        ----------
        .. [1] Jinguo Liu. "Linear Algebra Autodiff (complex valued)", blog post https://giggleliu.github.io/posts/2019-04-02-einsumbp/
        .. [2] Hai-Jun Liao, Jin-Guo Liu, Lei Wang, Tao Xiang. "Differentiable Programming Tensor Networks", arXiv:1903.09650v2
        """

        from pytensor.tensor.slinalg import solve_triangular

        (A,) = (cast(ptb.TensorVariable, x) for x in inputs)
        m, n = A.shape

        # Check if we have static shape info, if so we can get a better graph (avoiding the ifelse Op in the output)
        M_static, N_static = A.type.shape
        shapes_unknown = M_static is None or N_static is None

        def _H(x: ptb.TensorVariable):
            return x.conj().mT

        def _copyltu(x: ptb.TensorVariable):
            return ptb.tril(x, k=0) + _H(ptb.tril(x, k=-1))

        if self.mode == "raw":
            raise NotImplementedError("Gradient of qr not implemented for mode=raw")

        elif self.mode == "r":
            k = pt.minimum(m, n)

            # We need all the components of the QR to compute the gradient of A even if we only
            # use the upper triangular component in the cost function.
            props_dict = self._props_dict()
            props_dict["mode"] = "economic"
            props_dict["pivoting"] = False

            qr_op = type(self)(**props_dict)

            Q, R = qr_op(A)
            dQ = Q.zeros_like()

            # Unlike numpy.linalg.qr, scipy.linalg.qr returns the full (m,n) matrix when mode='r', *not* the (k,n)
            # matrix that is computed by mode='economic'. The gradient assumes that dR is of shape (k,n), so we need to
            # slice it to the first k rows. Note that if m <= n, then k = m, so this is safe in all cases.
            dR = cast(ptb.TensorVariable, output_grads[0][:k, :])

        else:
            Q, R = (cast(ptb.TensorVariable, x) for x in outputs)
            if self.mode == "full":
                qr_assert_op = Assert(
                    "Gradient of qr not implemented for m x n matrices with m > n and mode=full"
                )
                R = qr_assert_op(R, ptm.le(m, n))

            new_output_grads = []
            is_disconnected = [
                isinstance(x.type, DisconnectedType) for x in output_grads
            ]
            if all(is_disconnected):
                # This should never be reached by Pytensor
                return [disconnected_type()]  # pragma: no cover

            for disconnected, output_grad, output in zip(
                is_disconnected, output_grads, [Q, R], strict=True
            ):
                if disconnected:
                    new_output_grads.append(output.zeros_like())
                else:
                    new_output_grads.append(output_grad)

            (dQ, dR) = (cast(ptb.TensorVariable, x) for x in new_output_grads)

        if shapes_unknown or M_static >= N_static:
            # gradient expression when m >= n
            M = R @ _H(dR) - _H(dQ) @ Q
            K = dQ + Q @ _copyltu(M)
            A_bar_m_ge_n = _H(solve_triangular(R, _H(K)))

            if not shapes_unknown:
                return [A_bar_m_ge_n]

        # We have to trigger both branches if shapes_unknown is True, so this is purposefully not an elif branch
        if shapes_unknown or M_static < N_static:
            # gradient expression when m < n
            Y = A[:, m:]
            U = R[:, :m]
            dU, dV = dR[:, :m], dR[:, m:]
            dQ_Yt_dV = dQ + Y @ _H(dV)
            M = U @ _H(dU) - _H(dQ_Yt_dV) @ Q
            X_bar = _H(solve_triangular(U, _H(dQ_Yt_dV + Q @ _copyltu(M))))
            Y_bar = Q @ dV
            A_bar_m_lt_n = pt.concatenate([X_bar, Y_bar], axis=1)

            if not shapes_unknown:
                return [A_bar_m_lt_n]

        return [ifelse(ptm.ge(m, n), A_bar_m_ge_n, A_bar_m_lt_n)]


def qr(
    A: TensorLike,
    mode: Literal["full", "r", "economic", "raw", "complete", "reduced"] = "full",
    overwrite_a: bool = False,
    pivoting: bool = False,
    lwork: int | None = None,
):
    """
    QR Decomposition of input matrix `a`.

    The QR decomposition of a matrix `A` is a factorization of the form :math`A = QR`, where `Q` is an orthogonal
    matrix (:math:`Q Q^T = I`) and `R` is an upper triangular matrix.

    This decomposition is useful in various numerical methods, including solving linear systems and least squares
    problems.

    Parameters
    ----------
    A: TensorLike
        Input matrix of shape (M, N) to be decomposed.

    mode: str, one of "full", "economic", "r", or "raw"
        How the QR decomposition is computed and returned. Choosing the mode can avoid unnecessary computations,
        depending on which of the return matrices are needed. Given input matrix with shape  Choices are:

            - "full" (or "complete"): returns `Q` and `R` with dimensions `(M, M)` and `(M, N)`.
            - "economic" (or "reduced"): returns `Q` and `R` with dimensions `(M, K)` and `(K, N)`,
                                         where `K = min(M, N)`.
            - "r": returns only `R` with dimensions `(K, N)`.
            - "raw": returns `H` and `tau` with dimensions `(N, M)` and `(K,)`, where `H` is the matrix of
                     Householder reflections, and tau is the vector of Householder coefficients.

    pivoting: bool, default False
        If True, also return a vector of rank-revealing permutations `P` such that `A[:, P] = QR`.

    overwrite_a: bool, ignored
        Ignored. Included only for consistency with the function signature of `scipy.linalg.qr`. Pytensor will always
        automatically overwrite the input matrix `A` if it is safe to do sol.

    lwork: int, ignored
        Ignored. Included only for consistency with the function signature of `scipy.linalg.qr`. Pytensor will
        automatically determine the optimal workspace size for the QR decomposition.

    Returns
    -------
    Q or H: TensorVariable, optional
        A matrix with orthonormal columns. When mode = 'complete', it is the result is an orthogonal/unitary matrix
        depending on whether a is real/complex. The determinant may be either +/- 1 in that case. If
        mode = 'raw', it is the matrix of Householder reflections. If mode = 'r', Q is not returned.

    R or tau : TensorVariable, optional
        Upper-triangular matrix. If mode = 'raw', it is the vector of Householder coefficients.

    """
    # backwards compatibility from the numpy API
    if mode == "complete":
        mode = "full"
    elif mode == "reduced":
        mode = "economic"

    return Blockwise(QR(mode=mode, pivoting=pivoting, overwrite_a=False))(A)


class Schur(Op):
    """
    Schur Decomposition
    """

    __props__ = ("output", "overwrite_a", "sort")

    def __init__(
        self,
        output: Literal["real", "complex"] = "real",
        overwrite_a: bool = False,
        sort: Literal["lhp", "rhp", "iuc", "ouc"] | None = None,
    ):
        self.output = output
        self.gufunc_signature = "(m,m)->(m,m),(m,m)"
        self.overwrite_a = overwrite_a
        self.sort = sort
        self.destroy_map = {0: [0]} if overwrite_a else {}

        if output not in ["real", "complex"]:
            raise ValueError("output must be 'real' or 'complex'")
        if sort is not None and sort not in ("lhp", "rhp", "iuc", "ouc"):
            raise ValueError("sort must be None or one of ('lhp', 'rhp', 'iuc', 'ouc')")

    def make_sort_function(self):
        sort = self.sort
        sort_t = 1

        match sort:
            case None:
                sort_t = 0

                def sort_function(x, y=None):
                    return None

            case "lhp":

                def sort_function(x, y=None):
                    return x.real < 0.0

            case "rhp":

                def sort_function(x, y=None):
                    return x.real >= 0.0
            case "iuc":

                def sort_function(x, y=None):
                    z = x if y is None else x + y * 1j
                    return abs(z) <= 1.0
            case "ouc":

                def sort_function(x, y=None):
                    z = x if y is None else x + y * 1j
                    return abs(z) > 1.0
            case _:
                raise ValueError(
                    "sort must be None or one of ('lhp', 'rhp', 'iuc', 'ouc')"
                )

        return sort_function, sort_t

    def make_node(self, A):
        A = as_tensor_variable(A)
        assert A.ndim == 2

        out_dtype = A.dtype
        complex_input = out_dtype in ("complex64", "complex128")

        # Scipy behavior: output parameter only affects real inputs
        # Complex inputs always return complex output
        if self.output == "complex" and not complex_input:
            out_dtype = "complex64" if A.dtype == "float32" else "complex128"

        T = matrix(dtype=out_dtype, shape=A.type.shape)
        Z = matrix(dtype=out_dtype, shape=A.type.shape)

        return Apply(self, [A], [T, Z])

    def perform(self, node, inputs, outputs):
        (A,) = inputs
        (T_out, Z_out) = outputs
        overwrite_a = self.overwrite_a

        A_work = A
        if self.output == "complex" and not np.iscomplexobj(A):
            overwrite_a = False
            if A.dtype == np.float32:
                A_work = A.astype(np.complex64)
            else:
                A_work = A.astype(np.complex128)

        if self.output == "real" and np.iscomplexobj(A):
            overwrite_a = False

        (gees,) = scipy_linalg.get_lapack_funcs(("gees",), dtype=A_work.dtype)

        if A_work.size == 0:
            T_out[0] = np.empty_like(A_work, dtype=gees.dtype)
            Z_out[0] = np.empty_like(A_work, dtype=gees.dtype)
            return

        if not np.isfinite(A_work).all():
            T_out[0] = np.full(A_work.shape, np.nan, dtype=gees.dtype)
            Z_out[0] = np.full(A_work.shape, np.nan, dtype=gees.dtype)
            return

        sort_function, sort_t = self.make_sort_function()

        *_, work, _info = gees(
            sort_function, A_work, lwork=-1, overwrite_a=False, sort_t=sort_t
        )
        lwork = int(work[0].real)

        result = gees(
            sort_function,
            A_work,
            lwork=lwork,
            overwrite_a=overwrite_a,
            sort_t=sort_t,
        )

        if np.iscomplexobj(A_work):
            T, _sdim, _w, Z, _work, info = result
        else:
            T, _sdim, _wr, _wi, Z, _work, info = result

        if info != 0:
            T_out[0] = np.full(A_work.shape, np.nan, dtype=node.outputs[0].type.dtype)
            Z_out[0] = np.full(A_work.shape, np.nan, dtype=node.outputs[1].type.dtype)
        else:
            T_out[0] = T
            Z_out[0] = Z

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0], shapes[0]]

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        if not allowed_inplace_inputs:
            return self
        new_props = self._props_dict()  # type: ignore
        new_props["overwrite_a"] = True
        return type(self)(**new_props)


def schur(
    A: TensorLike,
    output: Literal["real", "complex"] = "real",
    sort: Literal["lhp", "rhp", "iuc", "ouc"] | None = None,
) -> tuple[TensorVariable, TensorVariable]:
    """
    Schur Decomposition of input matrix `A`.

    The Schur decomposition of a matrix `A` is a factorization of the form :math:`A = Z T Z^H`,
    where `Z` is a unitary matrix and `T` is either upper-triangular (for complex Schur form)
    or quasi-upper-triangular (for real Schur form with output='real').

    Parameters
    ----------
    A: TensorLike
        Input square matrix of shape (M, M) to be decomposed.

    output: str, one of "real" or "complex"
        For real-valued `A`, if output='real', then the Schur form is quasi-upper-triangular.
        If output='complex', the Schur form is upper-triangular. For complex-valued `A`,
        the Schur form is always upper-triangular regardless of the output parameter.

    sort: str or None, optional
        Specifies whether the upper eigenvalues should be sorted. Available options:

        - None (default): eigenvalues are not sorted
        - 'lhp': left half-plane (real(λ) < 0)
        - 'rhp': right half-plane (real(λ) >= 0)
        - 'iuc': inside unit circle (abs(λ) <= 1)
        - 'ouc': outside unit circle (abs(λ) > 1)

    Returns
    -------
    T : TensorVariable
        Schur form of A. An upper-triangular matrix (or quasi-upper-triangular if output='real').

    Z : TensorVariable
        Unitary Schur transformation matrix such that A = Z @ T @ Z.conj().T

    """
    return Blockwise(Schur(output=output, sort=sort))(A)  # type: ignore[return-value]


_deprecated_names = {
    "solve_continuous_lyapunov",
    "solve_discrete_are",
    "solve_discrete_lyapunov",
}


def __getattr__(name):
    if name in _deprecated_names:
        warnings.warn(
            f"{name} has been moved from tensor/slinalg.py as part of a reorganization "
            "of linear algebra routines in Pytensor. Imports from slinalg.py will fail in Pytensor 3.0.\n"
            f"Please use the stable user-facing linalg API: from pytensor.tensor.linalg import {name}",
            DeprecationWarning,
            stacklevel=2,
        )
        from pytensor.tensor._linalg.solve import linear_control

        return getattr(linear_control, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "block_diag",
    "cho_solve",
    "cholesky",
    "eigvalsh",
    "expm",
    "lu",
    "lu_factor",
    "lu_solve",
    "qr",
    "schur",
    "solve",
    "solve_triangular",
]
