from collections.abc import Sequence
from typing import cast

import numpy as np
from scipy import linalg as scipy_linalg
from scipy.linalg import get_lapack_funcs

from pytensor import tensor as pt
from pytensor.gradient import DisconnectedType
from pytensor.graph import Apply, Op
from pytensor.tensor import TensorLike
from pytensor.tensor import basic as ptb
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.type import matrix, tensor, vector
from pytensor.tensor.variable import TensorVariable


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

    def pullback(
        self,
        inputs: Sequence[ptb.Variable],
        outputs: Sequence[ptb.Variable],
        output_grads: Sequence[ptb.Variable],
    ) -> list[ptb.Variable]:
        r"""
        Derivation is due to Differentiation of Matrix Functionals Using Triangular Factorization
        F. R. De Hoog, R.S. Anderssen, M. A. Lukas
        """
        from pytensor.tensor._linalg.solve.triangular import solve_triangular

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

    def pullback(self, inputs, outputs, output_gradients):
        from pytensor.tensor._linalg.solve.triangular import solve_triangular

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
