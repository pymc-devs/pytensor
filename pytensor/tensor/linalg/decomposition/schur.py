from typing import Literal

import numpy as np
from scipy import linalg as scipy_linalg
from scipy.linalg import get_lapack_funcs

import pytensor
from pytensor.graph import Apply, Op
from pytensor.tensor import TensorLike
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg.dtype_utils import linalg_output_dtype
from pytensor.tensor.type import matrix, vector
from pytensor.tensor.variable import TensorVariable


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

        out_dtype = linalg_output_dtype(A.dtype)
        complex_input = np.dtype(out_dtype).kind == "c"

        # Scipy behavior: output parameter only affects real inputs
        # Complex inputs always return complex output
        if self.output == "complex" and not complex_input:
            out_dtype = "complex64" if out_dtype == "float32" else "complex128"

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


class QZ(Op):
    """
    QZ Decomposition
    """

    __props__ = (
        "complex_output",
        "overwrite_a",
        "overwrite_b",
        "sort",
        "return_eigenvalues",
    )

    def __init__(
        self,
        complex_output: bool = False,
        overwrite_a: bool = False,
        overwrite_b: bool = False,
        sort: Literal["lhp", "rhp", "iuc", "ouc"] | None = None,
        return_eigenvalues: bool = False,
    ):
        self.complex_output = complex_output
        self.overwrite_a = overwrite_a
        self.overwrite_b = overwrite_b
        self.sort = sort
        self.return_eigenvalues = return_eigenvalues

        if return_eigenvalues:
            self.gufunc_signature = "(m,m),(m,m)->(m,m),(m,m),(m),(m),(m,m),(m,m)"
        else:
            self.gufunc_signature = "(m,m),(m,m)->(m,m),(m,m),(m,m),(m,m)"

        self.destroy_map = {}
        if overwrite_a:
            self.destroy_map[0] = [0]
        if overwrite_b:
            self.destroy_map[1] = [1]

        if sort is not None and sort not in ("lhp", "rhp", "iuc", "ouc"):
            raise ValueError("sort must be None or one of ('lhp', 'rhp', 'iuc', 'ouc')")

    def make_sort_function(
        self, sort: Literal["lhp", "rhp", "iuc", "ouc", "none"] | None = None
    ):
        if sort is None:
            sort = self.sort
        sort_t = 1

        match sort:
            case None | "none":
                sort_t = 0

                def sort_function(alpha, beta):
                    """No sorting."""
                    return None

            case "lhp":

                def sort_function(alpha, beta):
                    """Sort eigenvalues with negative real part (left half-plane) to upper-left."""
                    out = np.empty(alpha.shape, dtype=bool)
                    nonzero = beta != 0
                    out[~nonzero] = False
                    out[nonzero] = (alpha[nonzero] / beta[nonzero]).real < 0.0
                    return out

            case "rhp":

                def sort_function(alpha, beta):
                    """Sort eigenvalues with positive real part (right half-plane) to upper-left."""
                    out = np.empty(alpha.shape, dtype=bool)
                    nonzero = beta != 0
                    out[~nonzero] = False
                    out[nonzero] = (alpha[nonzero] / beta[nonzero]).real > 0.0
                    return out

            case "iuc":

                def sort_function(alpha, beta):
                    """Sort eigenvalues inside the unit circle (abs(lambda) < 1) to upper-left."""
                    out = np.empty(alpha.shape, dtype=bool)
                    nonzero = beta != 0
                    out[~nonzero] = False
                    out[nonzero] = np.abs(alpha[nonzero] / beta[nonzero]) < 1.0
                    return out

            case "ouc":

                def sort_function(alpha, beta):
                    """Sort eigenvalues outside the unit circle (abs(lambda) > 1) to upper-left.

                    Infinite eigenvalues (beta=0, alpha != 0) are included."""
                    out = np.empty(alpha.shape, dtype=bool)
                    alpha_zero = alpha == 0
                    beta_zero = beta == 0
                    beta_nonzero = ~beta_zero

                    out[alpha_zero & beta_zero] = False
                    out[~alpha_zero & beta_zero] = True
                    out[beta_nonzero] = (
                        np.abs(alpha[beta_nonzero] / beta[beta_nonzero]) > 1.0
                    )

                    return out

            case _:
                raise ValueError(
                    "sort must be None or one of ('lhp', 'rhp', 'iuc', 'ouc', 'none')"
                )

        return sort_function, sort_t

    def make_node(self, A, B):
        A = as_tensor_variable(A)
        B = as_tensor_variable(B)
        assert A.ndim == 2
        assert B.ndim == 2

        out_dtype = linalg_output_dtype(A.dtype, B.dtype)

        complex_input = np.dtype(out_dtype).kind == "c"

        # Scipy behavior: output parameter only affects real inputs
        # Complex inputs always return complex output
        if self.complex_output and not complex_input:
            out_dtype = pytensor.scalar.upcast(out_dtype, "complex64")

        AA = matrix(dtype=out_dtype, shape=A.type.shape)
        BB = matrix(dtype=out_dtype, shape=B.type.shape)
        Q = matrix(dtype=out_dtype, shape=A.type.shape)
        Z = matrix(dtype=out_dtype, shape=A.type.shape)

        if self.return_eigenvalues:
            # Eigenvalues can be complex even for real matrices, so alpha is always complex
            # beta has the same dtype as the matrix outputs
            if complex_input or self.complex_output:
                alpha_dtype = out_dtype
            else:
                alpha_dtype = pytensor.scalar.upcast(out_dtype, "complex64")
            alpha = vector(dtype=alpha_dtype, shape=(A.type.shape[0],))
            beta = vector(dtype=out_dtype, shape=(A.type.shape[0],))

            return Apply(self, [A, B], [AA, BB, alpha, beta, Q, Z])

        else:
            return Apply(self, [A, B], [AA, BB, Q, Z])

    def perform(self, node, inputs, outputs):
        (A, B) = inputs

        if self.return_eigenvalues:
            (AA_out, BB_out, alpha_out, beta_out, Q_out, Z_out) = outputs
        else:
            (AA_out, BB_out, Q_out, Z_out) = outputs

        overwrite_a = self.overwrite_a
        overwrite_b = self.overwrite_b

        A_work = A
        B_work = B
        if self.complex_output and not np.iscomplexobj(A):
            overwrite_a = False
            if A.dtype == np.float32:
                A_work = A.astype(np.complex64)
            else:
                A_work = A.astype(np.complex128)

        if self.complex_output and not np.iscomplexobj(B):
            overwrite_b = False
            if B.dtype == np.float32:
                B_work = B.astype(np.complex64)
            else:
                B_work = B.astype(np.complex128)

        if not self.complex_output and np.iscomplexobj(A):
            overwrite_a = False

        if not self.complex_output and np.iscomplexobj(B):
            overwrite_b = False

        (gges,) = scipy_linalg.get_lapack_funcs(("gges",), dtype=A_work.dtype)
        gges_type = gges.typecode

        no_sort_fn, no_sort_t = self.make_sort_function(sort="none")

        # Workspace query
        *_, work, _info = gges(
            no_sort_fn,
            A_work,
            B_work,
            lwork=-1,
            overwrite_a=False,
            overwrite_b=False,
            sort_t=no_sort_t,
        )
        lwork = int(work[0].real)

        # This Op is a combination of scipy.linalg.qz and scipy.linalg.ordqz. They first call gges with no sorting,
        # then do the sorting in a second step if required
        AA, BB, _sdim, *ab, Q, Z, _work, info = gges(
            no_sort_fn,
            A_work,
            B_work,
            lwork=lwork,
            overwrite_a=overwrite_a,
            overwrite_b=overwrite_b,
            sort_t=no_sort_t,
        )

        # If this first pass failed, we skip the sorting step no matter what and return NaNs
        # TODO: When info > 0 and info < A.shape[0], gges fails to put A and B in Shur form but the eigenvalues
        #  are still valid. We could potentially still return something in this case.
        if info != 0:
            AA_out[0] = np.full(A_work.shape, np.nan, dtype=node.outputs[0].type.dtype)
            BB_out[0] = np.full(B_work.shape, np.nan, dtype=node.outputs[1].type.dtype)
            Q_out[0] = np.full(A_work.shape, np.nan, dtype=node.outputs[-2].type.dtype)
            Z_out[0] = np.full(A_work.shape, np.nan, dtype=node.outputs[-1].type.dtype)

            if self.return_eigenvalues:
                alpha_out[0] = np.full(
                    (A_work.shape[0],), np.nan, dtype=node.outputs[2].type.dtype
                )
                beta_out[0] = np.full(
                    (A_work.shape[0],), np.nan, dtype=node.outputs[3].type.dtype
                )

            return

        if self.sort is not None or self.return_eigenvalues:
            if gges_type == "s":
                _alphar, _alphai, beta = ab
                alpha = _alphar + np.complex64(1j) * _alphai
            elif gges_type == "d":
                _alphar, _alphai, beta = ab
                alpha = _alphar + 1j * _alphai
            else:
                alpha, beta = ab

        if self.sort is not None:
            sort_function, _ = self.make_sort_function()
            select = sort_function(alpha, beta)

            tgsen = get_lapack_funcs("tgsen", (AA, BB))
            lwork = 4 * AA.shape[0] + 16 if gges_type in "sd" else 1
            AA, BB, *ab, Q, Z, _, _, _, _, info = tgsen(
                select,
                AA,
                BB,
                Q,
                Z,
                ijob=0,
                lwork=lwork,
                liwork=1,
                overwrite_a=overwrite_a,
                overwrite_b=overwrite_b,
            )

            if gges_type == "s":
                alphar, alphai, beta = ab
                alpha = alphar + np.complex64(1j) * alphai
            elif gges_type == "d":
                alphar, alphai, beta = ab
                alpha = alphar + 1j * alphai
            else:
                alpha, beta = ab

        if info != 0:
            AA_out[0] = np.full(A_work.shape, np.nan, dtype=node.outputs[0].type.dtype)
            BB_out[0] = np.full(B_work.shape, np.nan, dtype=node.outputs[1].type.dtype)
            Q_out[0] = np.full(A_work.shape, np.nan, dtype=node.outputs[-2].type.dtype)
            Z_out[0] = np.full(A_work.shape, np.nan, dtype=node.outputs[-1].type.dtype)

            if self.return_eigenvalues:
                alpha_out[0] = np.full(
                    (A_work.shape[0],), np.nan, dtype=node.outputs[2].type.dtype
                )
                beta_out[0] = np.full(
                    (A_work.shape[0],), np.nan, dtype=node.outputs[3].type.dtype
                )
        else:
            AA_out[0] = AA
            BB_out[0] = BB
            Q_out[0] = Q
            Z_out[0] = Z

            if self.return_eigenvalues:
                alpha_out[0] = alpha
                beta_out[0] = beta

    def infer_shape(self, fgraph, node, shapes):
        A_shape, B_shape = shapes
        if self.return_eigenvalues:
            return [A_shape, B_shape, (A_shape[0],), (A_shape[0],), A_shape, B_shape]
        else:
            return [A_shape, B_shape, A_shape, B_shape]

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        if not allowed_inplace_inputs:
            return self
        new_props = self._props_dict()  # type: ignore
        if 0 in allowed_inplace_inputs:
            new_props["overwrite_a"] = True
        if 1 in allowed_inplace_inputs:
            new_props["overwrite_b"] = True
        return type(self)(**new_props)


def qz(
    A: TensorLike,
    B: TensorLike,
    output: Literal["real", "complex"] = "real",
    sort: Literal["lhp", "rhp", "iuc", "ouc"] | None = None,
    return_eigenvalues: bool = False,
) -> (
    tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable]
    | tuple[
        TensorVariable,
        TensorVariable,
        TensorVariable,
        TensorVariable,
        TensorVariable,
        TensorVariable,
    ]
):
    """
    QZ Decomposition of input matrix pair `(A, B)`.

    The QZ decomposition (also known as the generalized Schur decomposition) of a matrix pair
    `(A, B)` is a factorization of the form :math:`A = Q H Z^H` and :math:`B = Q K Z^H`,
    where `Q` and `Z` are unitary matrices, and `H` and `K` are upper-triangular matrices.

    Parameters
    ----------
    A: TensorLike
        First input square matrix of shape (M, M) to be decomposed.
    B: TensorLike
        Second input square matrix of shape (M, M) to be decomposed.
    output: str, one of "real" or "complex"
        For real-valued `A` and `B`, if output='real', then the Schur forms are quasi-upper-triangular.
        If output='complex', the Schur forms are upper-triangular. For complex-valued `A` and `B`,
        the Schur forms are always upper-triangular regardless of the output parameter.
    sort: str or None, optional
        Specifies whether the generalized eigenvalues should be sorted. Available options are:
        - None (default): eigenvalues are not sorted
        - 'lhp': left half-plane (real(λ) < 0)
        - 'rhp': right half-plane (real(λ) >= 0)
        - 'iuc': inside unit circle (abs(λ) <= 1)
        - 'ouc': outside unit circle (abs(λ) > 1)
    return_eigenvalues: bool, default False
        If True, the function also returns the generalized eigenvalues as two arrays `alpha` and `beta`,
        where the generalized eigenvalues are given by the ratio `alpha / beta`.

    Returns
    -------
    H : TensorVariable
        Schur form of A. An upper-triangular matrix (or quasi-upper-triangular if output='real').
    K : TensorVariable
        Schur form of B. An upper-triangular matrix (or quasi-upper-triangular if output='real').
    Q : TensorVariable
        Unitary matrix such that A = Q @ H @ Z.conj().T and B = Q @ K @ Z.conj().T.
    Z : TensorVariable
        Unitary matrix such that A = Q @ H @ Z.conj().T and B = Q @ K @ Z.conj().T.
    alpha : TensorVariable, optional
        Numerators of the generalized eigenvalues (returned if `return_eigenvalues` is True).
    beta : TensorVariable, optional
        Denominators of the generalized eigenvalues (returned if `return_eigenvalues` is True).

    Notes
    -----
    Unlike scipy.linalg.qz, the sort function is allowed. Behavior in this case follows that of scipy.linalg.ordqz.
    """
    if output not in ["real", "complex"]:
        raise ValueError("output must be 'real' or 'complex'")

    complex_output = output == "complex"
    qz_op = QZ(
        complex_output=complex_output, sort=sort, return_eigenvalues=return_eigenvalues
    )

    return Blockwise(qz_op)(A, B)  # type: ignore[return-value]


def ordqz(
    A: TensorLike,
    B: TensorLike,
    sort: Literal["lhp", "rhp", "iuc", "ouc"] | None = None,
    output: Literal["real", "complex"] = "real",
) -> (
    tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable]
    | tuple[
        TensorVariable,
        TensorVariable,
        TensorVariable,
        TensorVariable,
        TensorVariable,
        TensorVariable,
    ]
):
    """
    Ordered QZ Decomposition of input matrix pair `(A, B)`.

    Alias for `qz`. Included for API consistency with `scipy.linalg`. For details, see the docstring of
    `pytensor.linalg.qz`.
    """
    return qz(A, B, output=output, sort=sort, return_eigenvalues=True)
