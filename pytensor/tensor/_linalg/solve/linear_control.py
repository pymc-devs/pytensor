from typing import Literal, cast

import numpy as np
from scipy import linalg as scipy_linalg
from scipy.linalg import get_lapack_funcs

import pytensor
import pytensor.tensor.basic as ptb
from pytensor.compile.builders import OpFromGraph
from pytensor.graph import Apply, Op
from pytensor.tensor import TensorLike
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.functional import vectorize
from pytensor.tensor.nlinalg import kron, matrix_dot
from pytensor.tensor.reshape import join_dims
from pytensor.tensor.shape import reshape
from pytensor.tensor.slinalg import schur, solve
from pytensor.tensor.type import matrix
from pytensor.tensor.variable import TensorVariable


class TRSYL(Op):
    """
    Wrapper around LAPACK's `trsyl` function to solve the Sylvester equation:

        op(A) @ X + X @ op(B) = alpha * C

    Where `op(A)` is either `A` or `A^T`, depending on the options passed to `trsyl`. A and B must be
    in Schur canonical form: block upper triangular matrices with 1x1 and 2x2 blocks on the diagonal;
    each 2x2 diagonal block has its diagonal elements equal and its off-diagonal elements opposite in sign.

    This Op is not public facing. Instead, it is intended to be used as a building block for higher-level
    linear control solvers, such as `SolveSylvester` and `SolveContinuousLyapunov`.
    """

    __props__ = ("overwrite_c",)
    gufunc_signature = "(m,m),(n,n),(m,n)->(m,n)"

    def __init__(self, overwrite_c=False):
        self.overwrite_c = overwrite_c
        if self.overwrite_c:
            self.destroy_map = {0: [2]}

    def make_node(self, A, B, C):
        A = as_tensor_variable(A)
        B = as_tensor_variable(B)
        C = as_tensor_variable(C)

        out_dtype = pytensor.scalar.upcast(A.dtype, B.dtype, C.dtype)

        output_shape = list(C.type.shape)
        if output_shape[0] is None and A.type.shape[0] is not None:
            output_shape[0] = A.type.shape[0]
        if output_shape[1] is None and B.type.shape[0] is not None:
            output_shape[1] = B.type.shape[0]

        X = ptb.tensor(dtype=out_dtype, shape=tuple(output_shape))

        return Apply(self, [A, B, C], [X])

    def perform(self, node, inputs, outputs_storage):
        (A, B, C) = inputs
        X = outputs_storage[0]

        out_dtype = node.outputs[0].type.dtype
        (trsyl,) = get_lapack_funcs(("trsyl",), (A, B, C))

        if A.size == 0 or B.size == 0:
            return np.empty_like(C, dtype=out_dtype)

        Y, scale, info = trsyl(A, B, C, overwrite_c=self.overwrite_c)

        if info < 0:
            return np.full_like(C, np.nan, dtype=out_dtype)

        Y *= scale
        X[0] = Y

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[2]]

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        if not allowed_inplace_inputs:
            return self
        new_props = self._props_dict()  # type: ignore
        new_props["overwrite_c"] = True
        return type(self)(**new_props)


def _lop_solve_continuous_sylvester(inputs, outputs, output_grads):
    """
    Closed-form gradients for the solution for the Sylvester equation.

    Gradient computations come from Kao and Hennequin (2020), https://arxiv.org/pdf/2011.11430.pdf

    Note that these authors write the equation as AP + PB + C = 0. The code here follows scipy notation,
    so P = X and C = -Q. This change of notation requires minor adjustment to equations 10 and 11c
    """
    A, B, _ = inputs
    (dX,) = output_grads
    (X,) = outputs

    S = solve_sylvester(A.conj().mT, B.conj().mT, -dX)  # Eq 10
    A_bar = S @ X.conj().mT  # Eq 11a
    B_bar = X.conj().mT @ S  # Eq 11b
    Q_bar = -S  # Eq 11c

    return [A_bar, B_bar, Q_bar]


class SolveSylvester(OpFromGraph):
    """
    Wrapper Op for solving the continuous Sylvester equation :math:`AX + XB = C` for :math:`X`.
    """

    gufunc_signature = "(m,m),(n,n),(m,n)->(m,n)"


def solve_sylvester(A: TensorLike, B: TensorLike, Q: TensorLike) -> TensorVariable:
    """
    Solve the Sylvester equation :math:`AX + XB = C` for :math:`X`.

    Following scipy notation, this function solves the continuous-time Sylvester equation.

    Parameters
    ----------
    A: TensorLike
        Square matrix of shape ``M x M``.
    B: TensorLike
        Square matrix of shape ``N x N``.
    Q: TensorLike
        Matrix of shape ``M x N``.

    Returns
    -------
    X: TensorVariable
        Matrix of shape ``M x N``.
    """
    A = as_tensor_variable(A)
    B = as_tensor_variable(B)
    Q = as_tensor_variable(Q)

    A_matrix = matrix(dtype=A.dtype, shape=A.type.shape[-2:])
    B_matrix = matrix(dtype=B.dtype, shape=B.type.shape[-2:])
    Q_matrix = matrix(dtype=Q.dtype, shape=Q.type.shape[-2:])

    R, U = schur(A_matrix, output="real")
    S, V = schur(B_matrix, output="real")
    F = U.conj().mT @ Q_matrix @ V

    _trsyl = Blockwise(TRSYL())

    Y = _trsyl(R, S, F)
    X = U @ Y @ V.conj().mT

    op = SolveSylvester(
        inputs=[A_matrix, B_matrix, Q_matrix],
        outputs=[X],
        lop_overrides=_lop_solve_continuous_sylvester,
    )

    return cast(TensorVariable, Blockwise(op)(A, B, Q))


def solve_continuous_lyapunov(A: TensorLike, Q: TensorLike) -> TensorVariable:
    """
    Solve the continuous Lyapunov equation :math:`A X + X A^H + Q = 0`.

    Note that the lyapunov equation is a special case of the Sylvester equation, with :math:`B = A^H`. This function
    thus simply calls `solve_sylvester` with the appropriate arguments.

    Parameters
    ----------
    A: TensorLike
        Square matrix of shape ``N x N``.
    Q: TensorLike
        Square matrix of shape ``N x N``.

    Returns
    -------
    X: TensorVariable
        Square matrix of shape ``N x N``

    """
    A = as_tensor_variable(A)
    Q = as_tensor_variable(Q)

    return solve_sylvester(A, A.conj().mT, Q)


class SolveBilinearDiscreteLyapunov(OpFromGraph):
    """
    Wrapper Op for solving the discrete Lyapunov equation :math:`A X A^H - X = Q` for :math:`X`.

    Required so that backends that do not support method='bilinear' in `solve_discrete_lyapunov` can be rewritten
    to method='direct'.
    """


def solve_discrete_lyapunov(
    A: TensorLike,
    Q: TensorLike,
    method: Literal["direct", "bilinear"] = "bilinear",
) -> TensorVariable:
    """Solve the discrete Lyapunov equation :math:`A X A^H - X = Q`.

    Parameters
    ----------
    A: TensorLike
        Square matrix of shape N x N
    Q: TensorLike
        Square matrix of shape N x N
    method: str, one of ``"direct"`` or ``"bilinear"``
        Solver method used, . ``"direct"`` solves the problem directly via matrix inversion.  This has a pure
        PyTensor implementation and can thus be cross-compiled to supported backends, and should be preferred when
         ``N`` is not large. The direct method scales poorly with the size of ``N``, and the bilinear can be
        used in these cases.

    Returns
    -------
    X: TensorVariable
        Square matrix of shape ``N x N``. Solution to the Lyapunov equation

    """
    if method not in ["direct", "bilinear"]:
        raise ValueError(
            f'Parameter "method" must be one of "direct" or "bilinear", found {method}'
        )

    A = as_tensor_variable(A)
    Q = as_tensor_variable(Q)

    if method == "direct":
        vec_kron = vectorize(kron, signature="(n,n),(n,n)->(m,m)")
        AxA = vec_kron(A, A.conj())
        I = ptb.eye(AxA.shape[-1])

        vec_Q = join_dims(Q, start_axis=-2, n_axes=2)
        vec_X = solve(I - AxA, vec_Q, b_ndim=1)

        return reshape(vec_X, A.shape)

    elif method == "bilinear":
        I = ptb.eye(A.shape[-2])

        B_1 = A.conj().mT + I
        B_2 = A.conj().mT - I
        B = solve(B_1.mT, B_2.mT).mT

        AI_inv_Q = solve(A + I, Q)
        C = 2 * solve(B_1.mT, AI_inv_Q.mT).mT

        X = solve_continuous_lyapunov(B.conj().mT, -C)

        op = SolveBilinearDiscreteLyapunov(inputs=[A, Q], outputs=[X])
        return cast(TensorVariable, op(A, Q))

    else:
        raise ValueError(f"Unknown method {method}")


class SolveDiscreteARE(Op):
    __props__ = ("enforce_Q_symmetric",)
    gufunc_signature = "(m,m),(m,n),(m,m),(n,n)->(m,m)"

    def __init__(self, enforce_Q_symmetric: bool = False):
        self.enforce_Q_symmetric = enforce_Q_symmetric

    def make_node(self, A, B, Q, R):
        A = as_tensor_variable(A)
        B = as_tensor_variable(B)
        Q = as_tensor_variable(Q)
        R = as_tensor_variable(R)

        out_dtype = pytensor.scalar.upcast(A.dtype, B.dtype, Q.dtype, R.dtype)
        X = pytensor.tensor.matrix(dtype=out_dtype)

        return pytensor.graph.basic.Apply(self, [A, B, Q, R], [X])

    def perform(self, node, inputs, output_storage):
        A, B, Q, R = inputs
        X = output_storage[0]

        if self.enforce_Q_symmetric:
            Q = 0.5 * (Q + Q.T)

        out_dtype = node.outputs[0].type.dtype
        X[0] = scipy_linalg.solve_discrete_are(A, B, Q, R).astype(out_dtype)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]

    def grad(self, inputs, output_grads):
        # Gradient computations come from Kao and Hennequin (2020), https://arxiv.org/pdf/2011.11430.pdf
        A, B, Q, R = inputs

        (dX,) = output_grads
        X = self(A, B, Q, R)

        K_inner = R + matrix_dot(B.T, X, B)

        # K_inner is guaranteed to be symmetric, because X and R are symmetric
        K_inner_inv_BT = solve(K_inner, B.T, assume_a="sym")
        K = matrix_dot(K_inner_inv_BT, X, A)

        A_tilde = A - B.dot(K)

        dX_symm = 0.5 * (dX + dX.T)
        S = solve_discrete_lyapunov(A_tilde, dX_symm)

        A_bar = 2 * matrix_dot(X, A_tilde, S)
        B_bar = -2 * matrix_dot(X, A_tilde, S, K.T)
        Q_bar = S
        R_bar = matrix_dot(K, S, K.T)

        return [A_bar, B_bar, Q_bar, R_bar]


def solve_discrete_are(
    A: TensorLike,
    B: TensorLike,
    Q: TensorLike,
    R: TensorLike,
    enforce_Q_symmetric: bool = False,
) -> TensorVariable:
    """
    Solve the discrete Algebraic Riccati equation :math:`A^TXA - X - (A^TXB)(R + B^TXB)^{-1}(B^TXA) + Q = 0`.

    Discrete-time Algebraic Riccati equations arise in the context of optimal control and filtering problems, as the
    solution to Linear-Quadratic Regulators (LQR), Linear-Quadratic-Guassian (LQG) control problems, and as the
    steady-state covariance of the Kalman Filter.

    Such problems typically have many solutions, but we are generally only interested in the unique *stabilizing*
    solution. This stable solution, if it exists, will be returned by this function.

    Parameters
    ----------
    A: TensorLike
        Square matrix of shape M x M
    B: TensorLike
        Square matrix of shape M x M
    Q: TensorLike
        Symmetric square matrix of shape M x M
    R: TensorLike
        Square matrix of shape N x N
    enforce_Q_symmetric: bool
        If True, the provided Q matrix is transformed to 0.5 * (Q + Q.T) to ensure symmetry

    Returns
    -------
    X: TensorVariable
        Square matrix of shape M x M, representing the solution to the DARE
    """

    return cast(
        TensorVariable, Blockwise(SolveDiscreteARE(enforce_Q_symmetric))(A, B, Q, R)
    )


__all__ = [
    "solve_continuous_lyapunov",
    "solve_discrete_are",
    "solve_discrete_lyapunov",
    "solve_sylvester",
]
