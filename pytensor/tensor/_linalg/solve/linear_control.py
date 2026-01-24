from typing import Literal, cast

import numpy as np
from scipy.linalg import get_lapack_funcs

import pytensor
import pytensor.tensor.basic as ptb
import pytensor.tensor.math as ptm
from pytensor.compile.builders import OpFromGraph
from pytensor.graph import Apply, Op
from pytensor.tensor import TensorLike
from pytensor.tensor.basic import as_tensor_variable, zeros
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.functional import vectorize
from pytensor.tensor.nlinalg import kron, matrix_dot, norm
from pytensor.tensor.reshape import join_dims
from pytensor.tensor.shape import reshape
from pytensor.tensor.slinalg import lu, qr, qz, schur, solve, solve_triangular
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


def _lop_solve_discrete_are(inputs, outputs, output_grads):
    """
    Closed-form gradients for the solution for the discrete Algebraic Riccati equation.

    Gradient computations come from Kao and Hennequin (2020), https://arxiv.org/pdf/2011.11430.pdf
    """
    A, B, Q, R = inputs

    (dX,) = output_grads
    X = solve_discrete_are(A, B, Q, R)

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


class SolveDiscreteARE(OpFromGraph):
    """
    Wrapper Op for solving the discrete Algebraic Riccati equation
    :math:`A^TXA - X - (A^TXB)(R + B^TXB)^{-1}(B^TXA) + Q = 0` for :math:`X`.
    """

    gufunc_signature = "(m,m),(m,n),(m,m),(n,n)->(m,m)"


def solve_discrete_are(
    A: TensorLike,
    B: TensorLike,
    Q: TensorLike,
    R: TensorLike,
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

    Returns
    -------
    X: TensorVariable
        Square matrix of shape M x M, representing the solution to the DARE

    Notes
    -----
    This function is copied from the scipy implementation, found here: https://github.com/scipy/scipy/blob/892baa06054c31bed734423c0f53eaed52b1914b/scipy/linalg/_solvers.py#L687

    Notes are also adapted from the scipy documentation.

    The equation is solved by forming the extended symplectic matrix pencil as described in [1]_,
    :math: `H - \\lambda J`, given by the block matrices:

    .. math::
        H = \begin{bmatrix} A & 0 & B \\\\
                    -Q & I & 0 \\\\
                    0 & 0 & R \\end{bmatrix}
        , \\quad
        J = \begin{bmatrix} I & 0 & 0 \\\\
                    0 & A^H & 0 \\\\
                    0 & -B^H & 0 \\end{bmatrix}

    The stable invariant subspace of the pencil is then computed via the QZ decomposition. Failure conditions are
    linked to the symmetry of the solution matrix :math:`U_2 U_1^{-1}`, as described in [1]_ and [2]_. When the
    solution is not symmetric, NaNs are returned.

    [3]_ describes a balancing procedure for Hamiltonian matrices that can improve numerical stability. This procedure
    is not yet implemented in this function.

    References
    ----------
    .. [1]  P. van Dooren , "A Generalized Eigenvalue Approach For Solving
       Riccati Equations.", SIAM Journal on Scientific and Statistical
       Computing, Vol.2(2), :doi:`10.1137/0902010`

    .. [2] A.J. Laub, "A Schur Method for Solving Algebraic Riccati
       Equations.", Massachusetts Institute of Technology. Laboratory for
       Information and Decision Systems. LIDS-R ; 859. Available online :
       http://hdl.handle.net/1721.1/1301

    .. [3] P. Benner, "Symplectic Balancing of Hamiltonian Matrices", 2001,
       SIAM J. Sci. Comput., 2001, Vol.22(5), :doi:`10.1137/S1064827500367993`
    """
    A, B, Q, R = map(as_tensor_variable, (A, B, Q, R))
    is_complex = any(
        input_matrix.type.numpy_dtype.kind == "c" for input_matrix in (A, B, Q, R)
    )

    A_core = matrix(dtype=A.dtype, shape=A.type.shape[-2:])
    B_core = matrix(dtype=B.dtype, shape=B.type.shape[-2:])
    Q_core = matrix(dtype=Q.dtype, shape=Q.type.shape[-2:])
    R_core = matrix(dtype=R.dtype, shape=R.type.shape[-2:])

    # Given Zmm = zeros(m, m), Zmn = zeros(m, n), Znm = zeros(n, m), E = eye(m)
    # Construct the block matrix H of shape (2m + n, 2m + n):
    # H = block([[  A, Zmm,  B  ],
    #            [ -Q, E,    Zmn],
    #            [Znm, Znm,  R  ]])
    m, n = B_core.shape[-2:]

    H = zeros((2 * m + n, 2 * m + n), dtype=A.dtype)
    H = H[:m, :m].set(A_core)
    H = H[:m, 2 * m :].set(B_core)
    H = H[m : 2 * m, :m].set(-Q_core)
    H = H[m : 2 * m, m : 2 * m].set(ptb.eye(m))
    H = H[2 * m :, 2 * m :].set(R_core)

    # Construct block matrix J of shape (2m + n, 2m + n):
    # J = block([[  E,  Zmm,  Zmn],
    #            [Zmm,  A^H,  Zmn],
    #            [Znm, -B^H,  Zmn]])
    J = zeros((2 * m + n, 2 * m + n), dtype=A_core.dtype)
    J = J[:m, :m].set(ptb.eye(m))
    J = J[m : 2 * m, m : 2 * m].set(A_core.conj().T)
    J = J[2 * m :, m : 2 * m].set(-B_core.conj().T)

    # TODO: Implement balancing procedure from [3]_

    Q_of_QR, _ = qr(H[:, -n:], mode="full")

    H = Q_of_QR[:, n:].conj().T @ H[:, : 2 * m]
    J = Q_of_QR[:, n:].conj().T @ J[:, : 2 * m]
    *_, U = qz(
        H,
        J,
        sort="iuc",
        output="complex" if is_complex else "real",
        return_eigenvalues=False,
    )

    U00 = U[:m, :m]
    U10 = U[m:, :m]

    UP, UL, UU = lu(U00)  # type: ignore[misc]

    lhs = solve_triangular(
        UL.conj().T,
        solve_triangular(UU.conj().T, U10.conj().T, lower=True),
        unit_diagonal=True,
    )
    X = lhs.conj().T @ UP.conj().T

    U_sym = U00.conj().T @ U10
    norm_U_sym = norm(U_sym, ord=1)
    U_sym = U_sym - U_sym.conj().T
    sym_threshold = ptm.maximum(np.spacing(1000.0), 0.1 * norm_U_sym)

    result = ptb.switch(
        norm(U_sym, ord=1) > sym_threshold,
        ptb.full_like(X, np.nan),
        0.5 * (X + X.conj().T),
    )

    core_op = SolveDiscreteARE(
        inputs=[A_core, B_core, Q_core, R_core],
        outputs=[result],
        lop_overrides=_lop_solve_discrete_are,
    )

    return cast(TensorVariable, Blockwise(core_op)(A, B, Q, R))


__all__ = [
    "solve_continuous_lyapunov",
    "solve_discrete_are",
    "solve_discrete_lyapunov",
    "solve_sylvester",
]
