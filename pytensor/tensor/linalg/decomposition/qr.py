from typing import Literal, cast

import numpy as np
from scipy.linalg import get_lapack_funcs

from pytensor import ifelse
from pytensor import tensor as pt
from pytensor.gradient import DisconnectedType, disconnected_type
from pytensor.graph import Apply, Op
from pytensor.raise_op import Assert
from pytensor.tensor import TensorLike
from pytensor.tensor import basic as ptb
from pytensor.tensor import math as ptm
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.type import tensor


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

    def pullback(self, inputs, outputs, output_grads):
        """
        Reverse-mode gradient of the QR function.

        References
        ----------
        .. [1] Jinguo Liu. "Linear Algebra Autodiff (complex valued)", blog post https://giggleliu.github.io/posts/2019-04-02-einsumbp/
        .. [2] Hai-Jun Liao, Jin-Guo Liu, Lei Wang, Tao Xiang. "Differentiable Programming Tensor Networks", arXiv:1903.09650v2
        """
        from pytensor.tensor.linalg.solvers.triangular import solve_triangular

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
