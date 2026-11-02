import warnings
from typing import cast

import numpy as np

from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor import TensorLike
from pytensor.tensor import basic as ptb
from pytensor.tensor import math as ptm
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg._lazy import scipy_linalg
from pytensor.tensor.linalg.dtype_utils import linalg_output_dtype
from pytensor.tensor.symbolic import TensorSymbolicOp
from pytensor.tensor.type import matrix
from pytensor.tensor.variable import TensorVariable


class Expm(Op):
    """
    Compute the matrix exponential of a square array.
    """

    __props__ = ("overwrite_a",)
    gufunc_signature = "(m,m)->(m,m)"

    def __init__(self, overwrite_a: bool = False):
        self.overwrite_a = overwrite_a
        if self.overwrite_a:
            self.destroy_map = {0: [0]}

    def make_node(self, A):
        A = as_tensor_variable(A)
        assert A.ndim == 2

        dtype = linalg_output_dtype(A.type.dtype)
        expm = matrix(dtype=dtype, shape=A.type.shape)

        return Apply(self, [A], [expm])

    def perform(self, node, inputs, outputs):
        (A,) = inputs
        (expm,) = outputs
        expm[0] = scipy_linalg.expm(A)

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        if not allowed_inplace_inputs:
            return self
        new_props = self._props_dict()  # type: ignore
        new_props["overwrite_a"] = True
        return type(self)(**new_props)

    def pullback(self, inputs, outputs, output_grads):
        r"""Reverse-mode gradient via the augmented-matrix Fréchet derivative.

        The Fréchet derivative :math:`L_A(E)` of :math:`\exp(A)` is the
        upper-right block of one :math:`2n \times 2n` exponential [1]_:

            .. math:: \exp \begin{pmatrix} A & E \\ 0 & A \end{pmatrix}
                      = \begin{pmatrix} \exp(A) & L_A(E) \\ 0 & \exp(A) \end{pmatrix}.

        The Frobenius adjoint of :math:`E \mapsto L_A(E)` is
        :math:`Y \mapsto L_{A^T}(Y)`, so the pullback is recovered by placing
        :math:`A^T` on the diagonal and the cotangent :math:`\bar{A}` in place
        of :math:`E`.

        References
        ----------
        .. [1] Mathias, R. (1996). A chain rule for matrix functions and
               applications. *SIAM J. Matrix Anal. Appl.* 17(3), 610-620.
        """
        (A,) = inputs
        (A_bar,) = output_grads

        n = A.shape[-1]
        zero = ptb.zeros_like(A)
        top = ptb.join(-1, A.mT, A_bar)
        bot = ptb.join(-1, zero, A.mT)
        aug = ptb.join(-2, top, bot)

        return [expm(aug)[..., :n, n:]]

    def infer_shape(self, node, shapes):
        return [shapes[0]]


expm = Blockwise(Expm())


class Sqrtm(Op):
    """
    Compute the principal square root of a square array.
    """

    __props__ = ()
    gufunc_signature = "(m,m)->(m,m)"

    def make_node(self, A):
        A = as_tensor_variable(A)
        assert A.ndim == 2

        dtype = linalg_output_dtype(A.type.dtype)
        sqrtm = matrix(dtype=dtype, shape=A.type.shape)

        return Apply(self, [A], [sqrtm])

    def perform(self, node, inputs, outputs):
        (A,) = inputs
        (sqrtm,) = outputs
        out_dtype = node.outputs[0].type.numpy_dtype

        S = scipy_linalg.sqrtm(A)
        if S.dtype.kind == "c" and out_dtype.kind != "c":
            # Older scipy returns a complex dtype for any real matrix with complex
            # eigenvalues, so the dtype alone does not say whether a real root exists.
            # LAPACK reports the real eigenvalues of a real matrix with an exactly
            # zero imaginary part.
            eigenvalues = np.linalg.eigvals(A)
            if np.any((eigenvalues.imag == 0) & (eigenvalues.real < 0)):
                S = np.full(A.shape, np.nan, dtype=out_dtype)
            else:
                S = S.real

        sqrtm[0] = S.astype(out_dtype, copy=False)

    def pullback(self, inputs, outputs, output_grads):
        r"""Reverse-mode gradient via the Sylvester equation of the Fréchet derivative.

        Differentiating :math:`S S = A` gives :math:`S \, dS + dS \, S = dA`. The
        Frobenius adjoint of :math:`X \mapsto S X + X S` is
        :math:`Y \mapsto S^H Y + Y S^H`, so the pullback :math:`\bar{A}` solves

            .. math:: S^H \bar{A} + \bar{A} S^H = \bar{S}.

        The equation has a unique solution whenever no two eigenvalues of
        :math:`S` sum to zero, which holds for every principal root of a
        nonsingular matrix, including those with repeated eigenvalues.
        """
        from pytensor.tensor.linalg.solvers.linear_control import solve_sylvester

        (S,) = outputs
        (S_bar,) = output_grads
        S_H = S.conj().mT

        return [solve_sylvester(S_H, S_H, S_bar)]

    def infer_shape(self, node, shapes):
        return [shapes[0]]


def sqrtm(A: TensorLike) -> TensorVariable:
    """Principal square root of a square matrix.

    Same as ``scipy.linalg.sqrtm``, except that a real input always gives a real
    output. A real matrix with an eigenvalue on the negative real axis has no real
    principal square root, and the result is then filled with NaN. A singular
    ``A`` need not have a square root, and the result for one can differ between
    backends.

    When ``A`` is known to be symmetric or positive definite, for example through
    :func:`pytensor.assumptions.assume`, the root is computed from ``eigh(A)``.

    Parameters
    ----------
    A : TensorLike
        Square matrix of shape ``(..., M, M)``.

    Returns
    -------
    TensorVariable
        Matrix ``S`` of shape ``(..., M, M)`` with ``S @ S = A``.
    """
    return cast(TensorVariable, Blockwise(Sqrtm())(A))


class KroneckerProduct(TensorSymbolicOp):
    """Kronecker product Op."""

    def build_inner_graph(self, a, b):
        if a.ndim + b.ndim <= 2:
            raise TypeError(
                "kron: inputs dimensions must sum to 3 or more. "
                f"You passed {int(a.ndim)} and {int(b.ndim)}."
            )
        if a.ndim < b.ndim:
            a = ptb.expand_dims(a, tuple(range(b.ndim - a.ndim)))
        elif b.ndim < a.ndim:
            b = ptb.expand_dims(b, tuple(range(a.ndim - b.ndim)))
        a_reshaped = ptb.expand_dims(a, tuple(range(1, 2 * a.ndim, 2)))
        b_reshaped = ptb.expand_dims(b, tuple(range(0, 2 * b.ndim, 2)))
        out_shape = tuple(a.shape[i] * b.shape[i] for i in range(a.ndim))
        output = (a_reshaped * b_reshaped).reshape(out_shape)
        return [output]


_kron = KroneckerProduct()


def kron(a, b):
    """Kronecker product.

    Same as ``np.kron(a, b)``.

    Parameters
    ----------
    a : array_like
    b : array_like

    Returns
    -------
    array_like with a.ndim + b.ndim - 2 dimensions

    """
    return _kron(a, b)


def matrix_dot(*args):
    r"""Shorthand for product between several dots.

    Given :math:`N` matrices :math:`A_0, A_1, .., A_N`, ``matrix_dot`` will
    generate the matrix product between all in the given order, namely
    :math:`A_0 \cdot A_1 \cdot A_2 \cdot .. \cdot A_N`.

    .. deprecated::
        Use the ``@`` operator or ``pt.matmul`` instead.

    """
    warnings.warn(
        "matrix_dot is deprecated. Use the @ operator or pt.matmul instead.",
        FutureWarning,
        stacklevel=2,
    )

    rval = args[0]
    for a in args[1:]:
        rval = ptm.dot(rval, a)
    return rval


def matrix_power(M, n):
    r"""Raise a square matrix, ``M``, to the (integer) power ``n``.

    This implementation uses exponentiation by squaring which is
    significantly faster than the naive implementation.
    The time complexity for exponentiation by squaring is
    :math: `\mathcal{O}((n \log M)^k)`

    Parameters
    ----------
    M: TensorVariable
    n: int

    """
    if n < 0:
        from pytensor.tensor.linalg.inverse import pinv

        M = pinv(M)
        n = abs(n)

    # Shortcuts when 0 < n <= 3
    if n == 0:
        return ptb.eye(M.shape[-2])

    elif n == 1:
        return M

    elif n == 2:
        return ptm.dot(M, M)

    elif n == 3:
        return ptm.dot(ptm.dot(M, M), M)

    result = z = None

    while n > 0:
        z = M if z is None else ptm.dot(z, z)
        n, bit = divmod(n, 2)
        if bit:
            result = z if result is None else ptm.dot(result, z)

    return result
