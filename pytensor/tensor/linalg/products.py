from pytensor import tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor import basic as ptb
from pytensor.tensor import math as ptm
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg._lazy import scipy_linalg
from pytensor.tensor.symbolic import TensorSymbolicOp
from pytensor.tensor.type import matrix


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

        expm = matrix(dtype=A.dtype, shape=A.type.shape)

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
        from pytensor.tensor.linalg.solvers.general import solve

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

    """
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
