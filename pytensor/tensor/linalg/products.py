import scipy.linalg as scipy_linalg

from pytensor import tensor as pt
from pytensor.compile.builders import OpFromGraph
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor import basic as ptb
from pytensor.tensor import math as ptm
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.type import matrix


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


class KroneckerProduct(OpFromGraph):
    """
    Wrapper Op for Kronecker graphs
    """


def kron(a, b):
    """Kronecker product.

    Same as np.kron(a, b)

    Parameters
    ----------
    a: array_like
    b: array_like

    Returns
    -------
    array_like with a.ndim + b.ndim - 2 dimensions
    """
    a = as_tensor_variable(a)
    b = as_tensor_variable(b)

    if a is b:
        # In case a is the same as b, we need a different variable to build the OFG
        b = a.copy()

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
    output_out_of_shape = a_reshaped * b_reshaped
    output_reshaped = output_out_of_shape.reshape(out_shape)

    return KroneckerProduct(inputs=[a, b], outputs=[output_reshaped])(a, b)


def matrix_dot(*args):
    r"""Shorthand for product between several dots.

    Given :math:`N` matrices :math:`A_0, A_1, .., A_N`, ``matrix_dot`` will
    generate the matrix product between all in the given order, namely
    :math:`A_0 \cdot A_1 \cdot A_2 \cdot .. \cdot A_N`.

    """
    rval = args[0]
    for a in args[1:]:
        rval = ptm.matmul(rval, a)
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


class MultiDot(OpFromGraph):
    """Wrapper Op for a sequence of matrix multiplications.

    Used as a target for rewrites to re-order matrix multiplications for better performance.
    """


def multi_dot(matrices):
    """Compute the dot product of two or more matrices, selecting the fastest evaluation order automatically.

    The problem of optimal matrix multiplication ordering concerns how to place parenthesis in a sequence of matmuls
    to make computation as efficient as possible. For a discussion, see the wikipedia page: https://en.wikipedia.org/wiki/Matrix_chain_multiplication
    The following example from that page illustrates the point. Given 3 matrices A of shape (10, 30), B of shape
    (30, 5), and C of shape (5, 60), the product X = A @ B @ C can be performed in two ways: (A @ B) @ C or A @ (B @ C).

    The first way requires 10*30*5 + 10*5*60 = 4500 multiplications, while the second way requires
    30*5*60 + 10*30*60 = 27000 multiplications. Thus, the first way is much more efficient, and multi_dot will
    automatically select that way for you.

    The exact dynamic programming solution is used to find the optimal contraction path.

    Parameters
    ----------
    matrices : list of TensorVariable
        All inputs must be at least 2d, except for the first and last inputs, which can be 1d. If the first input is
        1d, it is treated as a row vector. If the last input is 1d, it is treated as a column vector.

    Returns
    -------
    result : TensorVariable
        The dot product of the input matrices, from left to right
    """
    if len(matrices) < 2:
        raise ValueError("multi_dot requires at least 2 matrices")

    matrices = [as_tensor_variable(a) for a in matrices]

    for i, a in enumerate(matrices):
        if a.ndim < 1:
            raise ValueError(f"multi_dot: array {i} is a scalar, expected at least 1-D")
        if a.ndim == 1 and 0 < i < len(matrices) - 1:
            raise ValueError(
                f"multi_dot: interior array {i} must be at least 2-D, got 1-D"
            )

    squeeze_first = matrices[0].ndim == 1
    squeeze_last = matrices[-1].ndim == 1
    if squeeze_first:
        matrices[0] = pt.expand_dims(matrices[0], axis=0)
    if squeeze_last:
        matrices[-1] = pt.expand_dims(matrices[-1], axis=1)

    if len(matrices) == 2:
        result = ptm.matmul(matrices[0], matrices[1])
    else:
        # Build naive inner graph, wrap in MultiDot
        inner_inputs = [a.type.clone()(name=f"i{i}") for i, a in enumerate(matrices)]
        inner_output = matrix_dot(*inner_inputs)
        result = MultiDot(inputs=inner_inputs, outputs=[inner_output])(*matrices)

    if squeeze_first:
        result = result[0] if result.ndim == 2 else result[..., 0, :]
    if squeeze_last:
        result = result[..., 0]

    return result
