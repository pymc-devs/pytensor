import numpy as np

from pytensor.gradient import DisconnectedType, disconnected_type
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor import basic as ptb
from pytensor.tensor import math as ptm
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg.dtype_utils import linalg_output_dtype
from pytensor.tensor.type import matrix


class MatrixPinv(Op):
    __props__ = ("hermitian",)
    gufunc_signature = "(m,n)->(n,m)"

    def __init__(self, hermitian):
        self.hermitian = hermitian

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        out_dtype = linalg_output_dtype(x.type.dtype)
        return Apply(
            self,
            [x],
            [matrix(shape=(x.type.shape[1], x.type.shape[0]), dtype=out_dtype)],
        )

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        z[0] = np.linalg.pinv(x, hermitian=self.hermitian)

    def pullback(self, inputs, outputs, g_outputs):
        r"""The gradient function should return

            .. math:: V\frac{\partial X^+}{\partial X},

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. According to `Wikipedia
        <https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse#Derivative>`_,
        this corresponds to

            .. math:: (-X^+ V^T X^+ + X^+ X^{+T} V (I - X X^+) + (I - X^+ X) V X^{+T} X^+)^T.
        """
        (x,) = inputs
        (z,) = outputs
        (gz,) = g_outputs

        from pytensor.tensor.linalg.products import matrix_dot

        x_dot_z = ptm.dot(x, z)
        z_dot_x = ptm.dot(z, x)

        grad = (
            -matrix_dot(z, gz.T, z)
            + matrix_dot(z, z.T, gz, (ptb.identity_like(x_dot_z) - x_dot_z))
            + matrix_dot((ptb.identity_like(z_dot_x) - z_dot_x), gz, z.T, z)
        ).T
        return [grad]

    def infer_shape(self, fgraph, node, shapes):
        return [list(reversed(shapes[0]))]


def pinv(x, hermitian=False):
    """Computes the pseudo-inverse of a matrix :math:`A`.

    The pseudo-inverse of a matrix :math:`A`, denoted :math:`A^+`, is
    defined as: "the matrix that 'solves' [the least-squares problem]
    :math:`Ax = b`," i.e., if :math:`\\bar{x}` is said solution, then
    :math:`A^+` is that matrix such that :math:`\\bar{x} = A^+b`.

    Note that :math:`Ax=AA^+b`, so :math:`AA^+` is close to the identity matrix.
    This method is not faster than `matrix_inverse`. Its strength comes from
    that it works for non-square matrices.
    If you have a square matrix though, `matrix_inverse` can be both more
    exact and faster to compute. Also this op does not get optimized into a
    solve op.

    """
    return Blockwise(MatrixPinv(hermitian=hermitian))(x)


class MatrixInverse(Op):
    r"""Computes the inverse of a matrix :math:`A`.

    Given a square matrix :math:`A`, ``matrix_inverse`` returns a square
    matrix :math:`A_{inv}` such that the dot product :math:`A \cdot A_{inv}`
    and :math:`A_{inv} \cdot A` equals the identity matrix :math:`I`.

    Notes
    -----
    When possible, the call to this op will be optimized to the call
    of ``solve``.

    """

    __props__ = ()
    gufunc_signature = "(m,m)->(m,m)"
    gufunc_spec = ("numpy.linalg.inv", 1, 1)

    def __init__(self):
        pass

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        out_dtype = linalg_output_dtype(x.type.dtype)
        return Apply(self, [x], [matrix(shape=x.type.shape, dtype=out_dtype)])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        z[0] = np.linalg.inv(x)

    def pullback(self, inputs, outputs, g_outputs):
        r"""The gradient function should return

            .. math:: V\frac{\partial X^{-1}}{\partial X},

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to

            .. math:: (X^{-1} \cdot V^{T} \cdot X^{-1})^T.

        """
        (x,) = inputs
        xi = self(x)
        (gz,) = g_outputs
        from pytensor.tensor.linalg.products import matrix_dot

        # ptm.dot(gz.T,xi)
        return [-matrix_dot(xi, gz.T, xi).T]

    def pushforward(self, inputs, outputs, eval_points):
        r"""The gradient function should return

            .. math:: \frac{\partial X^{-1}}{\partial X}V,

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to

            .. math:: X^{-1} \cdot V \cdot X^{-1}.

        """
        (x,) = inputs
        xi = self(x)
        (ev,) = eval_points
        if isinstance(ev.type, DisconnectedType):
            return [disconnected_type()]
        from pytensor.tensor.linalg.products import matrix_dot

        return [-matrix_dot(xi, ev, xi)]

    def infer_shape(self, fgraph, node, shapes):
        return shapes


inv = matrix_inverse = Blockwise(MatrixInverse())


class TensorInv(Op):
    """
    Class wrapper for tensorinv() function;
    PyTensor utilization of numpy.linalg.tensorinv;
    """

    __props__ = ("ind",)

    def __init__(self, ind=2):
        self.ind = ind

    def make_node(self, a):
        a = as_tensor_variable(a)
        out = a.type()
        return Apply(self, [a], [out])

    def perform(self, node, inputs, outputs):
        (a,) = inputs
        (x,) = outputs
        x[0] = np.linalg.tensorinv(a, self.ind)

    def infer_shape(self, fgraph, node, shapes):
        sp = shapes[0][self.ind :] + shapes[0][: self.ind]
        return [sp]


def tensorinv(a, ind=2):
    """
    PyTensor utilization of numpy.linalg.tensorinv;

    Compute the 'inverse' of an N-dimensional array.
    The result is an inverse for `a` relative to the tensordot operation
    ``tensordot(a, b, ind)``, i. e., up to floating-point accuracy,
    ``tensordot(tensorinv(a), a, ind)`` is the "identity" tensor for the
    tensordot operation.

    Parameters
    ----------
    a : array_like
        Tensor to 'invert'. Its shape must be 'square', i. e.,
        ``prod(a.shape[:ind]) == prod(a.shape[ind:])``.
    ind : int, optional
        Number of first indices that are involved in the inverse sum.
        Must be a positive integer, default is 2.

    Returns
    -------
    b : ndarray
        `a`'s tensordot inverse, shape ``a.shape[ind:] + a.shape[:ind]``.

    Raises
    ------
    LinAlgError
        If `a` is singular or not 'square' (in the above sense).
    """
    return TensorInv(ind)(a)
