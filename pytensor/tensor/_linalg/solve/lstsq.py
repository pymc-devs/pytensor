import numpy as np

from pytensor import scalar as ps
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.type import dmatrix, dvector, iscalar, matrix


class Lstsq(Op):
    __props__ = ()

    def make_node(self, x, y, rcond):
        x = as_tensor_variable(x)
        y = as_tensor_variable(y)
        rcond = as_tensor_variable(rcond)
        return Apply(
            self,
            [x, y, rcond],
            [
                dmatrix(),
                dvector(),
                iscalar(),
                dvector(),
            ],
        )

    def perform(self, node, inputs, outputs):
        zz = np.linalg.lstsq(inputs[0], inputs[1], inputs[2])
        outputs[0][0] = zz[0]
        outputs[1][0] = zz[1]
        outputs[2][0] = np.asarray(zz[2])
        outputs[3][0] = zz[3]


lstsq = Lstsq()


class TensorSolve(Op):
    """
    PyTensor utilization of numpy.linalg.tensorsolve
    Class wrapper for tensorsolve function.

    """

    __props__ = ("axes",)

    def __init__(self, axes=None):
        self.axes = axes

    def make_node(self, a, b):
        a = as_tensor_variable(a)
        b = as_tensor_variable(b)
        out_dtype = ps.upcast(a.dtype, b.dtype)
        x = matrix(dtype=out_dtype)
        return Apply(self, [a, b], [x])

    def perform(self, node, inputs, outputs):
        (
            a,
            b,
        ) = inputs
        (x,) = outputs
        x[0] = np.linalg.tensorsolve(a, b, self.axes)


def tensorsolve(a, b, axes=None):
    """
    PyTensor utilization of numpy.linalg.tensorsolve.

    Solve the tensor equation ``a x = b`` for x.
    It is assumed that all indices of `x` are summed over in the product,
    together with the rightmost indices of `a`, as is done in, for example,
    ``tensordot(a, x, axes=len(b.shape))``.

    Parameters
    ----------
    a : array_like
        Coefficient tensor, of shape ``b.shape + Q``. `Q`, a tuple, equals
        the shape of that sub-tensor of `a` consisting of the appropriate
        number of its rightmost indices, and must be such that
        ``prod(Q) == prod(b.shape)`` (in which sense `a` is said to be
        'square').
    b : array_like
        Right-hand tensor, which can be of any shape.
    axes : tuple of ints, optional
        Axes in `a` to reorder to the right, before inversion.
        If None (default), no reordering is done.
    Returns
    -------
    x : ndarray, shape Q
    Raises
    ------
    LinAlgError
        If `a` is singular or not 'square' (in the above sense).
    """

    return TensorSolve(axes)(a, b)
