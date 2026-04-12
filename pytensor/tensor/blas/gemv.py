"""BLAS GEMV operation: matrix-vector multiply with accumulation.

Computes: beta * y + alpha * dot(A, x)
"""

import numpy as np
from scipy.linalg import get_blas_funcs

from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.blas._core import must_initialize_y_gemv
from pytensor.tensor.type import DenseTensorType


class Gemv(Op):
    """
    expression is beta * y + alpha * A x

    A is matrix
    x, y are vectors
    alpha, beta are scalars
    output is a vector that can be inplace on y

    """

    __props__ = ("inplace",)

    def __init__(self, inplace):
        self.inplace = inplace
        if inplace:
            self.destroy_map = {0: [0]}

    def __str__(self):
        if self.inplace:
            return f"{self.__class__.__name__}{{inplace}}"
        else:
            return f"{self.__class__.__name__}{{no_inplace}}"

    def make_node(self, y, alpha, A, x, beta):
        y = as_tensor_variable(y)
        x = as_tensor_variable(x)
        A = as_tensor_variable(A)
        alpha = as_tensor_variable(alpha)
        beta = as_tensor_variable(beta)
        if y.dtype != A.dtype or y.dtype != x.dtype:
            raise TypeError(
                "Gemv requires matching dtypes", (y.dtype, A.dtype, x.dtype)
            )
        if A.ndim != 2:
            raise TypeError("gemv requires matrix for A", A.type)
        if x.ndim != 1:
            raise TypeError("gemv requires vector for x", x.type)
        if y.ndim != 1:
            raise TypeError("gemv requires vector for y", y.type)

        inputs = [y, alpha, A, x, beta]

        if any(not isinstance(i.type, DenseTensorType) for i in inputs):
            raise NotImplementedError("Only dense tensor types are supported")

        return Apply(self, inputs, [y.type()])

    def perform(self, node, inputs, out_storage):
        y, alpha, A, x, beta = inputs
        if (
            y.shape[0] != 0
            and x.shape[0] != 0
            and y.dtype in {"float32", "float64", "complex64", "complex128"}
        ):
            gemv = get_blas_funcs("gemv", dtype=y.dtype)

            if A.shape[0] != y.shape[0] or A.shape[1] != x.shape[0]:
                raise ValueError(
                    "Incompatible shapes for gemv "
                    f"(beta * y + alpha * dot(A, x)). y: {y.shape}, A: {A.shape}, x: {x.shape}"
                )

            if beta == 0 and must_initialize_y_gemv():
                # Most BLAS implementations of GEMV ignore y=nan when beta=0
                # PyTensor considers that the correct behavior,
                # and even exploits it to avoid copying or initializing outputs.
                # By deciding to exploit this, however, it becomes our responsibility
                # to ensure the behavior even in the rare cases BLAS deviates,
                # or users will get errors, even for graphs that had no nan to begin with.
                y.fill(0)

            # Here I suppose that A is in c order. If we don't make it
            #  explicitly as fortran order, scipy 0.7.2 seam to create
            #  a copy in fortran order instead of just reshaping it
            #  and using the trans flag.
            # If A is already in fortran order, make it in c order and using the
            #  trans flag don't seam to cause slowdown.
            # out_storage[0][0] = gemv(alpha, A, x, beta, y,
            #                         overwrite_y=self.inplace)
            out_storage[0][0] = gemv(
                alpha, A.T, x, beta, y, overwrite_y=self.inplace, trans=True
            )
        else:
            out = np.dot(A, x)
            if alpha != 1:
                out *= alpha
            if beta != 0:
                if beta != 1:
                    out += beta * y
                else:
                    out += y
            out_storage[0][0] = np.asarray(out, dtype=y.dtype)

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]]


gemv_no_inplace = Gemv(inplace=False)
gemv_inplace = Gemv(inplace=True)
# For the user interface. Opt will make them inplace later
gemv = gemv_no_inplace
