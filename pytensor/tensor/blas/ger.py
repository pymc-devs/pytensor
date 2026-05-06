from scipy.linalg import get_blas_funcs

from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.type import DenseTensorType


class Ger(Op):
    """
    BLAS defines general rank-1 update GER as A <- A + alpha x y'

    for matrix A, scalar alpha, vectors x and y.

    This interface to GER allows non-destructive operation on A via the
    `destructive` argument to the constructor.

    """

    __props__ = ("destructive",)

    def __init__(self, destructive):
        self.destructive = destructive
        if destructive:
            self.destroy_map = {0: [0]}

    def __str__(self):
        if self.destructive:
            return f"{self.__class__.__name__}{{destructive}}"
        else:
            return f"{self.__class__.__name__}{{non-destructive}}"

    def make_node(self, A, alpha, x, y):
        A = as_tensor_variable(A)
        y = as_tensor_variable(y)
        x = as_tensor_variable(x)
        alpha = as_tensor_variable(alpha)
        if not (A.dtype == x.dtype == y.dtype == alpha.dtype):
            raise TypeError(
                "ger requires matching dtypes", (A.dtype, alpha.dtype, x.dtype, y.dtype)
            )
        if alpha.ndim != 0:
            raise TypeError("ger requires scalar alpha", alpha.type)
        if A.ndim != 2:
            raise TypeError("ger requires matrix for A", A.type)
        if x.ndim != 1:
            raise TypeError("ger requires vector for x", x.type)
        if y.ndim != 1:
            raise TypeError("ger requires vector for y", y.type)

        if x.dtype not in ("float32", "float64", "complex64", "complex128"):
            raise TypeError("only float and complex types supported", x.dtype)

        inputs = [A, alpha, x, y]
        if any(not isinstance(i.type, DenseTensorType) for i in inputs):
            raise NotImplementedError("Only dense tensor types are supported")

        return Apply(self, inputs, [A.type()])

    def perform(self, node, inputs, output_storage):
        A, alpha, x, y = inputs
        if A.size:
            # GER doesn't handle zero-sized inputs
            ger_func = get_blas_funcs("ger", dtype=A.dtype)
            if A.flags["C_CONTIGUOUS"]:
                # Work on transposed system to avoid copying
                A = ger_func(alpha, y, x, a=A.T, overwrite_a=self.destructive).T
            else:
                A = ger_func(alpha, x, y, a=A, overwrite_a=self.destructive)
        output_storage[0][0] = A

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]]


ger = Ger(destructive=False)
ger_destructive = Ger(destructive=True)
