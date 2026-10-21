from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.linalg._lazy import scipy_linalg
from pytensor.tensor.type import DenseTensorType


class Ger(Op):
    r"""Rank-1 update of a matrix.

    .. math::

        A \leftarrow A + \alpha x y^{\top}

    for matrix :math:`A`, scalar :math:`\alpha` and vectors :math:`x` and :math:`y`.
    Constructed with ``inplace=True``, the output aliases ``A``'s storage and the op
    destroys it; otherwise ``A`` is left untouched.
    """

    __props__ = ("inplace",)
    gufunc_signature = "(m,n),(),(m),(n)->(m,n)"

    def __init__(self, inplace):
        self.inplace = inplace
        if inplace:
            self.destroy_map = {0: [0]}

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> Op:
        """``Ger`` updates ``A`` in place, so that is the only input it can destroy."""
        if 0 in allowed_inplace_inputs:
            return type(self)(inplace=True)
        return self

    def __str__(self):
        if self.inplace:
            return f"{self.__class__.__name__}{{inplace}}"
        else:
            return f"{self.__class__.__name__}{{no_inplace}}"

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
            ger_func = scipy_linalg.get_blas_funcs("ger", dtype=A.dtype)
            if A.flags["C_CONTIGUOUS"]:
                # Work on transposed system to avoid copying
                A = ger_func(alpha, y, x, a=A.T, overwrite_a=self.inplace).T
            else:
                A = ger_func(alpha, x, y, a=A, overwrite_a=self.inplace)
        output_storage[0][0] = A

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]]
