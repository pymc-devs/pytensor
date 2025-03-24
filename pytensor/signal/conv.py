from typing import TYPE_CHECKING, Literal

from numpy import convolve as numpy_convolve

from pytensor.graph import Apply, Op
from pytensor.scalar.basic import upcast
from pytensor.tensor.basic import as_tensor_variable, join, zeros
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.math import maximum, minimum
from pytensor.tensor.type import vector
from pytensor.tensor.variable import TensorVariable


if TYPE_CHECKING:
    from pytensor.tensor import TensorLike


class Conv1d(Op):
    __props__ = ("mode",)
    gufunc_signature = "(n),(k)->(o)"

    def __init__(self, mode: Literal["full", "valid"] = "full"):
        if mode not in ("full", "valid"):
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode

    def make_node(self, data, kernel):
        data = as_tensor_variable(data)
        kernel = as_tensor_variable(kernel)

        assert data.ndim == 1
        assert kernel.ndim == 1

        dtype = upcast(data.dtype, kernel.dtype)

        n = data.type.shape[0]
        k = kernel.type.shape[0]

        if n is None or k is None:
            out_shape = (None,)
        elif self.mode == "full":
            out_shape = (n + k - 1,)
        else:  # mode == "valid":
            out_shape = (max(n, k) - min(n, k) + 1,)

        out = vector(dtype=dtype, shape=out_shape)
        return Apply(self, [data, kernel], [out])

    def perform(self, node, inputs, outputs):
        data, kernel = inputs
        # We use numpy_convolve as that's what scipy would use if method="direct" was passed.
        # And mode != "same", which this Op doesn't cover anyway.
        outputs[0][0] = numpy_convolve(data, kernel, mode=self.mode)

    def infer_shape(self, fgraph, node, shapes):
        data_shape, kernel_shape = shapes
        n = data_shape[0]
        k = kernel_shape[0]
        if self.mode == "full":
            shape = n + k - 1
        else:  # mode == "valid":
            shape = maximum(n, k) - minimum(n, k) + 1
        return [[shape]]

    def L_op(self, inputs, outputs, output_grads):
        data, kernel = inputs
        [grad] = output_grads

        if self.mode == "full":
            valid_conv = type(self)(mode="valid")
            data_bar = valid_conv(grad, kernel[::-1])
            kernel_bar = valid_conv(grad, data[::-1])

        else:  # mode == "valid":
            full_conv = type(self)(mode="full")
            n = data.shape[0]
            k = kernel.shape[0]
            kmn = maximum(0, k - n)
            nkm = maximum(0, n - k)
            # We need mode="full" if k >= n else "valid" for data_bar (opposite for kernel_bar), but mode is not symbolic.
            # Instead, we always use mode="full" and slice the result so it behaves like "valid" for the input that's shorter.
            data_bar = full_conv(grad, kernel[::-1])
            data_bar = data_bar[kmn : data_bar.shape[0] - kmn]
            kernel_bar = full_conv(grad, data[::-1])
            kernel_bar = kernel_bar[nkm : kernel_bar.shape[0] - nkm]

        return [data_bar, kernel_bar]


def convolve(
    data: "TensorLike",
    kernel: "TensorLike",
    mode: Literal["full", "valid", "same"] = "full",
) -> TensorVariable:
    data = as_tensor_variable(data)
    kernel = as_tensor_variable(kernel)

    if mode == "same":
        # We implement "same" as "valid" with padded data.
        zeros_left = kernel.shape[0] // 2
        zeros_right = (kernel.shape[0] - 1) // 2
        data = join(
            0,
            zeros(zeros_left, dtype=kernel.dtype),
            data,
            zeros(zeros_right, dtype=kernel.dtype),
        )
        mode = "valid"

    return Blockwise(Conv1d(mode=mode))(data, kernel)
