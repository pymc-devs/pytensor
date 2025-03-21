from scipy.signal import convolve as scipy_convolve

import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.scalar.basic import upcast


class Conv1d(Op):
    __props__ = ("mode",)

    def __init__(self, mode="full"):
        self.mode = mode

    def make_node(self, data, kernel):
        data = pt.as_tensor_variable(data)
        kernel = pt.as_tensor_variable(kernel)

        assert data.ndim == 1
        assert kernel.ndim == 1

        dtype = upcast(data.dtype, kernel.dtype)

        n = data.type.shape[0]
        k = kernel.type.shape[0]

        if n is None or k is None:
            out_shape = (None,)
        elif self.mode == "full":
            out_shape = (n + k - 1,)
        elif self.mode == "valid":
            out_shape = (max(n, k) - min(n, k) + 1,)
        elif self.mode == "same":
            out_shape = (max(n, k),)

        out = pt.tensor(dtype=dtype, shape=out_shape)
        return Apply(self, [data, kernel], [out])

    def perform(self, node, inputs, outputs):
        data, kernel = inputs
        outputs[0][0] = scipy_convolve(data, kernel, mode=self.mode)

    def infer_shape(self, fgraph, node, shapes):
        data_shape, kernel_shape = shapes
        n = data_shape[0]
        k = kernel_shape[0]
        if self.mode == "full":
            shape = n + k - 1
        elif self.mode == "valid":
            shape = pt.maximum(n, k) - pt.minimum(n, k) + 1
        elif self.mode == "same":
            shape = pt.maximum(n, k)
        return [[shape]]

    def L_op(self, inputs, outputs, output_grads):
        data, kernel = inputs
        [grad] = output_grads

        if self.mode == "full":
            valid_conv = type(self)(mode="valid")
            data_bar = valid_conv(grad, kernel[::-1])
            kernel_bar = valid_conv(grad, data[::-1])

        elif self.mode == "valid":
            full_conv = type(self)(mode="full")
            n = data.shape[0]
            k = kernel.shape[0]
            kmn = pt.maximum(0, k - n)
            nkm = pt.maximum(0, n - k)
            # We need mode="full" if k >= n else "valid" for data_bar (opposite for kernel_bar), but mode is not symbolic.
            # Instead we always use mode="full" and slice the result so it behaves like "valid" for the input that's shorter.
            data_bar = full_conv(grad, kernel[::-1])
            data_bar = data_bar[kmn : data_bar.shape[0] - kmn]
            kernel_bar = full_conv(grad, data[::-1])
            kernel_bar = kernel_bar[nkm : kernel_bar.shape[0] - nkm]

        return [data_bar, kernel_bar]
