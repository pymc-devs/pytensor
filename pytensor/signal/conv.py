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
            out_shape = (n,)

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
            shape = n
        return [[shape]]

    def L_op(self, inputs, outputs, output_grads):
        data, kernel = inputs
        [grad] = output_grads

        if self.mode == "full":
            data_bar = convolve(grad, kernel[::-1], mode="valid")
            kernel_bar = convolve(grad, data[::-1], mode="valid")

        elif self.mode == "valid":
            n = data.shape[0]
            k = kernel.shape[0]
            kmn = pt.maximum(0, k - n)
            nkm = pt.maximum(0, n - k)
            # We need mode="full" if k >= n else "valid" for data_bar (opposite for kernel_bar), but mode is not symbolic.
            # Instead we always use mode="full" and slice the result so it behaves like "valid" for the input that's shorter.
            data_bar = convolve(grad, kernel[::-1], mode="full")
            data_bar = data_bar[kmn : data_bar.shape[0] - kmn]
            kernel_bar = convolve(grad, data[::-1], mode="full")
            kernel_bar = kernel_bar[nkm : kernel_bar.shape[0] - nkm]

        else:  # self.mode == "same"
            raise NotImplementedError("L_op not implemented for mode='same'")

        return [data_bar, kernel_bar]


def convolve(data, kernel, mode="full"):
    return Conv1d(mode)(data, kernel)
