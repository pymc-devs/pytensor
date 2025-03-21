from scipy.signal import convolve

import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.scalar.basic import upcast


class Conv1d(Op):
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
        outputs[0][0] = convolve(data, kernel, mode=self.mode)

    def L_op(self, inputs, outputs, output_grads):
        data, kernel = inputs
        [grad] = output_grads

        if self.mode == "full":
            data_bar = type(self)(mode="valid")(grad, kernel[::-1])
            kernel_bar = type(self)(mode="valid")(grad, data[::-1])

        return [data_bar, kernel_bar]
