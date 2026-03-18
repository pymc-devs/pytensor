import numpy as np

import pytensor
from pytensor import config
from pytensor.compile.io import In, Out
from pytensor.tensor.type import tensor3


class Reshape:
    """Benchmark reshape operations with multiple output shapes."""

    def setup(self):
        x = tensor3("x")
        self.x_val = np.random.random((2, 3, 4)).astype(config.floatX)

        y1 = x.reshape((6, 4))
        y2 = x.reshape((2, 12))
        y3 = x.reshape((-1,))

        self.reshape_fn = pytensor.function(
            [In(x, borrow=True)],
            [Out(y1, borrow=True), Out(y2, borrow=True), Out(y3, borrow=True)],
        )
        self.reshape_fn.trust_input = True

        # Warmup
        self.reshape_fn(self.x_val)

    def time_reshape(self):
        self.reshape_fn(self.x_val)
