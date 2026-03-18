import numpy as np

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import get_default_mode
from pytensor.tensor.type import tensor


class BlockDiagDot:
    """Benchmark block_diag @ vector with and without the rewrite optimization."""

    params = [[10, 100, 1000], [True, False]]
    param_names = ["size", "rewrite"]

    def setup(self, size, rewrite):
        rng = np.random.default_rng(sum(ord(c) for c in f"{size}_{rewrite}"))

        a_size = int(rng.uniform(1, int(0.8 * size)))
        b_size = int(rng.uniform(1, int(0.8 * (size - a_size))))
        c_size = size - a_size - b_size

        a = tensor("a", shape=(a_size, a_size))
        b = tensor("b", shape=(b_size, b_size))
        c = tensor("c", shape=(c_size, c_size))
        d = tensor("d", shape=(size,))

        x = pt.linalg.block_diag(a, b, c)
        out = x @ d

        mode = get_default_mode()
        if not rewrite:
            mode = mode.excluding("local_block_diag_dot_to_dot_block_diag")

        self.fn = pytensor.function([a, b, c, d], out, mode=mode)

        self.a_val = rng.normal(size=a.type.shape).astype(a.type.dtype)
        self.b_val = rng.normal(size=b.type.shape).astype(b.type.dtype)
        self.c_val = rng.normal(size=c.type.shape).astype(c.type.dtype)
        self.d_val = rng.normal(size=d.type.shape).astype(d.type.dtype)

        # Warmup
        self.fn(self.a_val, self.b_val, self.c_val, self.d_val)

    def time_block_diag_dot(self, size, rewrite):
        self.fn(self.a_val, self.b_val, self.c_val, self.d_val)
