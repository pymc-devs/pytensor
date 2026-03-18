import numpy as np

import pytensor
import pytensor.tensor.basic as ptb
from pytensor import Out
from pytensor.tensor.basic import join
from pytensor.tensor.subtensor import inc_subtensor, set_subtensor
from pytensor.tensor.type import matrices, vector, vectors


class AdvancedSubtensor1:
    """Benchmark advanced subtensor1 indexing."""

    params = [[True, False], [True, False]]
    param_names = ["static_shape", "gc"]

    def setup(self, static_shape, gc):
        x = vector("x", shape=(85 if static_shape else None,))
        x_values = np.random.normal(size=(85,)).astype(x.type.dtype)
        idxs_values = np.arange(85).repeat(11)
        out = x[idxs_values]
        self.fn = pytensor.function(
            [x],
            pytensor.Out(out, borrow=True),
            on_unused_input="ignore",
            trust_input=True,
        )
        self.fn.vm.allow_gc = gc
        self.x_values = x_values

        # Warmup
        self.fn(self.x_values)

    def time_advanced_subtensor1(self, static_shape, gc):
        self.fn(self.x_values)


class AdvancedIncSubtensor1:
    """Benchmark advanced inc/set subtensor1 operations."""

    params = [["inc_subtensor", "set_subtensor"], [True, False], [True, False]]
    param_names = ["func", "static_shape", "gc"]

    def setup(self, func, static_shape, gc):
        func_map = {"inc_subtensor": inc_subtensor, "set_subtensor": set_subtensor}
        subtensor_func = func_map[func]

        x = vector("x", shape=(85 if static_shape else None,))
        x_values = np.zeros((85,), dtype=x.type.dtype)
        buffer = ptb.zeros_like(x)
        y_values = np.random.normal(size=(85 * 11,)).astype(x.type.dtype)
        idxs_values = np.arange(85).repeat(11)
        out1 = subtensor_func(buffer[idxs_values], y_values)
        out2 = subtensor_func(buffer[idxs_values[::-1]], y_values)
        self.fn = pytensor.function(
            [x],
            [pytensor.Out(out1, borrow=True), pytensor.Out(out2, borrow=True)],
            on_unused_input="ignore",
            trust_input=True,
        )
        self.fn.vm.allow_gc = gc
        self.x_values = x_values

        # Warmup
        self.fn(self.x_values)

    def time_advanced_incsubtensor1(self, func, static_shape, gc):
        self.fn(self.x_values)


class JoinPerformance:
    """Benchmark join (concatenation) with various dimensions and memory layouts."""

    params = [[1, 2], [0, 1], ["C", "F", "Mixed"], [True, False]]
    param_names = ["ndim", "axis", "memory_layout", "gc"]

    def setup(self, ndim, axis, memory_layout, gc):
        # Skip invalid combinations
        if ndim == 1 and not (memory_layout == "C" and axis == 0):
            raise NotImplementedError("Skip invalid combination")

        n = 64
        if ndim == 1:
            inputs = vectors("abcdef")
        else:
            inputs = matrices("abcdef")

        out = join(axis, *inputs)
        self.fn = pytensor.function(inputs, Out(out, borrow=True), trust_input=True)
        self.fn.vm.allow_gc = gc

        test_values = [np.zeros((n, n)[:ndim], dtype=inputs[0].dtype) for _ in inputs]
        if memory_layout == "F":
            test_values = [np.asfortranarray(t) for t in test_values]
        elif memory_layout == "Mixed":
            test_values = [
                np.asfortranarray(t) if i % 2 else t for i, t in enumerate(test_values)
            ]
        self.test_values = test_values

        # Warmup
        self.fn(*self.test_values)

    def time_join(self, ndim, axis, memory_layout, gc):
        self.fn(*self.test_values)
