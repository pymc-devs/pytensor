import jax
import numpy as np

from pytensor import config
from pytensor.graph.fg import FunctionGraph
from pytensor.link.jax.ops import as_jax_op
from pytensor.tensor import tensor
from tests.link.jax.test_basic import compare_jax_and_py

def test_as_jax_op1():
    # 2 parameters input, single output
    rng = np.random.default_rng(14)
    x = tensor("a", shape=(2,))
    y = tensor("b", shape=(2,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    @as_jax_op
    def f(x, y):
        return jax.nn.sigmoid(x + y)

    out = f(x, y)

    fg = FunctionGraph([x, y], [out])
    fn, _ = compare_jax_and_py(fg, test_values)
