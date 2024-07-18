import numpy as np
import pytest

from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import get_test_value
from pytensor.tensor.math import Argmax, Max, maximum
from pytensor.tensor.math import max as pt_max
from pytensor.tensor.type import dvector, matrix, scalar, vector
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


def test_jax_max_and_argmax():
    # Test that a single output of a multi-output `Op` can be used as input to
    # another `Op`
    x = dvector()
    mx = Max([0])(x)
    amx = Argmax([0])(x)
    out = mx * amx
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(out_fg, [np.r_[1, 2]])


def test_dot():
    y = vector("y")
    y.tag.test_value = np.r_[1.0, 2.0].astype(config.floatX)
    x = vector("x")
    x.tag.test_value = np.r_[3.0, 4.0].astype(config.floatX)
    A = matrix("A")
    A.tag.test_value = np.empty((2, 2), dtype=config.floatX)
    alpha = scalar("alpha")
    alpha.tag.test_value = np.array(3.0, dtype=config.floatX)
    beta = scalar("beta")
    beta.tag.test_value = np.array(5.0, dtype=config.floatX)

    # This should be converted into a `Gemv` `Op` when the non-JAX compatible
    # optimizations are turned on; however, when using JAX mode, it should
    # leave the expression alone.
    out = y.dot(alpha * A).dot(x) + beta * y
    fgraph = FunctionGraph([y, x, A, alpha, beta], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = maximum(y, x)
    fgraph = FunctionGraph([y, x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = pt_max(y)
    fgraph = FunctionGraph([y], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])
