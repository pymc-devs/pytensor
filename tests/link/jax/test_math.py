import numpy as np
import pytest

from pytensor.configdefaults import config
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
    compare_jax_and_py([x], [out], [np.r_[1, 2]])


def test_dot():
    y = vector("y")
    y_test_value = np.r_[1.0, 2.0].astype(config.floatX)
    x = vector("x")
    x_test_value = np.r_[3.0, 4.0].astype(config.floatX)
    A = matrix("A")
    A_test_value = np.empty((2, 2), dtype=config.floatX)
    alpha = scalar("alpha")
    alpha_test_value = np.array(3.0, dtype=config.floatX)
    beta = scalar("beta")
    beta_test_value = np.array(5.0, dtype=config.floatX)

    # This should be converted into a `Gemv` `Op` when the non-JAX compatible
    # optimizations are turned on; however, when using JAX mode, it should
    # leave the expression alone.
    out = y.dot(alpha * A).dot(x) + beta * y
    compare_jax_and_py(
        [y, x, A, alpha, beta],
        out,
        [
            y_test_value,
            x_test_value,
            A_test_value,
            alpha_test_value,
            beta_test_value,
        ],
    )

    out = maximum(y, x)
    compare_jax_and_py([y, x], [out], [y_test_value, x_test_value])

    out = pt_max(y)
    compare_jax_and_py([y], [out], [y_test_value])
