from functools import partial

import numpy as np
import pytest

import pytensor.tensor as pt
import tests.unittest_tools as utt
from pytensor.configdefaults import config
from pytensor.tensor.linalg.products import expm, kron, sqrtm
from pytensor.tensor.type import matrix
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


def test_kron():
    x = matrix("x")
    y = matrix("y")
    z = kron(x, y)

    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)
    y_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)

    compare_jax_and_py([x, y], [z], [x_np, y_np])


def test_jax_expm():
    rng = np.random.default_rng(utt.fetch_seed())
    A = pt.tensor(name="A", shape=(5, 5))
    A_val = rng.normal(size=(5, 5)).astype(config.floatX)
    out = expm(A)

    compare_jax_and_py([A], [out], [A_val])


def test_jax_sqrtm():
    rng = np.random.default_rng(utt.fetch_seed())
    A = pt.tensor(name="A", shape=(None, 5, 5))
    A_val = rng.normal(size=(2, 5, 5)) + 5 * np.eye(5)
    A_val[1] = np.diag([4.0, -1.0, 9.0, 1.0, 2.0])

    rtol = 1e-3 if config.floatX == "float32" else 1e-7
    compare_jax_and_py(
        [A],
        [sqrtm(A)],
        [A_val.astype(config.floatX)],
        assert_fn=partial(np.testing.assert_allclose, rtol=rtol),
    )
