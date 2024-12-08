import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.compile.function import function
from pytensor.compile.sharedvalue import shared
from tests.link.pytorch.test_basic import pytorch_mode


torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "size,p",
    [
        ((1000,), 0.5),
        (
            (
                1000,
                4,
            ),
            0.5,
        ),
        ((10, 2), np.array([0.5, 0.3])),
        ((1000, 10, 2), np.array([0.5, 0.3])),
    ],
)
def test_random_bernoulli(size, p):
    rng = shared(np.random.default_rng(123))

    g = pt.random.bernoulli(p, size=size, rng=rng)
    g_fn = function([], g, mode=pytorch_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples.mean(axis=0), 0.5, 1)


@pytest.mark.parametrize(
    "size,n,p",
    [
        ((1000,), 10, 0.5),
        (
            (
                1000,
                4,
            ),
            10,
            0.5,
        ),
        (
            (
                1000,
                2,
            ),
            np.array([10, 40]),
            np.array([0.5, 0.3]),
        ),
    ],
)
def test_binomial(n, p, size):
    rng = shared(np.random.default_rng(123))
    g = pt.random.binomial(n, p, size=size, rng=rng)
    g_fn = function([], g, mode=pytorch_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples.mean(axis=0), n * p, rtol=0.1)
    np.testing.assert_allclose(samples.std(axis=0), np.sqrt(n * p * (1 - p)), rtol=0.1)
