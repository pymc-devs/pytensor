import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.compile.function import function
from pytensor.compile.sharedvalue import shared
from pytensor.link.pytorch.dispatch.basic import pytorch_typify
from tests.link.pytorch.test_basic import pytorch_mode


torch = pytest.importorskip("torch")


@pytest.mark.parametrize("update", [(True), (False)])
def test_random_updates(update):
    original = np.random.default_rng(seed=123)
    original_torch = pytorch_typify(original)
    rng = shared(original, name="rng", borrow=False)
    rv = pt.random.bernoulli(0.5, name="y", rng=rng)
    next_rng, x = rv.owner.outputs
    x.dprint()
    f = function([], x, updates={rng: next_rng} if update else None, mode="PYTORCH")
    draws = np.stack([f() for _ in range(5)])
    # assert we are getting different values
    if update:
        assert draws.sum() < 5 and draws.sum() >= 1
        # assert we have a new rng
        rng_value = rng.get_value(borrow=True)  # we can't copy torch generator
        assert torch.eq(rng_value.get_state(), original_torch.get_state())
    else:
        pass


@pytest.mark.parametrize(
    "size,p",
    [
        ((1000,), 0.5),
        (None, 0.5),
        ((1000, 4), 0.5),
        ((10, 2), np.array([0.5, 0.3])),
        ((1000, 10, 2), np.array([0.5, 0.3])),
    ],
)
def test_random_bernoulli(size, p):
    rng = shared(np.random.default_rng(123))

    g = pt.random.bernoulli(p, size=size, rng=rng)
    g_fn = function([], g, mode=pytorch_mode)
    samples = g_fn()
    samples_mean = samples.mean(axis=0) if samples.shape else samples
    np.testing.assert_allclose(samples_mean, 0.5, 1)


@pytest.mark.parametrize(
    "size,n,p,update",
    [
        ((1000,), 10, 0.5, False),
        ((1000, 4), 10, 0.5, False),
        ((1000, 2), np.array([10, 40]), np.array([0.5, 0.3]), True),
    ],
)
def test_binomial(size, n, p, update):
    rng = shared(np.random.default_rng(123))
    rv = pt.random.binomial(n, p, size=size, rng=rng)
    next_rng, *_ = rv.owner.inputs
    g_fn = function(
        [], rv, mode=pytorch_mode, updates={rng: next_rng} if update else None
    )
    samples = g_fn()
    if not update:
        np.testing.assert_allclose(samples, g_fn(), rtol=0.1)
        np.testing.assert_allclose(samples.mean(axis=0), n * p, rtol=0.1)
        np.testing.assert_allclose(
            samples.std(axis=0), np.sqrt(n * p * (1 - p)), rtol=0.2
        )
    else:
        second_samples = g_fn()
        np.testing.assert_array_equal(second_samples, samples)
