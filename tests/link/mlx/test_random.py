import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
import pytensor.tensor.random.basic as ptr
from pytensor.compile.maker import function
from pytensor.compile.sharedvalue import shared
from pytensor.tensor.random.utils import RandomStream


mx = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def _float32(monkeypatch):
    monkeypatch.setattr(pytensor.config, "floatX", "float32")


def compile_random_function(*args, mode="MLX", **kwargs):
    with pytest.warns(
        UserWarning, match=r"The RandomType SharedVariables \[.+\] will not be used"
    ):
        return function(*args, mode=mode, **kwargs)


def test_random_updates():
    """Successive calls draw new values and the original RNG is left untouched."""
    original_value = np.random.default_rng(seed=98)
    rng = shared(original_value, name="rng", borrow=False)
    next_rng, x = ptr.normal(0, 1, size=(3,), rng=rng).owner.outputs

    f = compile_random_function([], x, updates={rng: next_rng})
    res1, res2 = np.array(f()), np.array(f())
    assert not np.array_equal(res1, res2)

    # The shared RNG content must not be overwritten by the MLX key conversion.
    assert isinstance(rng.get_value(), np.random.Generator)
    assert rng.get_value().bit_generator.state == original_value.bit_generator.state


def test_random_RandomStream():
    """Two successive calls of a compiled graph should return different values."""
    srng = RandomStream(seed=123)
    out = srng.normal() - srng.normal()
    fn = compile_random_function([], out)
    assert np.array(fn()) != np.array(fn())


@pytest.mark.parametrize(
    "dist, params, size, expected_dtype",
    [
        (ptr.normal, (2.0, 3.0), (5,), "float32"),
        (ptr.uniform, (-1.0, 2.0), (5,), "float32"),
        (ptr.laplace, (0.0, 1.0), (4,), "float32"),
        (ptr.gumbel, (0.0, 1.0), (4,), "float32"),
        (ptr.integers, (0, 10), (6,), "int64"),
        (ptr.bernoulli, (0.7,), (8,), "int64"),
    ],
)
def test_distribution_shape_and_dtype(dist, params, size, expected_dtype):
    rng = shared(np.random.default_rng(0))
    out = dist(*params, size=size, rng=rng)
    assert out.type.dtype == expected_dtype

    f = function([], out, mode="MLX")
    res = np.array(f())
    assert res.shape == size
    assert res.dtype == np.dtype(expected_dtype)
    assert np.isfinite(res).all()


def test_categorical():
    p = np.array([0.1, 0.3, 0.6], dtype="float32")
    rng = shared(np.random.default_rng(0))
    out = ptr.categorical(p, size=(100_000,), rng=rng)

    f = function([], out, mode="MLX")
    res = np.array(f())
    assert res.shape == (100_000,)
    assert set(np.unique(res).tolist()) <= {0, 1, 2}
    freqs = np.array([(res == k).mean() for k in range(3)])
    np.testing.assert_allclose(freqs, p, atol=0.01)


def test_bernoulli_rate():
    rng = shared(np.random.default_rng(0))
    out = ptr.bernoulli(0.3, size=(100_000,), rng=rng)
    res = np.array(function([], out, mode="MLX")())
    assert set(np.unique(res).tolist()) <= {0, 1}
    np.testing.assert_allclose(res.mean(), 0.3, atol=0.01)


def test_integers_negative_low():
    # MLX `randint` truncates towards zero; assert the full range is covered.
    rng = shared(np.random.default_rng(0))
    out = ptr.integers(-5, 5, size=(50_000,), rng=rng)
    res = np.array(function([], out, mode="MLX")())
    assert res.dtype == np.int64
    assert set(np.unique(res).tolist()) == set(range(-5, 5))


@pytest.mark.parametrize("size", [(), (2, 3)])
def test_scalar_and_multidim_size(size):
    rng = shared(np.random.default_rng(0))
    res = np.array(function([], ptr.normal(0, 1, size=size, rng=rng), mode="MLX")())
    assert res.shape == size


def test_unimplemented_distribution_raises():
    rng = shared(np.random.default_rng(0))
    out = ptr.beta(1.0, 1.0, size=(3,), rng=rng)
    with pytest.raises(NotImplementedError, match="No MLX implementation"):
        function([], out, mode="MLX")


def test_size_not_statically_known_raises():
    x = pt.vector("x", dtype="float32")
    rng = shared(np.random.default_rng(0))
    out = ptr.normal(0, 1, size=x.shape, rng=rng)
    with pytest.raises(NotImplementedError, match="statically known"):
        function([x], out, mode="MLX")
