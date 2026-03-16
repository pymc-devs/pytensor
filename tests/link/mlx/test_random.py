import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import MLX, Mode
from pytensor.link.mlx.linker import MLXLinker
from pytensor.tensor.random.utils import RandomStream


mx = pytest.importorskip("mlx.core")

# MLX mode without mx.compile — needed for ops that use CPU streams internally
# (e.g. multivariate_normal, which uses SVD via mx.cpu stream and is
# incompatible with mx.compile's tracing).
MLX_NO_COMPILE = Mode(linker=MLXLinker(use_compile=False), optimizer=MLX.optimizer)


def test_normal_cumsum():
    out = pt.random.normal(size=(52,)).cumsum()
    result = out.eval(mode="MLX")
    assert isinstance(result, mx.array)
    assert result.shape == (52,)


def check_shape_and_dtype(
    make_rv, expected_shape, expected_dtype=None, n_evals=2, mode="MLX"
):
    """Compile and run an RV under MLX, assert shape and dtype, and verify
    that two successive draws differ (RNG state is properly threaded).

    Parameters
    ----------
    make_rv : callable(srng) -> rv_var
        Factory that creates the RV using the provided RandomStream.
    expected_shape : tuple
    expected_dtype : str or None
    n_evals : int
    mode : str or Mode
    """
    srng = RandomStream(seed=12345)
    rv = make_rv(srng)
    f = pytensor.function([], rv, mode=mode, updates=srng.updates())
    results = [np.array(f()) for _ in range(n_evals)]

    for r in results:
        assert r.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {r.shape}"
        )
        if expected_dtype is not None:
            assert r.dtype == np.dtype(expected_dtype), (
                f"Expected dtype {expected_dtype}, got {r.dtype}"
            )

    assert not np.array_equal(results[0], results[1]), (
        "Two draws were identical — RNG not advancing"
    )

    return results


def test_normal_shape_dtype():
    check_shape_and_dtype(
        lambda srng: srng.normal(loc=0.0, scale=1.0, size=(3, 4)),
        (3, 4),
        "float32",
    )


def test_normal_scalar():
    check_shape_and_dtype(
        lambda srng: srng.normal(loc=2.0, scale=0.5),
        (),
    )


def test_normal_array_params():
    result = pt.random.normal(loc=[0, 1], scale=[1.0, 0.3], size=(100, 2)).eval(
        mode="MLX"
    )
    assert result.shape == (100, 2)
    means = np.array(result).mean(axis=0)
    assert abs(means[0]) < 0.3
    assert abs(means[1] - 1.0) < 0.3


def test_uniform_shape_dtype():
    results = check_shape_and_dtype(
        lambda srng: srng.uniform(low=0.0, high=1.0, size=(10,)),
        (10,),
        "float32",
    )
    r = np.array(results[0])
    assert np.all(r >= 0.0)
    assert np.all(r < 1.0)


def test_bernoulli_shape():
    check_shape_and_dtype(
        lambda srng: srng.bernoulli(p=0.7, size=(5, 5)),
        (5, 5),
    )


def test_categorical_shape():
    probs = np.array([0.1, 0.4, 0.5], dtype=np.float32)
    results = check_shape_and_dtype(
        lambda srng: srng.categorical(p=probs, size=(8,)),
        (8,),
    )
    r = np.array(results[0])
    assert np.all(r < 3)
    assert np.all(r >= 0)


def test_mvnormal_shape():
    mean = np.zeros(4, dtype=np.float32)
    cov = np.eye(4, dtype=np.float32)
    # multivariate_normal uses SVD internally (CPU-only in MLX), which is
    # incompatible with mx.compile — use the no-compile mode.
    check_shape_and_dtype(
        lambda srng: srng.multivariate_normal(mean=mean, cov=cov, size=(6,)),
        (6, 4),
        "float32",
        mode=MLX_NO_COMPILE,
    )


def test_laplace_shape_dtype():
    check_shape_and_dtype(
        lambda srng: srng.laplace(loc=0.0, scale=1.0, size=(7,)),
        (7,),
        "float32",
    )


def test_gumbel_shape_dtype():
    check_shape_and_dtype(
        lambda srng: srng.gumbel(loc=0.0, scale=1.0, size=(6,)),
        (6,),
        "float32",
    )


def test_integers_shape():
    results = check_shape_and_dtype(
        lambda srng: srng.integers(low=0, high=10, size=(12,)),
        (12,),
    )
    r = np.array(results[0])
    assert np.all(r >= 0)
    assert np.all(r < 10)


def test_permutation_shape():
    x = np.arange(8, dtype=np.int32)
    results = check_shape_and_dtype(
        lambda srng: srng.permutation(x),
        (8,),
    )
    assert sorted(np.array(results[0]).tolist()) == list(range(8))


def test_gamma_not_implemented():
    srng = RandomStream(seed=1)
    rv = srng.gamma(shape=1.0, scale=1.0, size=(3,))
    with pytest.raises(NotImplementedError, match="No MLX implementation"):
        pytensor.function([], rv, mode="MLX", updates=srng.updates())


def test_beta_not_implemented():
    srng = RandomStream(seed=1)
    rv = srng.beta(alpha=2.0, beta=5.0, size=(3,))
    with pytest.raises(NotImplementedError, match="No MLX implementation"):
        pytensor.function([], rv, mode="MLX", updates=srng.updates())
