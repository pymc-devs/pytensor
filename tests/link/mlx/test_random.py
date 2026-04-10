import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.function import function
from pytensor.compile.sharedvalue import shared
from pytensor.tensor.random.utils import RandomStream


mx = pytest.importorskip("mlx.core")


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
    check_shape_and_dtype(
        lambda srng: srng.multivariate_normal(mean=mean, cov=cov, size=(6,)),
        (6, 4),
        "float32",
    )


@pytest.mark.parametrize("method", ["cholesky", "svd", "eigh"])
def test_mvnormal_decomposition_method(method):
    mean = np.zeros(4, dtype=np.float32)
    cov = np.eye(4, dtype=np.float32)
    check_shape_and_dtype(
        lambda srng: srng.multivariate_normal(
            mean=mean, cov=cov, size=(6,), method=method
        ),
        (6, 4),
        "float32",
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


def compile_shared_rng_function(*args, mode="MLX", **kwargs):
    with pytest.warns(
        UserWarning, match=r"The RandomType SharedVariables \[.+\] will not be used"
    ):
        return function(*args, mode=mode, **kwargs)


def test_random_updates():
    original_value = np.random.default_rng(seed=98)
    rng = shared(original_value, name="original_rng", borrow=False)
    next_rng, x = pt.random.normal(name="x", rng=rng).owner.outputs

    f = compile_shared_rng_function([], [x], updates={rng: next_rng})
    assert f() != f()

    # Check that the original shared variable was not overwritten when typifying
    assert all(
        a == b if not isinstance(a, np.ndarray) else np.array_equal(a, b)
        for a, b in zip(
            rng.get_value().bit_generator.state,
            original_value.bit_generator.state,
            strict=True,
        )
    )


@pytest.mark.parametrize("noise_first", (False, True))
def test_replaced_shared_rng_storage_order(noise_first):
    # Test that replacing the RNG variable in the linker does not cause
    # a disalignment between the compiled graph and the storage_map.

    mu = pytensor.shared(np.array(1.0), name="mu")
    rng = pytensor.shared(np.random.default_rng(123))
    next_rng, noise = pt.random.normal(rng=rng).owner.outputs

    out = noise * mu if noise_first else mu * noise

    updates = {
        mu: pt.grad(out, mu),
        rng: next_rng,
    }
    f = compile_shared_rng_function([], [out], updates=updates)

    # Confirm that input_storage type and fgraph input order are aligned
    for storage, fgraph_input in zip(
        f.input_storage, f.maker.fgraph.inputs, strict=True
    ):
        assert storage.type == fgraph_input.type

    assert mu.get_value() == 1
    f()
    assert mu.get_value() != 1


def test_replaced_shared_rng_storage_ordering_equality():
    """Test that storage identity comparison works when numpy arrays precede
    the RNG in input_storage (regression test for issue #314)."""
    pt_rng = RandomStream(1)

    batchshape = (3, 1, 4, 4)
    inp_shared = pytensor.shared(
        np.zeros(batchshape, dtype="float64"), name="inp_shared"
    )

    inp = pt.tensor4(dtype="float64", name="inp")
    inp_update = inp + pt_rng.normal(size=inp.shape, loc=5, scale=1e-5)

    fn = compile_shared_rng_function(
        inputs=[],
        outputs=[],
        updates={inp_shared: inp_update},
        givens={inp: inp_shared},
    )
    fn()
    np.testing.assert_allclose(np.array(inp_shared.get_value()), 5, rtol=1e-2)
    fn()
    np.testing.assert_allclose(np.array(inp_shared.get_value()), 10, rtol=1e-2)
