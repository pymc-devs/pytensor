import numpy as np
import pytest
import scipy.stats as stats

import pytensor
import pytensor.tensor as pt
import pytensor.tensor.random.basic as ptr
from pytensor import clone_replace
from pytensor.compile.function import function
from pytensor.compile.sharedvalue import SharedVariable, shared
from pytensor.graph.basic import Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.random.basic import RandomVariable
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.random.utils import RandomStream
from tests.link.jax.test_basic import compare_jax_and_py, jax_mode, set_test_value
from tests.tensor.random.test_basic import (
    batched_permutation_tester,
    batched_unweighted_choice_without_replacement_tester,
    batched_weighted_choice_without_replacement_tester,
)


jax = pytest.importorskip("jax")


from pytensor.link.jax.dispatch.random import numpyro_available  # noqa: E402


def compile_random_function(*args, mode="JAX", **kwargs):
    with pytest.warns(
        UserWarning, match=r"The RandomType SharedVariables \[.+\] will not be used"
    ):
        return function(*args, mode=mode, **kwargs)


def test_random_RandomStream():
    """Two successive calls of a compiled graph using `RandomStream` should
    return different values.

    """
    srng = RandomStream(seed=123)
    out = srng.normal() - srng.normal()

    fn = compile_random_function([], out, mode=jax_mode)
    jax_res_1 = fn()
    jax_res_2 = fn()

    assert not np.array_equal(jax_res_1, jax_res_2)


@pytest.mark.parametrize("rng_ctor", (np.random.default_rng,))
def test_random_updates(rng_ctor):
    original_value = rng_ctor(seed=98)
    rng = shared(original_value, name="original_rng", borrow=False)
    next_rng, x = pt.random.normal(name="x", rng=rng).owner.outputs

    f = compile_random_function([], [x], updates={rng: next_rng}, mode=jax_mode)
    assert f() != f()

    # Check that original rng variable content was not overwritten when calling jax_typify
    assert all(
        a == b if not isinstance(a, np.ndarray) else np.array_equal(a, b)
        for a, b in zip(
            rng.get_value().__getstate__(), original_value.__getstate__(), strict=True
        )
    )


@pytest.mark.parametrize("noise_first", (False, True))
def test_replaced_shared_rng_storage_order(noise_first):
    # Test that replacing the RNG variable in the linker does not cause
    # a disalignment between the compiled graph and the storage_map.

    mu = pytensor.shared(np.array(1.0), name="mu")
    rng = pytensor.shared(np.random.default_rng(123))
    next_rng, noise = pt.random.normal(rng=rng).owner.outputs

    if noise_first:
        out = noise * mu
    else:
        out = mu * noise

    updates = {
        mu: pt.grad(out, mu),
        rng: next_rng,
    }
    f = compile_random_function([], [out], updates=updates, mode="JAX")

    # The bug was found when noise used to be the first input of the fgraph
    # If this changes, the test may need to be tweaked to keep the save coverage
    assert isinstance(
        f.input_storage[1 - noise_first].type, RandomType
    ), "Test may need to be tweaked"

    # Confirm that input_storage type and fgraph input order are aligned
    for storage, fgrapn_input in zip(
        f.input_storage, f.maker.fgraph.inputs, strict=True
    ):
        assert storage.type == fgrapn_input.type

    assert mu.get_value() == 1
    f()
    assert mu.get_value() != 1


def test_replaced_shared_rng_storage_ordering_equality():
    """Test case described in issue #314.

    This happened when we tried to update the input storage after we clone the shared RNG.
    We used to call `input_storage.index(old_input_storage)` which would fail when the input_storage contained
    numpy arrays before the RNG value, which would fail the equality check.

    """
    pt_rng = RandomStream(1)

    batchshape = (3, 1, 4, 4)
    inp_shared = pytensor.shared(
        np.zeros(batchshape, dtype="float64"), name="inp_shared"
    )

    inp = pt.tensor4(dtype="float64", name="inp")
    inp_update = inp + pt_rng.normal(size=inp.shape, loc=5, scale=1e-5)

    # This function replaces inp by input_shared in the update expression
    # This is what caused the RNG to appear later than inp_shared in the input_storage

    fn = compile_random_function(
        inputs=[],
        outputs=[],
        updates={inp_shared: inp_update},
        givens={inp: inp_shared},
        mode="JAX",
    )
    fn()
    np.testing.assert_allclose(inp_shared.get_value(), 5, rtol=1e-3)
    fn()
    np.testing.assert_allclose(inp_shared.get_value(), 10, rtol=1e-3)


@pytest.mark.parametrize(
    "rv_op, dist_params, base_size, cdf_name, params_conv",
    [
        (
            ptr.beta,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                set_test_value(
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "beta",
            lambda *args: args,
        ),
        (
            ptr.cauchy,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                set_test_value(
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "cauchy",
            lambda *args: args,
        ),
        (
            ptr.exponential,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
            ],
            (2,),
            "expon",
            lambda *args: (0, args[0]),
        ),
        (
            ptr._gamma,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                set_test_value(
                    pt.dvector(),
                    np.array([0.5, 3.0], dtype=np.float64),
                ),
            ],
            (2,),
            "gamma",
            lambda a, b: (a, 0.0, b),
        ),
        (
            ptr.gumbel,
            [
                set_test_value(
                    pt.lvector(),
                    np.array([1, 2], dtype=np.int64),
                ),
                set_test_value(
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "gumbel_r",
            lambda *args: args,
        ),
        (
            ptr.laplace,
            [
                set_test_value(pt.dvector(), np.array([1.0, 2.0], dtype=np.float64)),
                set_test_value(pt.dscalar(), np.array(1.0, dtype=np.float64)),
            ],
            (2,),
            "laplace",
            lambda *args: args,
        ),
        (
            ptr.logistic,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                set_test_value(
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "logistic",
            lambda *args: args,
        ),
        (
            ptr.lognormal,
            [
                set_test_value(
                    pt.lvector(),
                    np.array([0, 0], dtype=np.int64),
                ),
                set_test_value(
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "lognorm",
            lambda mu, sigma: (sigma, 0, np.exp(mu)),
        ),
        (
            ptr.normal,
            [
                set_test_value(
                    pt.lvector(),
                    np.array([1, 2], dtype=np.int64),
                ),
                set_test_value(
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "norm",
            lambda *args: args,
        ),
        (
            ptr.pareto,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                set_test_value(
                    pt.dvector(),
                    np.array([2.0, 10.0], dtype=np.float64),
                ),
            ],
            (2,),
            "pareto",
            lambda shape, scale: (shape, 0.0, scale),
        ),
        (
            ptr.poisson,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([100000.0, 200000.0], dtype=np.float64),
                ),
            ],
            (2,),
            "poisson",
            lambda *args: args,
        ),
        (
            ptr.integers,
            [
                set_test_value(
                    pt.lscalar(),
                    np.array(0, dtype=np.int64),
                ),
                set_test_value(  # high-value necessary since test on cdf
                    pt.lscalar(),
                    np.array(1000, dtype=np.int64),
                ),
            ],
            (),
            "randint",
            lambda *args: args,
        ),
        (
            ptr.standard_normal,
            [],
            (2,),
            "norm",
            lambda *args: args,
        ),
        (
            ptr.t,
            [
                set_test_value(
                    pt.dscalar(),
                    np.array(2.0, dtype=np.float64),
                ),
                set_test_value(
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                set_test_value(
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "t",
            lambda *args: args,
        ),
        (
            ptr.uniform,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                set_test_value(
                    pt.dscalar(),
                    np.array(1000.0, dtype=np.float64),
                ),
            ],
            (2,),
            "uniform",
            lambda *args: args,
        ),
        (
            ptr.halfnormal,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([-1.0, 200.0], dtype=np.float64),
                ),
                set_test_value(
                    pt.dscalar(),
                    np.array(1000.0, dtype=np.float64),
                ),
            ],
            (2,),
            "halfnorm",
            lambda *args: args,
        ),
        (
            ptr.invgamma,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([10.4, 2.8], dtype=np.float64),
                ),
                set_test_value(
                    pt.dvector(),
                    np.array([3.4, 7.3], dtype=np.float64),
                ),
            ],
            (2,),
            "invgamma",
            lambda a, b: (a, 0, b),
        ),
        (
            ptr.chisquare,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([2.4, 4.9], dtype=np.float64),
                ),
            ],
            (2,),
            "chi2",
            lambda *args: args,
        ),
        (
            ptr.gengamma,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([10.4, 2.8], dtype=np.float64),
                ),
                set_test_value(
                    pt.dvector(),
                    np.array([3.4, 7.3], dtype=np.float64),
                ),
                set_test_value(
                    pt.dvector(),
                    np.array([0.9, 2.0], dtype=np.float64),
                ),
            ],
            (2,),
            "gengamma",
            lambda alpha, p, lambd: (alpha / p, p, 0, lambd),
        ),
        (
            ptr.wald,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([10.4, 2.8], dtype=np.float64),
                ),
                set_test_value(
                    pt.dvector(),
                    np.array([4.5, 2.0], dtype=np.float64),
                ),
            ],
            (2,),
            "invgauss",
            # https://stackoverflow.com/a/48603469
            lambda mean, scale: (mean / scale, 0, scale),
        ),
        pytest.param(
            ptr.vonmises,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([-0.5, 1.3], dtype=np.float64),
                ),
                set_test_value(
                    pt.dvector(),
                    np.array([5.5, 13.0], dtype=np.float64),
                ),
            ],
            (2,),
            "vonmises",
            lambda mu, kappa: (kappa, mu),
            marks=pytest.mark.skipif(
                not numpyro_available, reason="VonMises dispatch requires numpyro"
            ),
        ),
    ],
)
def test_random_RandomVariable(rv_op, dist_params, base_size, cdf_name, params_conv):
    """The JAX samplers are not one-to-one with NumPy samplers so we
    need to use a statistical test to make sure that the transpilation
    is correct.

    Parameters
    ----------
    rv_op
        The transpiled `RandomVariable` `Op`.
    dist_params
        The parameters passed to the op.

    """
    rng = shared(np.random.default_rng(29403))
    g = rv_op(*dist_params, size=(10000, *base_size), rng=rng)
    g_fn = compile_random_function(dist_params, g, mode=jax_mode)
    samples = g_fn(
        *[
            i.tag.test_value
            for i in g_fn.maker.fgraph.inputs
            if not isinstance(i, SharedVariable | Constant)
        ]
    )

    bcast_dist_args = np.broadcast_arrays(*[i.tag.test_value for i in dist_params])

    for idx in np.ndindex(*base_size):
        cdf_params = params_conv(*(arg[idx] for arg in bcast_dist_args))
        test_res = stats.cramervonmises(
            samples[(Ellipsis, *idx)], cdf_name, args=cdf_params
        )
        assert not np.isnan(test_res.statistic)
        assert test_res.pvalue > 0.01


@pytest.mark.parametrize(
    "rv_fn",
    [
        lambda param_that_implies_size: ptr.normal(
            loc=0, scale=pt.exp(param_that_implies_size)
        ),
        lambda param_that_implies_size: ptr.exponential(
            scale=pt.exp(param_that_implies_size)
        ),
        lambda param_that_implies_size: ptr.gamma(
            shape=1, scale=pt.exp(param_that_implies_size)
        ),
        lambda param_that_implies_size: ptr.t(
            df=3, loc=param_that_implies_size, scale=1
        ),
    ],
)
def test_size_implied_by_broadcasted_parameters(rv_fn):
    # We need a parameter with untyped shapes to test broadcasting does not result in identical draws
    param_that_implies_size = pt.matrix("param_that_implies_size", shape=(None, None))

    rv = rv_fn(param_that_implies_size)
    draws = rv.eval({param_that_implies_size: np.zeros((2, 2))}, mode=jax_mode)

    assert draws.shape == (2, 2)
    assert np.unique(draws).size == 4


@pytest.mark.parametrize("size", [(), (4,)])
def test_random_bernoulli(size):
    rng = shared(np.random.default_rng(123))
    g = pt.random.bernoulli(0.5, size=(1000, *size), rng=rng)
    g_fn = compile_random_function([], g, mode=jax_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples.mean(axis=0), 0.5, 1)


def test_random_mvnormal():
    rng = shared(np.random.default_rng(123))

    mu = np.ones(4)
    cov = np.eye(4)
    g = pt.random.multivariate_normal(mu, cov, size=(10000,), rng=rng)
    g_fn = compile_random_function([], g, mode=jax_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples.mean(axis=0), mu, atol=0.1)


@pytest.mark.parametrize(
    "parameter, size",
    [
        (np.ones(4), ()),
        (np.ones(4), (2, 4)),
    ],
)
def test_random_dirichlet(parameter, size):
    rng = shared(np.random.default_rng(123))
    g = pt.random.dirichlet(parameter, size=(1000, *size), rng=rng)
    g_fn = compile_random_function([], g, mode=jax_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples.mean(axis=0), 0.5, 1)


def test_random_choice():
    # `replace=True` and `p is None`
    rng = shared(np.random.default_rng(123))
    g = pt.random.choice(np.arange(4), size=10_000, rng=rng)
    g_fn = compile_random_function([], g, mode=jax_mode)
    samples = g_fn()
    assert samples.shape == (10_000,)
    # Elements are picked at equal frequency
    np.testing.assert_allclose(np.mean(samples == 3), 0.25, 2)

    # `replace=True` and `p is not None`
    rng = shared(np.random.default_rng(123))
    g = pt.random.choice(4, p=np.array([0.0, 0.5, 0.0, 0.5]), size=(5, 2), rng=rng)
    g_fn = compile_random_function([], g, mode=jax_mode)
    samples = g_fn()
    assert samples.shape == (5, 2)
    # Only odd numbers are picked
    assert np.all(samples % 2 == 1)

    # `replace=False` and `p is None`
    rng = shared(np.random.default_rng(123))
    g = pt.random.choice(np.arange(100), replace=False, size=(2, 49), rng=rng)
    g_fn = compile_random_function([], g, mode=jax_mode)
    samples = g_fn()
    assert samples.shape == (2, 49)
    # Elements are unique
    assert len(np.unique(samples)) == 98

    # `replace=False` and `p is not None`
    rng = shared(np.random.default_rng(123))
    g = pt.random.choice(
        8,
        p=np.array([0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0]),
        size=3,
        rng=rng,
        replace=False,
    )
    g_fn = compile_random_function([], g, mode=jax_mode)
    samples = g_fn()
    assert samples.shape == (3,)
    # Elements are unique
    assert len(np.unique(samples)) == 3
    # Only even numbers are picked
    assert np.all(samples % 2 == 0)


def test_random_categorical():
    rng = shared(np.random.default_rng(123))
    g = pt.random.categorical(0.25 * np.ones(4), size=(10000, 4), rng=rng)
    g_fn = compile_random_function([], g, mode=jax_mode)
    samples = g_fn()
    assert samples.shape == (10000, 4)
    np.testing.assert_allclose(samples.mean(axis=0), 6 / 4, 1)

    # Test zero probabilities
    g = pt.random.categorical([0, 0.5, 0, 0.5], size=(1000,), rng=rng)
    g_fn = compile_random_function([], g, mode=jax_mode)
    samples = g_fn()
    assert samples.shape == (1000,)
    assert np.all(samples % 2 == 1)


def test_random_permutation():
    array = np.arange(4)
    rng = shared(np.random.default_rng(123))
    g = pt.random.permutation(array, rng=rng)
    g_fn = compile_random_function([], g, mode=jax_mode)
    permuted = g_fn()
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(array, permuted)


@pytest.mark.parametrize(
    "batch_dims_tester",
    [
        batched_unweighted_choice_without_replacement_tester,
        batched_weighted_choice_without_replacement_tester,
        batched_permutation_tester,
    ],
)
def test_unnatural_batched_dims(batch_dims_tester):
    """Tests for RVs that don't have natural batch dims in JAX API."""
    batch_dims_tester(mode="JAX")


def test_random_geometric():
    rng = shared(np.random.default_rng(123))
    p = np.array([0.3, 0.7])
    g = pt.random.geometric(p, size=(10_000, 2), rng=rng)
    g_fn = compile_random_function([], g, mode=jax_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples.mean(axis=0), 1 / p, rtol=0.1)
    np.testing.assert_allclose(samples.std(axis=0), np.sqrt((1 - p) / p**2), rtol=0.1)


def test_negative_binomial():
    rng = shared(np.random.default_rng(123))
    n = np.array([10, 40])
    p = np.array([0.3, 0.7])
    g = pt.random.negative_binomial(n, p, size=(10_000, 2), rng=rng)
    g_fn = compile_random_function([], g, mode=jax_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples.mean(axis=0), n * (1 - p) / p, rtol=0.1)
    np.testing.assert_allclose(
        samples.std(axis=0), np.sqrt(n * (1 - p) / p**2), rtol=0.1
    )


@pytest.mark.skipif(not numpyro_available, reason="Binomial dispatch requires numpyro")
def test_binomial():
    rng = shared(np.random.default_rng(123))
    n = np.array([10, 40])
    p = np.array([0.3, 0.7])
    g = pt.random.binomial(n, p, size=(10_000, 2), rng=rng)
    g_fn = compile_random_function([], g, mode=jax_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples.mean(axis=0), n * p, rtol=0.1)
    np.testing.assert_allclose(samples.std(axis=0), np.sqrt(n * p * (1 - p)), rtol=0.1)


@pytest.mark.skipif(
    not numpyro_available, reason="BetaBinomial dispatch requires numpyro"
)
def test_beta_binomial():
    rng = shared(np.random.default_rng(123))
    n = np.array([10, 40])
    a = np.array([1.5, 13])
    b = np.array([0.5, 9])
    g = pt.random.betabinom(n, a, b, size=(10_000, 2), rng=rng)
    g_fn = compile_random_function([], g, mode=jax_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples.mean(axis=0), n * a / (a + b), rtol=0.1)
    np.testing.assert_allclose(
        samples.std(axis=0),
        np.sqrt((n * a * b * (a + b + n)) / ((a + b) ** 2 * (a + b + 1))),
        rtol=0.1,
    )


@pytest.mark.skipif(
    not numpyro_available, reason="Multinomial dispatch requires numpyro"
)
def test_multinomial():
    rng = shared(np.random.default_rng(123))
    n = np.array([10, 40])
    p = np.array([[0.3, 0.7, 0.0], [0.1, 0.4, 0.5]])
    g = pt.random.multinomial(n, p, size=(10_000, 2), rng=rng)
    g_fn = compile_random_function([], g, mode=jax_mode)
    samples = g_fn()
    np.testing.assert_allclose(samples.mean(axis=0), n[..., None] * p, rtol=0.1)
    np.testing.assert_allclose(
        samples.std(axis=0), np.sqrt(n[..., None] * p * (1 - p)), rtol=0.1
    )


@pytest.mark.skipif(not numpyro_available, reason="VonMises dispatch requires numpyro")
def test_vonmises_mu_outside_circle():
    # Scipy implementation does not behave as PyTensor/NumPy for mu outside the unit circle
    # We test that the random draws from the JAX dispatch work as expected in these cases
    rng = shared(np.random.default_rng(123))
    mu = np.array([-30, 40])
    kappa = np.array([100, 10])
    g = pt.random.vonmises(mu, kappa, size=(10_000, 2), rng=rng)
    g_fn = compile_random_function([], g, mode=jax_mode)
    samples = g_fn()
    np.testing.assert_allclose(
        samples.mean(axis=0), (mu + np.pi) % (2.0 * np.pi) - np.pi, rtol=0.1
    )

    # Circvar only does the correct thing in more recent versions of Scipy
    # https://github.com/scipy/scipy/pull/5747
    # np.testing.assert_allclose(
    #     stats.circvar(samples, axis=0),
    #     1 - special.iv(1, kappa) / special.iv(0, kappa),
    #     rtol=0.1,
    # )

    # For now simple compare with std from numpy draws
    rng = np.random.default_rng(123)
    ref_samples = rng.vonmises(mu, kappa, size=(10_000, 2))
    np.testing.assert_allclose(
        np.std(samples, axis=0), np.std(ref_samples, axis=0), rtol=0.1
    )


def test_random_unimplemented():
    """Compiling a graph with a non-supported `RandomVariable` should
    raise an error.

    """

    class NonExistentRV(RandomVariable):
        name = "non-existent"
        signature = "->()"
        dtype = "floatX"

        def __call__(self, size=None, **kwargs):
            return super().__call__(size=size, **kwargs)

        def rng_fn(cls, rng, size):
            return 0

    nonexistentrv = NonExistentRV()
    rng = shared(np.random.default_rng(123))
    out = nonexistentrv(rng=rng)
    fgraph = FunctionGraph([out.owner.inputs[0]], [out], clone=False)

    with pytest.raises(NotImplementedError):
        with pytest.warns(
            UserWarning, match=r"The RandomType SharedVariables \[.+\] will not be used"
        ):
            compare_jax_and_py(fgraph, [])


def test_random_custom_implementation():
    """We can register a JAX implementation for user-defined `RandomVariable`s"""

    class CustomRV(RandomVariable):
        name = "non-existent"
        signature = "->()"
        dtype = "floatX"

        def __call__(self, size=None, **kwargs):
            return super().__call__(size=size, **kwargs)

        def rng_fn(cls, rng, size):
            return 0

    from pytensor.link.jax.dispatch.random import jax_sample_fn

    @jax_sample_fn.register(CustomRV)
    def jax_sample_fn_custom(op, node):
        def sample_fn(rng, size, dtype, *parameters):
            return (rng, 0)

        return sample_fn

    nonexistentrv = CustomRV()
    rng = shared(np.random.default_rng(123))
    out = nonexistentrv(rng=rng)
    fgraph = FunctionGraph([out.owner.inputs[0]], [out], clone=False)
    with pytest.warns(
        UserWarning, match=r"The RandomType SharedVariables \[.+\] will not be used"
    ):
        compare_jax_and_py(fgraph, [])


def test_random_concrete_shape():
    """JAX should compile when a `RandomVariable` is passed a concrete shape.

    There are three quantities that JAX considers as concrete:
    1. Constants known at compile time;
    2. The shape of an array.
    3. `static_argnums` parameters
    This test makes sure that graphs with `RandomVariable`s compile when the
    `size` parameter satisfies either of these criteria.

    """
    rng = shared(np.random.default_rng(123))
    x_pt = pt.dmatrix()
    out = pt.random.normal(0, 1, size=x_pt.shape, rng=rng)
    jax_fn = compile_random_function([x_pt], out, mode=jax_mode)
    assert jax_fn(np.ones((2, 3))).shape == (2, 3)


def test_random_concrete_shape_from_param():
    rng = shared(np.random.default_rng(123))
    x_pt = pt.dmatrix()
    out = pt.random.normal(x_pt, 1, rng=rng)
    jax_fn = compile_random_function([x_pt], out, mode=jax_mode)
    assert jax_fn(np.ones((2, 3))).shape == (2, 3)


def test_random_concrete_shape_subtensor():
    """JAX should compile when a concrete value is passed for the `size` parameter.

    This test ensures that the `DimShuffle` `Op` used by PyTensor to turn scalar
    inputs into 1d vectors is replaced by an `Op` that turns concrete scalar
    inputs into tuples of concrete values using the `jax_size_parameter_as_tuple`
    rewrite.

    JAX does not accept scalars as `size` or `shape` arguments, so this is a
    slight improvement over their API.

    """
    rng = shared(np.random.default_rng(123))
    x_pt = pt.dmatrix()
    out = pt.random.normal(0, 1, size=x_pt.shape[1], rng=rng)
    jax_fn = compile_random_function([x_pt], out, mode=jax_mode)
    assert jax_fn(np.ones((2, 3))).shape == (3,)


def test_random_concrete_shape_subtensor_tuple():
    """JAX should compile when a tuple of concrete values is passed for the `size` parameter.

    This test ensures that the `MakeVector` `Op` used by PyTensor to turn tuple
    inputs into 1d vectors is replaced by an `Op` that turns a tuple of concrete
    scalar inputs into tuples of concrete values using the
    `jax_size_parameter_as_tuple` rewrite.

    """
    rng = shared(np.random.default_rng(123))
    x_pt = pt.dmatrix()
    out = pt.random.normal(0, 1, size=(x_pt.shape[0],), rng=rng)
    jax_fn = compile_random_function([x_pt], out, mode=jax_mode)
    assert jax_fn(np.ones((2, 3))).shape == (2,)


@pytest.mark.xfail(
    reason="`size_pt` should be specified as a static argument", strict=True
)
def test_random_concrete_shape_graph_input():
    rng = shared(np.random.default_rng(123))
    size_pt = pt.scalar()
    out = pt.random.normal(0, 1, size=size_pt, rng=rng)
    jax_fn = compile_random_function([size_pt], out, mode=jax_mode)
    assert jax_fn(10).shape == (10,)


def test_constant_shape_after_graph_rewriting():
    size = pt.vector("size", shape=(2,), dtype=int)
    x = pt.random.normal(size=size)
    assert x.type.shape == (None, None)

    with pytest.raises(TypeError):
        compile_random_function([size], x)([2, 5])

    # Rebuild with strict=False so output type is not updated
    # This reflects cases where size is constant folded during rewrites but the RV node is not recreated
    new_x = clone_replace(x, {size: pt.constant([2, 5])}, rebuild_strict=True)
    assert new_x.type.shape == (None, None)
    assert compile_random_function([], new_x)().shape == (2, 5)

    # Rebuild with strict=True, so output type is updated
    # This uses a different path in the dispatch implementation
    new_x = clone_replace(x, {size: pt.constant([2, 5])}, rebuild_strict=False)
    assert new_x.type.shape == (2, 5)
    assert compile_random_function([], new_x)().shape == (2, 5)
