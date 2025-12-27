import contextlib
from functools import partial

import numpy as np
import pytest
import scipy.stats as stats

import pytensor.tensor as pt
import pytensor.tensor.random.basic as ptr
from pytensor import shared
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.function import function
from pytensor.tensor.random.op import RandomVariableWithCoreShape
from tests.link.numba.test_basic import (
    compare_numba_and_py,
    numba_mode,
)
from tests.tensor.random.test_basic import (
    batched_permutation_tester,
    batched_unweighted_choice_without_replacement_tester,
    batched_weighted_choice_without_replacement_tester,
    create_mvnormal_cov_decomposition_method_test,
)


rng = np.random.default_rng(42849)


@pytest.mark.parametrize("mu_shape", [(), (3,), (5, 1)])
@pytest.mark.parametrize("sigma_shape", [(), (1,), (5, 3)])
@pytest.mark.parametrize("size_type", (None, "constant", "mutable"))
def test_random_size(mu_shape, sigma_shape, size_type):
    test_value_rng = np.random.default_rng(637)
    mu = test_value_rng.normal(size=mu_shape)
    sigma = np.exp(test_value_rng.normal(size=sigma_shape))

    # For testing
    rng = np.random.default_rng(123)
    pt_rng = shared(rng)
    if size_type is None:
        size = None
        pt_size = None
    elif size_type == "constant":
        size = (5, 3)
        pt_size = pt.as_tensor(size, dtype="int64")
    else:
        size = (5, 3)
        pt_size = shared(np.array(size, dtype="int64"), shape=(2,))

    next_rng, x = pt.random.normal(mu, sigma, rng=pt_rng, size=pt_size).owner.outputs
    fn = function([], x, updates={pt_rng: next_rng}, mode="NUMBA")

    res1 = fn()
    np.testing.assert_allclose(
        res1,
        rng.normal(mu, sigma, size=size),
    )

    res2 = fn()
    np.testing.assert_allclose(
        res2,
        rng.normal(mu, sigma, size=size),
    )

    pt_rng.set_value(np.random.default_rng(123))
    res3 = fn()
    np.testing.assert_array_equal(res1, res3)

    if size_type == "mutable" and len(mu_shape) < 2 and len(sigma_shape) < 2:
        pt_size.set_value(np.array((6, 3), dtype="int64"))
        res4 = fn()
        assert res4.shape == (6, 3)


def test_rng_copy():
    rng = shared(np.random.default_rng(123))
    x = pt.random.normal(rng=rng)

    fn = function([], x, mode="NUMBA")
    np.testing.assert_array_equal(fn(), fn())

    rng.type.values_eq(rng.get_value(), np.random.default_rng(123))


def test_rng_non_default_update():
    rng = shared(np.random.default_rng(1))
    rng_new = shared(np.random.default_rng(2))

    x = pt.random.normal(size=10, rng=rng)
    fn = function([], x, updates={rng: rng_new}, mode=numba_mode)

    ref = np.random.default_rng(1).normal(size=10)
    np.testing.assert_allclose(fn(), ref)

    ref = np.random.default_rng(2).normal(size=10)
    np.testing.assert_allclose(fn(), ref)
    np.testing.assert_allclose(fn(), ref)


def test_categorical_rv():
    """This is also a smoke test for a vector input scalar output RV"""
    p = np.array(
        [
            [
                [1.0, 0, 0, 0],
                [0.0, 1.0, 0, 0],
                [0.0, 0, 1.0, 0],
            ],
            [
                [0, 0, 0, 1.0],
                [0, 0, 0, 1.0],
                [0, 0, 0, 1.0],
            ],
        ]
    )
    x = pt.random.categorical(p=p, size=None)
    updates = {x.owner.inputs[0]: x.owner.outputs[0]}
    fn = function([], x, updates=updates, mode="NUMBA")
    res = fn()
    assert np.all(np.argmax(p, axis=-1) == res)

    # Batch size
    x = pt.random.categorical(p=p, size=(3, *p.shape[:-1]))
    fn = function([], x, updates=updates, mode="NUMBA")
    new_res = fn()
    assert new_res.shape == (3, *res.shape)
    for new_res_row in new_res:
        assert np.all(new_res_row == res)


def test_multivariate_normal():
    """This is also a smoke test for a multivariate RV"""
    rng = np.random.default_rng(123)

    x = pt.random.multivariate_normal(
        mean=np.zeros((3, 2)),
        cov=np.eye(2),
        rng=shared(rng),
    )

    fn = function([], x, mode="NUMBA")
    np.testing.assert_array_equal(
        fn(),
        rng.multivariate_normal(np.zeros(2), np.eye(2), size=(3,)),
    )


test_mvnormal_cov_decomposition_method = create_mvnormal_cov_decomposition_method_test(
    "NUMBA"
)


@pytest.mark.parametrize(
    "rv_op, dist_args, size",
    [
        (
            ptr.uniform,
            [
                (
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
                (
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.triangular,
            [
                (
                    pt.dscalar(),
                    np.array(-5.0, dtype=np.float64),
                ),
                (
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
                (
                    pt.dscalar(),
                    np.array(5.0, dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.lognormal,
            [
                (
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                (
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.pareto,
            [
                (
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                (
                    pt.dvector(),
                    np.array([2.0, 10.0], dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.exponential,
            [
                (
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.weibull,
            [
                (
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.logistic,
            [
                (
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                (
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.geometric,
            [
                (
                    pt.dvector(),
                    np.array([0.3, 0.4], dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.hypergeometric,
            [
                (
                    pt.lscalar(),
                    np.array(7, dtype=np.int64),
                ),
                (
                    pt.lscalar(),
                    np.array(8, dtype=np.int64),
                ),
                (
                    pt.lscalar(),
                    np.array(15, dtype=np.int64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.wald,
            [
                (
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                (
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.laplace,
            [
                (
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                (
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.binomial,
            [
                (
                    pt.lvector(),
                    np.array([1, 2], dtype=np.int64),
                ),
                (
                    pt.dscalar(),
                    np.array(0.9, dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.normal,
            [
                (
                    pt.lvector(),
                    np.array([1, 2], dtype=np.int64),
                ),
                (
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.poisson,
            [
                (
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
            ],
            None,
        ),
        (
            ptr.halfnormal,
            [
                (
                    pt.lvector(),
                    np.array([1, 2], dtype=np.int64),
                ),
                (
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            None,
        ),
        (
            ptr.bernoulli,
            [
                (
                    pt.dvector(),
                    np.array([0.1, 0.9], dtype=np.float64),
                ),
            ],
            None,
        ),
        (
            ptr.beta,
            [
                (
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                (
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
        ),
        (
            ptr._gamma,
            [
                (
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                (
                    pt.dvector(),
                    np.array([0.5, 3.0], dtype=np.float64),
                ),
            ],
            (2,),
        ),
        (
            ptr.chisquare,
            [
                (
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                )
            ],
            (2,),
        ),
        (
            ptr.negative_binomial,
            [
                (
                    pt.lvector(),
                    np.array([100, 200], dtype=np.int64),
                ),
                (
                    pt.dscalar(),
                    np.array(0.09, dtype=np.float64),
                ),
            ],
            (2,),
        ),
        (
            ptr.vonmises,
            [
                (
                    pt.dvector(),
                    np.array([-0.5, 0.5], dtype=np.float64),
                ),
                (
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
        ),
        (
            ptr.permutation,
            [
                (pt.dmatrix(), np.eye(5, dtype=np.float64)),
            ],
            (),
        ),
        (
            partial(ptr.choice, replace=True),
            [
                (pt.dmatrix(), np.eye(5, dtype=np.float64)),
            ],
            pt.as_tensor([2]),
        ),
        (
            # p must be passed by kwarg
            lambda a, p, size, rng: ptr.choice(
                a, p=p, size=size, replace=True, rng=rng
            ),
            [
                (pt.dmatrix(), np.eye(3, dtype=np.float64)),
                (pt.dvector(), np.array([0.25, 0.5, 0.25], dtype=np.float64)),
            ],
            (pt.as_tensor([2, 3])),
        ),
        pytest.param(
            partial(ptr.choice, replace=False),
            [
                (pt.dvector(), np.arange(5, dtype=np.float64)),
            ],
            pt.as_tensor([2]),
            marks=pytest.mark.xfail(
                AssertionError,
                reason="Not aligned with NumPy implementation",
            ),
        ),
        pytest.param(
            partial(ptr.choice, replace=False),
            [
                (pt.dmatrix(), np.eye(5, dtype=np.float64)),
            ],
            pt.as_tensor([2]),
            marks=pytest.mark.xfail(
                raises=AssertionError,
                reason="Not aligned with NumPy implementation",
            ),
        ),
        (
            # p must be passed by kwarg
            lambda a, p, size, rng: ptr.choice(
                a, p=p, size=size, replace=False, rng=rng
            ),
            [
                (pt.vector(), np.arange(5, dtype=np.float64)),
                (
                    pt.dvector(),
                    np.array([0.5, 0.0, 0.25, 0.0, 0.25], dtype=np.float64),
                ),
            ],
            pt.as_tensor([2]),
        ),
        pytest.param(
            # p must be passed by kwarg
            lambda a, p, size, rng: ptr.choice(
                a, p=p, size=size, replace=False, rng=rng
            ),
            [
                (pt.dmatrix(), np.eye(3, dtype=np.float64)),
                (pt.dvector(), np.array([0.25, 0.5, 0.25], dtype=np.float64)),
            ],
            (),
        ),
        pytest.param(
            # p must be passed by kwarg
            lambda a, p, size, rng: ptr.choice(
                a, p=p, size=size, replace=False, rng=rng
            ),
            [
                (pt.dmatrix(), np.eye(3, dtype=np.float64)),
                (pt.dvector(), np.array([0.25, 0.5, 0.25], dtype=np.float64)),
            ],
            (pt.as_tensor([2, 1])),
        ),
        (
            ptr.invgamma,
            [
                (
                    pt.dvector("shape"),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                (
                    pt.dvector("scale"),
                    np.array([0.5, 3.0], dtype=np.float64),
                ),
            ],
            (2,),
        ),
        (
            ptr.multinomial,
            [
                (
                    pt.lvector("n"),
                    np.array([1, 10, 1000], dtype=np.int64),
                ),
                (pt.dvector("p"), np.array([0.3, 0.7], dtype=np.float64)),
            ],
            None,
        ),
    ],
    ids=str,
)
def test_aligned_RandomVariable(rv_op, dist_args, size):
    """Tests for Numba samplers that are one-to-one with PyTensor's/NumPy's samplers."""
    dist_args, test_dist_args = zip(*dist_args, strict=True)
    rng = shared(np.random.default_rng(29402))
    g = rv_op(*dist_args, size=size, rng=rng)

    compare_numba_and_py(
        dist_args,
        [g],
        test_dist_args,
        eval_obj_mode=False,  # No python impl
    )


@pytest.mark.parametrize(
    "rv_op, dist_args, base_size, cdf_name, params_conv",
    [
        (
            ptr.cauchy,
            [
                (
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                (
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "cauchy",
            lambda *args: args,
        ),
        (
            ptr.gumbel,
            [
                (
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                (
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
            "gumbel_r",
            lambda *args: args,
        ),
        (
            ptr.t,
            [
                (pt.scalar(), np.array(np.e, dtype=np.float64)),
                (
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
                (
                    pt.dscalar(),
                    np.array(np.pi, dtype=np.float64),
                ),
            ],
            (2,),
            "t",
            lambda *args: args,
        ),
    ],
)
def test_unaligned_RandomVariable(rv_op, dist_args, base_size, cdf_name, params_conv):
    """Tests for Numba samplers that are not one-to-one with PyTensor's/NumPy's samplers."""
    dist_args, test_dist_args = zip(*dist_args, strict=True)
    rng = shared(np.random.default_rng(29402))
    g = rv_op(*dist_args, size=(2000, *base_size), rng=rng)
    g_fn = function(dist_args, g, mode=numba_mode)
    samples = g_fn(*test_dist_args)

    bcast_dist_args = np.broadcast_arrays(*test_dist_args)

    for idx in np.ndindex(*base_size):
        cdf_params = params_conv(*(arg[idx] for arg in bcast_dist_args))
        test_res = stats.cramervonmises(
            samples[(Ellipsis, *idx)], cdf_name, args=cdf_params
        )
        assert test_res.pvalue > 0.1


@pytest.mark.parametrize(
    "a, size, cm",
    [
        pytest.param(
            (
                pt.dvector(),
                np.array([100000, 1, 1], dtype=np.float64),
            ),
            None,
            contextlib.suppress(),
        ),
        pytest.param(
            (
                pt.dmatrix(),
                np.array(
                    [[100000, 1, 1], [1, 100000, 1], [1, 1, 100000]],
                    dtype=np.float64,
                ),
            ),
            (10, 3),
            contextlib.suppress(),
        ),
        pytest.param(
            (
                pt.dmatrix(),
                np.array(
                    [[100000, 1, 1], [1, 100000, 1], [1, 1, 100000]],
                    dtype=np.float64,
                ),
            ),
            (10, 4),
            pytest.raises(
                ValueError,
                match="Vectorized input 0 has an incompatible shape in axis 1\\.",
            ),
        ),
    ],
)
def test_DirichletRV(a, size, cm):
    a, a_val = a
    rng = shared(np.random.default_rng(29402))
    next_rng, g = ptr.dirichlet(a, size=size, rng=rng).owner.outputs
    g_fn = function([a], g, mode=numba_mode, updates={rng: next_rng})

    with cm:
        all_samples = [g_fn(a_val) for _ in range(1000)]
        exp_res = a_val / a_val.sum(-1)
        res = np.mean(all_samples, axis=tuple(range(0, a_val.ndim - 1)))
        assert np.allclose(res, exp_res, atol=1e-4)


def test_dirichlet_discrete_alpha():
    alpha = pt.lvector()
    g = ptr.dirichlet(alpha, size=100)
    fn = function([alpha], g, mode=numba_mode)
    res = fn(np.array([1, 1, 1], dtype=np.int64))
    assert res.dtype == np.float64
    np.testing.assert_allclose(res.sum(-1), 1.0)
    assert np.unique(res).size > 2  # Make sure we have more than just 0s and 1s


def test_cache_size_bcast_change():
    # Regression bug for caching with the same key in case where size is meaningful vs not
    alpha = pt.dvector()
    for s in (1, 2, 3):
        x = ptr.dirichlet(alpha, size=(s,))
        fn = function([alpha], x, mode=numba_mode)
        assert fn([0.2, 0.3, 0.5]).shape == (s, 3)


def test_rv_inside_ofg():
    rng_np = np.random.default_rng(562)
    rng = shared(rng_np)

    rng_dummy = rng.type()
    next_rng_dummy, rv_dummy = ptr.normal(
        0, 1, size=(3, 2), rng=rng_dummy
    ).owner.outputs
    out_dummy = rv_dummy.T

    next_rng, out = OpFromGraph([rng_dummy], [next_rng_dummy, out_dummy])(rng)
    fn = function([], out, updates={rng: next_rng}, mode=numba_mode)

    res1, res2 = fn(), fn()
    assert res1.shape == (2, 3)

    np.testing.assert_allclose(res1, rng_np.normal(0, 1, size=(3, 2)).T)
    np.testing.assert_allclose(res2, rng_np.normal(0, 1, size=(3, 2)).T)


@pytest.mark.parametrize(
    "batch_dims_tester",
    [
        batched_unweighted_choice_without_replacement_tester,
        batched_weighted_choice_without_replacement_tester,
        batched_permutation_tester,
    ],
)
def test_unnatural_batched_dims(batch_dims_tester):
    """Tests for RVs that don't have natural batch dims in Numba API."""
    batch_dims_tester(mode="NUMBA")


def test_repeated_args():
    v = pt.scalar()
    x = ptr.beta(v, v)
    fn, _ = compare_numba_and_py([v], [x], [0.5 * 1e6], eval_obj_mode=False)

    # Confirm we are testing a RandomVariable with repeated inputs
    final_node = fn.maker.fgraph.outputs[0].owner
    assert isinstance(final_node.op, RandomVariableWithCoreShape)
    assert final_node.inputs[-2] is final_node.inputs[-1]


def test_rv_fallback():
    """Test that random variables can fallback to object mode."""

    class CustomRV(ptr.RandomVariable):
        name = "custom"
        signature = "()->()"
        dtype = "float64"

        def rng_fn(self, rng, value, size=None):
            # Just return the value plus a random number
            return value + rng.standard_normal(size=size)

    custom_rv = CustomRV()

    rng = shared(np.random.default_rng(123))
    size = pt.scalar("size", dtype=int)
    next_rng, x = custom_rv(np.pi, size=(size,), rng=rng).owner.outputs

    fn = function([size], x, updates={rng: next_rng}, mode="NUMBA")

    result1 = fn(1)
    result2 = fn(1)
    assert result1.shape == (1,)
    assert result1 != result2

    large_sample = fn(1000)
    assert large_sample.shape == (1000,)
    np.testing.assert_allclose(large_sample.mean(), np.pi, rtol=1e-2)
    np.testing.assert_allclose(large_sample.std(), 1, rtol=1e-2)
