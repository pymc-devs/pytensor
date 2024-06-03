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
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Constant
from pytensor.graph.fg import FunctionGraph
from tests.link.numba.test_basic import (
    compare_numba_and_py,
    numba_mode,
    set_test_value,
)
from tests.tensor.random.test_basic import (
    batched_permutation_tester,
    batched_unweighted_choice_without_replacement_tester,
    batched_weighted_choice_without_replacement_tester,
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


@pytest.mark.parametrize(
    "rv_op, dist_args, size",
    [
        (
            ptr.uniform,
            [
                set_test_value(
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
                set_test_value(
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.triangular,
            [
                set_test_value(
                    pt.dscalar(),
                    np.array(-5.0, dtype=np.float64),
                ),
                set_test_value(
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
                set_test_value(
                    pt.dscalar(),
                    np.array(5.0, dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.lognormal,
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
            pt.as_tensor([3, 2]),
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
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.exponential,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.weibull,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
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
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.geometric,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([0.3, 0.4], dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        pytest.param(
            ptr.hypergeometric,
            [
                set_test_value(
                    pt.lscalar(),
                    np.array(7, dtype=np.int64),
                ),
                set_test_value(
                    pt.lscalar(),
                    np.array(8, dtype=np.int64),
                ),
                set_test_value(
                    pt.lscalar(),
                    np.array(15, dtype=np.int64),
                ),
            ],
            pt.as_tensor([3, 2]),
            marks=pytest.mark.xfail,  # Not implemented
        ),
        (
            ptr.wald,
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
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.laplace,
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
            pt.as_tensor([3, 2]),
        ),
        (
            ptr.binomial,
            [
                set_test_value(
                    pt.lvector(),
                    np.array([1, 2], dtype=np.int64),
                ),
                set_test_value(
                    pt.dscalar(),
                    np.array(0.9, dtype=np.float64),
                ),
            ],
            pt.as_tensor([3, 2]),
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
            pt.as_tensor(tuple(set_test_value(pt.lscalar(), v) for v in [3, 2])),
        ),
        (
            ptr.poisson,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                ),
            ],
            None,
        ),
        (
            ptr.halfnormal,
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
            None,
        ),
        (
            ptr.bernoulli,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([0.1, 0.9], dtype=np.float64),
                ),
            ],
            None,
        ),
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
        ),
        (
            ptr.chisquare,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                )
            ],
            (2,),
        ),
        (
            ptr.negative_binomial,
            [
                set_test_value(
                    pt.lvector(),
                    np.array([100, 200], dtype=np.int64),
                ),
                set_test_value(
                    pt.dscalar(),
                    np.array(0.09, dtype=np.float64),
                ),
            ],
            (2,),
        ),
        (
            ptr.vonmises,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([-0.5, 0.5], dtype=np.float64),
                ),
                set_test_value(
                    pt.dscalar(),
                    np.array(1.0, dtype=np.float64),
                ),
            ],
            (2,),
        ),
        (
            ptr.permutation,
            [
                set_test_value(pt.dmatrix(), np.eye(5, dtype=np.float64)),
            ],
            (),
        ),
        (
            partial(ptr.choice, replace=True),
            [
                set_test_value(pt.dmatrix(), np.eye(5, dtype=np.float64)),
            ],
            pt.as_tensor([2]),
        ),
        (
            # p must be passed by kwarg
            lambda a, p, size, rng: ptr.choice(
                a, p=p, size=size, replace=True, rng=rng
            ),
            [
                set_test_value(pt.dmatrix(), np.eye(3, dtype=np.float64)),
                set_test_value(
                    pt.dvector(), np.array([0.25, 0.5, 0.25], dtype=np.float64)
                ),
            ],
            (pt.as_tensor([2, 3])),
        ),
        pytest.param(
            partial(ptr.choice, replace=False),
            [
                set_test_value(pt.dvector(), np.arange(5, dtype=np.float64)),
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
                set_test_value(pt.dmatrix(), np.eye(5, dtype=np.float64)),
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
                set_test_value(pt.vector(), np.arange(5, dtype=np.float64)),
                set_test_value(
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
                set_test_value(pt.dmatrix(), np.eye(3, dtype=np.float64)),
                set_test_value(
                    pt.dvector(), np.array([0.25, 0.5, 0.25], dtype=np.float64)
                ),
            ],
            (),
        ),
        pytest.param(
            # p must be passed by kwarg
            lambda a, p, size, rng: ptr.choice(
                a, p=p, size=size, replace=False, rng=rng
            ),
            [
                set_test_value(pt.dmatrix(), np.eye(3, dtype=np.float64)),
                set_test_value(
                    pt.dvector(), np.array([0.25, 0.5, 0.25], dtype=np.float64)
                ),
            ],
            (pt.as_tensor([2, 1])),
        ),
    ],
    ids=str,
)
def test_aligned_RandomVariable(rv_op, dist_args, size):
    """Tests for Numba samplers that are one-to-one with PyTensor's/NumPy's samplers."""
    rng = shared(np.random.default_rng(29402))
    g = rv_op(*dist_args, size=size, rng=rng)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, SharedVariable | Constant)
        ],
        eval_obj_mode=False,  # No python impl
    )


@pytest.mark.parametrize(
    "rv_op, dist_args, base_size, cdf_name, params_conv",
    [
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
            ptr.gumbel,
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
            "gumbel_r",
            lambda *args: args,
        ),
    ],
)
def test_unaligned_RandomVariable(rv_op, dist_args, base_size, cdf_name, params_conv):
    """Tests for Numba samplers that are not one-to-one with PyTensor's/NumPy's samplers."""
    rng = shared(np.random.default_rng(29402))
    g = rv_op(*dist_args, size=(2000, *base_size), rng=rng)
    g_fn = function(dist_args, g, mode=numba_mode)
    samples = g_fn(
        *[
            i.tag.test_value
            for i in g_fn.maker.fgraph.inputs
            if not isinstance(i, SharedVariable | Constant)
        ]
    )

    bcast_dist_args = np.broadcast_arrays(*[i.tag.test_value for i in dist_args])

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
            set_test_value(
                pt.dvector(),
                np.array([100000, 1, 1], dtype=np.float64),
            ),
            None,
            contextlib.suppress(),
        ),
        pytest.param(
            set_test_value(
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
            set_test_value(
                pt.dmatrix(),
                np.array(
                    [[100000, 1, 1], [1, 100000, 1], [1, 1, 100000]],
                    dtype=np.float64,
                ),
            ),
            (10, 4),
            pytest.raises(
                ValueError,
                match="Vectorized input 0 has an incompatible shape in axis 1.",
            ),
        ),
    ],
)
def test_DirichletRV(a, size, cm):
    rng = shared(np.random.default_rng(29402))
    g = ptr.dirichlet(a, size=size, rng=rng)
    g_fn = function([a], g, mode=numba_mode)

    with cm:
        a_val = a.tag.test_value

        all_samples = []
        for i in range(1000):
            samples = g_fn(a_val)
            all_samples.append(samples)

        exp_res = a_val / a_val.sum(-1)
        res = np.mean(all_samples, axis=tuple(range(0, a_val.ndim - 1)))
        assert np.allclose(res, exp_res, atol=1e-4)


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
