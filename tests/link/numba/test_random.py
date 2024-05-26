import contextlib
from functools import partial

import numpy as np
import pytest
import scipy.stats as stats

import pytensor.tensor as pt
import pytensor.tensor.random.basic as ptr
from pytensor import shared
from pytensor.compile.function import function
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Constant
from pytensor.graph.fg import FunctionGraph
from tests.link.numba.test_basic import (
    compare_numba_and_py,
    eval_python_only,
    numba_mode,
    set_test_value,
)
from tests.tensor.random.test_basic import (
    batched_permutation_tester,
    batched_unweighted_choice_without_replacement_tester,
    batched_weighted_choice_without_replacement_tester,
)


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "rv_op, dist_args, size",
    [
        (
            ptr.normal,
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
            ptr.uniform,
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
        pytest.param(
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
            marks=pytest.mark.xfail(reason="Not implemented"),
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
        (
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
            ptr.randint,
            [
                set_test_value(
                    pt.lscalar(),
                    np.array(0, dtype=np.int64),
                ),
                set_test_value(
                    pt.lscalar(),
                    np.array(5, dtype=np.int64),
                ),
            ],
            pt.as_tensor([3, 2]),
        ),
        pytest.param(
            ptr.multivariate_normal,
            [
                set_test_value(
                    pt.dmatrix(),
                    np.array([[1, 2], [3, 4]], dtype=np.float64),
                ),
                set_test_value(
                    pt.tensor(dtype="float64", shape=(1, None, None)),
                    np.eye(2)[None, ...],
                ),
            ],
            pt.as_tensor(tuple(set_test_value(pt.lscalar(), v) for v in [4, 3, 2])),
            marks=pytest.mark.xfail(reason="Not implemented"),
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
                    pt.dvector(), np.array([0.5, 0.0, 0.5], dtype=np.float64)
                ),
            ],
            (),
        ),
        (
            partial(ptr.choice, replace=False),
            [
                set_test_value(pt.dvector(), np.arange(5, dtype=np.float64)),
            ],
            pt.as_tensor([2]),
        ),
        pytest.param(
            partial(ptr.choice, replace=False),
            [
                set_test_value(pt.dmatrix(), np.eye(5, dtype=np.float64)),
            ],
            pt.as_tensor([2]),
            marks=pytest.mark.xfail(
                raises=ValueError,
                reason="Numba random.choice does not support >=1D `a`",
            ),
        ),
        pytest.param(
            # p must be passed by kwarg
            lambda a, p, size, rng: ptr.choice(
                a, p=p, size=size, replace=False, rng=rng
            ),
            [
                set_test_value(pt.vector(), np.arange(5, dtype=np.float64)),
                # Boring p, because the variable is not truly "aligned"
                set_test_value(
                    pt.dvector(),
                    np.array([0.5, 0.0, 0.25, 0.0, 0.25], dtype=np.float64),
                ),
            ],
            (),
            marks=pytest.mark.xfail(
                raises=Exception,  # numba.TypeError
                reason="Numba random.choice does not support `p` parameter",
            ),
        ),
        pytest.param(
            # p must be passed by kwarg
            lambda a, p, size, rng: ptr.choice(
                a, p=p, size=size, replace=False, rng=rng
            ),
            [
                set_test_value(pt.dmatrix(), np.eye(3, dtype=np.float64)),
                # Boring p, because the variable is not truly "aligned"
                set_test_value(
                    pt.dvector(), np.array([0.5, 0.0, 0.5], dtype=np.float64)
                ),
            ],
            (),
            marks=pytest.mark.xfail(
                raises=ValueError,
                reason="Numba random.choice does not support >=1D `a`",
            ),
        ),
    ],
    ids=str,
)
def test_aligned_RandomVariable(rv_op, dist_args, size):
    """Tests for Numba samplers that are one-to-one with PyTensor's/NumPy's samplers."""
    rng = shared(np.random.RandomState(29402))
    g = rv_op(*dist_args, size=size, rng=rng)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, SharedVariable | Constant)
        ],
    )


@pytest.mark.parametrize(
    "rv_op, dist_args, base_size, cdf_name, params_conv",
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
            ptr.chisquare,
            [
                set_test_value(
                    pt.dvector(),
                    np.array([1.0, 2.0], dtype=np.float64),
                )
            ],
            (2,),
            "chi2",
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
            "nbinom",
            lambda *args: args,
        ),
        pytest.param(
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
            "vonmises_line",
            lambda mu, kappa: (kappa, mu),
            marks=pytest.mark.xfail(
                reason=(
                    "Numba's parameterization of `vonmises` does not match NumPy's."
                    "See https://github.com/numba/numba/issues/7886"
                )
            ),
        ),
    ],
)
def test_unaligned_RandomVariable(rv_op, dist_args, base_size, cdf_name, params_conv):
    """Tests for Numba samplers that are not one-to-one with PyTensor's/NumPy's samplers."""
    rng = shared(np.random.RandomState(29402))
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
    "dist_args, size, cm",
    [
        pytest.param(
            [
                set_test_value(
                    pt.dvector(),
                    np.array([100000, 1, 1], dtype=np.float64),
                ),
            ],
            None,
            contextlib.suppress(),
        ),
        pytest.param(
            [
                set_test_value(
                    pt.dmatrix(),
                    np.array(
                        [[100000, 1, 1], [1, 100000, 1], [1, 1, 100000]],
                        dtype=np.float64,
                    ),
                ),
            ],
            (10, 3),
            contextlib.suppress(),
        ),
        pytest.param(
            [
                set_test_value(
                    pt.dmatrix(),
                    np.array(
                        [[100000, 1, 1]],
                        dtype=np.float64,
                    ),
                ),
            ],
            (5, 4, 3),
            contextlib.suppress(),
        ),
        pytest.param(
            [
                set_test_value(
                    pt.dmatrix(),
                    np.array(
                        [[100000, 1, 1], [1, 100000, 1], [1, 1, 100000]],
                        dtype=np.float64,
                    ),
                ),
            ],
            (10, 4),
            pytest.raises(
                ValueError, match="objects cannot be broadcast to a single shape"
            ),
        ),
    ],
)
def test_CategoricalRV(dist_args, size, cm):
    rng = shared(np.random.RandomState(29402))
    g = ptr.categorical(*dist_args, size=size, rng=rng)
    g_fg = FunctionGraph(outputs=[g])

    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


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
            pytest.raises(ValueError, match="Parameters shape.*"),
        ),
    ],
)
def test_DirichletRV(a, size, cm):
    rng = shared(np.random.RandomState(29402))
    g = ptr.dirichlet(a, size=size, rng=rng)
    g_fn = function([a], g, mode=numba_mode)

    with cm:
        a_val = a.tag.test_value

        # For coverage purposes only...
        eval_python_only([a], [g], [a_val])

        all_samples = []
        for i in range(1000):
            samples = g_fn(a_val)
            all_samples.append(samples)

        exp_res = a_val / a_val.sum(-1)
        res = np.mean(all_samples, axis=tuple(range(0, a_val.ndim - 1)))
        assert np.allclose(res, exp_res, atol=1e-4)


def test_RandomState_updates():
    rng = shared(np.random.RandomState(1))
    rng_new = shared(np.random.RandomState(2))

    x = pt.random.normal(size=10, rng=rng)
    res = function([], x, updates={rng: rng_new}, mode=numba_mode)()

    ref = np.random.RandomState(2).normal(size=10)
    assert np.allclose(res, ref)


def test_random_Generator():
    rng = shared(np.random.default_rng(29402))
    g = ptr.normal(rng=rng)
    g_fg = FunctionGraph(outputs=[g])

    with pytest.raises(TypeError):
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "batch_dims_tester",
    [
        pytest.param(
            batched_unweighted_choice_without_replacement_tester,
            marks=pytest.mark.xfail(raises=NotImplementedError),
        ),
        pytest.param(
            batched_weighted_choice_without_replacement_tester,
            marks=pytest.mark.xfail(raises=NotImplementedError),
        ),
        batched_permutation_tester,
    ],
)
def test_unnatural_batched_dims(batch_dims_tester):
    """Tests for RVs that don't have natural batch dims in Numba API."""
    batch_dims_tester(mode="NUMBA", rng_ctor=np.random.RandomState)
