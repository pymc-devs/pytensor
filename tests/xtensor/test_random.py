import pytest


pytest.importorskip("xarray")
pytestmark = pytest.mark.filterwarnings("error")

import inspect
import re
from copy import deepcopy

import numpy as np

import pytensor.tensor.random as ptr
import pytensor.xtensor.random as pxr
from pytensor import config, function, shared
from pytensor.graph import rewrite_graph
from pytensor.graph.basic import equal_computations
from pytensor.tensor import broadcast_arrays, tensor
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.type import random_generator_type
from pytensor.xtensor import as_xtensor, xtensor
from pytensor.xtensor.random import (
    categorical,
    multinomial,
    multivariate_normal,
    normal,
)
from pytensor.xtensor.vectorization import XRV


def lower_rewrite(vars):
    return rewrite_graph(
        vars,
        include=(
            "lower_xtensor",
            "canonicalize",
        ),
    )


def test_all_basic_rvs_are_wrapped():
    # This ignores wrapper functions
    pxr_members = {name for name, _ in inspect.getmembers(pxr)}
    for name, op in inspect.getmembers(ptr.basic):
        if name in "_gamma":
            name = "gamma"
        if isinstance(op, RandomVariable) and name not in pxr_members:
            raise NotImplementedError(f"Variable {name} not implemented as XRV")


def test_updates():
    rng = shared(np.random.default_rng(40))
    next_rng, draws = normal(0, 1, rng=rng).owner.outputs
    fn = function([], [draws], updates=[(rng, next_rng)])
    res1, res2 = fn(), fn()

    rng = np.random.default_rng(40)
    expected_res1, expected_res2 = rng.normal(0, 1), rng.normal(0, 1)
    np.testing.assert_allclose(res1, expected_res1)
    np.testing.assert_allclose(res2, expected_res2)


def test_zero_inputs():
    class ZeroInputRV(RandomVariable):
        signature = "->()"
        dtype = "floatX"
        name = "ZeroInputRV"

        @classmethod
        def rng_fn(cls, rng, size=None):
            return rng.random(size=size)

    zero_input_rv = ZeroInputRV()
    zero_input_xrv = XRV(zero_input_rv, core_dims=((), ()), extra_dims=["a"])

    rng = random_generator_type("rng")
    a_size = xtensor("a_size", dims=(), dtype=int)
    rv = zero_input_xrv(rng, a_size)
    assert rv.type.dims == ("a",)
    assert rv.type.shape == (None,)

    rng_test = np.random.default_rng(12345)
    a_size_val = np.array(5)
    np.testing.assert_allclose(
        rv.eval({rng: rng_test, a_size: a_size_val}),
        rng_test.random(size=(a_size_val,)),
    )


def test_output_dim_does_not_map_from_input_dims():
    class NewDimRV(RandomVariable):
        signature = "()->(p)"
        dtype = "floatX"
        name = "NewDimRV"

        @classmethod
        def rng_fn(cls, rng, n, size=None):
            r = np.stack([n, n + 1], axis=-1)
            if size is None:
                return r
            return np.broadcast_to(r, (*size, 2))

        def _supp_shape_from_params(self, dist_params, param_shapes=None):
            return (2,)

    new_dim_rv = NewDimRV()
    new_dim_xrv = XRV(new_dim_rv, core_dims=(((),), ("p",)), extra_dims=["a"])

    a_size = xtensor("a_size", dims=(), dtype=int)
    rv = new_dim_xrv(None, a_size, 1)
    assert rv.type.dims == ("a", "p")
    assert rv.type.shape == (None, 2)

    a_size_val = np.array(5)
    np.testing.assert_allclose(
        rv.eval({a_size: a_size_val}), np.broadcast_to((1, 2), (a_size_val, 2))
    )


def test_dtype():
    x = normal(0, 1)
    assert x.type.dtype == config.floatX

    with config.change_flags(floatX="float64"):
        x = normal(0, 1)
    assert x.type.dtype == "float64"

    with config.change_flags(floatX="float32"):
        x = normal(0, 1)
    assert x.type.dtype == "float32"


def test_normal():
    rng = random_generator_type("rng")
    c_size = tensor("c_size", shape=(), dtype=int)
    mu = tensor("mu", shape=(3,))
    sigma = tensor("sigma", shape=(2,))

    mu_val = np.array([-10, 0.0, 10.0])
    sigma_val = np.array([1.0, 10.0])
    c_size_val = np.array(5)
    rng_val = np.random.default_rng(12345)

    c_size_xr = as_xtensor(c_size, name="c_size_xr")
    mu_xr = as_xtensor(mu, dims=("mu_dim",), name="mu_xr")
    sigma_xr = as_xtensor(sigma, dims=("sigma_dim",), name="sigma_xr")

    out = normal(mu_xr, sigma_xr, rng=rng)
    assert out.type.dims == ("mu_dim", "sigma_dim")
    assert out.type.shape == (3, 2)
    assert equal_computations(
        [lower_rewrite(out.values)],
        [rewrite_graph(ptr.normal(mu[:, None], sigma[None, :], rng=rng))],
    )

    out_eval = out.eval(
        {
            mu: mu_val,
            sigma: sigma_val,
            rng: rng_val,
        }
    )
    out_expected = deepcopy(rng_val).normal(mu_val[:, None], sigma_val[None, :])
    np.testing.assert_allclose(out_eval, out_expected)

    # Test with batch dimension
    out = normal(mu_xr, sigma_xr, extra_dims=dict(c_dim=c_size_xr), rng=rng)
    assert out.type.dims == ("c_dim", "mu_dim", "sigma_dim")
    assert out.type.shape == (None, 3, 2)
    lowered_size = (c_size, *broadcast_arrays(mu[:, None], sigma[None, :])[0].shape)
    assert equal_computations(
        [lower_rewrite(out.values)],
        [
            rewrite_graph(
                ptr.normal(mu[:, None], sigma[None, :], size=lowered_size, rng=rng)
            )
        ],
    )
    out_eval = out.eval(
        {
            mu: mu_val,
            sigma: sigma_val,
            c_size: c_size_val,
            rng: rng_val,
        }
    )
    out_expected = deepcopy(rng_val).normal(
        mu_val[:, None],
        sigma_val[None, :],
        size=(c_size_val, mu_val.shape[0], sigma_val.shape[0]),
    )
    np.testing.assert_allclose(out_eval, out_expected)

    # Test invalid core_dims
    with pytest.raises(
        ValueError,
        match=re.escape("normal needs 0 core_dims, but got 1"),
    ):
        normal(mu_xr, sigma_xr, core_dims=("a",), rng=rng)

    # Test Invalid extra_dims (conflicting with existing batch dims)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Size dimensions ['mu_dim'] conflict with parameter dimensions. They should be unique."
        ),
    ):
        pxr.normal(mu_xr, sigma_xr, extra_dims=dict(mu_dim=c_size_xr), rng=rng)


def test_categorical():
    rng = random_generator_type("rng")
    p = tensor("p", shape=(2, 3))
    c_size = tensor("c", shape=(), dtype=int)

    p_xr = as_xtensor(p, dims=("p", "batch_dim"), name="p_xr")
    c_size_xr = as_xtensor(c_size, name="c_size_xr")

    out = categorical(p_xr, core_dims=("p",), rng=rng)
    assert out.type.dims == ("batch_dim",)
    assert out.type.shape == (3,)
    assert equal_computations(
        [lower_rewrite(out.values)], [ptr.categorical(p.T, rng=rng)]
    )
    np.testing.assert_allclose(
        out.eval(
            {
                p: np.array([[1.0, 0], [0, 1.0], [1.0, 0]]).T,
                rng: np.random.default_rng(),
            }
        ),
        np.array([0, 1, 0]),
    )

    out = categorical(
        p_xr, core_dims=("p",), extra_dims=dict(cp1=c_size_xr + 1, c=c_size_xr), rng=rng
    )
    assert out.type.dims == ("cp1", "c", "batch_dim")
    assert out.type.shape == (None, None, 3)
    assert equal_computations(
        [lower_rewrite(out.values)],
        [
            rewrite_graph(
                ptr.categorical(
                    p.T, size=(1 + c_size, c_size, p[0].shape.squeeze()), rng=rng
                )
            )
        ],
    )
    np.testing.assert_allclose(
        out.eval(
            {
                p: np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]).T,
                c_size: np.array(5),
                rng: np.random.default_rng(),
            }
        ),
        np.broadcast_to([0, 1, 0], shape=(6, 5, 3)),
    )

    # Test invaild core dims
    with pytest.raises(
        ValueError, match="categorical needs 1 core_dims to be specified"
    ):
        categorical(p_xr, rng=rng)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "At least one core dim=('px',) missing from input p_xr with dims=('p', 'batch_dim')"
        ),
    ):
        categorical(p_xr, core_dims=("px",), rng=rng)


def test_multinomial():
    rng = random_generator_type("rng")
    n = tensor("n", shape=(2,))
    p = tensor("p", shape=(3, None))
    c_size = tensor("c", shape=(), dtype=int)
    n_xr = as_xtensor(n, dims=("a",), name="a_xr")
    p_xr = as_xtensor(p, dims=("p", "a"), name="p_xr")
    c_size_xr = as_xtensor(c_size, name="c_size_xr")
    a_size_xr = n_xr.sizes["a"]

    out = multinomial(n_xr, p_xr, core_dims=("p",), rng=rng)
    assert out.type.dims == ("a", "p")
    assert out.type.shape == (2, 3)
    assert equal_computations(
        [lower_rewrite(out.values)],
        [ptr.multinomial(n, p.T, size=None, rng=rng)],
    )
    # Test we can actually evaluate it
    np.testing.assert_allclose(
        out.eval(
            {
                n: [5, 10],
                p: np.array([[1.0, 0, 0], [0, 0, 1.0]]).T,
                rng: np.random.default_rng(),
            }
        ),
        np.array([[5, 0, 0], [0, 0, 10]]),
    )

    out = multinomial(
        n_xr, p_xr, core_dims=("p",), extra_dims=dict(c=c_size_xr), rng=rng
    )
    assert out.type.dims == ("c", "a", "p")
    assert equal_computations(
        [lower_rewrite(out.values)],
        [rewrite_graph(ptr.multinomial(n, p.T, size=(c_size, n.shape[0]), rng=rng))],
    )

    # Test we can actually evaluate it with extra_dims
    np.testing.assert_allclose(
        out.eval(
            {
                n: [5, 10],
                p: np.array([[1.0, 0, 0], [0, 0, 1.0]]).T,
                c_size: 5,
                rng: np.random.default_rng(),
            }
        ),
        np.broadcast_to(
            [[5, 0, 0], [0, 0, 10]],
            shape=(5, 2, 3),
        ),
    )

    # Test invalid core_dims
    with pytest.raises(
        ValueError, match="multinomial needs 1 core_dims to be specified"
    ):
        multinomial(n_xr, p_xr, rng=rng)

    with pytest.raises(ValueError, match="multinomial needs 1 core_dims, but got 2"):
        multinomial(n_xr, p_xr, core_dims=("p1", "p2"), rng=rng)

    with pytest.raises(
        ValueError, match=re.escape("Parameter a_xr has invalid core dimensions ['a']")
    ):
        # n cannot have a core dimension
        multinomial(n_xr, p_xr, core_dims=("a",), rng=rng)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "At least one core dim=('px',) missing from input p_xr with dims=('p', 'a')"
        ),
    ):
        multinomial(n_xr, p_xr, core_dims=("px",), rng=rng)

    # Test invalid extra_dims
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Size dimensions ['a'] conflict with parameter dimensions. They should be unique."
        ),
    ):
        multinomial(
            n_xr,
            p_xr,
            core_dims=("p",),
            extra_dims=dict(c=c_size_xr, a=a_size_xr),
            rng=rng,
        )


def test_multivariate_normal():
    rng = random_generator_type("rng")
    mu = tensor("mu", shape=(4, 2))
    cov = tensor("cov", shape=(2, 3, 2, 4))

    mu_xr = as_xtensor(mu, dims=("b1", "rows"), name="mu_xr")
    cov_xr = as_xtensor(cov, dims=("cols", "b2", "rows", "b1"), name="cov_xr")

    out = multivariate_normal(mu_xr, cov_xr, core_dims=("rows", "cols"), rng=rng)
    assert out.type.dims == ("b1", "b2", "rows")
    assert out.type.shape == (4, 3, 2)
    assert equal_computations(
        [lower_rewrite(out.values)],
        [ptr.multivariate_normal(mu[:, None], cov.transpose(3, 1, 2, 0), rng=rng)],
    )

    # Order of core_dims doesn't matter
    out = multivariate_normal(mu_xr, cov_xr, core_dims=("cols", "rows"), rng=rng)
    assert out.type.dims == ("b1", "b2", "rows")
    assert out.type.shape == (4, 3, 2)
    assert equal_computations(
        [lower_rewrite(out.values)],
        [ptr.multivariate_normal(mu[:, None], cov.transpose(3, 1, 2, 0), rng=rng)],
    )

    # Test method
    out = multivariate_normal(
        mu_xr, cov_xr, core_dims=("rows", "cols"), rng=rng, method="svd"
    )
    assert equal_computations(
        [lower_rewrite(out.values)],
        [
            ptr.multivariate_normal(
                mu[:, None], cov.transpose(3, 1, 2, 0), rng=rng, method="svd"
            )
        ],
    )

    # Test invalid core_dims
    with pytest.raises(
        TypeError,
        match=re.escape(
            "multivariate_normal() missing 1 required keyword-only argument: 'core_dims'"
        ),
    ):
        multivariate_normal(mu_xr, cov_xr)

    with pytest.raises(
        ValueError, match="multivariate_normal requires 2 core_dims, got 3"
    ):
        multivariate_normal(mu_xr, cov_xr, core_dims=("b1", "rows", "cols"))

    with pytest.raises(
        ValueError, match=re.escape("Operand has repeated dims ('rows', 'rows')")
    ):
        multivariate_normal(mu_xr, cov_xr, core_dims=("rows", "rows"))

    with pytest.raises(
        ValueError,
        match=re.escape("Parameter mu_xr has invalid core dimensions ['b1']"),
    ):
        # mu cannot have two core_dims
        multivariate_normal(mu_xr, cov_xr, core_dims=("rows", "b1"))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "At least one core dim=('rows', 'missing_cols') missing from input cov_xr with dims=('cols', 'b2', 'rows', 'b1')"
        ),
    ):
        # cov must have both core_dims
        multivariate_normal(mu_xr, cov_xr, core_dims=("rows", "missing_cols"))
