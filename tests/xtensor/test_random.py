import inspect
import re

import numpy as np
import pytest

import pytensor.tensor.random as ptr
import pytensor.xtensor.random as pxr
from pytensor.graph import rewrite_graph
from pytensor.graph.basic import equal_computations
from pytensor.tensor import tensor
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.type import random_generator_type
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.random import multinomial


def lower_rewrite(vars):
    return rewrite_graph(
        vars,
        include=(
            "xcanonicalize",
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


def test_normal():
    pass


def test_categorical():
    pass


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

    # Test wrong core_dims
    with pytest.raises(ValueError, match="multinomial needs 1 core_dims, but got 2"):
        multinomial(n_xr, p_xr, core_dims=("p1", "p2"), rng=rng)

    # Test invalid core_dims
    with pytest.raises(
        ValueError, match=re.escape("Parameter a_xr has invalid core dimensions ['a']")
    ):
        # n cannot have a core dimension
        multinomial(n_xr, p_xr, core_dims=("a",), rng=rng)

    # Test missing core_dims
    with pytest.raises(
        ValueError,
        match=re.escape(
            "At least one core dim=('px',) missing from input p_xr with dims=('p', 'a')"
        ),
    ):
        # p cannot have a core dimension
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
    pass


def test_output_dime_does_not_map_from_input_dime():
    pass


def test_zero_inputs():
    pass
