import pytest


pytest.importorskip("xarray")

import numpy as np

from pytensor.compile import function
from pytensor.xtensor.random import rng, shared_rng
from pytensor.xtensor.random.variable import (
    XRandomGeneratorSharedVariable,
    XRandomGeneratorVariable,
)
from pytensor.xtensor.type import XTensorVariable


def test_chained_rng():
    init_rng = rng(name="rng")
    next_rng, x = init_rng.normal(0, 1, extra_dims={"a": 3})
    assert isinstance(next_rng, XRandomGeneratorVariable)
    assert isinstance(x, XTensorVariable)
    final_rng, y = next_rng.normal(x, 1, extra_dims={"b": 5})
    assert isinstance(final_rng, XRandomGeneratorVariable)
    assert isinstance(y, XTensorVariable)

    fn = function([init_rng], [final_rng, x, y])

    rng_val = np.random.default_rng(123)
    next_rng_val, x_val, y_val = fn(rng_val)
    assert x_val.shape == (3,)
    assert y_val.shape == (5, 3)

    _, new_x_val, new_y_val = fn(next_rng_val)
    assert new_x_val.shape == (3,)
    assert new_y_val.shape == (5, 3)
    assert (new_x_val != x_val).all()
    assert (new_y_val != y_val).all()

    _, repeated_x_val, repeated_y_val = fn(rng_val)
    assert repeated_x_val.shape == (3,)
    assert repeated_y_val.shape == (5, 3)
    assert (repeated_x_val == x_val).all()
    assert (repeated_y_val == y_val).all()


def test_chained_shared_xrng():
    init_rng = shared_rng(seed=328, name="xrng")
    next_xrng, x = init_rng.normal(0, 1, extra_dims={"a": 3})
    assert isinstance(next_xrng, XRandomGeneratorVariable)
    assert isinstance(x, XTensorVariable)
    final_xrng, y = next_xrng.normal(x, 1, extra_dims={"b": 5})
    assert isinstance(next_xrng, XRandomGeneratorVariable)
    assert isinstance(x, XTensorVariable)

    fn = function([], [x, y], updates={init_rng: final_xrng})

    x_val, y_val = fn()
    assert x_val.shape == (3,)
    assert y_val.shape == (5, 3)

    new_x_val, new_y_val = fn()
    assert new_x_val.shape == (3,)
    assert new_y_val.shape == (5, 3)
    assert (new_x_val != x_val).all()
    assert (new_y_val != y_val).all()

    init_rng.set_value(seed=328)
    repeated_x_val, repeated_y_val = fn()
    assert repeated_x_val.shape == (3,)
    assert repeated_y_val.shape == (5, 3)
    assert (repeated_x_val == x_val).all()
    assert (repeated_y_val == y_val).all()


def test_shared_xrng_seed_reproducible():
    srng1 = shared_rng(seed=123)
    srng2 = shared_rng(seed=123)
    next1, x1 = srng1.normal(0, 1)
    next2, x2 = srng2.normal(0, 1)
    fn1 = function([], x1, updates={srng1: next1})
    fn2 = function([], x2, updates={srng2: next2})
    assert fn1() != fn1()
    assert fn2() != fn2()
    assert fn1() == fn2()


@pytest.mark.parametrize("borrow", (False, True))
def test_shared_xrng_construction(borrow):
    gen = np.random.default_rng(42)
    srng = shared_rng(gen, name="xrng", borrow=borrow)
    assert isinstance(srng, XRandomGeneratorSharedVariable)
    assert isinstance(srng, XRandomGeneratorVariable)

    gen_container = srng.container.storage[0]
    assert isinstance(gen_container, np.random.Generator)
    same = gen_container is gen
    assert same if borrow else not same

    gen_retrieved = srng.get_value(borrow=True)
    assert isinstance(gen_retrieved, np.random.Generator)
    same = gen_retrieved is gen
    assert same if borrow else not same

    srng = shared_rng(seed=42, name="xrng", borrow=borrow)
    assert isinstance(srng, XRandomGeneratorSharedVariable)
    assert isinstance(srng.get_value(borrow=borrow), np.random.Generator)

    srng = shared_rng(seed=None, name="xrng", borrow=borrow)
    assert isinstance(srng, XRandomGeneratorSharedVariable)
    assert isinstance(srng.get_value(borrow=borrow), np.random.Generator)


def test_shared_xrng_invalid_construction():
    with pytest.raises(ValueError, match="Must set one of value or seed"):
        shared_rng()
    with pytest.raises(ValueError, match="Cannot specify both value and seed"):
        shared_rng(value=np.random.default_rng(42), seed=42)
    with pytest.raises(TypeError, match=r"Expected numpy.random.Generator"):
        shared_rng(42)


@pytest.mark.parametrize("borrow", (False, True))
def test_shared_xrng_set_value(borrow):
    srng = shared_rng(seed=42)
    next_rng, x = srng.normal(0, 1)
    fn = function([], x, updates={srng: next_rng})
    v1 = fn()
    assert v1 != fn()

    new_gen = np.random.default_rng(42)
    srng.set_value(new_gen, borrow=borrow)
    retrieved_gen = srng.get_value(borrow=True)
    same = retrieved_gen == new_gen
    assert same if borrow else not same
    assert v1 == fn() != fn()

    srng.set_value(seed=42, borrow=borrow)
    assert v1 == fn() != fn()

    srng.set_value(seed=None, borrow=borrow)
    assert v1 != fn() != fn()


def test_shared_xrng_invalid_set_value():
    srng = shared_rng(seed=42)
    with pytest.raises(ValueError, match="Must set one of value or seed"):
        srng.set_value()
    with pytest.raises(ValueError, match="Cannot specify both new_value and seed"):
        srng.set_value(np.random.default_rng(42), seed=42)


def test_shared_xrng_get_value_return_internal_type():
    """Test that return_internal_type (no-op) works"""
    srng = shared_rng(seed=42)
    r_T = srng.get_value(borrow=True, return_internal_type=True)
    r_F = srng.get_value(borrow=False, return_internal_type=True)
    assert isinstance(r_T, np.random.Generator)
    assert isinstance(r_F, np.random.Generator)
    assert r_T is srng.container.storage[0]
    assert r_F is not srng.container.storage[0]
