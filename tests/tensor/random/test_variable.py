import numpy as np
import pytest

from pytensor import function, shared
from pytensor.tensor.random.variable import (
    RandomGeneratorSharedVariable,
    shared_rng,
)


@pytest.mark.parametrize("borrow", (False, True))
def test_pytensor_shared(borrow):
    """Test that generic shared(np.random.generator) returns a RandomGeneratorSharedVariable"""
    gen = np.random.default_rng(42)
    srng = shared(gen, borrow=borrow)
    assert isinstance(srng, RandomGeneratorSharedVariable)

    gen_container = srng.container.storage[0]
    same = gen_container is gen
    assert same if borrow else not same


def test_shared_rng_seed_reproducible():
    srng1 = shared_rng(seed=123)
    srng2 = shared_rng(seed=123)
    next1, x1 = srng1.normal()
    next2, x2 = srng2.normal()
    fn1 = function([], x1, updates={srng1: next1})
    fn2 = function([], x2, updates={srng2: next2})
    assert fn1() != fn1()
    assert fn2() != fn2()
    assert fn1() == fn2()


@pytest.mark.parametrize("borrow", (False, True))
def test_shared_rng_construction(borrow):
    gen = np.random.default_rng(42)
    srng = shared_rng(gen, name="rng", borrow=borrow)
    assert isinstance(srng, RandomGeneratorSharedVariable)

    gen_container = srng.container.storage[0]
    assert isinstance(gen_container, np.random.Generator)
    same = gen_container is gen
    assert same if borrow else not same

    gen_retrieved = srng.get_value(borrow=True)
    assert isinstance(gen_retrieved, np.random.Generator)
    same = gen_retrieved is gen
    assert same if borrow else not same

    srng = shared_rng(seed=42, name="rng", borrow=borrow)
    assert isinstance(srng, RandomGeneratorSharedVariable)
    assert isinstance(srng.get_value(borrow=borrow), np.random.Generator)

    srng = shared_rng(seed=None, name="rng", borrow=borrow)
    assert isinstance(srng, RandomGeneratorSharedVariable)
    assert isinstance(srng.get_value(borrow=borrow), np.random.Generator)


def test_shared_rng_invalid_construction():
    with pytest.raises(ValueError, match="Must set one of value or seed"):
        shared_rng()
    with pytest.raises(ValueError, match="Cannot specify both value and seed"):
        shared_rng(value=np.random.default_rng(42), seed=42)
    with pytest.raises(TypeError, match=r"Expected numpy.random.Generator"):
        shared_rng(value=42)


@pytest.mark.parametrize("borrow", (False, True))
def test_set_value(borrow):
    srng = shared_rng(seed=42)
    next_rng, x = srng.normal()
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


def test_invalid_set_value():
    srng = shared_rng(seed=42)
    with pytest.raises(ValueError, match="Must set one of value or seed"):
        srng.set_value()
    with pytest.raises(ValueError, match="Cannot specify both new_value and seed"):
        srng.set_value(np.random.default_rng(42), seed=42)


def test_get_value_return_internal_type():
    """Test that return_internal_type (no-op) works"""
    srng = shared_rng(seed=42)
    r_T = srng.get_value(borrow=True, return_internal_type=True)
    r_F = srng.get_value(borrow=False, return_internal_type=True)
    assert isinstance(r_T, np.random.Generator)
    assert isinstance(r_F, np.random.Generator)
    assert r_T is srng.container.storage[0]
    assert r_F is not srng.container.storage[0]
