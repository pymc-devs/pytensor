import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.looping import InnerShiftedArg, loop, shift
from pytensor.tensor.random.type import RandomGeneratorType


def test_simple_carry():
    x = pt.scalar("x")
    init = x

    def f(carry, x):
        x_t = carry
        return x_t * 2, None

    out, _ = loop(
        f,
        init=init,
        length=5,
    )

    np.testing.assert_allclose(out.eval({x: 2}), 2**6)


def test_structured_carry():
    x0 = pt.scalar("x0")
    init = ((x0,), x0 * 2)

    def f(carry, _):
        ((x_t,), y_t) = carry
        return ((x_t * 2,), y_t * 2), None

    ((x_final,), y_final), _ = loop(
        f,
        init=init,
        length=5,
    )

    np.testing.assert_allclose(x_final.eval({x0: 2}), 2**6)
    np.testing.assert_allclose(y_final.eval({x0: 2}), 2**7)


def test_simple_xs():
    x0 = pt.scalar("x0")
    beta = pt.vector("beta", shape=(5,))
    xs = beta

    def f(x_t, beta_t):
        return x_t * beta_t, None

    x_final, _ = loop(f, init=x0, xs=xs)

    np.testing.assert_allclose(
        x_final.eval({x0: 2, beta: [2, 1, 2, 1, 2]}),
        2**4,
    )


def test_structured_xs():
    x0 = pt.scalar("x0")
    beta = pt.vector("beta", shape=(5,))
    xs = (beta, (3 - beta,))

    def f(x_t, x):
        beta_t, (complement_beta_t,) = x
        return x_t * beta_t * complement_beta_t, None

    x_final, _ = loop(f, init=x0, xs=xs)

    np.testing.assert_allclose(
        x_final.eval({x0: 2, beta: [2, 1, 2, 1, 2]}),
        2**6,
    )


def test_simple_ys():
    xs = pt.vector("beta", shape=(5,))

    def f(_, x_t):
        return None, x_t * 2

    _, ys = loop(
        f,
        init=None,
        xs=xs,
    )

    xs_test = np.arange(
        5,
    )
    np.testing.assert_allclose(ys.eval({xs: xs_test}), xs_test * 2)


def test_structured_ys():
    xs = pt.vector("x", shape=(5,))

    def f(_, x_t):
        return None, (x_t * 2, ((x_t * 3,),))

    _, (ys, ((zs,),)) = loop(
        f,
        init=None,
        xs=xs,
    )

    xs_test = np.arange(
        5,
    )
    np.testing.assert_allclose(ys.eval({xs: xs_test}), xs_test * 2)
    np.testing.assert_allclose(zs.eval({xs: xs_test}), xs_test * 3)


def test_shifted_carry():
    x0 = pt.tensor("x0", shape=(2,))
    init = shift(x0, by=[-2, -1])

    def f(carry, xs):
        assert isinstance(carry, InnerShiftedArg)
        xtm2, xtm1 = carry
        return carry.push(xtm2 * xtm1), None

    x_final, _ = loop(
        f,
        init=init,
        length=3,
    )
    np.testing.assert_allclose(x_final.eval({x0: [1, 2]}), 8)


def test_shifted_xs():
    x0 = pt.scalar("x0")
    seq = pt.vector("seq", shape=(5,))
    xs = shift(seq, by=[0, 1])

    def f(carry, xs):
        assert isinstance(xs, InnerShiftedArg)
        xt, xtp1 = xs
        return carry + xt * xtp1, None

    x_final, _ = loop(
        f,
        init=x0,
        xs=xs,
    )

    seq_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # n_steps = 5 - 1 = 4 (due to tap offset)
    # step 0: 0 + 1*2 = 2
    # step 1: 2 + 2*3 = 8
    # step 2: 8 + 3*4 = 20
    # step 3: 20 + 4*5 = 40
    np.testing.assert_allclose(x_final.eval({x0: 0, seq: seq_val}), 40)


def test_shifted_xs_push_raises():
    seq = pt.vector("seq", shape=(5,))
    xs = shift(seq, by=[0, 1])

    def f(carry, xs):
        xs.push(xs[0] + xs[1])
        return carry, None

    with pytest.raises(ValueError, match="read-only"):
        loop(f, init=pt.scalar("x"), xs=xs)


def test_single_shift_carry():
    x0 = pt.tensor("x0", shape=(1,))
    init = shift(x0, by=-1)

    def f(carry, _):
        assert isinstance(carry, InnerShiftedArg)
        (xtm1,) = carry
        return carry.push(xtm1 * 2), None

    x_final, _ = loop(f, init=init, length=5)
    # x0=[3] → 6 → 12 → 24 → 48 → 96
    np.testing.assert_allclose(x_final.eval({x0: [3]}), 96)


def test_single_shift_xs():
    seq = pt.vector("seq", shape=(5,))
    xs = shift(seq, by=1)

    def f(_, xs):
        assert isinstance(xs, InnerShiftedArg)
        (xtp1,) = xs
        return None, xtp1 * 2

    _, ys = loop(f, init=None, xs=xs)

    seq_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # n_steps = 4 (seq[1:] has length 4)
    # ys = [2*2, 3*2, 4*2, 5*2] = [4, 6, 8, 10]
    np.testing.assert_allclose(ys.eval({seq: seq_val}), [4, 6, 8, 10])


def test_rng_carry():
    rng_init = RandomGeneratorType()("rng")

    def f(carry, _):
        rng = carry
        next_rng, sample = pt.random.normal(rng=rng).owner.outputs
        return next_rng, sample

    final_rng, samples = loop(f, init=rng_init, length=5)

    fn = pytensor.function([rng_init], [samples, final_rng])

    rng0 = np.random.default_rng(42)
    result1, rng1 = fn(rng0)
    assert result1.shape == (5,)
    assert not np.all(result1 == result1[0])

    # Same input RNG yields same results
    result2, _ = fn(rng0)
    np.testing.assert_array_equal(result1, result2)

    # Updated RNG yields fresh samples
    result3, _ = fn(rng1)
    assert not np.array_equal(result1, result3)


def test_rng_ys_trace_fails():
    rng_init = RandomGeneratorType()("rng")

    def f(carry, _):
        rng = carry
        next_rng, _sample = pt.random.normal(rng=rng).owner.outputs
        return next_rng, next_rng  # Trying to trace the RNG state as ys

    with pytest.raises(TypeError, match="ys outputs must be TensorVariables"):
        loop(f, init=rng_init, length=5)


def test_mixed_carry_and_xs():
    # Traced carry (TensorVariable)
    x0 = pt.scalar("x0")
    # Untraced carry (RNG)
    rng_init = RandomGeneratorType()("rng")
    # Shifted carry (mit_sot)
    y0 = pt.tensor("y0", shape=(2,))
    y_shifted = shift(y0, by=[-2, -1])

    # Regular xs + shifted xs
    seq = pt.vector("seq", shape=(6,))
    seq_shifted = shift(seq, by=[0, 1])

    init = (x0, rng_init, y_shifted)

    def f(carry, xs):
        x, rng, y_carry = carry
        xt, xtp1 = xs

        next_rng, noise = pt.random.normal(rng=rng).owner.outputs
        ytm2, ytm1 = y_carry
        new_y = ytm2 + ytm1

        return (x + xt + noise, next_rng, y_carry.push(new_y)), xt * xtp1

    (x_final, final_rng, y_final), products = loop(f, init=init, xs=seq_shifted)

    fn = pytensor.function(
        [x0, rng_init, y0, seq], [x_final, final_rng, y_final, products]
    )
    seq_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    rng0 = np.random.default_rng(42)
    x_final_val1, rng1, y_final_val, products_val = fn(0.0, rng0, [1.0, 1.0], seq_val)

    # n_steps = 5 (shift (0,1) on seq of length 6)
    # products: [1*2, 2*3, 3*4, 4*5, 5*6]
    np.testing.assert_allclose(products_val, [2, 6, 12, 20, 30])

    # y: fibonacci-like from [1, 1]
    # step 0: 1+1=2, step 1: 1+2=3, step 2: 2+3=5, step 3: 3+5=8, step 4: 5+8=13
    np.testing.assert_allclose(y_final_val, 13)

    # Same input RNG yields same results
    x_final_val2, _, _, _ = fn(0.0, rng0, [1.0, 1.0], seq_val)
    np.testing.assert_array_equal(x_final_val1, x_final_val2)

    # Updated RNG yields fresh samples (x_final includes noise, so it should differ)
    x_final_val3, _, _, _ = fn(0.0, rng1, [1.0, 1.0], seq_val)
    assert not np.array_equal(x_final_val1, x_final_val3)
