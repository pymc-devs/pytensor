from collections import namedtuple

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.looping import InnerShiftedArg, flatten_tree, loop, shift, unflatten_tree
from pytensor.tensor.random.type import RandomGeneratorType


State = namedtuple("State", "total count")


@pytest.mark.parametrize(
    "tree",
    [
        [],
        (),
        {},
        [1, [2, None], (), []],
        {"b": [2, None], "a": State(1, (3,))},
        State([1, 2], {"count": 3}),
    ],
)
@pytest.mark.parametrize("subsume_none", [False, True])
def test_tree_roundtrip(tree, subsume_none):
    leaves, spec = flatten_tree(tree, subsume_none=subsume_none)
    restored = unflatten_tree(leaves, spec)
    assert restored == tree
    assert type(restored) is type(tree)


@pytest.mark.parametrize("leaves", [[], [1, 2, 3]])
def test_unflatten_wrong_leaf_count(leaves):
    _, spec = flatten_tree(State(1, 2))
    with pytest.raises(ValueError, match="entries than expected"):
        unflatten_tree(leaves, spec)


def test_dictionary_and_list_loop():
    x = pt.vector("x")
    initial = pt.scalar("initial")

    def step(carry, slices):
        total = carry["state"].total + slices[0]
        count = carry["state"].count + slices[1]["weight"]
        # Return keys in a different order from init.
        return {"empty": [], "state": State(total, count)}, [total, {"count": count}]

    final, traces = loop(
        step,
        init={"state": State(initial, pt.zeros_like(initial)), "empty": []},
        xs=[x, {"weight": pt.ones_like(x)}],
    )
    assert isinstance(final["state"], State)
    assert final["empty"] == []
    assert isinstance(traces, list)
    fn = pytensor.function(
        [initial, x],
        [final["state"].total, final["state"].count, traces[0], traces[1]["count"]],
    )
    total, count, totals, counts = fn(10.0, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(total, 16)
    np.testing.assert_allclose(count, 3)
    np.testing.assert_allclose(totals, [11, 13, 16])
    np.testing.assert_allclose(counts, [1, 2, 3])


@pytest.mark.parametrize("mode", ["FAST_COMPILE", "FAST_RUN", "NUMBA"])
@pytest.mark.parametrize("steps", [0, 1, 3])
@pytest.mark.parametrize("explicit_full_gradient", [False, True])
def test_carry_gradient(mode, steps, explicit_full_gradient):
    x = pt.vector("x")
    length = pt.iscalar("length")
    kwargs = {"truncate_gradient": -1} if explicit_full_gradient else {}
    final, ys = loop(
        lambda carry, _: (carry * 2, carry * 2), x, length=length, **kwargs
    )
    fn = pytensor.function(
        [x, length], [final, pt.grad(final.sum() + ys.sum(), x)], mode=mode
    )
    value, gradient = fn(np.array([1.0, 2.0]), steps)
    np.testing.assert_allclose(value, np.array([1.0, 2.0]) * 2**steps)
    np.testing.assert_allclose(
        gradient, 2**steps + sum(2**i for i in range(1, steps + 1))
    )


@pytest.mark.parametrize("truncate_gradient", [-1, 2])
def test_loop_gradients_match_scan(truncate_gradient):
    initial = pt.scalar("initial")
    xs = pt.vector("xs")
    scale = pt.scalar("scale")

    def step(carry, x):
        value = carry * scale + x**2
        return value, value**2

    final, ys = loop(step, initial, xs, truncate_gradient=truncate_gradient)
    reference, reference_ys = pytensor.scan(
        lambda x, carry, scale: step(carry, x),
        sequences=xs,
        outputs_info=[initial, None],
        non_sequences=[scale],
        truncate_gradient=truncate_gradient,
        return_updates=False,
    )
    inputs = [initial, xs, scale]
    actual_gradients = pt.grad(final + ys.sum(), inputs)
    reference_gradients = pt.grad(reference[-1] + reference_ys.sum(), inputs)
    fn = pytensor.function(inputs, [*actual_gradients, *reference_gradients])
    actual = fn(1.0, np.array([0.1, 0.2, 0.3]), 2.0)
    for gradient, reference_gradient in zip(actual[:3], actual[3:], strict=True):
        np.testing.assert_allclose(gradient, reference_gradient)


@pytest.mark.parametrize("mode", ["FAST_COMPILE", "FAST_RUN", "NUMBA"])
def test_stop_condition(mode):
    initial = pt.scalar("initial")
    final, ys = loop(
        lambda carry, _: (carry + 1, carry + 1, carry + 1 >= 3), initial, length=10
    )
    fn = pytensor.function([initial], [final, ys], mode=mode)
    final_value, trace = fn(0.0)
    np.testing.assert_allclose(final_value, 3)
    np.testing.assert_allclose(trace, [1, 2, 3])
    final_value, trace = fn(4.0)
    np.testing.assert_allclose(final_value, 5)
    np.testing.assert_allclose(trace, [5])


@pytest.mark.parametrize("stop", [pt.vector(dtype="bool"), pt.as_tensor(1)])
def test_invalid_stop_condition(stop):
    with pytest.raises(TypeError, match="scalar boolean"):
        loop(lambda carry, _: (carry, None, stop), pt.scalar(), length=3)


@pytest.mark.parametrize("by", [[], [0, 0], [1, 0], [-1, 0], [0.5]])
def test_invalid_shifts(by):
    with pytest.raises(ValueError, match="by"):
        shift(pt.vector(), by=by)


def test_invalid_shift_context():
    with pytest.raises(ValueError, match="Init shifts must be negative"):
        loop(lambda carry, _: (carry, None), shift(pt.ones(1), by=0), length=1)
    with pytest.raises(ValueError, match="Sequence shifts must be non-negative"):
        loop(lambda carry, xs: (carry, xs[0]), None, shift(pt.ones(1)), length=1)
    with pytest.raises(ValueError, match="at least 2"):
        shift([1.0], by=-2)


@pytest.mark.parametrize(
    "by, expected_gradient", [((-1,), [8, 0, 0]), ((-2, -1), [2, 3, 0])]
)
def test_shifted_carry_ignores_extra_history(by, expected_gradient):
    history = pt.vector("history")

    def step(carry, _):
        value = carry[0] * 2 if len(carry) == 1 else carry[0] + carry[1]
        return carry.push(value), None

    final, _ = loop(step, shift(history, by=by), length=3)
    fn = pytensor.function([history], [final, pt.grad(final, history)])
    value, gradient = fn([1.0, 2.0, 99.0])
    np.testing.assert_allclose(value, 8)
    np.testing.assert_allclose(gradient, expected_gradient)


@pytest.mark.parametrize(
    "update", [{"other": pt.scalar()}, State(pt.scalar(), pt.scalar())]
)
def test_carry_structure_mismatch(update):
    with pytest.raises(ValueError, match="does not match the init expression"):
        loop(lambda carry, _: (update, None), {"value": pt.scalar()}, length=3)


@pytest.mark.parametrize("length", [1.5, [3], True])
def test_invalid_length(length):
    with pytest.raises(TypeError, match=r"scalar integer|Cannot cast True"):
        loop(lambda carry, _: (carry, None), pt.scalar(), length=length)


def test_legacy_scan_deprecation():
    initial = pt.scalar("initial")
    with pytest.warns(DeprecationWarning, match="pytensor.scan is deprecated"):
        values = pytensor.scan(
            lambda carry: carry * 2,
            outputs_info=initial,
            n_steps=3,
            return_updates=False,
        )
    np.testing.assert_allclose(values.eval({initial: 1.0}), [2, 4, 8])


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


@pytest.mark.parametrize("mode", ["FAST_RUN", "NUMBA"])
def test_rng_carry(mode):
    rng_init = RandomGeneratorType()("rng")

    def f(carry, _):
        rng = carry
        next_rng, sample = pt.random.normal(rng=rng).owner.outputs
        return next_rng, sample

    final_rng, samples = loop(f, init=rng_init, length=5)

    fn = pytensor.function([rng_init], [samples, final_rng], mode=mode)

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


@pytest.mark.parametrize("mode", ["FAST_RUN", "NUMBA"])
def test_mixed_carry_and_xs(mode):
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
        [x0, rng_init, y0, seq], [x_final, final_rng, y_final, products], mode=mode
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
