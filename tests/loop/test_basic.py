import numpy as np

import pytensor
from pytensor import config, function, grad, shared
from pytensor.loop.basic import filter, map, reduce, scan
from pytensor.scan import until
from pytensor.tensor import arange, eq, scalar, vector, zeros
from pytensor.tensor.random import normal


def test_scan_with_sequences():
    xs = vector("xs")
    ys = vector("ys")
    zs = scan(
        fn=lambda x, y: x * y,
        sequences=[xs, ys],
    )
    pytensor.dprint(ys, print_type=True)
    np.testing.assert_almost_equal(
        zs.eval(
            {
                xs: np.arange(10, dtype=config.floatX),
                ys: np.arange(10, dtype=config.floatX),
            }
        ),
        np.arange(10) ** 2,
    )


def test_scan_with_carried_and_non_carried_states():
    x = scalar("x")
    [ys1, ys2] = scan(
        fn=lambda xtm1: (xtm1 + 1, (xtm1 + 1) * 2),
        init_states=[x, None],
        n_steps=10,
    )
    fn = function([x], [ys1, ys2])
    res = fn(-1)
    np.testing.assert_almost_equal(res[0], np.arange(10))
    np.testing.assert_almost_equal(res[1], np.arange(10) * 2)


def test_scan_with_sequence_and_carried_state():
    xs = vector("xs")
    ys = scan(
        fn=lambda x, ytm1: (ytm1 + 1) * x,
        init_states=[zeros(())],
        sequences=[xs],
    )
    fn = function([xs], ys)
    np.testing.assert_almost_equal(fn([1, 2, 3]), [1, 4, 15])


def test_scan_taking_grads_wrt_non_sequence():
    # Tests sequence + non-carried state
    xs = vector("xs")
    ys = xs**2

    J = scan(
        lambda i, ys, xs: grad(ys[i], wrt=xs),
        sequences=arange(ys.shape[0]),
        non_sequences=[ys, xs],
    )

    f = pytensor.function([xs], J)
    np.testing.assert_array_equal(f([4, 4]), np.c_[[8, 0], [0, 8]])


def test_scan_taking_grads_wrt_sequence():
    # This is not possible with the old Scan
    xs = vector("xs")
    ys = xs**2

    J = scan(
        lambda y, xs: grad(y, wrt=xs),
        sequences=[ys],
        non_sequences=[xs],
    )

    f = pytensor.function([xs], J)
    np.testing.assert_array_equal(f([4, 4]), np.c_[[8, 0], [0, 8]])


def test_while_scan():
    xs = scan(
        fn=lambda x: (x + 1, until((x + 1) >= 9)),
        init_states=[-1],
        n_steps=20,
    )

    f = pytensor.function([], xs)
    np.testing.assert_array_equal(f(), np.arange(10))


def test_scan_rvs():
    rng = shared(np.random.default_rng(123))
    test_rng = np.random.default_rng(123)

    def normal_fn(prev_rng):
        next_rng, x = normal(rng=prev_rng).owner.outputs
        return next_rng, x

    [rngs, xs] = scan(
        fn=normal_fn,
        init_states=[rng, None],
        n_steps=5,
    )
    fn = function([], xs, updates={rng: rngs[-1]})

    for i in range(3):
        res = fn()
        np.testing.assert_almost_equal(res, test_rng.normal(size=5))


def test_map():
    xs = vector("xs")
    ys = map(
        fn=lambda x: x * 100,
        sequences=xs,
    )
    np.testing.assert_almost_equal(
        ys.eval({xs: np.arange(10, dtype=config.floatX)}), np.arange(10) * 100
    )


def test_reduce():
    xs = vector("xs")
    y = reduce(
        fn=lambda x, acc: acc + x,
        init_states=zeros(()),
        sequences=xs,
    )
    np.testing.assert_almost_equal(
        y.eval({xs: np.arange(10, dtype=config.floatX)}), np.arange(10).cumsum()[-1]
    )


def test_filter():
    xs = vector("xs")
    ys = filter(
        fn=lambda x: eq(x % 2, 0),
        sequences=xs,
    )
    np.testing.assert_array_equal(
        ys.eval({xs: np.arange(0, 20, dtype=config.floatX)}), np.arange(0, 20, 2)
    )
