import numpy as np

import pytensor
from pytensor.loop.basic import map, reduce, scan
from pytensor.tensor import vector, zeros


def test_scan():
    xs = vector("xs")
    ys = vector("ys")
    _, (zs,) = scan(
        fn=lambda x, y: x * y,
        sequences=[xs, ys],
    )
    pytensor.dprint(ys, print_type=True)
    np.testing.assert_almost_equal(
        zs.eval({xs: np.arange(10), ys: np.arange(10)}),
        np.arange(10) ** 2,
    )


def test_map():
    xs = vector("xs")
    ys = map(
        fn=lambda x: x * 100,
        sequences=xs,
    )
    np.testing.assert_almost_equal(ys.eval({xs: np.arange(10)}), np.arange(10) * 100)


def test_reduce():
    xs = vector("xs")
    y = reduce(
        fn=lambda acc, x: acc + x,
        init_states=zeros(()),
        sequences=xs,
    )
    np.testing.assert_almost_equal(
        y.eval({xs: np.arange(10)}), np.arange(10).cumsum()[-1]
    )
