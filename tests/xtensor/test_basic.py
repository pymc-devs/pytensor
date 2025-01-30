import numpy as np

from pytensor import function
from pytensor.xtensor.basic import add, exp
from pytensor.xtensor.type import xtensor


def test_add():
    x = xtensor("x", dims=("city",), shape=(None,))
    y = xtensor("y", dims=("country",), shape=(4,))
    z = add(exp(x), exp(y))
    assert z.type.dims == ("city", "country")
    assert z.type.shape == (None, 4)

    fn = function([x, y], z)
    # fn.dprint(print_type=True)

    np.testing.assert_allclose(
        fn(x=np.zeros(3), y=np.zeros(4)),
        np.full((3, 4), 2.0),
    )
