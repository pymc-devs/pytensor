import numpy as np

from pytensor import function
from pytensor.xtensor.basic import Rename
from pytensor.xtensor.type import xtensor


def test_shape_feature_does_not_see_xop():
    CALLED = False

    x = xtensor("x", dims=("a",), dtype="int64")

    class XOpWithBadInferShape(Rename):
        def infer_shape(self, node, inputs, outputs):
            global CALLED
            CALLED = True
            raise NotImplementedError()

    test_xop = XOpWithBadInferShape(new_dims=("b",))

    out = test_xop(x) - test_xop(x)
    assert out.dims == ("b",)

    fn = function([x], out)
    np.testing.assert_allclose(fn([1, 2, 3]), [0, 0, 0])
    assert not CALLED
