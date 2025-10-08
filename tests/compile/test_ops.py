import pickle

import numpy as np
import pytest

from pytensor import function
from pytensor.compile.ops import as_op, wrap_py
from pytensor.tensor.type import dmatrix, dvector
from tests import unittest_tools as utt


@wrap_py([dmatrix, dmatrix], dmatrix)
def mul(a, b):
    """
    This is for test_pickle, since the function still has to be
    reachable from pickle (as in it cannot be defined inline)
    """
    return a * b


class TestOpDecorator(utt.InferShapeTester):
    def test_1arg(self):
        x = dmatrix("x")

        @wrap_py(dmatrix, dvector)
        def cumprod(x):
            return np.cumprod(x)

        fn = function([x], cumprod(x))
        r = fn([[1.5, 5], [2, 2]])
        r0 = np.array([1.5, 7.5, 15.0, 30.0])

        assert np.allclose(r, r0), (r, r0)

    def test_deprecation(self):
        x = dmatrix("x")

        with pytest.warns(FutureWarning):

            @as_op(dmatrix, dvector)
            def cumprod(x):
                return np.cumprod(x)

        fn = function([x], cumprod(x))
        r = fn([[1.5, 5], [2, 2]])
        r0 = np.array([1.5, 7.5, 15.0, 30.0])

        assert np.allclose(r, r0), (r, r0)

    def test_2arg(self):
        x = dmatrix("x")
        x.tag.test_value = np.zeros((2, 2))
        y = dvector("y")
        y.tag.test_value = [0, 0, 0, 0]

        @wrap_py([dmatrix, dvector], dvector)
        def cumprod_plus(x, y):
            return np.cumprod(x) + y

        fn = function([x, y], cumprod_plus(x, y))
        r = fn([[1.5, 5], [2, 2]], [1, 100, 2, 200])
        r0 = np.array([2.5, 107.5, 17.0, 230.0])

        assert np.allclose(r, r0), (r, r0)

    def test_infer_shape(self):
        x = dmatrix("x")
        x.tag.test_value = np.zeros((2, 2))
        y = dvector("y")
        y.tag.test_value = [0, 0, 0, 0]

        def infer_shape(fgraph, node, shapes):
            _x, y = shapes
            return [y]

        @wrap_py([dmatrix, dvector], dvector, infer_shape)
        def cumprod_plus(x, y):
            return np.cumprod(x) + y

        self._compile_and_check(
            [x, y],
            [cumprod_plus(x, y)],
            [[[1.5, 5], [2, 2]], [1, 100, 2, 200]],
            cumprod_plus.__class__,
            warn=False,
        )

    def test_pickle(self):
        x = dmatrix("x")
        y = dmatrix("y")

        m = mul(x, y)

        s = pickle.dumps(m)
        m2 = pickle.loads(s)

        assert m2.owner.op == m.owner.op
