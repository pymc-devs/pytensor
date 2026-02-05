"""This file don't test everything. It only test one past crash error."""

import pytest

import pytensor
from pytensor import as_symbolic
from pytensor.graph.basic import Constant
from pytensor.tensor.math import argmax
from pytensor.tensor.type import vector
from pytensor.tensor.type_other import NoneConst, NoneTypeT


def test_none_Constant():
    # FIXME: This is a poor test.

    # Tests equals
    # We had an error in the past with unpickling

    o1 = Constant(NoneTypeT(), None, name="NoneConst")
    o2 = Constant(NoneTypeT(), None, name="NoneConst")
    assert o1.equals(o2)
    assert NoneConst.equals(o1)
    assert o1.equals(NoneConst)
    assert NoneConst.equals(o2)
    assert o2.equals(NoneConst)

    # This trigger equals that returned the wrong answer in the past.
    import pickle

    import pytensor

    x = vector("x")
    y = argmax(x)
    kwargs = {}
    # We can't pickle DebugMode
    if pytensor.config.mode in ["DebugMode", "DEBUG_MODE"]:
        kwargs = {"mode": "FAST_RUN"}
    f = pytensor.function([x], [y], **kwargs)
    pickle.loads(pickle.dumps(f))


def test_slice_handling():
    from pytensor.tensor.type import iscalar

    i = iscalar()
    x = vector("x")

    result = x[0:i]
    f = pytensor.function([x, i], result)

    import numpy as np

    test_val = np.arange(10)
    assert np.array_equal(f(test_val, 5), test_val[0:5])


def test_as_symbolic():
    res = as_symbolic(None)
    assert res is NoneConst

    with pytest.raises(NotImplementedError):
        as_symbolic(slice(1, 2))

    from pytensor.tensor.type import iscalar

    with pytest.raises(NotImplementedError):
        as_symbolic(slice(iscalar()))
