"""This file don't test everything. It only test one past crash error."""

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

    x = vector("x")
    y = argmax(x)
    kwargs = {}
    # We can't pickle DebugMode
    if pytensor.config.mode in ["DebugMode", "DEBUG_MODE"]:
        kwargs = {"mode": "FAST_RUN"}
    f = pytensor.function([x], [y], **kwargs)
    pickle.loads(pickle.dumps(f))


def test_as_symbolic():
    # Remove this when xtensor is not using symbolic slices
    from pytensor.tensor.type import iscalar
    from pytensor.tensor.type_other import SliceConstant, slicetype

    res = as_symbolic(None)
    assert res is NoneConst

    res = as_symbolic(slice(1, 2))
    assert isinstance(res, SliceConstant)
    assert res.type == slicetype
    assert res.data == slice(1, 2)

    i = iscalar()
    res = as_symbolic(slice(i))
    assert res.owner is not None
