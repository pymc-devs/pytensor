import pytensor.tensor as pt
from pytensor.assumptions import (
    LOWER_TRIANGULAR,
    SYMMETRIC,
    UPPER_TRIANGULAR,
    FactState,
)
from pytensor.assumptions.specify import assume
from pytensor.tensor.reshape import join_dims, split_dims
from tests.assumptions.conftest import make_fgraph


def test_join_dims_clear_of_core_forwards_property():
    x = pt.tensor("x", shape=(2, 3, 4, 4))
    x_lower = assume(x, lower_triangular=True)
    y = join_dims(x_lower, start_axis=0, n_axes=2)
    _, af = make_fgraph(y)
    assert af.check(y, LOWER_TRIANGULAR)


def test_join_dims_reaching_core_axis_is_unknown():
    x = pt.tensor("x", shape=(2, 3, 4, 4))
    x_sym = assume(x, symmetric=True)
    y = join_dims(x_sym, start_axis=1, n_axes=2)
    _, af = make_fgraph(y)
    assert af.get(y, SYMMETRIC) == FactState.UNKNOWN


def test_split_dims_batch_axis_forwards_property():
    x = pt.tensor("x", shape=(6, 4, 4))
    x_upper = assume(x, upper_triangular=True)
    y = split_dims(x_upper, shape=(2, 3), axis=0)
    _, af = make_fgraph(y)
    assert af.check(y, UPPER_TRIANGULAR)


def test_split_dims_core_axis_is_unknown():
    x = pt.tensor("x", shape=(5, 6, 4))
    x_sym = assume(x, symmetric=True)
    y = split_dims(x_sym, shape=(2, 3), axis=1)
    _, af = make_fgraph(y)
    assert af.get(y, SYMMETRIC) == FactState.UNKNOWN
