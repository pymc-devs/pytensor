import pytensor.tensor as pt
from pytensor.assumptions import DIAGONAL, SYMMETRIC, FactState
from pytensor.assumptions.specify import assume
from tests.assumptions.conftest import make_fgraph


def test_specify_shape_forwards_property():
    x = pt.matrix("x", shape=(4, 4))
    x_diag = assume(x, diagonal=True)
    y = pt.specify_shape(x_diag, (4, 4))
    _, af = make_fgraph(y)
    assert af.check(y, DIAGONAL)


def test_reshape_rechunk_batch_axes_forwards_property():
    x = pt.tensor("x", shape=(6, 4, 4))
    x_sym = assume(x, symmetric=True)
    y = pt.reshape(x_sym, (2, 3, 4, 4))
    _, af = make_fgraph(y)
    assert af.check(y, SYMMETRIC)


def test_reshape_changing_trailing_dims_is_unknown():
    x = pt.matrix("x", shape=(4, 4))
    x_sym = assume(x, symmetric=True)
    y = pt.reshape(x_sym, (2, 8))
    _, af = make_fgraph(y)
    assert af.get(y, SYMMETRIC) == FactState.UNKNOWN


def test_reshape_unknown_trailing_dim_is_unknown():
    """A statically-unknown trailing dim cannot be confirmed unchanged."""
    x = pt.tensor("x", shape=(6, 4, None))
    x_sym = assume(x, symmetric=True)
    y = pt.reshape(x_sym, (2, 3, 4, x.shape[2]))
    _, af = make_fgraph(y)
    assert af.get(y, SYMMETRIC) == FactState.UNKNOWN
