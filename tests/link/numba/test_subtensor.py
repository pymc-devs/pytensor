import numpy as np
import pytest

from pytensor.graph import FunctionGraph
from pytensor.tensor import as_tensor
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
    advanced_inc_subtensor1,
    advanced_set_subtensor1,
    advanced_subtensor1,
    inc_subtensor,
    set_subtensor,
)
from tests.link.numba.test_basic import compare_numba_and_py


rng = np.random.default_rng(sum(map(ord, "Numba subtensors")))


@pytest.mark.parametrize(
    "x, indices",
    [
        (as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), (1,)),
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            (slice(None)),
        ),
        (as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), (1, 2, 0)),
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            (slice(1, 2), 1, slice(None)),
        ),
    ],
)
def test_Subtensor(x, indices):
    """Test NumPy's basic indexing."""
    out_pt = x[indices]
    assert isinstance(out_pt.owner.op, Subtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_numba_and_py(out_fg, [])


@pytest.mark.parametrize(
    "x, indices",
    [
        (as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), ([1, 2],)),
        (as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), ([1],)),
    ],
)
def test_AdvancedSubtensor1(x, indices):
    """Test NumPy's advanced indexing in one dimension."""
    out_pt = advanced_subtensor1(x, *indices)
    assert isinstance(out_pt.owner.op, AdvancedSubtensor1)
    out_fg = FunctionGraph([], [out_pt])
    compare_numba_and_py(out_fg, [])


def test_AdvancedSubtensor1_out_of_bounds():
    out_pt = advanced_subtensor1(np.arange(3), [4])
    assert isinstance(out_pt.owner.op, AdvancedSubtensor1)
    out_fg = FunctionGraph([], [out_pt])
    with pytest.raises(IndexError):
        compare_numba_and_py(out_fg, [])


@pytest.mark.parametrize(
    "x, indices",
    [
        (as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), ([1, 2], [2, 3])),
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            ([1, 2], slice(None), [3, 4]),
        ),
    ],
)
def test_AdvancedSubtensor(x, indices):
    """Test NumPy's advanced indexing in more than one dimension."""
    out_pt = x[indices]
    assert isinstance(out_pt.owner.op, AdvancedSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_numba_and_py(out_fg, [])


@pytest.mark.parametrize(
    "x, y, indices",
    [
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            as_tensor(np.array(10)),
            (1,),
        ),
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            as_tensor(rng.poisson(size=(4, 5))),
            (slice(None)),
        ),
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            as_tensor(np.array(10)),
            (1, 2, 0),
        ),
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            as_tensor(rng.poisson(size=(1, 5))),
            (slice(1, 2), 1, slice(None)),
        ),
    ],
)
def test_IncSubtensor(x, y, indices):
    out_pt = set_subtensor(x[indices], y)
    assert isinstance(out_pt.owner.op, IncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_numba_and_py(out_fg, [])

    out_pt = inc_subtensor(x[indices], y)
    assert isinstance(out_pt.owner.op, IncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_numba_and_py(out_fg, [])

    x_pt = x.type()
    out_pt = set_subtensor(x_pt[indices], y, inplace=True)
    assert isinstance(out_pt.owner.op, IncSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_numba_and_py(out_fg, [x.data])


@pytest.mark.parametrize(
    "x, y, indices",
    [
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            as_tensor(rng.poisson(size=(2, 4, 5))),
            ([1, 2],),
        ),
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            as_tensor(rng.poisson(size=(2, 4, 5))),
            ([1, 1],),
        ),
        # Broadcasting values
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            as_tensor(rng.poisson(size=(1, 4, 5))),
            ([0, 2, 0],),
        ),
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            as_tensor(rng.poisson(size=(5,))),
            ([0, 2],),
        ),
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            as_tensor(rng.poisson(size=())),
            ([2, 0],),
        ),
        (
            as_tensor(np.arange(5)),
            as_tensor(rng.poisson(size=())),
            ([2, 0],),
        ),
    ],
)
def test_AdvancedIncSubtensor1(x, y, indices):
    out_pt = advanced_set_subtensor1(x, y, *indices)
    assert isinstance(out_pt.owner.op, AdvancedIncSubtensor1)
    out_fg = FunctionGraph([], [out_pt])
    compare_numba_and_py(out_fg, [])

    out_pt = advanced_inc_subtensor1(x, y, *indices)
    assert isinstance(out_pt.owner.op, AdvancedIncSubtensor1)
    out_fg = FunctionGraph([], [out_pt])
    compare_numba_and_py(out_fg, [])

    # With symbolic inputs
    x_pt = x.type()
    y_pt = y.type()

    out_pt = AdvancedIncSubtensor1(inplace=True)(x_pt, y_pt, *indices)
    assert isinstance(out_pt.owner.op, AdvancedIncSubtensor1)
    out_fg = FunctionGraph([x_pt, y_pt], [out_pt])
    compare_numba_and_py(out_fg, [x.data, y.data])

    out_pt = AdvancedIncSubtensor1(set_instead_of_inc=True, inplace=True)(
        x_pt, y_pt, *indices
    )
    assert isinstance(out_pt.owner.op, AdvancedIncSubtensor1)
    out_fg = FunctionGraph([x_pt, y_pt], [out_pt])
    compare_numba_and_py(out_fg, [x.data, y.data])


@pytest.mark.parametrize(
    "x, y, indices",
    [
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            as_tensor(rng.poisson(size=(2, 5))),
            ([1, 2], [2, 3]),
        ),
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            as_tensor(rng.poisson(size=(2, 4))),
            ([1, 2], slice(None), [3, 4]),
        ),
        pytest.param(
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            as_tensor(rng.poisson(size=(2, 5))),
            ([1, 1], [2, 2]),
        ),
    ],
)
def test_AdvancedIncSubtensor(x, y, indices):
    out_pt = set_subtensor(x[indices], y)
    assert isinstance(out_pt.owner.op, AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_numba_and_py(out_fg, [])

    out_pt = inc_subtensor(x[indices], y)
    assert isinstance(out_pt.owner.op, AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_pt])
    compare_numba_and_py(out_fg, [])

    x_pt = x.type()
    out_pt = set_subtensor(x_pt[indices], y)
    # Inplace isn't really implemented for `AdvancedIncSubtensor`, so we just
    # hack it on here
    out_pt.owner.op.inplace = True
    assert isinstance(out_pt.owner.op, AdvancedIncSubtensor)
    out_fg = FunctionGraph([x_pt], [out_pt])
    compare_numba_and_py(out_fg, [x.data])
