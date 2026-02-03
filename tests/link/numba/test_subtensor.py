import contextlib

import numpy as np
import pytest

import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor import Mode
from pytensor.basic import as_symbolic
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
from tests.link.numba.test_basic import compare_numba_and_py, numba_mode


rng = np.random.default_rng(sum(map(ord, "Numba subtensors")))


@pytest.mark.parametrize("step", [None, 1, 2, -2, "x"], ids=lambda x: f"step={x}")
@pytest.mark.parametrize("stop", [None, 10, "x"], ids=lambda x: f"stop={x}")
@pytest.mark.parametrize("start", [None, 0, 3, "x"], ids=lambda x: f"start={x}")
def test_slice(start, stop, step):
    x = ps.int64("x")

    sym_slice = as_symbolic(
        slice(
            x if start == "x" else start,
            x if stop == "x" else stop,
            x if step == "x" else step,
        )
    )

    no_opt_mode = Mode(linker="numba", optimizer=None)
    evaled_slice = sym_slice.eval({x: -5}, on_unused_input="ignore", mode=no_opt_mode)
    assert isinstance(evaled_slice, slice)
    if start == "x":
        assert evaled_slice.start == -5
    elif start is None and (evaled_slice.step is None or evaled_slice.step > 0):
        # Numba can convert to 0 (and sometimes does) in this case
        assert evaled_slice.start in (None, 0)
    else:
        assert evaled_slice.start == start

    if stop == "x":
        assert evaled_slice.stop == -5
    else:
        assert evaled_slice.stop == stop

    if step == "x":
        assert evaled_slice.step == -5
    elif step is None:
        # Numba can convert to 1 (and sometimes does) in this case
        assert evaled_slice.step in (None, 1)
    else:
        assert evaled_slice.step == step


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
    compare_numba_and_py([], [out_pt], [])


@pytest.mark.parametrize(
    "x, indices",
    [
        (pt.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), ([1, 2],)),
        (pt.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), ([1],)),
    ],
)
def test_AdvancedSubtensor1(x, indices):
    """Test NumPy's advanced indexing in one dimension."""
    out_pt = advanced_subtensor1(x, *indices)
    assert isinstance(out_pt.owner.op, AdvancedSubtensor1)
    compare_numba_and_py([], [out_pt], [])


def test_AdvancedSubtensor1_out_of_bounds():
    out_pt = advanced_subtensor1(np.arange(3), [4])
    assert isinstance(out_pt.owner.op, AdvancedSubtensor1)
    with pytest.raises(IndexError):
        compare_numba_and_py([], [out_pt], [])


@pytest.mark.parametrize(
    "x, indices",
    [
        # Single vector indexing
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            (0, [1, 2, 2, 3]),
        ),
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            (np.array([True, False, False])),
        ),
        # Single multidimensional indexing
        (
            as_tensor(np.arange(3 * 3).reshape((3, 3))),
            (np.eye(3).astype(int)),
        ),
        (
            as_tensor(np.arange(3 * 3).reshape((3, 3))),
            (np.eye(3).astype(bool)),
        ),
        (
            as_tensor(np.arange(3 * 3 * 2).reshape((3, 3, 2))),
            (np.eye(3).astype(int)),
        ),
        (
            as_tensor(np.arange(3 * 3 * 2).reshape((3, 3, 2))),
            (np.eye(3).astype(bool)),
        ),
        (
            as_tensor(np.arange(2 * 3 * 3).reshape((2, 3, 3))),
            (slice(2, None), np.eye(3).astype(int)),
        ),
        (
            as_tensor(np.arange(2 * 3 * 3).reshape((2, 3, 3))),
            (slice(2, None), np.eye(3).astype(bool)),
        ),
        # Multiple vector indexing
        (
            pt.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            ([1, 2], [2, 3]),
        ),
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            (slice(None), [1, 2], [3, 4]),
        ),
        (
            as_tensor(np.arange(3 * 5 * 7).reshape((3, 5, 7))),
            ([1, 2], [3, 4], [5, 6]),
        ),
        # Non-consecutive vector indexing
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            ([1, 2], slice(None), [3, 4]),
        ),
        # Multiple multidimensional integer indexing
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            ([[1, 2], [2, 1]], [[0, 0], [0, 0]]),
        ),
        (
            as_tensor(np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))),
            (slice(None), [[1, 2], [2, 1]], slice(None), [[0, 0], [0, 0]]),
        ),
        # Multiple multidimensional indexing with broadcasting
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            ([[1, 2], [2, 1]], [0, 0]),
        ),
        # multiple multidimensional integer indexing mixed with basic indexing
        (
            as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            ([[1, 2], [2, 1]], slice(1, None), [[0, 0], [0, 0]]),
        ),
    ],
)
@pytest.mark.filterwarnings("error")  # Raise if we did not expect objmode to be needed
def test_AdvancedSubtensor(x, indices):
    """Test NumPy's advanced indexing in more than one dimension."""
    x_pt = x.type()
    out_pt = x_pt[indices]
    assert isinstance(out_pt.owner.op, AdvancedSubtensor)
    compare_numba_and_py(
        [x_pt],
        [out_pt],
        [x.data],
        # Specialize allows running boolean indexing without falling back to object mode
        # Thanks to bool_idx_to_nonzero rewrite
        numba_mode=numba_mode.including("specialize"),
    )


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
    compare_numba_and_py([], [out_pt], [])

    out_pt = inc_subtensor(x[indices], y)
    assert isinstance(out_pt.owner.op, IncSubtensor)
    compare_numba_and_py([], [out_pt], [])

    x_pt = x.type()
    out_pt = set_subtensor(x_pt[indices], y, inplace=True)
    assert isinstance(out_pt.owner.op, IncSubtensor)
    compare_numba_and_py([x_pt], [out_pt], [x.data], inplace=True)


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
    compare_numba_and_py([], [out_pt], [])

    out_pt = advanced_inc_subtensor1(x, y, *indices)
    assert isinstance(out_pt.owner.op, AdvancedIncSubtensor1)
    compare_numba_and_py([], [out_pt], [])

    # With symbolic inputs
    x_pt = x.type()
    y_pt = y.type()

    out_pt = AdvancedIncSubtensor1(inplace=True)(x_pt, y_pt, *indices)
    assert isinstance(out_pt.owner.op, AdvancedIncSubtensor1)
    compare_numba_and_py([x_pt, y_pt], [out_pt], [x.data, y.data], inplace=True)

    out_pt = AdvancedIncSubtensor1(set_instead_of_inc=True, inplace=True)(
        x_pt, y_pt, *indices
    )
    assert isinstance(out_pt.owner.op, AdvancedIncSubtensor1)
    compare_numba_and_py([x_pt, y_pt], [out_pt], [x.data, y.data], inplace=True)


@pytest.mark.parametrize(
    "x, y, indices, duplicate_indices, duplicate_indices_require_obj_mode",
    [
        (
            np.arange(3 * 4 * 5).reshape((3, 4, 5)),
            -np.arange(3 * 5).reshape(3, 5),
            (slice(None, None, 2), [1, 2, 3]),  # Mixed basic and vector index
            False,
            False,
        ),
        (
            np.arange(3 * 4 * 5).reshape((3, 4, 5)),
            np.array(-99),  # Broadcasted value
            (
                slice(None, None, 2),
                [1, 2, 3],
                -1,
            ),  # Mixed basic and broadcasted vector idx
            False,
            False,
        ),
        (
            np.arange(3 * 4 * 5).reshape((3, 4, 5)),
            np.array(-99),  # Broadcasted value
            (slice(None, None, 2), [1, 2, 3]),  # Mixed basic and vector idx
            False,
            False,
        ),
        (
            np.arange(3 * 4 * 5).reshape((3, 4, 5)),
            -np.arange(4 * 5).reshape(4, 5),
            (0, [1, 2, 2, 3]),  # Broadcasted vector index with repeated values
            True,
            False,
        ),
        (
            np.arange(3 * 4 * 5).reshape((3, 4, 5)),
            np.array([-99]),  # Broadcasted value
            (0, [1, 2, 2, 3]),  # Broadcasted vector index with repeated values
            True,
            False,
        ),
        (
            np.arange(3 * 4 * 5).reshape((3, 4, 5)),
            -np.arange(1 * 4 * 5).reshape(1, 4, 5),
            (np.array([True, False, False])),  # Broadcasted boolean index
            False,
            False,
        ),
        (
            np.arange(3 * 3).reshape((3, 3)),
            -np.arange(3),
            (np.eye(3).astype(bool)),  # Boolean index
            False,
            False,
        ),
        (
            np.arange(3 * 3 * 5).reshape((3, 3, 5)),
            rng.poisson(size=(3, 2)),
            (
                np.eye(3).astype(bool),
                slice(-2, None),
            ),  # Boolean index, mixed with basic index
            False,
            False,
        ),
        (
            np.arange(3 * 4 * 5).reshape((3, 4, 5)),
            rng.poisson(size=(2, 5)),
            ([1, 2], [2, 3]),  # 2 vector indices
            False,
            False,
        ),
        (
            np.arange(3 * 4 * 5).reshape((3, 4, 5)),
            rng.poisson(size=(3, 2)),
            (slice(None), [1, 2], [2, 3]),  # 2 vector indices
            False,
            False,
        ),
        (
            np.arange(3 * 4 * 6).reshape((3, 4, 6)),
            rng.poisson(size=(2,)),
            ([1, 2], [2, 3], [4, 5]),  # 3 vector indices
            False,
            False,
        ),
        (
            np.arange(3 * 4 * 5).reshape((3, 4, 5)),
            np.array(-99),  # Broadcasted value
            ([1, 2], [2, 3]),  # 2 vector indices
            False,
            False,
        ),
        (
            np.arange(3 * 4 * 5).reshape((3, 4, 5)),
            rng.poisson(size=(2, 4)),
            ([1, 2], slice(None), [3, 4]),  # Non-consecutive vector indices
            False,
            False,
        ),
        (
            np.arange(3 * 4 * 5).reshape((3, 4, 5)),
            rng.poisson(size=(2, 2)),
            (
                slice(1, None),
                [1, 2],
                [3, 4],
            ),  # Mixed double vector index and basic index
            False,
            False,
        ),
        (
            np.arange(5),
            rng.poisson(size=(2, 2)),
            ([[1, 2], [2, 3]]),  # matrix index
            False,
            False,
        ),
        (
            np.arange(3 * 5).reshape((3, 5)),
            rng.poisson(size=(2, 2, 2)),
            (slice(1, 3), [[1, 2], [2, 3]]),  # matrix index, mixed with basic index
            False,
            False,
        ),
        (
            np.arange(3 * 5).reshape((3, 5)),
            rng.poisson(size=(1, 2, 2)),  # Same as before, but Y broadcasts
            (slice(1, 3), [[1, 2], [2, 3]]),
            False,
            False,
        ),
        (
            np.arange(3 * 4 * 5).reshape((3, 4, 5)),
            rng.poisson(size=(2, 5)),
            ([1, 1], [2, 2]),  # Repeated indices
            True,
            True,
        ),
        (
            np.arange(3 * 4 * 5).reshape((3, 4, 5)),
            rng.poisson(size=(3, 2, 2)),
            (slice(None), [[1, 2], [2, 1]], [[2, 3], [0, 0]]),  # 2 matrix indices
            False,
            False,
        ),
    ],
)
@pytest.mark.parametrize("inplace", (False, True))
@pytest.mark.filterwarnings("error")  # Raise if we did not expect objmode to be needed
def test_AdvancedIncSubtensor(
    x,
    y,
    indices,
    duplicate_indices,
    duplicate_indices_require_obj_mode,
    inplace,
):
    # Need rewrite to support certain forms of advanced indexing without object mode
    mode = numba_mode.including("specialize")

    x_pt = pt.as_tensor(x).type("x")
    y_pt = pt.as_tensor(y).type("y")

    out_pt = set_subtensor(x_pt[indices], y_pt, inplace=inplace)
    assert isinstance(out_pt.owner.op, AdvancedIncSubtensor)

    fn, _ = compare_numba_and_py(
        [x_pt, y_pt], out_pt, [x, y], numba_mode=mode, inplace=inplace
    )

    if inplace:
        # Test updates inplace
        x_orig = x.copy()
        fn(x, y + 1)
        assert not np.all(x == x_orig)

    out_pt = inc_subtensor(x_pt[indices], y_pt, inplace=inplace)
    assert isinstance(out_pt.owner.op, AdvancedIncSubtensor)

    fn, _ = compare_numba_and_py(
        [x_pt, y_pt], out_pt, [x, y], numba_mode=mode, inplace=inplace
    )
    if inplace:
        # Test updates inplace
        x_orig = x.copy()
        fn(x, y)
        assert not np.all(x == x_orig)

    if duplicate_indices:
        # If inc_subtensor is called with `ignore_duplicates=True`, and it's not one of the cases supported by Numba
        # We have to fall back to obj_mode
        out_pt = inc_subtensor(
            x_pt[indices], y_pt, inplace=inplace, ignore_duplicates=True
        )
        assert isinstance(out_pt.owner.op, AdvancedIncSubtensor)

        with (
            pytest.warns(
                UserWarning,
                match="Numba will use object mode to run AdvancedIncSubtensor's perform method",
            )
            if duplicate_indices_require_obj_mode
            else contextlib.nullcontext()
        ):
            fn, _ = compare_numba_and_py(
                [x_pt, y_pt], out_pt, [x, y], numba_mode=mode, inplace=inplace
            )
        if inplace:
            # Test updates inplace
            x_orig = x.copy()
            fn(x, y)
            assert not np.all(x == x_orig)


def test_advanced_indexing_with_newaxis_fallback_obj_mode():
    # This should be automatically solved with https://github.com/pymc-devs/pytensor/issues/1564
    # After which we can add these parametrizations to the relevant tests above
    x = pt.matrix("x")
    out = x[None, [0, 1, 2], [0, 1, 2]]
    with pytest.warns(
        UserWarning,
        match=r"Numba will use object mode to run AdvancedSubtensor's perform method",
    ):
        compare_numba_and_py([x], [out], [np.random.normal(size=(4, 4))])

    out = x[None, [0, 1, 2], [0, 1, 2]].inc(5)
    with pytest.warns(
        UserWarning,
        match=r"Numba will use object mode to run AdvancedIncSubtensor's perform method",
    ):
        compare_numba_and_py([x], [out], [np.random.normal(size=(4, 4))])
