import contextlib

import numpy as np
import pytest

import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor import Mode
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
    """Test slicing with scalar variables in Numba."""
    x_scalar = ps.int64("x")
    data = pt.arange(20)

    tslice = slice(
        x_scalar if start == "x" else start,
        x_scalar if stop == "x" else stop,
        x_scalar if step == "x" else step,
    )

    # Apply slice to tensor
    out_pt = data[tslice]
    assert isinstance(out_pt.owner.op, Subtensor)

    # Compare numba and Python execution
    no_opt_mode = Mode(linker="numba", optimizer=None)
    result = out_pt.eval({x_scalar: -5}, on_unused_input="ignore", mode=no_opt_mode)

    # Compute expected result
    expected_slice = slice(
        -5 if start == "x" else start,
        -5 if stop == "x" else stop,
        -5 if step == "x" else step,
    )
    expected = np.arange(20)[expected_slice]

    assert np.array_equal(result, expected)


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
        # Thanks to ravel_multidimensional_bool_idx rewrite
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


def test_advanced_indexing_with_newaxis():
    x = pt.matrix("x")
    out = x[None, [0, 1, 2], [0, 1, 2]]
    compare_numba_and_py([x], [out], [np.random.normal(size=(4, 4))])

    out = x[None, [0, 1, 2], [0, 1, 2]].inc(5)
    compare_numba_and_py([x], [out], [np.random.normal(size=(4, 4))])


def test_advanced_boolean_indexing_multi_dim():
    """Test boolean indexing where the mask consumes multiple dimensions.

    A 2D boolean mask indexing a 3D tensor will consume the first 2 dimensions,
    resulting in a flattened selection along those dims.
    """
    # 2D mask that consumes 2 dimensions of a 3D tensor
    mask = np.array(
        [[True, False, True], [False, False, True]]
    )  # shape (2, 3) -> 3 True values
    val_data = np.arange(24).reshape((2, 3, 4)).astype("float64")

    val = pt.tensor("val", shape=(2, 3, 4), dtype="float64")

    # Basic boolean indexing with 2D mask - mask consumes dims 0,1
    out = val[mask]
    compare_numba_and_py([val], [out], [val_data])

    # Boolean indexing with 2D mask combined with newaxis and ellipsis
    # val[mask, None, ..., None] should produce shape (3, 1, 4, 1)
    out_with_newaxis = val[mask, None, ..., None]
    compare_numba_and_py([val], [out_with_newaxis], [val_data])

    # Boolean indexing with set_subtensor
    y = pt.tensor("y", shape=(3, 4), dtype="float64")
    y_data = np.ones((3, 4)) * 99
    out_set = pt.set_subtensor(val[mask], y)
    compare_numba_and_py([val, y], [out_set], [val_data.copy(), y_data])
