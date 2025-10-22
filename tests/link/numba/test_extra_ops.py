import contextlib

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from pytensor.raise_op import assert_op
from pytensor.tensor import extra_ops
from tests.link.numba.test_basic import compare_numba_and_py


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "val",
    [
        (pt.lscalar(), np.array(6, dtype="int64")),
    ],
)
def test_Bartlett(val):
    val, test_val = val
    g = extra_ops.bartlett(val)

    compare_numba_and_py(
        [val],
        g,
        [test_val],
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, atol=1e-15),
    )


@pytest.mark.parametrize(
    "val, axis, mode",
    [
        (
            (pt.matrix(), np.arange(3, dtype=config.floatX).reshape((3, 1))),
            1,
            "add",
        ),
        (
            (pt.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))),
            0,
            "add",
        ),
        (
            (pt.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))),
            1,
            "add",
        ),
        (
            (pt.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))),
            0,
            "mul",
        ),
        (
            (pt.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))),
            1,
            "mul",
        ),
        # Regression tests for https://github.com/pymc-devs/pytensor/issues/1689
        (
            (pt.vector(), np.arange(6, dtype=config.floatX)),
            0,
            "add",
        ),
        (
            (pt.vector(), np.arange(6, dtype=config.floatX)),
            0,
            "mul",
        ),
    ],
)
def test_CumOp(val, axis, mode):
    val, test_val = val
    g = extra_ops.CumOp(axis=axis, mode=mode)(val)

    compare_numba_and_py(
        [val],
        g,
        [test_val],
    )


def test_FillDiagonal():
    a = pt.lmatrix("a")
    test_a = np.zeros((10, 2), dtype="int64")

    val = pt.lscalar("val")
    test_val = np.array(1, dtype="int64")

    g = extra_ops.FillDiagonal()(a, val)

    compare_numba_and_py(
        [a, val],
        g,
        [test_a, test_val],
    )


@pytest.mark.parametrize(
    "a, val, offset",
    [
        (
            (pt.lmatrix(), np.zeros((10, 2), dtype="int64")),
            (pt.lscalar(), np.array(1, dtype="int64")),
            (pt.lscalar(), np.array(-1, dtype="int64")),
        ),
        (
            (pt.lmatrix(), np.zeros((10, 2), dtype="int64")),
            (pt.lscalar(), np.array(1, dtype="int64")),
            (pt.lscalar(), np.array(0, dtype="int64")),
        ),
        (
            (pt.lmatrix(), np.zeros((10, 3), dtype="int64")),
            (pt.lscalar(), np.array(1, dtype="int64")),
            (pt.lscalar(), np.array(1, dtype="int64")),
        ),
    ],
)
def test_FillDiagonalOffset(a, val, offset):
    a, test_a = a
    val, test_val = val
    offset, test_offset = offset
    g = extra_ops.FillDiagonalOffset()(a, val, offset)

    compare_numba_and_py(
        [a, val, offset],
        g,
        [test_a, test_val, test_offset],
    )


@pytest.mark.parametrize(
    "arr, shape, mode, order, exc",
    [
        (
            tuple((pt.lscalar(), v) for v in np.array([0])),
            (pt.lvector(), np.array([2])),
            "raise",
            "C",
            None,
        ),
        (
            tuple((pt.lscalar(), v) for v in np.array([0, 0, 3])),
            (pt.lvector(), np.array([2, 3, 4])),
            "raise",
            "C",
            None,
        ),
        (
            tuple((pt.lvector(), v) for v in np.array([[0, 1], [2, 0], [1, 3]])),
            (pt.lvector(), np.array([2, 3, 4])),
            "raise",
            "C",
            None,
        ),
        (
            tuple((pt.lvector(), v) for v in np.array([[0, 1], [2, 0], [1, 3]])),
            (pt.lvector(), np.array([2, 3, 4])),
            "raise",
            "F",
            NotImplementedError,
        ),
        (
            tuple(
                (pt.lvector(), v) for v in np.array([[0, 1, 2], [2, 0, 3], [1, 3, 5]])
            ),
            (pt.lvector(), np.array([2, 3, 4])),
            "raise",
            "C",
            ValueError,
        ),
        (
            tuple(
                (pt.lvector(), v) for v in np.array([[0, 1, 2], [2, 0, 3], [1, 3, 5]])
            ),
            (pt.lvector(), np.array([2, 3, 4])),
            "wrap",
            "C",
            None,
        ),
        (
            tuple(
                (pt.lvector(), v) for v in np.array([[0, 1, 2], [2, 0, 3], [1, 3, 5]])
            ),
            (pt.lvector(), np.array([2, 3, 4])),
            "clip",
            "C",
            None,
        ),
    ],
)
def test_RavelMultiIndex(arr, shape, mode, order, exc):
    arr, test_arr = zip(*arr, strict=True)
    shape, test_shape = shape
    g = extra_ops.RavelMultiIndex(mode, order)(*arr, shape)

    cm = contextlib.suppress() if exc is None else pytest.raises(exc)
    with cm:
        compare_numba_and_py(
            [*arr, shape],
            g,
            [*test_arr, test_shape],
        )


@pytest.mark.parametrize(
    "x, repeats, axis, exc",
    [
        (
            (pt.lvector(), np.arange(2, dtype="int64")),
            (pt.lvector(), np.array([1, 3], dtype="int64")),
            0,
            None,
        ),
        (
            (pt.lmatrix(), np.zeros((2, 2), dtype="int64")),
            (pt.lvector(), np.array([1, 3], dtype="int64")),
            0,
            UserWarning,
        ),
    ],
)
def test_Repeat(x, repeats, axis, exc):
    x, test_x = x
    repeats, test_repeats = repeats
    g = extra_ops.Repeat(axis)(x, repeats)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            [x, repeats],
            g,
            [test_x, test_repeats],
        )


@pytest.mark.parametrize(
    "x, axis, return_index, return_inverse, return_counts, exc",
    [
        (
            (pt.lscalar(), np.array(1, dtype="int64")),
            None,
            False,
            False,
            False,
            None,
        ),
        (
            (pt.lvector(), np.array([1, 1, 2], dtype="int64")),
            None,
            False,
            False,
            False,
            None,
        ),
        (
            (pt.lmatrix(), np.array([[1, 1], [2, 2]], dtype="int64")),
            None,
            False,
            False,
            False,
            None,
        ),
        (
            (pt.lmatrix(), np.array([[1, 1], [1, 1], [2, 2]], dtype="int64")),
            0,
            False,
            False,
            False,
            UserWarning,
        ),
        (
            (pt.lmatrix(), np.array([[1, 1], [1, 1], [2, 2]], dtype="int64")),
            0,
            True,
            True,
            True,
            UserWarning,
        ),
    ],
)
def test_Unique(x, axis, return_index, return_inverse, return_counts, exc):
    x, test_x = x
    g = extra_ops.Unique(return_index, return_inverse, return_counts, axis)(x)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            [x],
            g,
            [test_x],
        )


@pytest.mark.parametrize(
    "arr, shape, order, exc",
    [
        (
            (pt.lvector(), np.array([9, 15, 1], dtype="int64")),
            pt.as_tensor([2, 3, 4]),
            "C",
            None,
        ),
        (
            (pt.lvector(), np.array([1, 0], dtype="int64")),
            pt.as_tensor([2]),
            "C",
            None,
        ),
        (
            (pt.lvector(), np.array([9, 15, 1], dtype="int64")),
            pt.as_tensor([2, 3, 4]),
            "F",
            NotImplementedError,
        ),
    ],
)
def test_UnravelIndex(arr, shape, order, exc):
    arr, test_arr = arr
    g = extra_ops.UnravelIndex(order)(arr, shape)

    cm = contextlib.suppress() if exc is None else pytest.raises(exc)
    with cm:
        compare_numba_and_py(
            [arr],
            g,
            [test_arr],
        )


@pytest.mark.parametrize(
    "a, v, side, sorter, exc",
    [
        (
            (pt.vector(), np.array([1.0, 2.0, 3.0], dtype=config.floatX)),
            (pt.matrix(), rng.random((3, 2)).astype(config.floatX)),
            "left",
            None,
            None,
        ),
        pytest.param(
            (
                pt.vector(),
                np.array([0.29769574, 0.71649186, 0.20475563]).astype(config.floatX),
            ),
            (
                pt.matrix(),
                np.array(
                    [
                        [0.18847123, 0.39659508],
                        [0.56220006, 0.57428752],
                        [0.86720994, 0.44522637],
                    ]
                ).astype(config.floatX),
            ),
            "left",
            None,
            None,
        ),
        (
            (pt.vector(), np.array([1.0, 2.0, 3.0], dtype=config.floatX)),
            (pt.matrix(), rng.random((3, 2)).astype(config.floatX)),
            "right",
            (pt.lvector(), np.array([0, 2, 1])),
            UserWarning,
        ),
    ],
)
def test_Searchsorted(a, v, side, sorter, exc):
    a, test_a = a
    v, test_v = v
    if sorter is not None:
        sorter, test_sorter = sorter

    g = extra_ops.SearchsortedOp(side)(a, v, sorter)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            [a, v] if sorter is None else [a, v, sorter],
            g,
            [test_a, test_v] if sorter is None else [test_a, test_v, test_sorter],
        )


def test_check_and_raise():
    x = pt.vector()
    x_test_value = np.array([1.0, 2.0], dtype=config.floatX)

    out = assert_op(x.sum(), np.array(True))

    compare_numba_and_py([x], out, [x_test_value])
