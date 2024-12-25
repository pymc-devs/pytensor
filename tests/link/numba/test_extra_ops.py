import contextlib

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from pytensor.tensor import extra_ops
from tests.link.numba.test_basic import compare_numba_and_py


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "val,test_values",
    [
        (pt.lscalar(), [np.array(6, dtype="int64")]),
    ],
)
def test_Bartlett(val, test_values):
    g = extra_ops.bartlett(val)
    inputs = [val]

    compare_numba_and_py(
        inputs,
        [g],
        test_values,
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, atol=1e-15),
    )


@pytest.mark.parametrize(
    "val, axis, mode,test_values",
    [
        (pt.matrix(), 1, "add", [np.arange(3, dtype=config.floatX).reshape((3, 1))]),
        (
            pt.dtensor3(),
            -1,
            "add",
            [np.arange(30, dtype=config.floatX).reshape((2, 3, 5))],
        ),
        ((pt.matrix()), 0, "add", [np.arange(6, dtype=config.floatX).reshape((3, 2))]),
        (pt.matrix(), 1, "add", [np.arange(6, dtype=config.floatX).reshape((3, 2))]),
        (pt.matrix(), None, "add", [np.arange(6, dtype=config.floatX).reshape((3, 2))]),
        (pt.matrix(), 0, "mul", [np.arange(6, dtype=config.floatX).reshape((3, 2))]),
        (pt.matrix(), 1, "mul", [np.arange(6, dtype=config.floatX).reshape((3, 2))]),
        (pt.matrix(), None, "mul", [np.arange(6, dtype=config.floatX).reshape((3, 2))]),
    ],
)
def test_CumOp(val, axis, mode, test_values):
    g = extra_ops.CumOp(axis=axis, mode=mode)(val)
    inputs = [val]

    compare_numba_and_py(
        inputs,
        [g],
        test_values,
    )


@pytest.mark.parametrize(
    "a, val, test_values",
    [
        (
            pt.lmatrix(),
            pt.lscalar(),
            [np.zeros((10, 2), dtype="int64"), np.array(1, dtype="int64")],
        )
    ],
)
def test_FillDiagonal(a, val, test_values):
    g = extra_ops.FillDiagonal()(a, val)
    inputs = [a, val]

    compare_numba_and_py(
        inputs,
        [g],
        test_values,
    )


@pytest.mark.parametrize(
    "a, val, offset, test_values",
    [
        (
            pt.lmatrix(),
            pt.lscalar(),
            pt.lscalar(),
            [
                np.zeros((10, 2), dtype="int64"),
                np.array(1, dtype="int64"),
                np.array(-1, dtype="int64"),
            ],
        ),
        (
            pt.lmatrix(),
            pt.lscalar(),
            pt.lscalar(),
            [
                np.zeros((10, 2), dtype="int64"),
                np.array(1, dtype="int64"),
                np.array(0, dtype="int64"),
            ],
        ),
        (
            pt.lmatrix(),
            pt.lscalar(),
            pt.lscalar(),
            [
                np.zeros((10, 3), dtype="int64"),
                np.array(1, dtype="int64"),
                np.array(1, dtype="int64"),
            ],
        ),
    ],
)
def test_FillDiagonalOffset(a, val, offset, test_values):
    g = extra_ops.FillDiagonalOffset()(a, val, offset)
    inputs = [a, val, offset]

    compare_numba_and_py(
        inputs,
        [g],
        test_values,
    )


@pytest.mark.parametrize(
    "arr, shape, mode, order, exc,  test_values",
    [
        (
            tuple(pt.lscalar() for v in np.array([0])),
            pt.lvector(),
            "raise",
            "C",
            None,
            [0, np.array([2])],
        ),
        (
            tuple(pt.lscalar() for v in np.array([0, 0, 3])),
            pt.lvector(),
            "raise",
            "C",
            None,
            [0, 0, 3, np.array([2, 3, 4])],
        ),
        (
            tuple(pt.lvector() for v in np.array([[0, 1], [2, 0], [1, 3]])),
            pt.lvector(),
            "raise",
            "C",
            None,
            [np.array([0, 1]), np.array([2, 0]), np.array([1, 3]), np.array([2, 3, 4])],
        ),
        (
            tuple(pt.lvector() for v in np.array([[0, 1], [2, 0], [1, 3]])),
            pt.lvector(),
            "raise",
            "F",
            NotImplementedError,
            [np.array([0, 1]), np.array([2, 0]), np.array([1, 3]), np.array([2, 3, 4])],
        ),
        (
            tuple(pt.lvector() for v in np.array([[0, 1, 2], [2, 0, 3], [1, 3, 5]])),
            pt.lvector(),
            "raise",
            "C",
            ValueError,
            [
                np.array([0, 1, 2]),
                np.array([2, 0, 3]),
                np.array([1, 3, 5]),
                np.array([2, 3, 4]),
            ],
        ),
        (
            tuple(pt.lvector() for v in np.array([[0, 1, 2], [2, 0, 3], [1, 3, 5]])),
            pt.lvector(),
            "wrap",
            "C",
            None,
            [
                np.array([0, 1, 2]),
                np.array([2, 0, 3]),
                np.array([1, 3, 5]),
                np.array([2, 3, 4]),
            ],
        ),
        (
            tuple(pt.lvector() for v in np.array([[0, 1, 2], [2, 0, 3], [1, 3, 5]])),
            pt.lvector(),
            "clip",
            "C",
            None,
            [
                np.array([0, 1, 2]),
                np.array([2, 0, 3]),
                np.array([1, 3, 5]),
                np.array([2, 3, 4]),
            ],
        ),
    ],
)
def test_RavelMultiIndex(arr, shape, mode, order, exc, test_values):
    g = extra_ops.RavelMultiIndex(mode, order)(*((*arr, shape)))
    inputs = [*arr, shape]

    cm = contextlib.suppress() if exc is None else pytest.raises(exc)
    with cm:
        compare_numba_and_py(
            inputs,
            [g],
            test_values,
        )


@pytest.mark.parametrize(
    "x, repeats, axis, exc,test_values",
    [
        (
            pt.lscalar(),
            pt.lscalar(),
            None,
            None,
            [np.array(1, dtype="int64"), np.array(0, dtype="int64")],
        ),
        (
            pt.lmatrix(),
            pt.lscalar(),
            None,
            None,
            [np.zeros((2, 2), dtype="int64"), np.array(1, dtype="int64")],
        ),
        (
            pt.lvector(),
            pt.lvector(),
            None,
            None,
            [np.arange(2, dtype="int64"), np.array([1, 1], dtype="int64")],
        ),
        (
            pt.lmatrix(),
            pt.lscalar(),
            0,
            UserWarning,
            [np.zeros((2, 2), dtype="int64"), np.array(1, dtype="int64")],
        ),
    ],
)
def test_Repeat(x, repeats, axis, exc, test_values):
    g = extra_ops.Repeat(axis)(x, repeats)
    inputs = [x, repeats]

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            inputs,
            [g],
            test_values,
        )


@pytest.mark.parametrize(
    "x, axis, return_index, return_inverse, return_counts, exc, test_values",
    [
        (pt.lscalar(), None, False, False, False, None, [np.array(1, dtype="int64")]),
        (
            pt.lvector(),
            None,
            False,
            False,
            False,
            None,
            [np.array([1, 1, 2], dtype="int64")],
        ),
        (
            pt.lmatrix(),
            None,
            False,
            False,
            False,
            None,
            [np.array([[1, 1], [2, 2]], dtype="int64")],
        ),
        (
            pt.lmatrix(),
            0,
            False,
            False,
            False,
            UserWarning,
            [np.array([[1, 1], [1, 1], [2, 2]], dtype="int64")],
        ),
        (
            pt.lmatrix(),
            0,
            True,
            True,
            True,
            UserWarning,
            [np.array([[1, 1], [1, 1], [2, 2]], dtype="int64")],
        ),
    ],
)
def test_Unique(x, axis, return_index, return_inverse, return_counts, exc, test_values):
    g = extra_ops.Unique(return_index, return_inverse, return_counts, axis)(x)
    inputs = [x]
    outputs = []
    if isinstance(g, list):
        outputs = g
    else:
        outputs = [g]

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            inputs,
            outputs,
            test_values,
        )


@pytest.mark.parametrize(
    "arr, shape, order, exc,test_values",
    [
        (
            pt.lvector(),
            pt.as_tensor([2, 3, 4]),
            "C",
            None,
            [np.array([9, 15, 1], dtype="int64")],
        ),
        (pt.lvector(), pt.as_tensor([2]), "C", None, [np.array([1, 0], dtype="int64")]),
        (
            pt.lvector(),
            pt.as_tensor([2, 3, 4]),
            "F",
            NotImplementedError,
            [np.array([9, 15, 1], dtype="int64")],
        ),
    ],
)
def test_UnravelIndex(arr, shape, order, exc, test_values):
    g = extra_ops.UnravelIndex(order)(arr, shape)
    inputs = [arr]
    outputs = []
    if isinstance(g, list):
        outputs = g
    else:
        outputs = [g]

    cm = contextlib.suppress() if exc is None else pytest.raises(exc)
    with cm:
        compare_numba_and_py(
            inputs,
            outputs,
            test_values,
        )


@pytest.mark.parametrize(
    "a, v, side, sorter, exc,test_values",
    [
        (
            pt.vector(),
            pt.matrix(),
            "left",
            None,
            None,
            [
                np.array([1.0, 2.0, 3.0], dtype=config.floatX),
                rng.random((3, 2)).astype(config.floatX),
            ],
        ),
        pytest.param(
            pt.vector(),
            pt.matrix(),
            "left",
            None,
            None,
            [
                np.array([0.29769574, 0.71649186, 0.20475563]).astype(config.floatX),
                np.array(
                    [
                        [0.18847123, 0.39659508],
                        [0.56220006, 0.57428752],
                        [0.86720994, 0.44522637],
                    ]
                ).astype(config.floatX),
            ],
        ),
        (
            pt.vector(),
            pt.matrix(),
            "right",
            pt.lvector(),
            UserWarning,
            [
                np.array([1.0, 2.0, 3.0], dtype=config.floatX),
                rng.random((3, 2)).astype(config.floatX),
                np.array([0, 2, 1]),
            ],
        ),
    ],
)
def test_Searchsorted(a, v, side, sorter, exc, test_values):
    g = extra_ops.SearchsortedOp(side)(a, v, sorter)
    inputs = [a, v]

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            inputs,
            [g],
            test_values,
        )
