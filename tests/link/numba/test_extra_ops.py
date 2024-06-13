import contextlib

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor import extra_ops
from tests.link.numba.test_basic import compare_numba_and_py, set_test_value


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "test_values",
    [
        set_test_value(pt.lscalar(), np.array(6, dtype="int64")),
    ],
)
def test_Bartlett(test_values):
    val = next(iter(test_values.keys()))
    g = extra_ops.bartlett(val)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            test_values[i]
            for i in test_values
            if not isinstance(i, SharedVariable | Constant)
        ],
        assert_fn=lambda x, y: np.testing.assert_allclose(x, y, atol=1e-15),
    )


@pytest.mark.parametrize(
    "test_values, axis, mode",
    [
        (
            set_test_value(
                pt.matrix(), np.arange(3, dtype=config.floatX).reshape((3, 1))
            ),
            1,
            "add",
        ),
        (
            set_test_value(
                pt.dtensor3(), np.arange(30, dtype=config.floatX).reshape((2, 3, 5))
            ),
            -1,
            "add",
        ),
        (
            set_test_value(
                pt.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))
            ),
            0,
            "add",
        ),
        (
            set_test_value(
                pt.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))
            ),
            1,
            "add",
        ),
        (
            set_test_value(
                pt.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))
            ),
            None,
            "add",
        ),
        (
            set_test_value(
                pt.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))
            ),
            0,
            "mul",
        ),
        (
            set_test_value(
                pt.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))
            ),
            1,
            "mul",
        ),
        (
            set_test_value(
                pt.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))
            ),
            None,
            "mul",
        ),
    ],
)
def test_CumOp(test_values, axis, mode):
    val = next(iter(test_values.keys()))
    g = extra_ops.CumOp(axis=axis, mode=mode)(val)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            test_values[i]
            for i in test_values
            if not isinstance(i, SharedVariable | Constant)
        ],
    )


@pytest.mark.parametrize(
    "inputs",
    [
        (
            set_test_value(pt.lmatrix(), np.zeros((10, 2), dtype="int64")),
            set_test_value(pt.lscalar(), np.array(1, dtype="int64")),
        )
    ],
)
def test_FillDiagonal(inputs):
    print(inputs)
    # assert 0
    test_values = {k: v for d in inputs for k, v in d.items()}
    inputs = list(test_values.keys())
    g = extra_ops.FillDiagonal()(*inputs)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            test_values[i]
            for i in test_values
            if not isinstance(i, SharedVariable | Constant)
        ],
    )


@pytest.mark.parametrize(
    "inputs",
    [
        (
            set_test_value(pt.lmatrix(), np.zeros((10, 2), dtype="int64")),
            set_test_value(pt.lscalar(), np.array(1, dtype="int64")),
            set_test_value(pt.lscalar(), np.array(-1, dtype="int64")),
        ),
        (
            set_test_value(pt.lmatrix(), np.zeros((10, 2), dtype="int64")),
            set_test_value(pt.lscalar(), np.array(1, dtype="int64")),
            set_test_value(pt.lscalar(), np.array(0, dtype="int64")),
        ),
        (
            set_test_value(pt.lmatrix(), np.zeros((10, 3), dtype="int64")),
            set_test_value(pt.lscalar(), np.array(1, dtype="int64")),
            set_test_value(pt.lscalar(), np.array(1, dtype="int64")),
        ),
    ],
)
def test_FillDiagonalOffset(inputs):
    test_values = {k: v for d in inputs for k, v in d.items()}
    inputs = list(test_values.keys())
    g = extra_ops.FillDiagonalOffset()(*inputs)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            test_values[i]
            for i in test_values
            if not isinstance(i, SharedVariable | Constant)
        ],
    )


@pytest.mark.parametrize(
    "arr, shape, mode, order, exc",
    [
        (
            tuple(set_test_value(pt.lscalar(), v) for v in np.array([0])),
            set_test_value(pt.lvector(), np.array([2])),
            "raise",
            "C",
            None,
        ),
        (
            tuple(set_test_value(pt.lscalar(), v) for v in np.array([0, 0, 3])),
            set_test_value(pt.lvector(), np.array([2, 3, 4])),
            "raise",
            "C",
            None,
        ),
        (
            tuple(
                set_test_value(pt.lvector(), v)
                for v in np.array([[0, 1], [2, 0], [1, 3]])
            ),
            set_test_value(pt.lvector(), np.array([2, 3, 4])),
            "raise",
            "C",
            None,
        ),
        (
            tuple(
                set_test_value(pt.lvector(), v)
                for v in np.array([[0, 1], [2, 0], [1, 3]])
            ),
            set_test_value(pt.lvector(), np.array([2, 3, 4])),
            "raise",
            "F",
            NotImplementedError,
        ),
        (
            tuple(
                set_test_value(pt.lvector(), v)
                for v in np.array([[0, 1, 2], [2, 0, 3], [1, 3, 5]])
            ),
            set_test_value(pt.lvector(), np.array([2, 3, 4])),
            "raise",
            "C",
            ValueError,
        ),
        (
            tuple(
                set_test_value(pt.lvector(), v)
                for v in np.array([[0, 1, 2], [2, 0, 3], [1, 3, 5]])
            ),
            set_test_value(pt.lvector(), np.array([2, 3, 4])),
            "wrap",
            "C",
            None,
        ),
        (
            tuple(
                set_test_value(pt.lvector(), v)
                for v in np.array([[0, 1, 2], [2, 0, 3], [1, 3, 5]])
            ),
            set_test_value(pt.lvector(), np.array([2, 3, 4])),
            "clip",
            "C",
            None,
        ),
    ],
)
def test_RavelMultiIndex(arr, shape, mode, order, exc):
    test_values = {k: v for d in arr for k, v in d.items()}
    arr = tuple(test_values.keys())
    test_values.update(shape)
    shape = next(iter(shape.keys()))
    g = extra_ops.RavelMultiIndex(mode, order)(*((*arr, shape)))
    g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.raises(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                test_values[i]
                for i in test_values
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "test_values, repeats, axis, exc",
    [
        (
            set_test_value(pt.lscalar(), np.array(1, dtype="int64")),
            set_test_value(pt.lscalar(), np.array(0, dtype="int64")),
            None,
            None,
        ),
        (
            set_test_value(pt.lmatrix(), np.zeros((2, 2), dtype="int64")),
            set_test_value(pt.lscalar(), np.array(1, dtype="int64")),
            None,
            None,
        ),
        (
            set_test_value(pt.lvector(), np.arange(2, dtype="int64")),
            set_test_value(pt.lvector(), np.array([1, 1], dtype="int64")),
            None,
            None,
        ),
        (
            set_test_value(pt.lmatrix(), np.zeros((2, 2), dtype="int64")),
            set_test_value(pt.lscalar(), np.array(1, dtype="int64")),
            0,
            UserWarning,
        ),
    ],
)
def test_Repeat(test_values, repeats, axis, exc):
    x = next(iter(test_values.keys()))
    test_values.update(repeats)
    repeats = next(iter(repeats.keys()))
    g = extra_ops.Repeat(axis)(x, repeats)
    g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                test_values[i]
                for i in test_values
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "test_values, axis, return_index, return_inverse, return_counts, exc",
    [
        (
            set_test_value(pt.lscalar(), np.array(1, dtype="int64")),
            None,
            False,
            False,
            False,
            None,
        ),
        (
            set_test_value(pt.lvector(), np.array([1, 1, 2], dtype="int64")),
            None,
            False,
            False,
            False,
            None,
        ),
        (
            set_test_value(pt.lmatrix(), np.array([[1, 1], [2, 2]], dtype="int64")),
            None,
            False,
            False,
            False,
            None,
        ),
        (
            set_test_value(
                pt.lmatrix(), np.array([[1, 1], [1, 1], [2, 2]], dtype="int64")
            ),
            0,
            False,
            False,
            False,
            UserWarning,
        ),
        (
            set_test_value(
                pt.lmatrix(), np.array([[1, 1], [1, 1], [2, 2]], dtype="int64")
            ),
            0,
            True,
            True,
            True,
            UserWarning,
        ),
    ],
)
def test_Unique(test_values, axis, return_index, return_inverse, return_counts, exc):
    x = next(iter(test_values.keys()))
    g = extra_ops.Unique(return_index, return_inverse, return_counts, axis)(x)

    if isinstance(g, list):
        g_fg = FunctionGraph(outputs=g)
    else:
        g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                test_values[i]
                for i in test_values
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "test_values, shape, order, exc",
    [
        (
            set_test_value(pt.lvector(), np.array([9, 15, 1], dtype="int64")),
            pt.as_tensor([2, 3, 4]),
            "C",
            None,
        ),
        (
            set_test_value(pt.lvector(), np.array([1, 0], dtype="int64")),
            pt.as_tensor([2]),
            "C",
            None,
        ),
        (
            set_test_value(pt.lvector(), np.array([9, 15, 1], dtype="int64")),
            pt.as_tensor([2, 3, 4]),
            "F",
            NotImplementedError,
        ),
    ],
)
def test_UnravelIndex(test_values, shape, order, exc):
    arr = next(iter(test_values.keys()))
    g = extra_ops.UnravelIndex(order)(arr, shape)

    if isinstance(g, list):
        g_fg = FunctionGraph(outputs=g)
    else:
        g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.raises(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                test_values[i]
                for i in test_values
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "inputs, side, sorter, exc",
    [
        (
            [
                set_test_value(
                    pt.vector(), np.array([1.0, 2.0, 3.0], dtype=config.floatX)
                ),
                set_test_value(pt.matrix(), rng.random((3, 2)).astype(config.floatX)),
            ],
            "left",
            None,
            None,
        ),
        pytest.param(
            [
                set_test_value(
                    pt.vector(),
                    np.array([0.29769574, 0.71649186, 0.20475563]).astype(
                        config.floatX
                    ),
                ),
                set_test_value(
                    pt.matrix(),
                    np.array(
                        [
                            [0.18847123, 0.39659508],
                            [0.56220006, 0.57428752],
                            [0.86720994, 0.44522637],
                        ]
                    ).astype(config.floatX),
                ),
            ],
            "left",
            None,
            None,
        ),
        (
            [
                set_test_value(
                    pt.vector(), np.array([1.0, 2.0, 3.0], dtype=config.floatX)
                ),
                set_test_value(pt.matrix(), rng.random((3, 2)).astype(config.floatX)),
            ],
            "right",
            set_test_value(pt.lvector(), np.array([0, 2, 1])),
            UserWarning,
        ),
    ],
)
def test_Searchsorted(inputs, side, sorter, exc):
    test_values = {k: v for d in inputs for k, v in d.items()}
    inputs = list(test_values.keys())
    if isinstance(sorter, dict):
        test_values.update(sorter)
        sorter = next(iter(sorter.keys()))
    inputs.append(sorter)
    g = extra_ops.SearchsortedOp(side)(*inputs)
    g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                test_values[i]
                for i in test_values
                if not isinstance(i, SharedVariable | Constant)
            ],
        )
