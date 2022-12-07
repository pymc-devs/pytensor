import jax.errors
import numpy as np
import pytest

import pytensor
import pytensor.tensor.basic as at
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import get_test_value
from pytensor.tensor.type import iscalar, matrix, scalar, vector
from tests.link.jax.test_basic import compare_jax_and_py


def test_jax_Alloc():
    x = at.alloc(0.0, 2, 3)
    x_fg = FunctionGraph([], [x])

    (jax_res,) = compare_jax_and_py(x_fg, [])

    assert jax_res.shape == (2, 3)

    x = at.alloc(1.1, 2, 3)
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])

    x = at.AllocEmpty("float32")(2, 3)
    x_fg = FunctionGraph([], [x])

    def compare_shape_dtype(x, y):
        (x,) = x
        (y,) = y
        return x.shape == y.shape and x.dtype == y.dtype

    compare_jax_and_py(x_fg, [], assert_fn=compare_shape_dtype)

    a = scalar("a")
    x = at.alloc(a, 20)
    x_fg = FunctionGraph([a], [x])

    compare_jax_and_py(x_fg, [10.0])

    a = vector("a")
    x = at.alloc(a, 20, 10)
    x_fg = FunctionGraph([a], [x])

    compare_jax_and_py(x_fg, [np.ones(10, dtype=config.floatX)])


def test_jax_MakeVector():
    x = at.make_vector(1, 2, 3)
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])


def test_arange():
    out = at.arange(1, 10, 2)
    fgraph = FunctionGraph([], [out])
    compare_jax_and_py(fgraph, [])


def test_arange_nonconcrete():
    """JAX cannot JIT-compile `jax.numpy.arange` when arguments are not concrete values."""

    a = scalar("a")
    a.tag.test_value = 10
    out = at.arange(a)

    with pytest.raises(NotImplementedError):
        fgraph = FunctionGraph([a], [out])
        compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_jax_Join():
    a = matrix("a")
    b = matrix("b")

    x = at.join(0, a, b)
    x_fg = FunctionGraph([a, b], [x])
    compare_jax_and_py(
        x_fg,
        [
            np.c_[[1.0, 2.0, 3.0]].astype(config.floatX),
            np.c_[[4.0, 5.0, 6.0]].astype(config.floatX),
        ],
    )
    compare_jax_and_py(
        x_fg,
        [
            np.c_[[1.0, 2.0, 3.0]].astype(config.floatX),
            np.c_[[4.0, 5.0]].astype(config.floatX),
        ],
    )

    x = at.join(1, a, b)
    x_fg = FunctionGraph([a, b], [x])
    compare_jax_and_py(
        x_fg,
        [
            np.c_[[1.0, 2.0, 3.0]].astype(config.floatX),
            np.c_[[4.0, 5.0, 6.0]].astype(config.floatX),
        ],
    )
    compare_jax_and_py(
        x_fg,
        [
            np.c_[[1.0, 2.0], [3.0, 4.0]].astype(config.floatX),
            np.c_[[5.0, 6.0]].astype(config.floatX),
        ],
    )


class TestJaxSplit:
    def test_basic(self):
        a = matrix("a")
        a_splits = at.split(a, splits_size=[1, 2, 3], n_splits=3, axis=0)
        fg = FunctionGraph([a], a_splits)
        compare_jax_and_py(
            fg,
            [
                np.zeros((6, 4)).astype(config.floatX),
            ],
        )

        a = matrix("a", shape=(6, None))
        a_splits = at.split(a, splits_size=[2, a.shape[0] - 2], n_splits=2, axis=0)
        fg = FunctionGraph([a], a_splits)
        compare_jax_and_py(
            fg,
            [
                np.zeros((6, 4)).astype(config.floatX),
            ],
        )

    def test_runtime_errors(self):
        a = matrix("a")

        a_splits = at.split(a, splits_size=[2, 2, 2], n_splits=2, axis=0)
        fn = pytensor.function([a], a_splits, mode="JAX")
        with pytest.raises(
            ValueError, match="Length of splits is not equal to n_splits"
        ):
            fn(np.zeros((6, 4), dtype=pytensor.config.floatX))

        a_splits = at.split(a, splits_size=[2, 4], n_splits=3, axis=0)
        fn = pytensor.function([a], a_splits, mode="JAX")
        with pytest.raises(
            ValueError, match="Length of splits is not equal to n_splits"
        ):
            fn(np.zeros((6, 4), dtype=pytensor.config.floatX))

        a_splits = at.split(a, splits_size=[2, 4], n_splits=2, axis=0)
        fn = pytensor.function([a], a_splits, mode="JAX")
        with pytest.raises(
            ValueError, match="Split sizes do not sum up to input length along axis: 7"
        ):
            fn(np.zeros((7, 4), dtype=pytensor.config.floatX))

        a_splits = at.split(a, splits_size=[2, -4, 8], n_splits=3, axis=0)
        fn = pytensor.function([a], a_splits, mode="JAX")
        with pytest.raises(
            ValueError,
            match="Split sizes cannot be negative",
        ):
            fn(np.zeros((6, 4), dtype=pytensor.config.floatX))

    def test_jax_split_not_supported(self):
        a = matrix("a", shape=(6, None))

        a_splits = at.split(a, splits_size=[2, a.shape[1] - 2], n_splits=2, axis=1)
        with pytest.warns(
            UserWarning, match="Split node does not have constant split positions."
        ):
            fn = pytensor.function([a], a_splits, mode="JAX")
        # It raises an informative ConcretizationTypeError, but there's an AttributeError that surpsasses it
        with pytest.raises(AttributeError):
            fn(np.zeros((6, 4), dtype=pytensor.config.floatX))

        split_axis = iscalar("split_axis")
        a_splits = at.split(a, splits_size=[2, 4], n_splits=2, axis=split_axis)
        with pytest.warns(UserWarning, match="Split node does not have constant axis."):
            fn = pytensor.function([a, split_axis], a_splits, mode="JAX")
        with pytest.raises(jax.errors.TracerIntegerConversionError):
            fn(np.zeros((6, 6), dtype=pytensor.config.floatX), 0)


def test_jax_eye():
    """Tests jaxification of the Eye operator"""
    out = at.eye(3)
    out_fg = FunctionGraph([], [out])

    compare_jax_and_py(out_fg, [])
