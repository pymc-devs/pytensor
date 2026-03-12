import re

import numpy as np
import pytest

from pytensor.compile import get_mode


jax = pytest.importorskip("jax")
from jax import errors

import pytensor
import pytensor.tensor as pt
import pytensor.tensor.basic as ptb
from pytensor.configdefaults import config
from pytensor.tensor.type import iscalar, matrix, scalar, vector
from tests.link.jax.test_basic import compare_jax_and_py
from tests.tensor.test_basic import check_alloc_runtime_broadcast


def test_jax_Alloc():
    x = ptb.alloc(0.0, 2, 3)

    _, [jax_res] = compare_jax_and_py([], [x], [])

    assert jax_res.shape == (2, 3)

    x = ptb.alloc(1.1, 2, 3)

    compare_jax_and_py([], [x], [])

    x = ptb.AllocEmpty("float32")(2, 3)

    def compare_shape_dtype(x, y):
        assert x.shape == y.shape and x.dtype == y.dtype

    compare_jax_and_py([], [x], [], assert_fn=compare_shape_dtype)

    a = scalar("a")
    x = ptb.alloc(a, 20)

    compare_jax_and_py([a], [x], [10.0])

    a = vector("a")
    x = ptb.alloc(a, 20, 10)

    compare_jax_and_py([a], [x], [np.ones(10, dtype=config.floatX)])


def test_alloc_runtime_broadcast():
    check_alloc_runtime_broadcast(get_mode("JAX"))


def test_jax_MakeVector():
    x = ptb.make_vector(1, 2, 3)

    compare_jax_and_py([], [x], [])


def test_arange():
    out = ptb.arange(1, 10, 2)

    compare_jax_and_py([], [out], [])


def test_arange_of_shape():
    x = vector("x")
    out = ptb.arange(1, x.shape[-1], 2)
    compare_jax_and_py([x], [out], [np.zeros((5,))], jax_mode="JAX")


def test_arange_nonconcrete():
    """JAX cannot JIT-compile `jax.numpy.arange` when arguments are not concrete values."""

    a = scalar("a")
    a_test_value = 10
    out = ptb.arange(a)

    with pytest.raises(NotImplementedError):
        compare_jax_and_py([a], [out], [a_test_value])


def test_jax_Join():
    a = matrix("a")
    b = matrix("b")

    x = ptb.join(0, a, b)
    compare_jax_and_py(
        [a, b],
        [x],
        [
            np.c_[[1.0, 2.0, 3.0]].astype(config.floatX),
            np.c_[[4.0, 5.0, 6.0]].astype(config.floatX),
        ],
    )
    compare_jax_and_py(
        [a, b],
        [x],
        [
            np.c_[[1.0, 2.0, 3.0]].astype(config.floatX),
            np.c_[[4.0, 5.0]].astype(config.floatX),
        ],
    )

    x = ptb.join(1, a, b)
    compare_jax_and_py(
        [a, b],
        [x],
        [
            np.c_[[1.0, 2.0, 3.0]].astype(config.floatX),
            np.c_[[4.0, 5.0, 6.0]].astype(config.floatX),
        ],
    )
    compare_jax_and_py(
        [a, b],
        [x],
        [
            np.c_[[1.0, 2.0], [3.0, 4.0]].astype(config.floatX),
            np.c_[[5.0, 6.0]].astype(config.floatX),
        ],
    )


class TestJaxSplit:
    def test_basic(self):
        a = matrix("a")
        a_splits = ptb.split(a, splits_size=[1, 2, 3], n_splits=3, axis=0)
        compare_jax_and_py(
            [a],
            a_splits,
            [
                np.zeros((6, 4)).astype(config.floatX),
            ],
        )

        a = matrix("a", shape=(6, None))
        a_splits = ptb.split(a, splits_size=[2, a.shape[0] - 2], n_splits=2, axis=0)
        compare_jax_and_py(
            [a],
            a_splits,
            [
                np.zeros((6, 4)).astype(config.floatX),
            ],
        )

    def test_runtime_errors(self):
        a = matrix("a")

        a_splits = ptb.split(a, splits_size=[2, 2, 2], n_splits=2, axis=0)
        fn = pytensor.function([a], a_splits, mode="JAX")
        with pytest.raises(
            ValueError, match="Length of splits is not equal to n_splits"
        ):
            fn(np.zeros((6, 4), dtype=pytensor.config.floatX))

        # This check is triggered at compile time if splits_size has incompatible static length
        splits_size = vector("splits_size", shape=(None,), dtype=int)
        a_splits = ptb.split(a, splits_size=splits_size, n_splits=3, axis=0)
        fn = pytensor.function([a, splits_size], a_splits, mode="JAX")
        with pytest.raises(
            ValueError, match="Length of splits is not equal to n_splits"
        ):
            fn(np.zeros((6, 4), dtype=pytensor.config.floatX), [2, 2])

        a_splits = ptb.split(a, splits_size=[2, 4], n_splits=2, axis=0)
        fn = pytensor.function([a], a_splits, mode="JAX")
        with pytest.raises(
            ValueError, match="Split sizes do not sum up to input length along axis: 7"
        ):
            fn(np.zeros((7, 4), dtype=pytensor.config.floatX))

        a_splits = ptb.split(a, splits_size=[2, -4, 8], n_splits=3, axis=0)
        fn = pytensor.function([a], a_splits, mode="JAX")
        with pytest.raises(
            ValueError,
            match="Split sizes cannot be negative",
        ):
            fn(np.zeros((6, 4), dtype=pytensor.config.floatX))

    def test_jax_split_not_supported(self):
        a = matrix("a", shape=(6, None))

        a_splits = ptb.split(a, splits_size=[2, a.shape[1] - 2], n_splits=2, axis=1)
        with pytest.warns(
            UserWarning, match="Split node does not have constant split positions."
        ):
            fn = pytensor.function([a], a_splits, mode="JAX")
        # This test used to raise AttributeError in previous versions of JAX.
        # Now it raises `TracerIntegerConversionError`.
        # We accept both errors for backwards compatibility.
        with pytest.raises((AttributeError, errors.TracerIntegerConversionError)):
            fn(np.zeros((6, 4), dtype=pytensor.config.floatX))

        split_axis = iscalar("split_axis")
        a_splits = ptb.split(a, splits_size=[2, 4], n_splits=2, axis=split_axis)
        with pytest.warns(UserWarning, match="Split node does not have constant axis."):
            fn = pytensor.function([a, split_axis], a_splits, mode="JAX")
        # Same reasoning as above to accept both errors.
        with pytest.raises((AttributeError, errors.TracerIntegerConversionError)):
            fn(np.zeros((6, 6), dtype=pytensor.config.floatX), 0)


def test_jax_eye():
    """Tests jaxification of the Eye operator"""
    out = ptb.eye(3)

    compare_jax_and_py([], [out], [])


def test_tri():
    out = ptb.tri(10, 10, 0)

    compare_jax_and_py([], [out], [])


def test_tri_nonconcrete():
    """JAX cannot JIT-compile `jax.numpy.tri` when arguments are not concrete values."""

    m, n, k = (
        scalar("a", dtype="int64"),
        scalar("n", dtype="int64"),
        scalar("k", dtype="int64"),
    )
    m_test_value = 10
    n_test_value = 10
    k_test_value = 0

    out = ptb.tri(m, n, k)

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "JAX requires the arguments of `jax.numpy.arange` to be constants"
        ),
    ):
        compare_jax_and_py([m, n, k], [out], [m_test_value, n_test_value, k_test_value])


def test_jax_roll():
    # Test roll with dynamic shift (the main bug fix)
    x = pt.dmatrix("x")
    shift = pt.iscalar("shift")
    data = np.arange(16, dtype=float).reshape(4, 4)

    # axis=0, dynamic shift
    out = pt.roll(x, shift=shift, axis=0)
    compare_jax_and_py([x, shift], [out], [data, 2])

    # axis=1, dynamic shift
    out = pt.roll(x, shift=shift, axis=1)
    compare_jax_and_py([x, shift], [out], [data, 3])

    # negative shift
    out = pt.roll(x, shift=shift, axis=0)
    compare_jax_and_py([x, shift], [out], [data, -1])

    # axis=None (flatten then roll)
    out = pt.roll(x, shift=shift, axis=None)
    compare_jax_and_py([x, shift], [out], [data, 2])

    # shift larger than axis size
    out = pt.roll(x, shift=shift, axis=0)
    compare_jax_and_py([x, shift], [out], [data, 10])
