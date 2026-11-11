"""`Array`: loading, storing and slicing elements in generated code."""

import ctypes

import numpy as np
import pytest


ir = pytest.importorskip("pytensor_llvm.ir")

from pytensor.link.llvm.codegen.emit import Array  # noqa: E402
from pytensor.link.llvm.codegen.num import Num  # noqa: E402
from pytensor.link.llvm.layout import DYNAMIC, PACKED, Layout  # noqa: E402


def run(build, *arrays):
    """Compile `build(b, *Arrays)` (strides and lengths as constants) and run it on
    `arrays`; returns what the function returns (a double)."""
    m = ir.Module("array")
    f = m.function("f", [ir.ptr] * len(arrays), ir.f64, exported=True)
    b = f.builder()
    wrapped = [
        Array(
            str(a.dtype), p, [b.i64(n) for n in a.shape], [b.i64(s) for s in a.strides]
        )
        for a, p in zip(arrays, f.params, strict=True)
    ]
    b.ret(build(b, *wrapped))
    c = m.compile(cache=False)
    fn = ctypes.CFUNCTYPE(ctypes.c_double, *[ctypes.c_void_p] * len(arrays))(
        c.address("f")
    )
    return fn(*[a.ctypes.data for a in arrays])


def test_load_and_store_follow_the_strides():
    x = np.arange(24.0).reshape(4, 6)[::-1, ::2]  # negative and non-unit strides
    out = np.zeros((3, 4))[:, ::-1]

    def build(b, xa, oa):
        with b.loop(xa.shape[0]) as i:
            with b.loop(xa.shape[1]) as j:
                oa.store(b, xa.load(b, i, j) * 2.0, j, i)
        return (xa.load(b, 1, 2) + xa.load(b, 3, 0)).value  # Python ints as indices

    assert run(build, x, out) == x[1, 2] + x[3, 0]
    np.testing.assert_array_equal(out, 2 * x.T)


@pytest.mark.parametrize("dtype", ["float32", "int16", "uint8", "bool"])
def test_elements_are_nums_of_the_dtype(dtype):
    x = (np.arange(1, 6) * 50).astype(dtype)
    out = np.zeros(5, dtype=dtype)

    def build(b, xa, oa):
        with b.loop(5) as i:
            v = xa.load(b, i)
            assert isinstance(v, Num) and v.dtype == dtype
            oa.store(b, v, b.sub(4, i))  # reversed
        oa.store(b, 1, 0)  # a Python number takes the dtype
        oa.store(b, xa.load(b, 1).value, 1)  # a raw IR value
        return b.f64(0.0)

    run(build, x, out)
    want = x[::-1].copy()
    want[0], want[1] = 1, x[1]
    np.testing.assert_array_equal(out, want)


def test_at_gives_sub_arrays():
    x = np.arange(60.0).reshape(3, 4, 5)[:, ::-1]
    out = np.zeros(5)

    def build(b, xa, oa):
        row = xa.at(b, 2).at(b, 1)  # x[2, 1]
        assert row.ndim == 1 and row.shape[0].const_int == 5
        column = xa.at(b, 3, axis=-1)  # x[:, :, 3]
        assert [n.const_int for n in column.shape] == [3, 4]
        with b.loop(5) as i:
            oa.store(b, row.load(b, i), i)
        return column.load(b, 1, 2).value

    assert run(build, x, out) == x[1, 2, 3]
    np.testing.assert_array_equal(out, x[2, 1])


def test_at_keeps_what_the_layout_says():
    m = ir.Module("array")
    f = m.function("f", [ir.ptr, ir.i64, ir.i64], ir.f64)
    b = f.builder()
    p, n, i = f.params
    c = Array("float64", p, [n, b.i64(4)], [b.i64(32), b.i64(8)], Layout.c(2))
    assert c.at(b, i).layout.is_c((4,))  # a row of a C-contiguous matrix is contiguous
    column = c.at(b, i, axis=1)  # a column is strided
    assert str(column.layout) == "[32]" and column.strides[0].const_int == 32
    unknown = Array("float64", p, [n, n], [n, n])
    assert unknown.at(b, 0).layout == Layout.make((None,), (0,), (DYNAMIC,))
    fortran = Array(
        "float64",
        p,
        [b.i64(3), n],
        [b.i64(8), b.i64(24)],
        Layout.make((3, None), (1, 0), (PACKED, PACKED)),
    )
    assert fortran.at(b, i, axis=1).layout.is_c((3,))  # a column of a Fortran matrix


def test_mistakes_raise():
    m = ir.Module("array")
    f = m.function("f", [ir.ptr, ir.i64], ir.f64)
    b = f.builder()
    p, n = f.params
    a = Array("float64", p, [n, n], [n, n])
    with pytest.raises(ValueError, match="1 indices for an array with 2 axes"):
        a.load(b, 0)
    with pytest.raises(TypeError, match="convert with astype"):
        a.store(b, Num.const(b, "int64", 1), 0, 0)
    with pytest.raises(IndexError):
        a.at(b, 0, axis=2)
