"""`Num`: operators on values with a dtype, checked against numpy."""

import ctypes
import operator

import numpy as np
import pytest


ir = pytest.importorskip("pytensor_llvm.ir")

from pytensor.link.llvm.codegen.num import Num  # noqa: E402
from pytensor.link.llvm.codegen.types import SUPPORTED_DTYPES, value_type  # noqa: E402


CTYPES = {
    "float64": ctypes.c_double,
    "float32": ctypes.c_float,
    "int64": ctypes.c_int64,
    "int32": ctypes.c_int32,
    "int16": ctypes.c_int16,
    "int8": ctypes.c_int8,
    "uint8": ctypes.c_uint8,
    "bool": ctypes.c_bool,
}

VALUES = {
    "f": [0.0, 1.5, -2.25, 7.0, -0.5, np.inf, np.nan],
    "i": [0, 1, -1, 3, -7, 100, -128, 127],
    "u": [0, 1, 3, 7, 100, 255],
    "b": [False, True],
}


def compile_binary(dtype, fn, out=None):
    """A C function (x, y) -> fn(Num x, Num y) of `dtype`."""
    out = out or dtype
    m = ir.Module("num")
    f = m.function("f", [value_type(dtype)] * 2, value_type(out), exported=True)
    b = f.builder()
    x, y = (Num(b, p, dtype) for p in f.params)
    r = fn(x, y)
    assert r.dtype == out
    b.ret(r.value)
    c = m.compile(cache=False)
    cf = ctypes.CFUNCTYPE(CTYPES[out], CTYPES[dtype], CTYPES[dtype])(c.address("f"))
    cf._keep = c
    return cf


def check(dtype, fn, ref=None, out=None, skip=lambda x, y: False):
    cf = compile_binary(dtype, fn, out)
    ref = ref or fn
    dt = np.dtype(dtype)
    vals = VALUES[dt.kind]
    with np.errstate(all="ignore"):
        for xv in vals:
            for yv in vals:
                x, y = dt.type(xv), dt.type(yv)
                if skip(x, y):
                    continue
                want = np.asarray(ref(x, y)).astype(out or dtype)
                got = np.asarray(cf(x.item(), y.item())).astype(out or dtype)
                np.testing.assert_array_equal(
                    got, want, err_msg=f"{dtype}: {xv!r}, {yv!r}"
                )


NUMERIC = [d for d in SUPPORTED_DTYPES if d != "bool"]
INTS = [d for d in NUMERIC if not d.startswith("float")]


@pytest.mark.parametrize("dtype", NUMERIC)
@pytest.mark.parametrize(
    "op", [operator.add, operator.sub, operator.mul], ids=["add", "sub", "mul"]
)
def test_arithmetic(dtype, op):
    check(dtype, op)


@pytest.mark.parametrize("dtype", NUMERIC)
def test_floor_division_and_modulo_follow_numpy(dtype):
    # numpy: integers give 0 for a zero divisor, MIN // -1 wraps. Floats are
    # floor(x / y) and x - floor(x / y) * y (like PyTensor's C code), which differs
    # from numpy for infinite operands.
    def skip(x, y):
        return dtype.startswith("float") and not (np.isfinite(x) and np.isfinite(y))

    check(dtype, operator.floordiv, skip=skip)
    check(dtype, operator.mod, skip=skip)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_true_division(dtype):
    check(dtype, operator.truediv)
    check(dtype, lambda x, y: 1.0 / y)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize(
    "op",
    [operator.lt, operator.le, operator.gt, operator.ge],
    ids=["lt", "le", "gt", "ge"],
)
def test_comparisons(dtype, op):
    check(dtype, op, out="bool")


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_eq_ne_minimum_maximum(dtype):
    check(dtype, lambda x, y: x.eq(y), operator.eq, out="bool")
    check(dtype, lambda x, y: x.ne(y), operator.ne, out="bool")
    check(dtype, lambda x, y: x.maximum(y), np.maximum)
    check(dtype, lambda x, y: x.minimum(y), np.minimum)


@pytest.mark.parametrize("dtype", [*INTS, "bool"])
def test_bits(dtype):
    check(dtype, operator.and_)
    check(dtype, operator.or_)
    check(dtype, operator.xor)
    check(dtype, lambda x, y: ~x & y)


def test_booleans_add_and_multiply_like_numpy():
    check("bool", operator.add, np.logical_or)
    check("bool", operator.mul, np.logical_and)


@pytest.mark.parametrize("dtype", NUMERIC)
def test_unary_select_and_literals(dtype):
    signed = not dtype.startswith("uint")
    check(dtype, lambda x, y: abs(x))
    if signed:
        check(dtype, lambda x, y: -x)
    check(
        dtype,
        lambda x, y: (x < y).select(x, y + 1),
        lambda x, y: np.where(x < y, x, y + type(y)(1)),
    )
    check(
        dtype,
        lambda x, y: (x > 3).select(1, y),
        lambda x, y: np.where(x > 3, type(y)(1), y),
    )
    check(
        dtype, lambda x, y: 2 * x - 1 + y, lambda x, y: type(x)(2) * x - type(x)(1) + y
    )
    check(dtype, lambda x, y: 3 - x, lambda x, y: type(x)(3) - x)


@pytest.mark.parametrize("src", SUPPORTED_DTYPES)
@pytest.mark.parametrize("dst", SUPPORTED_DTYPES)
def test_astype(src, dst):
    # (float -> integer is only defined for values that fit, and never for NaN)
    def skip(x, y):
        if not src.startswith("float") or dst.startswith("float") or dst == "bool":
            return False
        return not (np.isfinite(x) and np.iinfo(dst).min <= x <= np.iinfo(dst).max)

    check(src, lambda x, y: x.astype(dst), lambda x, y: x, out=dst, skip=skip)


@pytest.mark.parametrize("src", ["float64", "float32"])
@pytest.mark.parametrize("dst", INTS)
def test_float_to_int_saturates(src, dst):
    info = np.iinfo(dst)
    cf = compile_binary(src, lambda x, y: x.astype(dst), out=dst)
    values = [0.0, -0.5, 0.9, -1.0, 127.9, 200.5, 300.0, -300.0, 4e4, -4e4, 3e9, -3e9]
    values += [1e30, -1e30, np.inf, -np.inf, np.nan, info.min, info.max]
    for v in values:
        x = np.dtype(src).type(v).item()
        # out of range (or infinite) saturates, NaN gives 0, the rest truncates
        want = (
            0
            if np.isnan(x)
            else info.max
            if x == np.inf
            else info.min
            if x == -np.inf
            else int(x)
        )
        assert cf(x, x) == min(max(want, info.min), info.max), (src, dst, x)


def test_mistakes_raise():
    m = ir.Module("num")
    f = m.function("f", [ir.i64, ir.f64, ir.i32], ir.i64)
    b = f.builder()
    i, x, j = (
        Num(b, f.params[0], "int64"),
        Num(b, f.params[1], "float64"),
        Num(b, f.params[2], "int32"),
    )
    with pytest.raises(TypeError, match="convert one with astype"):
        i + j
    with pytest.raises(TypeError, match="the float 1.5"):
        i + 1.5
    with pytest.raises(TypeError, match="not a Python condition"):
        if i < 3:
            pass
    with pytest.raises(TypeError, match="wrap IR values"):
        i + f.params[0]
    with pytest.raises(TypeError, match="compare it first"):
        i.select(i, i)
    with pytest.raises(NotImplementedError, match="not defined for int64"):
        i / 2
    with pytest.raises(NotImplementedError, match="not defined for float64"):
        x & x
    # `==` is identity, so Nums can be dictionary keys
    assert (i == i) is True and (i == Num(b, f.params[0], "int64")) is False
    assert {i: 1}[i] == 1


def test_constants_stay_constants():
    m = ir.Module("num")
    b = m.function("f", [ir.i64], ir.i64).builder()
    n = Num.const(b, "int64", 6)
    assert (
        (n * 2 + 1).const_value == 13
        and (n // 4).const_value == 1
        and (n % 4).const_value == 2
    )
    assert (n < 7).const_value is True and (n < 7).select(n, 0).const_value == 6
    assert (
        Num.const(b, "uint8", 255).const_value == 255
        and Num.const(b, "float64", 0.5).const_value == 0.5
    )
