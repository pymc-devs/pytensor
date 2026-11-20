"""GammaLn and Psi for float32 and float64, ported from the Numba kernels
(`pytensor.link.numba.dispatch.scalar._gamma_kernels`).

For 0 < x: [1, 64) is a table lookup and a polynomial, with no libm call; below 1,
lgamma(x + 1) - log(x) and psi(x + 1) - 1 / x; from 64, Stirling's series. float32 has
its own shorter tables and computes in float32. The code is built with only `contract`
(`b.strict_fp`): reassociation would drop x0's low part. x <= 0 and NaN go to the Rust
functions.
"""

from __future__ import annotations

import numpy as np

from pytensor.link.llvm.codegen.num import Num
from pytensor.link.llvm.codegen.types import ir, value_type
from pytensor.scalar import _gamma_tables as gt


def _tables(dtype):
    if dtype == "float32":
        return gt.LGAMMA32, gt.PSI32, gt.X0_HI32, gt.X0_LO32, gt.STIRLING32
    return gt.LGAMMA, gt.PSI, gt.X0_HI, gt.X0_LO, gt.STIRLING


def _horner(coefs, s):
    """sum_k coefs[k] s^k; `coefs` are Nums or Python numbers (the top one a Num or
    `s` decides the dtype)."""
    p = coefs[-1] if isinstance(coefs[-1], Num) else Num.const(s.b, s.dtype, coefs[-1])
    for c in coefs[-2::-1]:
        p = p * s + c
    return p


def _panel_series(b, table, y):
    """The Chebyshev panel of y in [1, 64) of `table` (32 panels an octave)."""
    dtype, t = y.dtype, value_type(y.dtype)
    ncols = table.shape[1]

    def load(ptr, i):
        return Num(b, b.load(t, b.gep(t, ptr, i.value)), dtype)

    e = (y >= 2.0).astype("int64")
    for edge in (4.0, 8.0, 16.0, 32.0):
        e = e + (y >= edge).astype("int64")
    inv_pow2 = b.const_array(t, [2.0**-k for k in range(6)])
    u = (y * load(inv_pow2, e) - 1.0) * 32.0
    m = Num(b, b.fptosi(u.value, ir.i64), "int64")
    s = (u - m.astype(dtype)) * 2.0 - 1.0
    ptr = b.const_array(t, table.ravel().tolist())
    base = (e * 32 + m) * ncols
    return _horner([load(ptr, base + k) for k in range(ncols)], s)


def _positive_or_rust(b, x, which, large, table):
    """large(x) for x >= 64, table(x) for 0 < x < 64, else the Rust function."""

    def small():
        return b.if_else(
            (x > 0.0).value,
            lambda: table(x).value,
            lambda: b.rust_math(which, x.value),
        )

    (r,) = b.if_else((x >= 64.0).value, lambda: large(x).value, small)
    return r


def _strict(fn):
    def wrapper(b, x, dtype):
        prev = b.strict_fp(True)
        try:
            return fn(b, Num(b, x, dtype), dtype)
        finally:
            b.strict_fp(prev)

    return wrapper


def _log(x):
    return Num(x.b, x.b.libm(ir.Libm.Log, [x.value]), x.dtype)


@_strict
def gammaln(b, x, dtype):
    lgamma_table, _, _, _, stirling = _tables(dtype)

    def large(x):
        inv = 1.0 / x
        series = inv * _horner([float(c) for c in stirling], inv * inv)
        return (x - 0.5) * (_log(x) - 1.0) + float(gt.STIRLING_C) + series

    def table(x):
        ge1 = x >= 1.0
        # the table holds lgamma / ((x - 1) (x - 2)) on [1, 4)
        f = _panel_series(b, lgamma_table, ge1.select(x, x + 1.0))

        def above_one():
            return (x >= 4.0).select(f, (x - 1.0) * (x - 2.0) * f).value

        def below_one():
            return (x * (x - 1.0) * f - _log(x)).value

        (r,) = b.if_else(ge1.value, above_one, below_one)
        return Num(b, r, dtype)

    return _positive_or_rust(b, x, ir.RustMath.GammaLn, large, table)


@_strict
def psi(b, x, dtype):
    _, psi_table, x0_hi, x0_lo, stirling = _tables(dtype)
    # d/dx sum_k A_k x^-(2k + 1) = -x^-2 sum_k (2k + 1) A_k x^-2k
    dstirling = (stirling * np.arange(1, 2 * stirling.size, 2)).astype(dtype)

    def large(x):
        inv = 1.0 / x
        p = inv * inv
        return _log(x) - inv * 0.5 - p * _horner([float(c) for c in dstirling], p)

    def table(x):
        ge1 = x >= 1.0
        y = ge1.select(x, x + 1.0)
        r = _panel_series(b, psi_table, y)
        # r (y - x0) with x0 in two parts; y - x0_hi is exact
        r = (y < 2.0).select(r * (y - float(x0_hi)) - r * float(x0_lo), r)
        return ge1.select(r, r - 1.0 / x)

    return _positive_or_rust(b, x, ir.RustMath.Psi, large, table)
