"""Parallel loops: the builder's regions, the runtime's parallel_for, generated loops."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
import pytensor.tensor.random.basic as rb
from pytensor.compile.mode import get_mode


ir = pytest.importorskip("pytensor_llvm.ir")
pytensor_llvm = pytest.importorskip("pytensor_llvm")

MODE = get_mode("LLVM").clone(link_kwargs={"numba_fallback": False})


def test_region_captures_and_results():
    """A region filling out[i] = a * x[i] + b with captured a, b and pointers; chunks
    of 7 over 100 elements; the result is checked in the parent."""
    m = ir.Module("par")
    f = m.function("f", [ir.ptr, ir.ptr, ir.ptr, ir.ptr, ir.ptr, ir.ptr, ir.ptr], ir.i32, exported=True)
    b = f.builder()
    ctx, args = f.params[0], f.params[1]
    x = b.load(ir.ptr, b.gep_bytes(args, b.const_int(ir.i64, 0)))
    out = b.load(ir.ptr, b.gep_bytes(args, b.const_int(ir.i64, 40)))
    a = b.fmul(b.const_float(ir.f64, 1.5), b.const_float(ir.f64, 2.0))  # 3.0 (a value of f)
    n = b.const_int(ir.i64, 100)
    with b.parallel(ctx, n, b.const_int(ir.i64, 7)) as r:
        with b.loop(b.sub(r.end, r.start)) as j:
            i = b.add(r.start, j)
            v = b.load(ir.f64, b.gep(ir.f64, x, i))
            b.store(b.fadd(b.fmul(a, v), b.const_float(ir.f64, 1.0)), b.gep(ir.f64, out, i))
    ok = b.icmp(ir.ICmp.EQ, r.result, b.const_int(ir.i32, 0))
    b.ret(b.zext(b.xor(ok, b.const_bool(True)), ir.i32))
    c = m.compile(cache=False)
    xv = np.arange(100.0)
    (res,) = ir.call_kernel(c.address("f"), [xv], [np.empty(100)])
    np.testing.assert_allclose(res, 3.0 * xv + 1.0)


def test_values_of_the_body_dont_leak_out():
    m = ir.Module("leak")
    f = m.function("f", [ir.ptr], ir.i32)
    b = f.builder()
    with b.parallel(f.params[0], b.const_int(ir.i64, 10), b.const_int(ir.i64, 2)) as r:
        inside = b.add(r.start, r.end)
    with pytest.raises(ValueError, match="different function"):
        b.add(inside, inside)


def big(shape):
    return np.random.default_rng(0).normal(size=shape)


@pytest.mark.parametrize("static", [True, False])
def test_large_elemwise_and_reductions(static):
    shape = (3000, 400)
    x = pt.tensor("x", shape=shape if static else (None, None))
    outs = [pt.exp(x) * 2 + 1, pt.exp(x).sum(axis=1), (x * 3).T + 1]
    f = pytensor.function([x], outs, mode=MODE)
    xv = big(shape)
    res = f(xv)
    np.testing.assert_allclose(res[0], np.exp(xv) * 2 + 1)
    np.testing.assert_allclose(res[1], np.exp(xv).sum(axis=1))
    np.testing.assert_allclose(res[2], (xv * 3).T + 1)
    # the same without parallelism
    f.maker.linker.program.set_parallel(False)
    for r, e in zip(f(xv), res, strict=True):
        np.testing.assert_array_equal(r, e)


def test_small_loops_have_no_region():
    from pytensor.link.llvm.codegen.function import build_node_function

    small = pt.tensor("s", shape=(10, 10))
    large = pt.tensor("l", shape=(1000, 1000))
    for var, expected in [(small, False), (large, True), (pt.dmatrix("d"), True)]:
        m = ir.Module("t")
        build_node_function(m, pt.exp(var).owner, "node")
        assert ("ptl_rt_parallel_for" in m.ir()) is expected


def test_random_draws_dont_depend_on_parallelism():
    rng = pytensor.shared(np.random.default_rng(0))
    a = pt.dvector("a")
    x = rb.gamma(a, 1.0, size=(500, 2000), rng=rng)  # a per-element sampler in the loop
    f = pytensor.function([a], x, mode=MODE)
    p = f.maker.linker.program
    av = np.linspace(0.5, 5, 2000)
    p.seed(3)
    r1 = f(av)
    p.set_parallel(False)
    p.seed(3)
    r2 = f(av)
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_allclose((r1 / av).mean(), 1.0, rtol=0.02)  # gamma(a, rate 1) has mean a


def test_errors_in_a_chunk():
    rng = pytensor.shared(np.random.default_rng(0))
    a = pt.dvector("a")
    f = pytensor.function([a], rb.gamma(a, 1.0, size=(500, 2000), rng=rng), mode=MODE)
    av = np.ones(2000)
    av[1777] = -1.0
    with pytest.raises(Exception, match="shape <= 0"):
        f(av)
    f(np.ones(2000))  # still usable


def test_thread_count():
    assert pytensor_llvm.threads() >= 1
    with pytest.raises(RuntimeError, match="already has"):
        pytensor_llvm.set_threads(pytensor_llvm.threads() + 1)
