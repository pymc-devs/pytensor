"""Ops without an LLVM implementation run through their numba implementation."""

import ctypes

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.link.llvm.linker import LLVMLinker


native = pytest.importorskip("pytensor_llvm")
pytest.importorskip("numba")

LLVM = Mode(linker=LLVMLinker(), optimizer="fast_run")
rng = np.random.default_rng(3)


def compiled(f):
    return f.maker.linker.compiled


def foreign_steps(f):
    return [d for d in compiled(f).info()["schedule"] if d.startswith("foreign:")]


def test_fallback_ops_mixed_with_native_kernels():
    x = pt.matrix("x")
    s = pt.lvector("s")
    outs = [
        pt.repeat(x[:, 0], s[:4]) * 2,  # numba -> native Elemwise, in place on the numba result
        pt.repeat(x[0], s[:5]).sum(),  # numba -> native Reduce
        pt.argmax(x, axis=1),
        pt.argsort(pt.exp(x), axis=1),  # native Elemwise feeding a numba op
        pt.linalg.det(x[:4, :4]),
    ]
    f = pytensor.function([x, s], outs, mode=LLVM)
    assert len(foreign_steps(f)) >= 4
    v = rng.normal(size=(4, 5))
    sv = np.array([1, 0, 3, 2, 1])
    res = f(v, sv)
    ref = pytensor.function([x, s], outs, mode="FAST_COMPILE")(v, sv)
    for r, e in zip(res, ref, strict=True):
        np.testing.assert_allclose(r, e)
    # strided input into a numba op
    vt = rng.normal(size=(5, 4)).T
    np.testing.assert_allclose(f(vt, sv)[1], np.repeat(vt[0], sv).sum())


def test_fallback_scalar_inputs_and_outputs():
    x = pt.vector("x")
    k = pt.lscalar("k")
    f = pytensor.function([x, k], [x[k], x[k:] * x[k]], mode=LLVM)
    v = rng.normal(size=6)
    a, b = f(v, 2)
    assert a == pytest.approx(v[2])
    np.testing.assert_allclose(b, v[2:] * v[2])


def test_fallback_exception_is_reported():
    x = pt.vector("x")
    s = pt.lvector("s")
    f = pytensor.function([x, s], pt.repeat(x, s) * 2, mode=LLVM)
    assert foreign_steps(f)
    np.testing.assert_allclose(f(np.arange(3.0), np.array([1, 0, 2])), [0.0, 4.0, 4.0])
    with pytest.raises(Exception, match="Repeat.*numba"):
        f(np.arange(3.0), np.array([1, 2]))
    np.testing.assert_allclose(f(np.arange(2.0), np.array([0, 2])), [2.0, 2.0])


def test_fallback_can_be_disabled():
    x = pt.matrix("x")
    mode = Mode(linker=LLVMLinker(numba_fallback=False), optimizer="fast_run")
    with pytest.raises(NotImplementedError, match="No LLVM"):
        pytensor.function([x], pt.argmax(x, axis=0), mode=mode)


def test_fallback_through_c_api_without_gil():
    from tests.link.llvm.test_c_api import PtArray, api, as_ptarrays  # noqa: F401

    x = pt.matrix("x")
    out = (pt.argsort(x, axis=1) * 2).sum(axis=0)
    f = pytensor.function([x], out, mode=LLVM)
    assert foreign_steps(f)
    v = rng.normal(size=(30, 7))
    fn = compiled(f).c_function()
    a = api()
    call_t = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(PtArray), ctypes.POINTER(PtArray))
    new_t = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)
    free_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    # ctypes.CFUNCTYPE callables release the GIL while the native call runs
    st = new_t(a.state_new)(fn)
    ins, keep = as_ptarrays([v])
    res = np.empty(7, dtype="int64")
    outs, keep2 = as_ptarrays([res])
    assert call_t(a.call)(st, ins, outs) == 0
    np.testing.assert_allclose(res, (np.argsort(v, axis=1) * 2).sum(axis=0))
    free_t(a.state_free)(st)
    free_t(a.function_decref)(fn)
