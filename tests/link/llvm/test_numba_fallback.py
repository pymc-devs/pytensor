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
    outs = [
        pt.cumprod(x, axis=0) * 2,  # numba -> native Elemwise, in place on the numba result
        pt.cumsum(x, axis=0).sum(),  # numba -> native Reduce
        pt.argmax(x, axis=1),
        pt.argsort(pt.exp(x), axis=1),  # native Elemwise feeding a numba op
        pt.diagonal(x).T,
    ]
    f = pytensor.function([x], outs, mode=LLVM)
    assert len(foreign_steps(f)) >= 4
    v = rng.normal(size=(4, 5))
    ref = [np.cumprod(v, 0) * 2, np.cumsum(v, 0).sum(), np.argmax(v, 1), np.argsort(np.exp(v), 1), np.diagonal(v)]
    for r, e in zip(f(v), ref, strict=True):
        np.testing.assert_allclose(r, e)
    # strided input into a numba op
    vt = rng.normal(size=(5, 4)).T
    np.testing.assert_allclose(f(vt)[1], np.cumsum(vt, 0).sum())


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
    a, b = pt.split(x, s, n_splits=2)
    f = pytensor.function([x, s], [a * 2, b], mode=LLVM)
    assert foreign_steps(f)
    r = f(np.arange(5.0), np.array([2, 3]))
    np.testing.assert_allclose(r[0], [0.0, 2.0])
    with pytest.raises(Exception, match="Split.*numba"):
        f(np.arange(5.0), np.array([2, 2]))
    np.testing.assert_allclose(f(np.arange(5.0), np.array([1, 4]))[1], [1.0, 2.0, 3.0, 4.0])


def test_fallback_can_be_disabled():
    x = pt.matrix("x")
    mode = Mode(linker=LLVMLinker(numba_fallback=False), optimizer="fast_run")
    with pytest.raises(NotImplementedError, match="CumOp|Cumsum|No LLVM"):
        pytensor.function([x], pt.cumsum(x, axis=0), mode=mode)


def test_fallback_through_c_api_without_gil():
    from tests.link.llvm.test_c_api import PtArray, api, as_ptarrays  # noqa: F401

    x = pt.matrix("x")
    out = (pt.cumsum(x, axis=1) * 2).sum(axis=0)
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
    res = np.empty(7)
    outs, keep2 = as_ptarrays([res])
    assert call_t(a.call)(st, ins, outs) == 0
    np.testing.assert_allclose(res, (np.cumsum(v, axis=1) * 2).sum(axis=0))
    free_t(a.state_free)(st)
    free_t(a.function_decref)(fn)
