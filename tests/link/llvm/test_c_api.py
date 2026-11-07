"""Calling compiled functions through the C API, from C."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.link.llvm.linker import LLVMLinker


native = pytest.importorskip("pytensor_llvm")
LLVM = Mode(linker=LLVMLinker(), optimizer="fast_run")


class PtArray(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("owner", ctypes.c_void_p),
        ("ndim", ctypes.c_int64),
    ]


class PtTypeInfo(ctypes.Structure):
    _fields_ = [("dtype", ctypes.c_int32), ("ndim", ctypes.c_int32), ("static_shape", ctypes.POINTER(ctypes.c_int64))]


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("capi")
    (d / "pytensor_llvm.h").write_text(native.c_header())
    lib = d / "libharness.so"
    src = Path(__file__).with_name("c_harness.c")
    subprocess.run([cc, "-O2", "-shared", "-fPIC", f"-I{d}", str(src), "-o", str(lib)], check=True)
    h = ctypes.CDLL(str(lib))
    h.pt_run_once.restype = ctypes.c_int
    h.pt_run_once.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(PtArray), ctypes.POINTER(PtArray), ctypes.c_char_p, ctypes.c_size_t]
    h.pt_bench.restype = ctypes.c_double
    h.pt_bench.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(PtArray), ctypes.c_int, ctypes.c_long, ctypes.c_char_p, ctypes.c_size_t]
    return h


def as_ptarrays(arrays):
    """PtArray views of numpy arrays (keeps shape/stride buffers alive)."""
    keep = []
    arr = (PtArray * max(len(arrays), 1))()
    for i, a in enumerate(arrays):
        sh = (ctypes.c_int64 * max(a.ndim, 1))(*a.shape)
        st = (ctypes.c_int64 * max(a.ndim, 1))(*a.strides)
        keep += [a, sh, st]
        arr[i] = PtArray(a.ctypes.data, sh, st, None, a.ndim)
    return arr, keep


class Api(ctypes.Structure):
    _fields_ = [("abi_version", ctypes.c_uint32), ("api_version", ctypes.c_uint32)] + [
        (name, ctypes.c_void_p)
        for name in [
            "function_incref", "function_decref", "n_inputs", "n_outputs", "input_type", "output_type",
            "state_new", "state_free", "call", "last_error", "buffer_decref",
        ]
    ]


def api():
    return Api.from_address(native.c_api_table())


def _model():
    mu = pt.dscalar("mu")
    log_sigma = pt.dscalar("log_sigma")
    data = pt.vector("data")
    sigma = pt.exp(log_sigma)
    logp = (-0.5 * ((data - mu) / sigma) ** 2 - pt.log(sigma)).sum() - 0.5 * mu**2 + log_sigma
    grads = pytensor.grad(logp, [mu, log_sigma])
    f = pytensor.function([mu, log_sigma, data], [logp, *grads, data * 2], mode=LLVM)
    vals = [np.array(0.3), np.array(-0.2), np.random.default_rng(0).normal(size=200)]
    return f, vals


def test_api_table_and_introspection():
    a = api()
    assert a.abi_version == native.ABI_VERSION and a.api_version == 1
    f, _ = _model()
    fn = f.maker.linker.compiled.c_function()
    n_in = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)(a.n_inputs)
    n_out = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)(a.n_outputs)
    in_type = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(PtTypeInfo))(a.input_type)
    assert n_in(fn) == 3 and n_out(fn) == 4
    t = PtTypeInfo()
    assert in_type(fn, 2, ctypes.byref(t)) == 0
    assert (t.dtype, t.ndim, t.static_shape[0]) == (0, 1, -1)
    assert in_type(fn, 3, ctypes.byref(t)) != 0
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(a.function_decref)(fn)


@pytest.mark.parametrize("provided", [True, False])
def test_call_from_c(harness, provided):
    f, vals = _model()
    ref = f(*vals)
    fn = f.maker.linker.compiled.c_function()
    a = api()
    ins, keep = as_ptarrays(vals)
    if provided:
        outs_np = [np.empty_like(r) for r in ref]
        outs, keep2 = as_ptarrays(outs_np)
    else:
        outs = (PtArray * len(ref))()
        keep2 = []
        for k, r in enumerate(ref):
            sh, st = (ctypes.c_int64 * 4)(), (ctypes.c_int64 * 4)()
            keep2 += [sh, st]
            outs[k] = PtArray(None, sh, st, None, r.ndim)
    err = ctypes.create_string_buffer(512)
    assert harness.pt_run_once(ctypes.addressof(a), fn, ins, outs, err, 512) == 0, err.value
    decref = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(a.buffer_decref)
    for k, r in enumerate(ref):
        if provided:
            got = outs_np[k]
            assert outs[k].owner is None
        else:
            o = outs[k]
            shape = tuple(o.shape[d] for d in range(o.ndim))
            strides = tuple(o.strides[d] for d in range(o.ndim))
            buf = (ctypes.c_char * (int(np.prod(shape)) * 8 or 8)).from_address(o.data)
            got = np.ndarray(shape, dtype=r.dtype, buffer=buf, strides=strides).copy()
            assert o.owner is not None
            decref(o.owner)
        np.testing.assert_allclose(got, r, rtol=1e-13)
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(a.function_decref)(fn)


def test_c_errors_are_reported(harness):
    x = pt.vector("x")
    y = pt.vector("y")
    f = pytensor.function([x, y], x + y, mode=LLVM)
    fn = f.maker.linker.compiled.c_function()
    a = api()
    ins, keep = as_ptarrays([np.zeros(3), np.zeros(4)])
    out_np = np.empty(3)
    outs, keep2 = as_ptarrays([out_np])
    err = ctypes.create_string_buffer(512)
    assert harness.pt_run_once(ctypes.addressof(a), fn, ins, outs, err, 512) != 0
    assert b"mismatch" in err.value
    # wrong provided output shape
    ins, keep = as_ptarrays([np.zeros(3), np.zeros(3)])
    outs, keep2 = as_ptarrays([np.empty(5)])
    assert harness.pt_run_once(ctypes.addressof(a), fn, ins, outs, err, 512) != 0
    assert b"provided output" in err.value
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(a.function_decref)(fn)


def test_function_outlives_python_object(harness):
    f, vals = _model()
    ref = f(*vals)
    fn = f.maker.linker.compiled.c_function()
    del f
    import gc

    gc.collect()
    a = api()
    ins, keep = as_ptarrays(vals)
    outs_np = [np.empty_like(r) for r in ref]
    outs, keep2 = as_ptarrays(outs_np)
    err = ctypes.create_string_buffer(512)
    assert harness.pt_run_once(ctypes.addressof(a), fn, ins, outs, err, 512) == 0
    np.testing.assert_allclose(outs_np[0], ref[0])
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(a.function_decref)(fn)


def test_bench_runs(harness):
    f, vals = _model()
    fn = f.maker.linker.compiled.c_function()
    a = api()
    ins, keep = as_ptarrays(vals)
    err = ctypes.create_string_buffer(512)
    for provided in (1, 0):
        ns = harness.pt_bench(ctypes.addressof(a), fn, ins, provided, 1000, err, 512)
        assert ns > 0, err.value
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(a.function_decref)(fn)
