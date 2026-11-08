"""Tests for the runtime model: output modes, layouts, ownership and the builder API."""

import gc

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.link.llvm.linker import LLVMLinker


native = pytest.importorskip("pytensor_llvm")

LLVM = Mode(linker=LLVMLinker(), optimizer="fast_run")
rng = np.random.default_rng(1)


def compiled(f):
    return f.maker.linker.compiled


def plan(f):
    return {name.split(" ")[1].strip("()") + ":" + name.split(" ")[0]: (st, lay) for name, st, lay in compiled(f).plan()}


# ------------------------------------------------------------------ layouts


def test_fortran_inputs_take_fast_path_and_are_correct():
    x = pt.matrix("x")
    y = pt.matrix("y")
    f = pytensor.function([x, y], pt.exp(x) * y + 1, mode=LLVM)
    for make in (np.ascontiguousarray, np.asfortranarray):
        a, b = make(rng.normal(size=(30, 40))), make(rng.normal(size=(30, 40)))
        np.testing.assert_allclose(f(a, b), np.exp(a) * b + 1)
    # mixed layouts take the strided path
    a, b = np.asfortranarray(rng.normal(size=(30, 40))), rng.normal(size=(30, 40))
    np.testing.assert_allclose(f(a, b), np.exp(a) * b + 1)


def test_planner_chooses_fortran_output_for_transposed_intermediates():
    x = pt.matrix("x")
    y = pt.matrix("y")
    # the transpose can't be lifted through Sort, so the Elemwise sees an F view of a
    # C intermediate and the planner gives it an F output
    t = pt.sort(x, axis=1)
    u = pt.exp(t.T) * 2 + 1
    out = u.sum(axis=1) + pt.log1p(y**2).sum(axis=0)
    f = pytensor.function([x, y], [out, u], mode=LLVM)
    layouts = [lay for name, st, lay in compiled(f).plan() if "Elemwise" in name]
    assert "F" in layouts
    a, b = rng.normal(size=(5, 7)), rng.normal(size=(5, 7))
    ref_u = np.exp(np.sort(a, axis=1).T) * 2 + 1
    r_out, r_u = f(a, b)
    np.testing.assert_allclose(r_out, ref_u.sum(1) + np.log1p(b**2).sum(0))
    np.testing.assert_allclose(r_u, ref_u)
    assert r_u.flags.f_contiguous  # returned zero-copy in the layout the planner chose


def test_reduce_over_fortran_input():
    x = pt.tensor3("x")
    f = pytensor.function([x], [x.sum(axis=0), x.sum(axis=(1, 2)), x.max(axis=2)], mode=LLVM)
    v = np.asfortranarray(rng.normal(size=(4, 5, 6)))
    r = f(v)
    np.testing.assert_allclose(r[0], v.sum(0))
    np.testing.assert_allclose(r[1], v.sum((1, 2)))
    np.testing.assert_allclose(r[2], v.max(2))


# ------------------------------------------------------------------ output modes


def test_arange_kernel_allocated_and_feeds_kernels():
    n = pt.scalar("n", dtype="int64")
    x = pt.vector("x")
    r = pt.arange(n)
    out = (pt.cast(r, "float64") * 2.0).sum() + x.sum()
    f = pytensor.function([n, x], [r, out, pt.arange(0.5, n, 0.25)], mode=LLVM)
    for nv in [0, 1, 7, 1000]:
        a, b, c = f(np.int64(nv), np.ones(3))
        np.testing.assert_array_equal(a, np.arange(nv))
        assert b == pytest.approx(2 * np.arange(nv).sum() + 3)
        np.testing.assert_allclose(c, np.arange(0.5, nv, 0.25))
    st = {s for _, s, _ in compiled(f).plan()}
    assert "kernel-allocated" in st


def test_arange_negative_step_and_errors():
    a, b, s = pt.lscalars("a", "b", "s")
    f = pytensor.function([a, b, s], pt.arange(a, b, s), mode=LLVM)
    np.testing.assert_array_equal(f(10, 0, -3), np.arange(10, 0, -3))
    np.testing.assert_array_equal(f(0, 10, -3), np.arange(0, 10, -3))
    with pytest.raises(Exception, match="step"):
        f(0, 10, 0)
    # the function is still usable after a kernel error
    np.testing.assert_array_equal(f(0, 5, 2), np.arange(0, 5, 2))


def test_alloc_and_inplace_take_over():
    x = pt.vector("x")
    n = pt.lscalar("n")
    a = pt.alloc(x, n, x.shape[0])  # kernel-allocated, shape from values
    out = pt.exp(a) + 1.0  # can take over the allocated buffer in place
    f = pytensor.function([x, n], out.sum(axis=0), mode=LLVM)
    xv = rng.normal(size=4)
    np.testing.assert_allclose(f(xv, 3), (np.exp(np.broadcast_to(xv, (3, 4))) + 1).sum(0))
    storages = [s for _, s, _ in compiled(f).plan()]
    if "kernel-allocated" in storages:
        assert any(s.startswith("in-place, takes") for s in storages)


def test_alloc_shape_error():
    x = pt.vector("x")
    n = pt.lscalar("n")
    f = pytensor.function([x, n], pt.alloc(x, n, 3) * 2, mode=LLVM)
    with pytest.raises(Exception, match="broadcast"):
        f(np.ones(4), 2)
    np.testing.assert_allclose(f(np.ones(3), 2), np.full((2, 3), 2.0))


def test_view_outputs_are_copied_and_inputs_untouched():
    x = pt.matrix("x")
    f = pytensor.function([x], [x.T, pt.exp(x).T, x], mode=LLVM)
    v = rng.normal(size=(3, 4))
    v0 = v.copy()
    a, b, c = f(v)
    np.testing.assert_allclose(a, v.T)
    np.testing.assert_allclose(b, np.exp(v).T)
    assert not np.shares_memory(a, v) and not np.shares_memory(c, v)
    np.testing.assert_array_equal(v, v0)


def test_outputs_are_zero_copy_and_independent_across_calls():
    x = pt.vector("x")
    f = pytensor.function([x], pt.exp(x), mode=LLVM)
    a = f(np.zeros(1000))
    b = f(np.ones(1000))
    assert type(a.base).__name__ == "PyCapsule"
    np.testing.assert_allclose(a, 1.0)
    np.testing.assert_allclose(b, np.e)
    a[:] = 5.0  # writable, owned by the caller
    del b
    gc.collect()
    np.testing.assert_allclose(a, 5.0)


def test_assert_is_a_view_and_raises():
    x = pt.vector("x")
    y = pt.vector("y")
    f = pytensor.function([x, y], x + y, mode=LLVM)
    with pytest.raises(Exception, match="mismatch|broadcast"):
        f(np.zeros(3), np.zeros(4))
    np.testing.assert_allclose(f(np.ones(3), np.ones(3)), 2.0)


# ------------------------------------------------------------------ builder API


def test_builder_api_directly():
    sg = native.ScalarGraph(["float64", "float64"])
    e = sg.apply("exp", [sg.input(1)], "float64")
    s = sg.apply("add", [sg.input(0), e, sg.constant(1.0, "float64")], "float64")
    sg.set_outputs([s])

    g = native.GraphBuilder()
    x = g.variable("float64", [False])
    y = g.variable("float64", [False])
    g.add_input(x)
    g.add_input(y)
    z = g.variable("float64", [False])
    g.elemwise(sg, [x, y], [z])
    t = g.variable("float64", [])
    g.reduce("add", [0], z, t)
    g.add_output(z)
    g.add_output(t)
    fn = g.compile()
    a, b = rng.normal(size=10), rng.normal(size=10)
    zr, tr = fn(a, b)
    np.testing.assert_allclose(zr, a + np.exp(b) + 1)
    assert tr == pytest.approx((a + np.exp(b) + 1).sum())


def test_builder_validates_eagerly():
    g = native.GraphBuilder()
    x = g.variable("float64", [False])
    y = g.variable("float64", [False, False])
    sg = native.ScalarGraph(["float64"])
    sg.set_outputs([sg.apply("exp", [sg.input(0)], "float64")])
    with pytest.raises(ValueError, match="not defined"):
        g.elemwise(sg, [x], [g.variable("float64", [False])])
    g.add_input(x)
    with pytest.raises(ValueError, match="dims"):
        g.elemwise(sg, [x], [y])
    with pytest.raises(NotImplementedError, match="scalar op"):
        sg.apply("frobnicate", [sg.input(0)], "float64")
    with pytest.raises(ValueError, match="takes"):
        sg.apply("exp", [sg.input(0), sg.input(0)], "float64")
    with pytest.raises(NotImplementedError, match="dtype"):
        g.variable("complex128", [])


# ------------------------------------------------------------------ segments


def _modes():
    return {
        seg: Mode(linker=LLVMLinker(segments=seg), optimizer="fast_run") for seg in (False, True)
    }


def _segments(f):
    return [d for d in compiled(f).info()["schedule"] if d.startswith("Segment")]


def test_segment_small_chain_matches_unsegmented():
    # reductions between the elemwise ops keep fusion from merging them into one node
    a = pt.dscalar("a")
    v = pt.vector("v", shape=(3,))
    x = pt.vector("x")
    s = pt.exp(v).sum()
    t = pt.sigmoid(s) * a
    u = (t * v - pt.log1p(v**2)).max()
    outs = [t, u, x * u]
    vals = [np.array(0.3), rng.normal(size=3), rng.normal(size=50)]
    res = {}
    for seg, mode in _modes().items():
        f = pytensor.function([a, v, x], outs, mode=mode)
        res[seg] = f(*vals)
        if seg:
            assert _segments(f)
            n_seg = len(compiled(f).info()["schedule"])
        else:
            n_plain = len(compiled(f).info()["schedule"])
    assert n_seg < n_plain
    for r0, r1 in zip(res[False], res[True], strict=True):
        np.testing.assert_allclose(r0, r1, rtol=1e-13)


def test_segment_assert_reports_member_message():
    from pytensor.raise_op import Assert

    a = pt.dscalar("a")
    checked = Assert("a must be positive")(a, a > 0)
    out = pt.exp(checked) * 2 + pt.log(checked)
    f = pytensor.function([a], out, mode=_modes()[True])
    assert any("Assert" in d for d in _segments(f))
    with pytest.raises(Exception, match="a must be positive"):
        f(np.array(-1.0))
    np.testing.assert_allclose(f(np.array(1.0)), np.exp(1.0) * 2)


def test_segment_escaping_views_and_outputs():
    a, b = pt.dscalars("a", "b")
    v = pt.vector("v")
    s = pt.exp(a) + b  # 0-d, escapes as an output
    r = s.dimshuffle("x")  # view, used inside and as an output
    w = r * 2 + pt.log(b**2 + 1)  # small elemwise on the view
    outs = [s, r, w, v * r]
    vals = [np.array(0.5), np.array(-2.0), rng.normal(size=10)]
    res = {seg: pytensor.function([a, b, v], outs, mode=m)(*vals) for seg, m in _modes().items()}
    for r0, r1 in zip(res[False], res[True], strict=True):
        assert r0.shape == r1.shape
        np.testing.assert_allclose(r0, r1, rtol=1e-13)


def test_segment_threshold_and_disable():
    x = pt.vector("x", shape=(3,))
    y = pt.vector("y", shape=(3,))
    out = (pt.exp(x) * y).sum() + pt.log1p(x**2).sum()
    vals = [rng.normal(size=3), rng.normal(size=3)]
    on = pytensor.function([x, y], out, mode=Mode(linker=LLVMLinker(segment_max_elems=64), optimizer="fast_run"))
    off = pytensor.function([x, y], out, mode=Mode(linker=LLVMLinker(segment_max_elems=2), optimizer="fast_run"))
    assert _segments(on) and not _segments(off)
    np.testing.assert_allclose(on(*vals), off(*vals), rtol=1e-13)


def test_perf_map_written_when_enabled(tmp_path):
    # PYTENSOR_ENABLE_PERF is read once per process, so check in a fresh one
    import os
    import subprocess
    import sys

    code = """
import os, numpy as np, pytensor, pytensor.tensor as pt
x = pt.vector("x")
f = pytensor.function([x], (pt.exp(x) * 2).sum(), mode="LLVM")
f(np.ones(3))
print(open(f"/tmp/perf-{os.getpid()}.map").read())
os.unlink(f"/tmp/perf-{os.getpid()}.map")
"""
    env = dict(os.environ, PYTENSOR_ENABLE_PERF="1", PTLLVM_CACHE_DIR=str(tmp_path))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
    lines = [line.split(" ", 2) for line in out.splitlines() if line.strip()]
    assert lines
    for start, size, name in lines:
        assert int(start, 16) > 0 and int(size, 16) > 0
        assert name.startswith("ptk_")
    assert any("[" in name for *_, name in lines)  # entry points carry the kernel description
    # frame pointers are part of the cache key: profiling objects are separate files
    assert all(p.suffix == ".o" for p in tmp_path.iterdir())


def test_small_graph_with_slices_join_alloc_is_one_segment():
    # nutpie-style: slice a joined parameter vector, compute, join the gradients
    import pytensor
    import pytensor.tensor as pt

    x = pt.vector("x", shape=(10,))
    mu, ltau, z = x[0], x[1], x[2:]
    z2 = x[::-3]  # negative step
    logp = -0.5 * mu**2 + ltau - 0.5 * (z**2).sum() - pt.exp(ltau) * (z2 * z[:4]).sum()
    grad = pytensor.grad(logp, x)
    out = pt.join(0, grad[:3], pt.alloc(logp, 2), grad[3:])
    f = pytensor.function([x], [logp, out], mode="LLVM")
    ref = pytensor.function([x], [logp, out], mode="FAST_COMPILE")
    xv = np.random.default_rng(0).normal(size=10)
    for r, e in zip(f(xv), ref(xv), strict=True):
        np.testing.assert_allclose(r, e, rtol=1e-12)
    sched = f.maker.linker.compiled.info()["schedule"]
    assert len([d for d in sched if not d.startswith("view")]) == 1, sched


def test_static_inc_and_set_subtensor_in_segments():
    import pytensor
    import pytensor.tensor as pt

    x = pt.matrix("x", shape=(4, 3))
    y = pt.vector("y", shape=(3,))
    outs = [
        pt.inc_subtensor(x[1], y),
        pt.set_subtensor(x[::2, 1:], y[:2]),  # broadcast over rows
        pt.inc_subtensor(x[-1, ::-1], y * 2) + 1,
        pt.set_subtensor(x[:, 0], 5.0),
    ]
    f = pytensor.function([x, y], outs, mode="LLVM")
    ref = pytensor.function([x, y], outs, mode="FAST_COMPILE")
    vals = [np.arange(12.0).reshape(4, 3), np.array([10.0, 20.0, 30.0])]
    for r, e in zip(f(*vals), ref(*vals), strict=True):
        np.testing.assert_allclose(r, e)
    sched = f.maker.linker.compiled.info()["schedule"]
    assert not [d for d in sched if "inc_subtensor" in d], sched


def test_shape_overflow_is_an_error_not_a_crash():
    import pytensor
    import pytensor.tensor as pt

    n = pt.lscalar("n")
    f = pytensor.function([n], pt.alloc(np.float64(1.0), n, n), mode="LLVM")
    with pytest.raises(Exception, match="too big"):
        f(2**32)  # n*n wraps to 0 in int64
    with pytest.raises(Exception, match="negative"):
        f(-1)
    assert f(3).shape == (3, 3)


def test_concurrent_calls_from_threads():
    # used to deadlock: the call state's lock was held while the GIL was released
    import threading

    import pytensor
    import pytensor.tensor as pt

    x = pt.vector("x")
    f = pytensor.function([x], pt.exp(x).sum(), mode="LLVM")
    xv = np.ones(100_000)
    results = []

    def work():
        results.extend(float(f(xv)) for _ in range(20))

    ts = [threading.Thread(target=work) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in ts), "deadlock"
    np.testing.assert_allclose(results, np.e * 100_000)
    assert len(results) == 80


def test_byte_swapped_and_misaligned_inputs():
    import pytensor
    import pytensor.tensor as pt

    x = pt.vector("x")
    f = pytensor.function([x], x * 2, mode="LLVM", trust_input=True)
    be = np.arange(4.0).astype(">f8")
    np.testing.assert_allclose(f(be), [0, 2, 4, 6])
    raw = np.zeros(4 * 8 + 1, dtype=np.uint8)
    mis = raw[1:].view(np.float64)
    mis[:] = 1.5
    np.testing.assert_allclose(f.maker.linker.compiled(mis)[0], 3.0)


def test_checked_mode_runs_clean(tmp_path):
    # PTLLVM_CHECKED is read once per process: run a mix of kernels in a fresh one
    import os
    import subprocess
    import sys

    code = """
import numpy as np, pytensor, pytensor.tensor as pt
x = pt.matrix("x"); i = pt.lvector("i")
y = pt.exp(x[i]).sum(0) @ x.T + pt.cumsum(x, 0)[-1].sum()
outs = [y, pytensor.grad(y.sum(), x), pt.sort(x, axis=1), pt.linalg.cholesky(x.T @ x + 3 * pt.eye(4))]
f = pytensor.function([x, i], outs, mode="LLVM")
v = np.random.default_rng(0).normal(size=(5, 4)); iv = np.array([0, 4, 2, 2])
r = f(v, iv)
ref = pytensor.function([x, i], outs, mode="FAST_COMPILE")(v, iv)
assert all(np.allclose(a, b) for a, b in zip(r, ref))
print("ok")
"""
    env = dict(os.environ, PTLLVM_CHECKED="1", PTLLVM_CACHE_DIR=str(tmp_path))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-2000:]
