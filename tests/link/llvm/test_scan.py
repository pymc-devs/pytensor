"""Native Scan: fast path (step compiled as one segment) and general path (inner program)."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.link.llvm.linker import LLVM_REWRITES, LLVMLinker
from pytensor.scan.utils import until


pytest.importorskip("pytensor_llvm")


def llvm(**kw):
    return Mode(
        linker=LLVMLinker(numba_fallback=False, **kw),
        optimizer=RewriteDatabaseQuery(include=["fast_run", *LLVM_REWRITES]),
    )


FAST = llvm()
GENERAL = llvm(segments=False)  # no segments -> every step runs as an inner program
REF = Mode(linker="py", optimizer="fast_run")
rng = np.random.default_rng(5)


def scan(*args, **kwargs):
    return pytensor.scan(*args, return_updates=False, **kwargs)


def scan_steps(f):
    return [d for d in f.maker.linker.compiled.info()["schedule"] if d.startswith("Scan")]


def compare(inputs, outputs, values, modes=(FAST, GENERAL)):
    ref = pytensor.function(inputs, outputs, mode=REF, on_unused_input="ignore")(*values)
    if not isinstance(ref, list):
        ref = [ref]
    fs = []
    for mode in modes:
        f = pytensor.function(inputs, outputs, mode=mode, on_unused_input="ignore")
        res = f(*values)
        if not isinstance(res, list):
            res = [res]
        for r, e in zip(res, ref, strict=True):
            assert r.shape == e.shape and r.dtype == e.dtype, (r.shape, e.shape, r.dtype, e.dtype)
            np.testing.assert_allclose(r, e, rtol=1e-12, atol=1e-14)
        assert scan_steps(f), "Scan was not run natively"
        fs.append(f)
    return fs


def test_sit_sot_with_sequence_and_non_sequence():
    x0, rho = pt.dscalars("x0", "rho")
    eps = pt.vector("eps")
    xs = scan(lambda e, x, r: r * x + e, sequences=[eps], outputs_info=[x0], non_sequences=[rho])
    fast, general = compare([x0, rho, eps], [xs, (xs**2).sum()], [np.array(0.3), np.array(0.9), rng.normal(size=50)])
    assert "fast path" in scan_steps(fast)[0]
    assert "general path" in scan_steps(general)[0]


def test_mit_sot_taps():
    x = pt.vector("x")  # initial taps
    eps = pt.vector("eps")
    xs = scan(
        lambda e, x3, x1: 0.5 * x1 - 0.2 * x3 + e,
        sequences=[eps],
        outputs_info=[{"initial": x, "taps": [-3, -1]}],
    )
    compare([x, eps], xs, [rng.normal(size=3), rng.normal(size=40)])


def test_nit_sot_map_over_matrix_rows():
    m = pt.matrix("m")
    w = pt.vector("w")
    # sort is not pushed out of the loop by the scan rewrites
    ys = scan(lambda row, w: pt.sort(pt.exp(row) * w)[::-1], sequences=[m], non_sequences=[w])
    compare([m, w], [ys, ys.sum()], [rng.normal(size=(7, 4)), rng.normal(size=4)])


def test_multiple_outputs_mixed_kinds():
    x0 = pt.dvector("x0")
    seq = pt.matrix("seq")

    def step(s, x):
        new = 0.9 * x + s
        return new, new.sum(), (new**2).max()

    xs, sums, maxs = scan(step, sequences=[seq], outputs_info=[x0, None, None])
    compare([x0, seq], [xs, sums, maxs], [rng.normal(size=3), rng.normal(size=(20, 3))])


def test_while_loop_truncates():
    w0 = pt.dscalar("w0")
    ws = scan(lambda w: (w * 2, until(w > 1000)), outputs_info=[w0], n_steps=100)
    f, _ = compare([w0], ws, [np.array(1.5)])
    assert f(np.array(1.5)).shape[0] < 100


def test_only_last_value_uses_circular_buffer():
    x0, rho = pt.dscalars("x0", "rho")
    eps = pt.vector("eps")
    xs = scan(lambda e, x, r: r * x + e, sequences=[eps], outputs_info=[x0], non_sequences=[rho])
    # only the last state is needed: the save-memory rewrites shrink the buffer
    compare([x0, rho, eps], [xs[-1], xs[-2]], [np.array(0.3), np.array(0.7), rng.normal(size=33)])


def test_gradient_of_scan_mit_mot():
    x0, rho = pt.dscalars("x0", "rho")
    eps = pt.vector("eps")
    xs = scan(lambda e, x, r: r * x + pt.tanh(e), sequences=[eps], outputs_info=[x0], non_sequences=[rho])
    logp = (-0.5 * xs**2).sum()
    compare([x0, rho, eps], [logp, *pytensor.grad(logp, [x0, rho, eps])], [np.array(0.3), np.array(0.8), rng.normal(size=60)])


def test_general_path_for_large_or_dynamic_steps():
    A = pt.matrix("A")
    x0 = pt.vector("x0")
    seq = pt.matrix("seq")
    xs = scan(lambda s, x, A: pt.tanh(A @ x) + s, sequences=[seq], outputs_info=[x0], non_sequences=[A])
    (f,) = compare([A, x0, seq], xs, [rng.normal(size=(5, 5)) * 0.3, rng.normal(size=5), rng.normal(size=(12, 5))], modes=(FAST,))
    assert "general path" in scan_steps(f)[0]  # Dot isn't segment material


def test_zero_steps_and_n_steps_input():
    x0 = pt.dscalar("x0")
    n = pt.lscalar("n")
    xs = scan(lambda x: x + 1, outputs_info=[x0], n_steps=n)
    for nv in [0, 1, 5]:
        ref = pytensor.function([x0, n], xs, mode=REF)(np.array(2.0), nv)
        for mode in (FAST, GENERAL):
            np.testing.assert_allclose(pytensor.function([x0, n], xs, mode=mode)(np.array(2.0), nv), ref)


def test_nested_scan():
    x = pt.matrix("x")

    def outer_step(row):
        inner = scan(lambda v, acc: acc + v, sequences=[row], outputs_info=[pt.zeros((), dtype=row.dtype)])
        return inner[-1]

    sums = scan(outer_step, sequences=[x])
    compare([x], sums, [rng.normal(size=(6, 8))], modes=(FAST,))


def test_shared_inner_graph_inlined_twice():
    # merged nodes can share one OpFromGraph; inlining both must not mix up ids
    from pytensor.compile.builders import OpFromGraph

    x = pt.vector("x")
    a = pt.vector("a")
    op = OpFromGraph([a], [pt.exp(a)[1:] * 2])
    out = op(x) + op(x[::-1])
    f = pytensor.function([x], out, mode=FAST)
    xv = rng.normal(size=6)
    np.testing.assert_allclose(f(xv), np.exp(xv)[1:] * 2 + np.exp(xv[::-1])[1:] * 2)
