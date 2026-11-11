"""Scan on LLVM: as a generated loop (its body inlined, kernels called for the rest), and as
the Rust kernel running its body as a subprogram."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode, get_mode
from pytensor.scan.utils import until


pytest.importorskip("pytensor_llvm")

LLVM = get_mode("LLVM")
GEN = LLVM.clone(link_kwargs={"numba_fallback": False})
KERNEL = LLVM.clone(link_kwargs={"numba_fallback": False, "generate_scan": False})
REF = Mode(linker="py", optimizer="fast_run")
rng = np.random.default_rng(5)


def scan(*args, **kwargs):
    return pytensor.scan(*args, return_updates=False, **kwargs)


def scan_tasks(f):
    """Descriptions of the program's Scan tasks."""
    dump = f.maker.linker.program.dump()
    return [line for line in dump.split("tasks:")[1].splitlines() if "Scan" in line]


def generated(f):
    """Are the top-level Scan tasks generated loops? (A generated task is described by
    the op, "Scan{...}"; the kernel's by just "Scan". Scans inside units are skipped.)"""
    tops = [s.split(")  ", 1)[1] for s in scan_tasks(f) if s.split(")  ", 1)[1].startswith("Scan")]
    return all(t.startswith("Scan{") for t in tops) and any(t.startswith("Scan{") for t in tops) if tops else None


def compare(inputs, outputs, values, modes=(GEN, KERNEL)):
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
        assert scan_tasks(f), "no Scan task"
        g = generated(f)
        assert g is None or g == (mode is not KERNEL), scan_tasks(f)
        fs.append(f)
    return fs


def test_sit_sot_with_sequence_and_non_sequence():
    x0, rho = pt.dscalars("x0", "rho")
    eps = pt.vector("eps")
    xs = scan(lambda e, x, r: r * x + e, sequences=[eps], outputs_info=[x0], non_sequences=[rho])
    compare([x0, rho, eps], [xs, (xs**2).sum()], [np.array(0.3), np.array(0.9), rng.normal(size=50)])


def test_ir_includes_the_modules_of_the_loop_body():
    # the kernel runs the body as a subprogram, planned by a planner of its own: its
    # generated modules are listed with the program's
    eps = pt.vector("eps")
    xs = scan(lambda e, x: pt.exp(x) * 0.5 + e, sequences=[eps], outputs_info=[pt.zeros(())])
    f = pytensor.function([eps], xs, mode=KERNEL)
    stats = f.maker.linker.stats
    assert len(stats.modules) > stats.n_modules
    assert "define" in f.maker.linker.ir()


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
    # sort (a Rust kernel, called from the loop) isn't pushed out by the scan rewrites
    ys = scan(lambda row, w: pt.sort(pt.exp(row) * w)[::-1], sequences=[m], non_sequences=[w])
    compare([m, w], [ys, ys.sum()], [rng.normal(size=(7, 4)), rng.normal(size=4)])


def test_nit_sot_rows_of_dynamic_shape():
    # rows without a static shape: the buffer is allocated by the first iteration
    m = pt.matrix("m")
    k = pt.lscalar("k")
    ys = scan(lambda row, k: pt.sort(row)[:k], sequences=[m], non_sequences=[k])
    compare([m, k], ys, [rng.normal(size=(5, 6)), np.int64(3)])
    compare([m, k], ys, [rng.normal(size=(0, 6)), np.int64(3)])


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
    # only the last states are needed: the save-memory rewrites shrink the buffer
    compare([x0, rho, eps], [xs[-1], xs[-2]], [np.array(0.3), np.array(0.7), rng.normal(size=33)])


def test_gradient_of_scan_mit_mot():
    x0, rho = pt.dscalars("x0", "rho")
    eps = pt.vector("eps")
    xs = scan(lambda e, x, r: r * x + pt.tanh(e), sequences=[eps], outputs_info=[x0], non_sequences=[rho])
    logp = (-0.5 * xs**2).sum()
    compare([x0, rho, eps], [logp, *pytensor.grad(logp, [x0, rho, eps])], [np.array(0.3), np.array(0.8), rng.normal(size=60)])


def test_kernel_inside_the_loop_body():
    A = pt.matrix("A")
    x0 = pt.vector("x0")
    seq = pt.matrix("seq")
    xs = scan(lambda s, x, A: pt.tanh(A @ x) + s, sequences=[seq], outputs_info=[x0], non_sequences=[A])
    out = [xs, pytensor.grad(xs.sum(), A)]
    compare([A, x0, seq], out, [rng.normal(size=(5, 5)) * 0.3, rng.normal(size=5), rng.normal(size=(12, 5))])


def test_zero_steps_and_n_steps_input():
    x0 = pt.dscalar("x0")
    n = pt.lscalar("n")
    xs = scan(lambda x: x + 1, outputs_info=[x0], n_steps=n)
    for nv in [0, 1, 5]:
        ref = pytensor.function([x0, n], xs, mode=REF)(np.array(2.0), nv)
        for mode in (GEN, KERNEL):
            np.testing.assert_allclose(pytensor.function([x0, n], xs, mode=mode)(np.array(2.0), nv), ref)


def test_n_steps_shorter_than_the_sequence():
    x = pt.vector("x")
    n = pt.lscalar("n")
    ys = scan(lambda v, acc: acc * 0.5 + v, sequences=[x], outputs_info=[pt.zeros(())], n_steps=n)
    compare([x, n], ys, [np.arange(4.0), np.int64(3)])


def test_nested_scan():
    x = pt.matrix("x")

    def outer_step(row):
        inner = scan(lambda v, acc: acc + v, sequences=[row], outputs_info=[pt.zeros((), dtype=row.dtype)])
        return inner[-1]

    sums = scan(outer_step, sequences=[x])
    compare([x], sums, [rng.normal(size=(6, 8))])


def test_loop_body_with_an_interpreted_op_from_graph():
    from pytensor.compile.builders import OpFromGraph

    a = pt.vector("a")
    op = OpFromGraph([a], [pt.exp(a) * 2])
    m = pt.matrix("m")
    ys = scan(lambda row: op(row).sum(), sequences=[m])
    compare([m], ys, [rng.normal(size=(4, 3))])


def test_loop_body_with_a_numba_kernel():
    # a node without an LLVM implementation in the loop body is a numba kernel
    m = pt.matrix("m")
    ys = scan(lambda row: pt.linalg.norm(pt.outer(row, row), ord="nuc"), sequences=[m])
    compare([m], ys, [rng.normal(size=(3, 4))], modes=(LLVM,))


def test_array_constant_in_the_loop_body():
    m = pt.matrix("m")
    c = np.arange(4.0)
    ys = scan(lambda row: (row * pt.constant(c)).sum(), sequences=[m])
    compare([m], ys, [rng.normal(size=(5, 4))])


def test_untraced_state_of_dynamic_shape():
    # only the last state is used: it becomes an untraced state, here a vector of
    # dynamic length (read from its output slot in every step)
    x0 = pt.vector("x0")
    seq = pt.matrix("seq")
    xs = scan(lambda s, x: pt.tanh(x) + s, sequences=[seq], outputs_info=[x0])
    f, _ = compare([x0, seq], xs[-1], [rng.normal(size=4), rng.normal(size=(6, 4))])
    from pytensor.scan.op import Scan

    (node,) = [n for n in f.maker.fgraph.toposort() if isinstance(n.op, Scan)]
    assert node.op.info.n_untraced_sit_sot == 1
