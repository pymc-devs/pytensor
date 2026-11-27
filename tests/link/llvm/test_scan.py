"""Scan on LLVM: a generated loop (its body inlined, kernels called for the rest)."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode, get_mode
from pytensor.scan.utils import until


pytest.importorskip("pytensor_llvm")

LLVM = get_mode("LLVM")
GEN = LLVM.clone(link_kwargs={"numba_fallback": False})
REF = Mode(linker="py", optimizer="fast_run")
rng = np.random.default_rng(5)


def scan(*args, **kwargs):
    return pytensor.scan(*args, return_updates=False, **kwargs)


def scan_steps(f):
    """The program's items that run a Scan."""
    dump = f.maker.linker.program.dump()
    return [line for line in dump.split("steps:")[1].splitlines() if "Scan" in line]


def compare(inputs, outputs, values, modes=(GEN,)):
    ref = pytensor.function(inputs, outputs, mode=REF, on_unused_input="ignore")(
        *values
    )
    if not isinstance(ref, list):
        ref = [ref]
    fs = []
    for mode in modes:
        f = pytensor.function(inputs, outputs, mode=mode, on_unused_input="ignore")
        res = f(*values)
        if not isinstance(res, list):
            res = [res]
        for r, e in zip(res, ref, strict=True):
            assert r.shape == e.shape and r.dtype == e.dtype, (
                r.shape,
                e.shape,
                r.dtype,
                e.dtype,
            )
            np.testing.assert_allclose(r, e, rtol=1e-12, atol=1e-14)
        assert scan_steps(f), "no Scan step"
        fs.append(f)
    return fs


def test_sit_sot_with_sequence_and_non_sequence():
    x0, rho = pt.dscalars("x0", "rho")
    eps = pt.vector("eps")
    xs = scan(
        lambda e, x, r: r * x + e,
        sequences=[eps],
        outputs_info=[x0],
        non_sequences=[rho],
    )
    compare(
        [x0, rho, eps],
        [xs, (xs**2).sum()],
        [np.array(0.3), np.array(0.9), rng.normal(size=50)],
    )


def test_loop_without_scratch_memory_calls_no_runtime_functions():
    # a body taking no scratch memory needs no scratch calls around the loop
    x0, rho = pt.dscalars("x0", "rho")
    eps = pt.vector("eps")
    xs = scan(
        lambda e, x, r: r * x + e,
        sequences=[eps],
        outputs_info=[x0],
        non_sequences=[rho],
    )
    f = pytensor.function([x0, rho, eps], xs, mode=GEN)
    ir = f.maker.linker.ir()
    kernel = ir[ir.index("; ===== Scan") :].split("; =====")[1].split("define")[1]
    loop = kernel.split("\nfold:")[1].split("\nfold.exit:")[0]
    assert "@ptl_rt_scratch" not in loop


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
    ys = scan(
        lambda row, w: pt.sort(pt.exp(row) * w)[::-1], sequences=[m], non_sequences=[w]
    )
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
    (f,) = compare([w0], ws, [np.array(1.5)])
    assert f(np.array(1.5)).shape[0] < 100


def test_only_last_value_uses_circular_buffer():
    x0, rho = pt.dscalars("x0", "rho")
    eps = pt.vector("eps")
    xs = scan(
        lambda e, x, r: r * x + e,
        sequences=[eps],
        outputs_info=[x0],
        non_sequences=[rho],
    )
    # only the last states are needed: the save-memory rewrites shrink the buffer
    compare(
        [x0, rho, eps],
        [xs[-1], xs[-2]],
        [np.array(0.3), np.array(0.7), rng.normal(size=33)],
    )


def test_circular_buffer_of_dynamic_length():
    # whether the row written is one the iteration reads is decided at run time
    # (k=1: the buffer has the two rows tap -2 needs, and the row read is the one
    # written)
    x = pt.matrix("x", shape=(None, 4))
    k = pt.lscalar("k")
    ys = scan(
        lambda x2: x2 * 2 + 1, outputs_info=[{"initial": x, "taps": [-2]}], n_steps=7
    )
    for kv in (1, 2, 3):
        compare([x, k], ys[-k:], [np.arange(8.0).reshape(2, 4), np.int64(kv)])


def test_gradient_of_scan_mit_mot():
    x0, rho = pt.dscalars("x0", "rho")
    eps = pt.vector("eps")
    xs = scan(
        lambda e, x, r: r * x + pt.tanh(e),
        sequences=[eps],
        outputs_info=[x0],
        non_sequences=[rho],
    )
    logp = (-0.5 * xs**2).sum()
    compare(
        [x0, rho, eps],
        [logp, *pytensor.grad(logp, [x0, rho, eps])],
        [np.array(0.3), np.array(0.8), rng.normal(size=60)],
    )


def test_kernel_inside_the_loop_body():
    A = pt.matrix("A")
    x0 = pt.vector("x0")
    seq = pt.matrix("seq")
    xs = scan(
        lambda s, x, A: pt.tanh(A @ x) + s,
        sequences=[seq],
        outputs_info=[x0],
        non_sequences=[A],
    )
    out = [xs, pytensor.grad(xs.sum(), A)]
    compare(
        [A, x0, seq],
        out,
        [rng.normal(size=(5, 5)) * 0.3, rng.normal(size=5), rng.normal(size=(12, 5))],
    )


def test_zero_steps_and_n_steps_input():
    x0 = pt.dscalar("x0")
    n = pt.lscalar("n")
    xs = scan(lambda x: x + 1, outputs_info=[x0], n_steps=n)
    for nv in [0, 1, 5]:
        ref = pytensor.function([x0, n], xs, mode=REF)(np.array(2.0), nv)
        np.testing.assert_allclose(
            pytensor.function([x0, n], xs, mode=GEN)(np.array(2.0), nv), ref
        )


def test_n_steps_shorter_than_the_sequence():
    x = pt.vector("x")
    n = pt.lscalar("n")
    ys = scan(
        lambda v, acc: acc * 0.5 + v,
        sequences=[x],
        outputs_info=[pt.zeros(())],
        n_steps=n,
    )
    compare([x, n], ys, [np.arange(4.0), np.int64(3)])


def test_nested_scan():
    x = pt.matrix("x")

    def outer_step(row):
        inner = scan(
            lambda v, acc: acc + v,
            sequences=[row],
            outputs_info=[pt.zeros((), dtype=row.dtype)],
        )
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
    (f,) = compare([x0, seq], xs[-1], [rng.normal(size=4), rng.normal(size=(6, 4))])
    from pytensor.scan.op import Scan

    (node,) = [n for n in f.maker.fgraph.toposort() if isinstance(n.op, Scan)]
    assert node.op.info.n_untraced_sit_sot == 1


def test_untraced_state_that_changes_shape():
    # the state grows by one element per step: every step reallocates it (the
    # reference Scan can't run this, so the result is compared with numpy)
    x0 = pt.vector("x0")
    seq = pt.vector("seq")
    xs = scan(
        lambda s, x: pt.concatenate([x, x[-1:] + s]),
        sequences=[seq],
        outputs_info=[x0],
    )
    f = pytensor.function([x0, seq], xs[-1], mode=GEN)
    for n in [0, 1, 5, 40]:
        steps = np.arange(float(n))
        want = np.arange(3.0)
        for s in steps:
            want = np.concatenate([want, want[-1:] + s])
        np.testing.assert_array_equal(f(np.arange(3.0), steps), want)


def test_nit_sot_buffer_shorter_than_n_steps():
    # only the last outputs are needed: the save-memory rewrites shrink the nit-sot
    # buffer, written circularly
    x = pt.vector("x")

    def step(v, acc):
        new = acc + v
        return new, pt.sin(new)

    _, ys = scan(step, sequences=[x], outputs_info=[pt.zeros(()), None])
    compare([x], [ys[-1], ys[-3]], [rng.normal(size=50)])


def test_body_output_is_a_view_of_the_state():
    # the step writes a view of the state it read into memory that state is in: an
    # untraced state, and a mit-sot buffer whose row read by tap -2 is the row written
    x0 = pt.vector("x0")
    xs = scan(lambda x: x[::-1], outputs_info=[x0], n_steps=3)
    compare([x0], xs[-1], [np.arange(4.0)])
    x = pt.matrix("x")
    ys = scan(
        lambda x2: x2[::-1], outputs_info=[{"initial": x, "taps": [-2]}], n_steps=5
    )
    compare([x], ys[-1], [np.arange(8.0).reshape(2, 4)])
