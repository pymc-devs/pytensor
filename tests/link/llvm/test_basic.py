import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.builders import OpFromGraph, Execution
from pytensor.compile.mode import get_mode


pytest.importorskip("pytensor_llvm")

LLVM = get_mode("LLVM")
REF = "FAST_COMPILE"
rng = np.random.default_rng(1)


def compare(inputs, outputs, values, mode=LLVM, rtol=1e-10, atol=1e-12):
    f = pytensor.function(inputs, outputs, mode=mode)
    g = pytensor.function(inputs, outputs, mode=REF)
    res, ref = f(*values), g(*values)
    if not isinstance(res, list):
        res, ref = [res], [ref]
    for a, b in zip(res, ref, strict=True):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    return f


def info(f):
    p = f.maker.linker.program
    return {"tasks": p.tasks, "n_tasks": p.n_tasks, "arena_bytes": p.arena_bytes, "stats": f.maker.linker.stats}


def test_elemwise_reduce_dot_dynamic_shapes():
    x = pt.matrix("x")
    y = pt.vector("y")
    out = [(pt.exp(x) * y).sum(axis=1) + pt.log1p(x.T).mean(), x @ y, pt.max(x, axis=0)]
    X, Y = rng.random((5, 3)), rng.random(3)
    compare([x, y], out, [X, Y])
    # strided and transposed inputs
    compare([x, y], out, [rng.random((10, 3))[::2], rng.random(6)[::2]])


@pytest.mark.parametrize("dtype", ["float32", "int64", "int32", "bool"])
def test_dtypes(dtype):
    x = pt.vector("x", dtype=dtype)
    y = pt.vector("y", dtype=dtype)
    if dtype == "bool":
        outs = [x & y, pt.any(x), pt.all(y)]
        vals = [rng.random(7) > 0.5, rng.random(7) > 0.3]
    else:
        outs = [x * y + x, pt.maximum(x, y), x.sum(), pt.max(y)]
        vals = [(rng.random(7) * 10).astype(dtype), (rng.random(7) * 10).astype(dtype)]
    compare([x, y], outs, vals, rtol=1e-5)


def test_small_static_graph_is_one_unit():
    x = pt.tensor("x", shape=(3, 4))
    y = pt.tensor("y", shape=(4,))
    w = pt.sum(pt.exp(x) * y) * 2 + pt.max(x)
    f = compare([x, y], w, [rng.random((3, 4)), rng.random(4)])
    i = info(f)
    assert i["n_tasks"] == 1
    assert i["tasks"][0].startswith("unit[")


def test_arena_reuses_temporaries():
    # large static intermediates aren't grouped: each is a step whose output lives in
    # the arena, which reuses memory by liveness
    n = 10_000
    x = pt.tensor("x", shape=(n,))
    a = pt.exp(x)
    b = pt.sort(a)  # not fusable: forces a materialized intermediate
    c = pt.log1p(b)
    d = pt.sort(c)
    e = pt.sqrt(d)
    f = compare([x], pt.sort(e), [rng.normal(size=n)])
    i = info(f)
    temporaries = len(i["tasks"]) - 1
    assert temporaries >= 4
    assert n * 8 <= i["arena_bytes"] < temporaries * n * 8


def test_compiled_ofg_calls_kernels():
    A = pt.matrix("A")
    v = pt.vector("v")
    inner = OpFromGraph([A, v], [((A @ v) ** 2).sum(), A @ v], execution=Execution.COMPILED)
    out = inner(A, v)
    f = compare([A, v], out, [rng.random((4, 3)), rng.random(3)])
    tasks = info(f)["tasks"]
    # Dot can't be generated yet, so the unit runs its nodes as separate tasks
    assert any("Dot" in s for s in tasks)


def test_ir_of_the_generated_code():
    x = pt.vector("x")
    f = pytensor.function([x], pt.exp(x) * 2 + 1, mode=LLVM)
    linker = f.maker.linker
    assert len(linker.stats.modules) == linker.stats.n_modules >= 1
    text = linker.ir()
    assert text.startswith("; =====") and "define" in text
    assert "define" in linker.ir(optimized=True)


def test_interpreted_ofg():
    x = pt.vector("x")
    inner = OpFromGraph([x], [pt.exp(x).sum(), x * 2], execution=Execution.INTERPRETED)
    out = inner(x)
    f = compare([x], [out[0], out[1] + 1], [rng.random(6)])
    assert any(s.startswith("program:") for s in info(f)["tasks"])


def test_nested_ofg_kinds():
    x = pt.vector("x")
    compiled = OpFromGraph([x], [pt.exp(x) + 1], execution=Execution.COMPILED)
    y = pt.vector("y")
    interp = OpFromGraph([y], [compiled(y).sum() * pt.log1p(y)], execution=Execution.INTERPRETED)
    z = pt.vector("z")
    outer = OpFromGraph([z], [interp(z) - 1], execution=Execution.COMPILED)
    compare([z], outer(z), [rng.random(5)])


def test_numba_fallback_inside_compiled_unit():
    x = pt.vector("x")
    # cumsum has no native implementation yet: called from the unit via the trampoline
    inner = OpFromGraph([x], [pt.cumsum(pt.exp(x)) * 2], execution=Execution.COMPILED)
    compare([x], inner(x), [rng.random(8)])


def test_shape_error_inside_unit_and_function_still_works():
    x = pt.vector("x")
    y = pt.vector("y")
    inner = OpFromGraph([x, y], [pt.exp(x) + y, (x * y).sum()], execution=Execution.COMPILED)
    f = pytensor.function([x, y], inner(x, y), mode=LLVM)
    with pytest.raises(Exception, match="shape mismatch"):
        f(np.ones(3), np.ones(4))
    a, b = f(np.ones(3), np.ones(3))
    np.testing.assert_allclose(a, np.e + 1)
    np.testing.assert_allclose(b, 3.0)


def test_outputs_alias_inputs_views_and_each_other():
    x = pt.matrix("x")
    s = pt.exp(x)
    f = pytensor.function([x], [x, x.T, s, s, s.T], mode=LLVM)
    X = rng.random((2, 3))
    r = f(X)
    np.testing.assert_allclose(r[0], X)
    np.testing.assert_allclose(r[1], X.T)
    np.testing.assert_allclose(r[2], np.exp(X))
    np.testing.assert_allclose(r[4], np.exp(X).T)
    assert not np.shares_memory(r[0], X)
    r[2][...] = 0  # outputs are independent of later calls
    np.testing.assert_allclose(f(X)[2], np.exp(X))


def test_kernel_outputs_of_units():
    # dynamic shapes inside a compiled unit: outputs are allocated by the unit
    x = pt.vector("x")
    inner = OpFromGraph([x], [pt.exp(x), x.sum()], execution=Execution.COMPILED)
    f = compare([x], inner(x), [rng.random(11)])
    for n in [0, 1, 1000]:
        r = f(np.ones(n))
        assert r[0].shape == (n,)


def test_repeated_calls_and_threads():
    from concurrent.futures import ThreadPoolExecutor

    x = pt.vector("x")
    f = pytensor.function([x], [pt.exp(x).sum(), x * 2], mode=LLVM)
    vals = [rng.random(50_000) for _ in range(16)]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda v: f(v), vals))
    for v, (s, d) in zip(vals, res, strict=True):
        np.testing.assert_allclose(s, np.exp(v).sum())
        np.testing.assert_allclose(d, v * 2)


def test_profiling_with_subprograms():
    import pytensor

    x = pt.dmatrix("x")
    w = pt.dvector("w")
    res = pytensor.scan(lambda r, acc: acc + pt.exp(r) @ w, sequences=[x], outputs_info=[pt.zeros(())], return_updates=False)
    # the Scan kernel runs its loop body as a subprogram (a generated Scan has none)
    mode = get_mode("LLVM").clone(link_kwargs={"generate_scan": False})
    f = pytensor.function([x, w], [res[-1], pt.sort(x, axis=None)], mode=mode)
    p = f.maker.linker.program
    xv, wv = np.ones((5, 3)), np.ones(3)
    f(xv, wv)
    assert all(secs == 0 for *_, secs in p.profile())
    p.set_profiling(True)
    for _ in range(3):
        f(xv, wv)
    rows = p.profile()
    top = [r for r in rows if r[0] == 0]
    nested = [r for r in rows if r[0] == 1]
    assert top and nested
    assert all(calls == 3 for _, _, calls, _ in top)
    assert all(calls == 15 for _, _, calls, _ in nested)  # the loop body ran 5 times per call
    assert sum(s for *_, s in top) > 0
    assert "Scan" in p.profile_report()
    p.reset_profile()
    p.set_profiling(False)
    f(xv, wv)
    assert all(secs == 0 and calls == 0 for _, _, calls, secs in p.profile())


def test_scalars_around_a_scan():
    # scalars computed from `n` are used before the Scan (x[:n]) and after it (the
    # gradient's IncSubtensor): they can't be one compiled unit with the Scan in between
    x = pt.vector("x")
    n = pt.iscalar("n")
    ys = pytensor.scan(
        lambda xt, acc: acc * 0.5 + xt, sequences=[x[:n]], outputs_info=[pt.zeros(())], return_updates=False
    )
    compare([x, n], [ys.sum(), pt.grad(ys.sum(), x)], [np.arange(5.0), np.int32(3)])


def test_kernel_reading_a_scalar():
    # a Rust kernel (Scan) reading a ScalarType value (here a non-sequence, as rewrites
    # produce for merged gradient scans) gets it as a 0-d tensor, made in the scalars'
    # compiled unit
    from pytensor.compile.mode import Mode
    from pytensor.graph.replace import clone_replace
    from pytensor.scalar import ScalarType, bool as scalar_bool
    from pytensor.scan.op import Scan
    from pytensor.tensor.basic import scalar_from_tensor, tensor_from_scalar

    x = pt.vector("x")
    ys = pytensor.scan(
        lambda acc, f: pt.switch(f, acc * 0.5 + 1, acc - 1),
        outputs_info=[pt.zeros(())],
        non_sequences=[pt.gt(x.sum(), 0)],
        n_steps=4,
        return_updates=False,
    )
    node = ys.owner.inputs[0].owner
    while not isinstance(node.op, Scan):
        node = next(i.owner for i in node.inputs if i.owner is not None)
    op = node.op
    # the same Scan with the flag as a ScalarType non-sequence
    s = scalar_bool()
    inner = op.fgraph.unfreeze()
    inner_in = list(inner.inputs)
    inner_out = clone_replace(list(inner.outputs), {inner_in[-1]: tensor_from_scalar(s)})
    new_op = Scan([*inner_in[:-1], s], inner_out, op.info, mode=op.mode)
    flag = scalar_from_tensor(node.inputs[-1])
    new = new_op(*node.inputs[:-1], flag, return_list=True)
    out = new[0][1:].sum() + x.sum()

    f = pytensor.function([x], out, mode=Mode(linker=LLVM.linker, optimizer=None))
    scans = [nd for nd in f.maker.fgraph.toposort() if isinstance(nd.op, Scan)]
    assert any(isinstance(i.type, ScalarType) for i in scans[0].inputs)
    g = pytensor.function([x], out, mode=REF)
    for v in (np.arange(3.0), -np.arange(3.0)):
        np.testing.assert_allclose(f(v), g(v))


def test_scalar_graph_input():
    from pytensor.scalar import float64
    from pytensor.tensor.basic import tensor_from_scalar

    s = float64("s")
    x = pt.vector("x")
    out = x * tensor_from_scalar(s * 2)
    f = pytensor.function([s, x], out, mode=LLVM)
    np.testing.assert_allclose(f(1.5, np.arange(3.0)), np.arange(3.0) * 3)
