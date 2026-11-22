import re

import numpy as np
import pytest

import pytensor
import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor.compile.builders import Execution, OpFromGraph
from pytensor.compile.mode import get_mode
from pytensor.compile.ops import deep_copy_op
from pytensor.graph.destroyhandler import DestroyHandler
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.rewriting.units import GroupIntoUnits


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
    return {
        "steps": p.steps,
        "n_steps": p.n_steps,
        "arena_bytes": p.arena_bytes,
        "stats": f.maker.linker.stats,
    }


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
    w = pt.join(0, x[1:], y[None]).T.sum(axis=0)
    f = compare([x, y], w, [rng.random((3, 4)), rng.random(4)])
    i = info(f)
    assert i["n_steps"] == 1
    assert i["steps"][0].startswith("unit[")


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
    temporaries = len(i["steps"]) - 1
    assert temporaries >= 4
    assert n * 8 <= i["arena_bytes"] < temporaries * n * 8


def test_compiled_ofg_calls_kernels():
    A = pt.matrix("A")
    v = pt.vector("v")
    inner = OpFromGraph(
        [A, v], [((A @ v) ** 2).sum(), A @ v], execution=Execution.COMPILED
    )
    out = inner(A, v)
    f = compare([A, v], out, [rng.random((4, 3)), rng.random(3)])
    steps = info(f)["steps"]
    # Dot can't be generated yet, so the unit runs its nodes as separate steps
    assert any("Dot" in s for s in steps)


def test_ir_of_the_generated_code():
    x = pt.vector("x")
    f = pytensor.function([x], pt.exp(x) * 2 + 1, mode=LLVM)
    linker = f.maker.linker
    # the items' modules and the program's function
    assert len(linker.stats.modules) == linker.stats.n_modules + 1 >= 2
    text = linker.ir()
    assert text.startswith("; =====") and "define" in text
    assert "define" in linker.ir(optimized=True)


def test_debug_info():
    """With debug information, every instruction is located at the Python code that
    emitted it, in the function of its node (also in units and parallel bodies); the
    remarks say where the instructions they are about come from."""
    x = pt.dmatrix("x")
    y = pt.dvector("y")
    unit = OpFromGraph(
        [y], [pt.cumsum(pt.exp(y))], execution=Execution.COMPILED, inline=False
    )
    mode = LLVM.clone(link_kwargs={"debug_info": True})
    f = pytensor.function([x, y], [(pt.log(x) * y).sum(axis=1), unit(y)], mode=mode)
    xv, yv = rng.random((300, 400)), rng.random(400)
    out = f(xv, yv)
    np.testing.assert_allclose(out[0], (np.log(xv) * yv).sum(axis=1))
    np.testing.assert_allclose(out[1], np.cumsum(np.exp(yv)))

    linker = f.maker.linker
    text = linker.ir(optimized=True, locations=True)
    assert "!dbg" not in text and "!DI" not in text
    # (inside the unit, nodes are named after its inner graph)
    assert re.search(r"; (?:array|context)\.py:\d+ \S+ <.* < Exp\(i0\) < .* < unit: Exp", text)
    assert "; parallel body of Exp(i0)" in text
    remarks = linker.ir(optimized=True, remarks=True)
    assert re.search(r";     at .*fused\.py:\d+ .* < FusedElemwise", remarks)


def test_optimization_remarks(capfd):
    """The remarks name the instruction of the IR before optimization they are about;
    the IR after them is the optimized IR. Compiling afterwards prints no remarks."""
    x = pt.vector("x")
    f = pytensor.function([x], x * 2 + 1, mode=LLVM)
    m = f.maker.linker.stats.modules[0]
    remarks, optimized = m.optimized_ir(remarks=True).split("\n\n", 1)
    assert optimized == m.optimized_ir()
    lines = remarks.splitlines()
    k = next(k for k, line in enumerate(lines) if "vectorized loop" in line)
    about = next(line for line in reversed(lines[:k]) if line.startswith("; @"))
    assert re.match(r"; @[\w.]+ %[\w.]+: br ", about)
    pytensor.function([x], x * 3, mode=LLVM)
    assert "remark" not in capfd.readouterr().err


@pytest.mark.parametrize(
    "fn", [pt.exp, pt.exp2, pt.log, pt.log2, pt.log10, pt.sin, pt.cos, pt.tanh]
)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_math_functions_use_vector_library(fn, dtype):
    # LLVM only vectorizes math calls to libmvec through vector-function-abi-variant
    # attributes (not through intrinsics), so each of these must keep them
    x = pt.vector("x", dtype=dtype)
    f = pytensor.function([x], fn(x) * 2, mode=LLVM)
    linker = f.maker.linker
    if "vector-function-abi-variant" not in linker.ir():
        pytest.skip("no vector library on this target")
    assert re.search(
        r"call [\w ]*<\d+ x (double|float)> @_ZGV", linker.ir(optimized=True)
    )


def test_units_respect_inplace_orderings():
    # `s` is inplace on `a` and so must run after the view `a.T`, while `exp(s)` is again
    # groupable: one unit holding `a.T` and `exp(s)` would depend on itself through `s`
    x = pt.tensor("x", shape=(2, 3))
    y = pt.tensor("y", shape=(3,))
    a = pt.sort(x, axis=1)
    s = pt.set_subtensor(a[0], y)
    out = [a.T * 2, pt.exp(s) + 1]
    compare([x, y], out, [rng.random((2, 3)), rng.random(3)])


def test_units_declare_their_views():
    # a unit holding `a.T` returns a view of `a`: what reads it still runs before the
    # node that overwrites `a`
    x = pt.tensor("x", shape=(200, 3))
    v = pt.vector("v", shape=(200,))
    a = pt.sort(x, axis=0)
    out = a.T @ v
    read = out.owner.inputs[0].owner  # the Matmul (under a Squeeze)
    destroy = Elemwise(ps.exp, inplace_pattern={0: 0})(a).owner
    fgraph = FunctionGraph([x, v], [out, destroy.out], clone=False)
    fgraph.attach_feature(DestroyHandler())
    GroupIntoUnits().apply(fgraph)
    assert any(isinstance(n.op, OpFromGraph) for n in fgraph.apply_nodes)
    assert read in fgraph.orderings()[destroy]


def test_units_include_copies():
    x = pt.tensor("x", shape=(4, 5))
    out = [deep_copy_op(x[1:].sum(axis=0)), x.sum(axis=1)]
    f = compare([x], out, [rng.random((4, 5))])
    s = info(f)["stats"]
    assert (s.n_units, s.n_modules, s.n_rust_kernels) == (1, 1, 0)


def test_nested_ofg_kinds():
    x = pt.vector("x")
    compiled = OpFromGraph([x], [pt.exp(x) + 1], execution=Execution.COMPILED)
    y = pt.vector("y")
    inlined = OpFromGraph([y], [compiled(y).sum() * pt.log1p(y)])
    z = pt.vector("z")
    outer = OpFromGraph([z], [inlined(z) - 1], execution=Execution.COMPILED)
    compare([z], outer(z), [rng.random(5)])


def test_numba_fallback_inside_compiled_unit():
    x = pt.vector("x")
    # cumsum has no native implementation yet: called from the unit via the trampoline
    inner = OpFromGraph([x], [pt.cumsum(pt.exp(x)) * 2], execution=Execution.COMPILED)
    compare([x], inner(x), [rng.random(8)])


def test_shape_error_inside_unit_and_function_still_works():
    x = pt.vector("x")
    y = pt.vector("y")
    inner = OpFromGraph(
        [x, y], [pt.exp(x) + y, (x * y).sum()], execution=Execution.COMPILED
    )
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


def test_profiling():
    x = pt.dvector("x")
    f = pytensor.function(
        [x], [pt.cumsum(pt.exp(x)).sum(), pt.sort(x)], mode=get_mode("LLVM")
    )
    p = f.maker.linker.program
    xv = np.ones(5)
    f(xv)
    assert all(secs == 0 for *_, secs in p.profile())
    p.set_profiling(True)
    for _ in range(3):
        f(xv)
    rows = p.profile()
    assert all(calls == 3 for _, calls, _ in rows)
    assert sum(s for *_, s in rows) > 0
    assert "3 calls" in p.profile_report()
    p.reset_profile()
    p.set_profiling(False)
    f(xv)
    assert all(secs == 0 and calls == 0 for _, calls, secs in p.profile())


def test_scalars_around_a_scan():
    # scalars computed from `n` are used before the Scan (x[:n]) and after it (the
    # gradient's IncSubtensor): they can't be one compiled unit with the Scan in between
    x = pt.vector("x")
    n = pt.iscalar("n")
    ys = pytensor.scan(
        lambda xt, acc: acc * 0.5 + xt,
        sequences=[x[:n]],
        outputs_info=[pt.zeros(())],
        return_updates=False,
    )
    compare([x, n], [ys.sum(), pt.grad(ys.sum(), x)], [np.arange(5.0), np.int32(3)])


def test_kernel_reading_a_scalar():
    # a node reading a ScalarType value (here a Scan with a non-sequence, as rewrites
    # produce for merged gradient scans) is in a compiled unit with the scalar nodes
    # computing it: no ScalarType value is left outside units
    from pytensor.compile.mode import Mode
    from pytensor.graph.replace import clone_replace
    from pytensor.scalar import ScalarType
    from pytensor.scalar import bool as scalar_bool
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
    inner_out = clone_replace(
        list(inner.outputs), {inner_in[-1]: tensor_from_scalar(s)}
    )
    new_op = Scan([*inner_in[:-1], s], inner_out, op.info, mode=op.mode)
    flag = scalar_from_tensor(node.inputs[-1])
    new = new_op(*node.inputs[:-1], flag, return_list=True)
    out = new[0][1:].sum() + x.sum()

    f = pytensor.function([x], out, mode=Mode(linker=LLVM.linker, optimizer=None))
    assert not any(
        isinstance(o.type, ScalarType)
        for nd in f.maker.fgraph.apply_nodes
        for o in nd.outputs
    )
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
