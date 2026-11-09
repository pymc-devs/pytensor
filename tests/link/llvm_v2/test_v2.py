import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.builders import OpFromGraph, UnitKind
from pytensor.compile.mode import get_mode


pytest.importorskip("pytensor_llvm")

V2 = get_mode("LLVM_V2")
REF = "FAST_COMPILE"
rng = np.random.default_rng(1)


def compare(inputs, outputs, values, mode=V2, rtol=1e-10, atol=1e-12):
    f = pytensor.function(inputs, outputs, mode=mode)
    g = pytensor.function(inputs, outputs, mode=REF)
    res, ref = f(*values), g(*values)
    if not isinstance(res, list):
        res, ref = [res], [ref]
    for a, b in zip(res, ref, strict=True):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    return f


def info(f):
    p = f.maker.linker.compiled
    return {"steps": p.steps, "n_steps": p.n_steps, "arena_bytes": p.arena_bytes, "stats": f.maker.linker.stats}


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
    inner = OpFromGraph([A, v], [((A @ v) ** 2).sum(), A @ v], unit=UnitKind.COMPILED)
    out = inner(A, v)
    f = compare([A, v], out, [rng.random((4, 3)), rng.random(3)])
    steps = info(f)["steps"]
    # Dot can't be generated yet, so the unit runs its nodes as separate steps
    assert any("Dot" in s for s in steps)


def test_interpreted_ofg():
    x = pt.vector("x")
    inner = OpFromGraph([x], [pt.exp(x).sum(), x * 2], unit=UnitKind.INTERPRETED)
    out = inner(x)
    f = compare([x], [out[0], out[1] + 1], [rng.random(6)])
    assert any(s.startswith("program:") for s in info(f)["steps"])


def test_nested_ofg_kinds():
    x = pt.vector("x")
    compiled = OpFromGraph([x], [pt.exp(x) + 1], unit=UnitKind.COMPILED)
    y = pt.vector("y")
    interp = OpFromGraph([y], [compiled(y).sum() * pt.log1p(y)], unit=UnitKind.INTERPRETED)
    z = pt.vector("z")
    outer = OpFromGraph([z], [interp(z) - 1], unit=UnitKind.COMPILED)
    compare([z], outer(z), [rng.random(5)])


def test_numba_fallback_inside_compiled_unit():
    x = pt.vector("x")
    # cumsum has no native implementation yet: called from the unit via the trampoline
    inner = OpFromGraph([x], [pt.cumsum(pt.exp(x)) * 2], unit=UnitKind.COMPILED)
    compare([x], inner(x), [rng.random(8)])


def test_shape_error_inside_unit_and_function_still_works():
    x = pt.vector("x")
    y = pt.vector("y")
    inner = OpFromGraph([x, y], [pt.exp(x) + y, (x * y).sum()], unit=UnitKind.COMPILED)
    f = pytensor.function([x, y], inner(x, y), mode=V2)
    with pytest.raises(Exception, match="shape mismatch"):
        f(np.ones(3), np.ones(4))
    a, b = f(np.ones(3), np.ones(3))
    np.testing.assert_allclose(a, np.e + 1)
    np.testing.assert_allclose(b, 3.0)


def test_outputs_alias_inputs_views_and_each_other():
    x = pt.matrix("x")
    s = pt.exp(x)
    f = pytensor.function([x], [x, x.T, s, s, s.T], mode=V2)
    X = rng.random((2, 3))
    r = f(X)
    np.testing.assert_allclose(r[0], X)
    np.testing.assert_allclose(r[1], X.T)
    np.testing.assert_allclose(r[2], np.exp(X))
    np.testing.assert_allclose(r[4], np.exp(X).T)
    assert not np.shares_memory(r[0], X)
    r[2][...] = 0  # outputs are independent of later calls
    np.testing.assert_allclose(f(X)[2], np.exp(X))


def test_callee_outputs_of_units():
    # dynamic shapes inside a compiled unit: outputs are allocated by the unit
    x = pt.vector("x")
    inner = OpFromGraph([x], [pt.exp(x), x.sum()], unit=UnitKind.COMPILED)
    f = compare([x], inner(x), [rng.random(11)])
    for n in [0, 1, 1000]:
        r = f(np.ones(n))
        assert r[0].shape == (n,)


def test_repeated_calls_and_threads():
    from concurrent.futures import ThreadPoolExecutor

    x = pt.vector("x")
    f = pytensor.function([x], [pt.exp(x).sum(), x * 2], mode=V2)
    vals = [rng.random(50_000) for _ in range(16)]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda v: f(v), vals))
    for v, (s, d) in zip(vals, res, strict=True):
        np.testing.assert_allclose(s, np.exp(v).sum())
        np.testing.assert_allclose(d, v * 2)


def test_profiling_with_nested_programs():
    import pytensor

    x = pt.dmatrix("x")
    w = pt.dvector("w")
    res = pytensor.scan(lambda r, acc: acc + pt.exp(r) @ w, sequences=[x], outputs_info=[pt.zeros(())], return_updates=False)
    f = pytensor.function([x, w], [res[-1], pt.sort(x, axis=None)], mode="LLVM_V2")
    p = f.maker.linker.compiled
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
    assert all(calls == 15 for _, _, calls, _ in nested)  # the step program ran 5 times per call
    assert sum(s for *_, s in top) > 0
    assert "Scan" in p.profile_report()
    p.reset_profile()
    p.set_profiling(False)
    f(xv, wv)
    assert all(secs == 0 and calls == 0 for _, _, calls, secs in p.profile())
