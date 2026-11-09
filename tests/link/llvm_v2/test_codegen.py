import numpy as np
import pytest

import pytensor
import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor.graph.basic import Constant


ir = pytest.importorskip("pytensor_llvm.ir")

from pytensor.link.llvm_v2.codegen import OutMode, compile_node  # noqa: E402


rng = np.random.default_rng(42)


def run_node(node, *values, provided=None, provided_layouts=None):
    """Compile `node` alone and call its C ABI wrapper on `values`."""
    c = compile_node(node, provided=provided_layouts)
    outs = []
    for k, (o, spec) in enumerate(zip(node.outputs, c.signature.outputs, strict=True)):
        if spec.mode is OutMode.PROVIDED:
            arr = None if provided is None else provided[k]
            outs.append(np.empty(o.type.shape, dtype=o.type.dtype) if arr is None else arr)
        else:
            outs.append((np.dtype(o.type.dtype), o.type.ndim))
    # constants are inputs of the node function too
    ins = []
    it = iter(values)
    for i in node.inputs:
        ins.append(np.asarray(i.data if isinstance(i, Constant) else next(it), dtype=i.type.dtype))
    return ir.call_abi(c.address, ins, outs, c.workspace)


def reference(out, inputs, values):
    f = pytensor.function(inputs, out, mode="FAST_COMPILE", on_unused_input="ignore")
    return f(*values)


def check(out, inputs, values, rtol=1e-12, atol=1e-14):
    """Run the node computing `out`, on its actual inputs (evaluated by reference)."""
    node = out.owner
    node_inputs = [i for i in node.inputs if not isinstance(i, Constant)]
    in_values = reference(node_inputs, inputs, values) if node_inputs else []
    (res,) = run_node(node, *in_values)
    np.testing.assert_allclose(res, reference(out, inputs, values), rtol=rtol, atol=atol)
    return res


# ------------------------------------------------------------------ the ir wrapper


def test_ir_fold_libm_and_compile_cache():
    def build():
        m = ir.Module("sumsq")
        f = m.function("sumsq", [ir.ptr, ir.i64], ir.f64, exported=True)
        b = f.builder()
        x, n = f.params

        def body(i, a):
            v = b.load(ir.f64, b.gep(ir.f64, x, i))
            return [b.fadd(a, b.fmul(v, v))]

        (acc,) = b.fold(n, [b.const_float(ir.f64, 0.0)], body)
        b.ret(b.libm(ir.Libm.Log1p, [acc]))
        return m.compile()

    c1, c2 = build(), build()
    assert c1.address("sumsq") == c2.address("sumsq")
    assert c2.source == "in process"


@pytest.mark.parametrize(
    "bad",
    [
        lambda b, a, k: b.fadd(a, k),
        lambda b, a, k: b.add(a, a),
        lambda b, a, k: b.ret(k),
        lambda b, a, k: b.load(ir.f64, a),
        lambda b, a, k: b.select(a, a, a),
        lambda b, a, k: b.trunc(k, ir.i64),
        lambda b, a, k: b.extract(a, 0),
        lambda b, a, k: b.const_int(ir.i8, 300),
        lambda b, a, k: b.intrinsic(ir.Intrinsic.Exp, [k]),
    ],
)
def test_ir_rejects_bad_instructions(bad):
    m = ir.Module("bad")
    g = m.function("g", [ir.f64, ir.i64], ir.f64)
    b = g.builder()
    a, k = g.params
    with pytest.raises(ValueError):
        bad(b, a, k)


def test_ir_rejects_values_from_elsewhere():
    m = ir.Module("m1")
    f = m.function("f", [ir.f64], ir.f64)
    g = m.function("g", [ir.f64], ir.f64)
    bf, bg = f.builder(), g.builder()
    (x,) = f.params
    with pytest.raises(ValueError, match="different function"):
        bg.ret(x)
    other = ir.Module("m2").function("h", [ir.f64], ir.f64)
    with pytest.raises(ValueError, match="different module"):
        bf.ret(other.params[0])
    bf.ret(x)
    with pytest.raises(ValueError, match="ended"):
        bf.fadd(x, x)


def test_ir_invalid_module_is_an_error():
    m = ir.Module("noret")
    f = m.function("f", [], ir.i32, exported=True)
    f.builder()  # body never returns
    with pytest.raises(ValueError, match="invalid IR"):
        m.compile()


# ------------------------------------------------------------------ Elemwise


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_elemwise_float_ops(dtype):
    x = pt.vector("x", dtype=dtype)
    y = pt.vector("y", dtype=dtype)
    xv = (rng.random(37) + 0.5).astype(dtype)
    yv = (rng.random(37) + 0.5).astype(dtype)
    tol = dict(rtol=1e-5, atol=1e-6) if dtype == "float32" else {}
    for e in [
        x + y,
        x * y - x,
        x / y,
        x**y,
        pt.exp(x),
        pt.log1p(x),
        pt.sqrt(x),
        pt.tanh(x),
        pt.arctan2(x, y),
        pt.maximum(x, y),
        pt.sigmoid(x),
        pt.softplus(x - 1),
        pt.gammaln(x),
        pt.erf(x - 1),
        pt.switch(x > y, x, y),
        x // y,
        x % y,
    ]:
        check(e, [x, y], [xv, yv], **tol)


@pytest.mark.parametrize("dtype", ["int64", "int32", "int8", "uint8"])
def test_elemwise_int_ops(dtype):
    x = pt.vector("x", dtype=dtype)
    y = pt.vector("y", dtype=dtype)
    info = np.iinfo(dtype)
    lo, hi = max(info.min, -100), min(info.max, 100)
    xv = rng.integers(lo, hi, 50).astype(dtype)
    yv = rng.integers(lo, hi, 50).astype(dtype)
    yv[:5] = 0
    if info.min < 0:
        xv[5], yv[5] = info.min, -1
    for e in [x + y, x * y, x - y, x // y, x % y, pt.maximum(x, y), pt.minimum(x, y), x < y, pt.eq(x, y), abs(x), x & y]:
        check(e, [x, y], [xv, yv])


def test_elemwise_bool_and_mixed_dtypes():
    x = pt.vector("x", dtype="bool")
    y = pt.vector("y", dtype="int32")
    z = pt.vector("z", dtype="float32")
    xv = rng.random(20) > 0.5
    yv = rng.integers(-5, 5, 20).astype("int32")
    zv = rng.random(20).astype("float32")
    check(x & (y > 0), [x, y], [xv, yv])
    check(y * z + x, [x, y, z], [xv, yv, zv], rtol=1e-6)
    check(pt.switch(x, y, z), [x, y, z], [xv, yv, zv])


def test_elemwise_broadcasting_and_strides():
    x = pt.matrix("x")
    r = pt.row("r")
    c = pt.col("c")
    s = pt.scalar("s")
    e = pt.exp(x) * r + c - s
    vals = [rng.random((10, 8))[::2, 1::2], rng.random((1, 4)), rng.random((5, 1)), 2.0]
    check(e, [x, r, c, s], vals)
    check(e, [x, r, c, s], [v.T.copy().T if hasattr(v, "T") else v for v in vals])


def test_elemwise_composite():
    x = pt.vector("x")
    y = pt.vector("y")
    a, bb = ps.float64(), ps.float64()
    comp = ps.Composite([a, bb], [ps.exp(a) * bb + ps.log1p(bb), ps.sqr(a - bb)])
    outs = pt.elemwise.Elemwise(comp)(x, y)
    node = outs[0].owner
    xv, yv = rng.random(13), rng.random(13)
    r1, r2 = run_node(node, xv, yv)
    np.testing.assert_allclose(r1, np.exp(xv) * yv + np.log1p(yv))
    np.testing.assert_allclose(r2, (xv - yv) ** 2)


def test_elemwise_shape_mismatch_raises():
    x = pt.vector("x")
    y = pt.vector("y")
    with pytest.raises(RuntimeError, match="shape mismatch"):
        run_node((x + y).owner, np.ones(3), np.ones(4))


def test_provided_output_with_any_strides():
    from pytensor.link.llvm_v2.layout import Layout

    x = pt.tensor("x", shape=(4, 6))
    node = pt.exp(x).owner
    big = np.zeros((8, 12))
    view = big[::2, ::2]  # a strided slice of a larger array
    xv = rng.random((4, 6))
    # code generated for provided memory of any strides
    (res,) = run_node(node, xv, provided=[view], provided_layouts=[Layout.dynamic(2)])
    np.testing.assert_allclose(big[::2, ::2], np.exp(xv))
    assert big[1::2].sum() == 0 and big[:, 1::2].sum() == 0
    assert res is view


def test_scalar_type_nodes_have_no_c_abi():
    a = ps.float64("a")
    node = ps.exp(a).owner
    with pytest.raises(ValueError, match="no C ABI"):
        compile_node(node)


# ------------------------------------------------------------------ CAReduce


@pytest.mark.parametrize("axis", [None, 0, 1, (0, 2), 2])
@pytest.mark.parametrize("op", [pt.sum, pt.max, pt.min, pt.prod])
def test_careduce(op, axis):
    x = pt.tensor3("x")
    xv = rng.random((4, 5, 6)) + 0.5
    check(op(x, axis=axis), [x], [xv])
    check(op(x, axis=axis), [x], [xv[:, ::-1, ::2]])


def test_careduce_bool_and_int_accumulators():
    x = pt.matrix("x", dtype="bool")
    xv = rng.random((5, 7)) > 0.3
    check(pt.all(x, axis=1), [x], [xv])
    check(pt.any(x), [x], [xv])
    y = pt.matrix("y", dtype="int8")
    yv = rng.integers(-100, 100, (6, 9)).astype("int8")
    check(pt.sum(y, axis=0), [y], [yv])  # accumulates in int64
    check(pt.max(y), [y], [yv])


def test_careduce_acc_dtype_uses_workspace():
    x = pt.tensor("x", shape=(30, 4), dtype="float32")
    e = pt.sum(x, axis=0, acc_dtype="float64")
    node = e.owner
    c = compile_node(node)
    assert c.workspace >= 4 * 8
    xv = rng.random((30, 4)).astype("float32")
    (res,) = run_node(node, xv)
    np.testing.assert_allclose(res, xv.astype("float64").sum(0), rtol=1e-6)


def test_empty_arrays():
    x = pt.matrix("x")
    z = np.zeros((0, 3))
    for e in [pt.exp(x), pt.sum(x, axis=1), pt.sum(x), pt.prod(x, axis=0), pt.max(x, axis=1)]:
        (res,) = run_node(e.owner, z)
        np.testing.assert_allclose(res, reference(e, [x], [z]))
    for e in [pt.max(x, axis=0), pt.max(x)]:
        with pytest.raises(RuntimeError, match="zero-size array"):
            run_node(e.owner, z)


def test_alias_metadata_on_array_accesses():
    # inputs and written outputs get scopes and TBAA tags; the gather's inner row copy
    # then doesn't reload the index (and vectorizes)
    from pytensor.link.llvm_v2.codegen.function import build_node_function

    x = pt.matrix("x")
    i = pt.vector("i", dtype="int32")
    node = x[i].owner
    m = ir.Module("alias_test")
    build_node_function(m, node, "gather")
    text = m.ir()
    assert "!alias.scope" in text and "!noalias" in text and "!tbaa" in text
    assert "pt.alias_test.gather.i0" in text and "pt.alias_test.gather.o0" in text
