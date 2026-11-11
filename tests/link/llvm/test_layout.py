"""Static layouts: the rules, the specialization of generated code, the checks."""

import ctypes

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import get_mode


ir = pytest.importorskip("pytensor_llvm.ir")

from pytensor.link.llvm.codegen.function import build_kernel_wrapper, build_node_function  # noqa: E402
from pytensor.link.llvm.codegen.function import node_outputs  # noqa: E402
from pytensor.link.llvm.layout import DYNAMIC, PACKED, Layout  # noqa: E402


LLVM = get_mode("LLVM")
CHECKED = LLVM.clone(link_kwargs={"check_layouts": True, "numba_fallback": False})
C = Layout.c
rng = np.random.default_rng(3)


def view_layout(out, *ins):
    """The layout `llvm_outputs` declares for `out` given its node's input layouts."""
    node = out.owner
    ins = list(ins) + [None] * (len(node.inputs) - len(ins))
    return node_outputs(node, ins)[node.outputs.index(out)].layout


# ------------------------------------------------------------------ rules


def test_view_rules():
    x = pt.dmatrix("x")
    s = pt.tensor("s", shape=(4, 5, 6))
    assert str(view_layout(x.T, C(2))) == "[P0, P1]"
    assert str(view_layout(x[:, :3], C(2))) == "[?, P0]"
    assert view_layout(x[1:], C(2)) == C(2)
    assert view_layout(x[0], C(2)) == C(1)
    assert str(view_layout(x[:, 0], C(2))) == "[?]"
    assert str(view_layout(x[:, ::2], C(2))) == "[?, 16]"
    # static shapes give static strides
    assert str(view_layout(s[:, ::2], C(3))) == "[240, 96, P0]"
    assert str(view_layout(s[:, 1], C(3))) == "[240, P0]"
    assert view_layout(s[1:3], C(3)) == C(3)
    assert str(view_layout(s.dimshuffle(2, 1, 0)[0], view_layout(s.dimshuffle(2, 1, 0), C(3)))) == "[48, 240]"
    # length-1 axes don't matter
    assert view_layout(x.dimshuffle(0, "x", 1), C(2)) == C(3)
    b = pt.tensor("b", shape=(None, 1, None))
    assert view_layout(b.dimshuffle(0, 2), C(3)) == C(2)
    # a slice that keeps the static length keeps the layout
    assert view_layout(s[0:4], C(3)) == C(3)
    # nothing known stays nothing known
    assert view_layout(x.T, Layout.dynamic(2)) == Layout.dynamic(2)


def test_satisfied_implies_join():
    a = np.zeros((4, 5))
    sl = a[:, :3]
    lay = Layout.make((None, None), (0, 1), (DYNAMIC, PACKED))
    assert lay.satisfied_by(sl.shape, sl.strides, 8)
    assert not C(2).satisfied_by(sl.shape, sl.strides, 8)
    assert C(2).satisfied_by(a.shape, a.strides, 8)
    assert C(2).satisfied_by((0, 3), (999, 8), 8)  # empty
    assert C(2).satisfied_by((3, 1), (8, 999), 8)  # length-1 axis
    assert C(2).implies(lay, (None, None), 8) and not lay.implies(C(2), (None, None), 8)
    assert C(2).join(lay, (None, None), 8) == lay
    assert C(2).join(Layout.make((None, None), (1, 0), (PACKED, PACKED)), (None, None), 8) == Layout.dynamic(2)


# ------------------------------------------------------------------ programs


def check_against_numpy(inputs, outputs, values, mode=CHECKED):
    f = pytensor.function(inputs, outputs, mode=mode)
    ref = pytensor.function(inputs, outputs, mode="FAST_COMPILE")
    for r, e in zip(f(*values), ref(*values), strict=True):
        np.testing.assert_allclose(r, e, rtol=1e-12)
    return f


def test_views_feed_specialized_code():
    x = pt.dmatrix("x")
    y = pt.tensor("y", shape=(6, 7))
    outs = [
        pt.exp(x[:, 1:]).sum(axis=1),
        (x.T * 2).sum(axis=0),
        pt.exp(x[::2]) + 1,
        pt.log1p(y[:, ::3] ** 2).sum(),
        (y.T[2:] @ y[:, :3]),
        pt.exp(y[2]) * y[0, :],
    ]
    values = [rng.normal(size=(6, 7)), rng.normal(size=(6, 7))]
    check_against_numpy([x, y], outs, values)
    # with C-contiguous inputs, views of them are known too: tasks are specialized
    f = check_against_numpy([x, y], outs, values, mode=CHECKED.clone(link_kwargs={"input_layout": "c"}))
    dump = f.maker.linker.program.dump()
    assert "[?, P0]" in dump and "[P0, P1]" in dump, dump


@pytest.mark.parametrize("input_layout", ["any", "c"])
def test_non_contiguous_inputs(input_layout):
    x = pt.dmatrix("x")
    mode = CHECKED.clone(link_kwargs={"input_layout": input_layout})
    f = pytensor.function([x], [pt.exp(x) + x[:, ::-1], x.sum(axis=0)], mode=mode)
    base = rng.normal(size=(8, 10))
    for v in (base, base.T, base[::2, 1:7], np.asfortranarray(base)):
        r = f(v)
        np.testing.assert_allclose(r[0], np.exp(v) + v[:, ::-1])
        np.testing.assert_allclose(r[1], v.sum(axis=0))


def test_c_inputs_are_checked_by_the_c_api():
    from tests.link.llvm.test_c_api import Api, PtArray, as_ptarrays

    x = pt.dvector("x")
    f = pytensor.function([x], pt.exp(x), mode=LLVM.clone(link_kwargs={"input_layout": "c"}))
    p = f.maker.linker.program
    assert "(C-contiguous)" in p.dump()
    fn = p.c_function()
    a = Api.from_address(p.c_api_table())
    new = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(a.session_new)
    call = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(PtArray), ctypes.POINTER(PtArray))(a.call)
    err = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p)(a.last_error)
    st = new(fn)
    v = np.arange(10.0)
    out = np.empty(5)
    ins, k1 = as_ptarrays([v[::2]])
    outs, k2 = as_ptarrays([out])
    assert call(st, ins, outs) != 0 and b"must be C-contiguous" in err(st)
    ins, k1 = as_ptarrays([v[:5]])
    assert call(st, ins, outs) == 0
    np.testing.assert_allclose(out, np.exp(v[:5]))
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(a.session_free)(st)
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(a.function_decref)(fn)


def test_scan_body_gets_row_layouts():
    x = pt.dmatrix("x")
    w = pt.dvector("w")
    res = pytensor.scan(
        lambda r, acc: acc + pt.exp(r[1:]) @ w[1:], sequences=[x], outputs_info=[pt.zeros(())], return_updates=False
    )
    values = [rng.normal(size=(5, 4)), rng.normal(size=4)]
    check_against_numpy([x, w], [res], values)
    # a transposed sequence: rows aren't contiguous, and that's known too
    res = pytensor.scan(lambda r, acc: acc + pt.exp(r).sum(), sequences=[x.T], outputs_info=[pt.zeros(())], return_updates=False)
    check_against_numpy([x], [res], values[:1])


def test_layout_check_catches_a_wrong_claim():
    x = pt.dmatrix("x")
    node = pt.exp(x).owner
    m = ir.Module("wrong_claim")
    nf = build_node_function(m, node, "node", in_layouts=[C(2)])
    build_kernel_wrapper(m, nf, "kernel", check_layouts=True)
    c = m.compile(cache=False)
    v = rng.normal(size=(3, 4))
    (r,) = ir.call_kernel(c.address("kernel"), [v], [(np.dtype("float64"), 2)])
    np.testing.assert_allclose(r, np.exp(v))
    with pytest.raises(Exception, match="layout check"):
        ir.call_kernel(c.address("kernel"), [v.T], [(np.dtype("float64"), 2)])


def test_specialized_signature_passes_fewer_strides():
    x = pt.dmatrix("x")
    node = pt.exp(x).owner
    from pytensor.link.llvm.codegen import node_signature

    any_ = node_signature(node)
    c = node_signature(node, in_layouts=[C(2)])
    assert len(any_.inputs[0].stride_axes) == 2 and c.inputs[0].stride_axes == ()
    sl = node_signature(node, in_layouts=[Layout.make((None, None), (0, 1), (DYNAMIC, PACKED))])
    assert sl.inputs[0].stride_axes == (0,)


# ------------------------------------------------------------------ the feature and the rewrite


def test_feature_predicts_and_invalidates():
    from pytensor.graph.fg import FunctionGraph
    from pytensor.link.llvm.layout import LayoutFeature

    x = pt.dmatrix("x")
    t = x.T
    y = pt.exp(t[1:])
    z = y.T[:, :2]
    fg = FunctionGraph([x], [z], clone=False)
    fg.attach_feature(LayoutFeature(inputs_c=True))
    assert fg.llvm_layout(x) == C(2)
    assert str(fg.llvm_layout(t)) == "[P0, P1]"
    assert str(fg.llvm_layout(t[1:].owner.outputs[0])) == "[P0, ?]"
    assert fg.llvm_layout(y) == C(2)
    assert str(fg.llvm_layout(z)) == "[P0, P1]"  # slicing the outermost packed axis keeps it packed
    # replacing y by a transposed (C-contiguous) input changes everything downstream
    u = pt.dmatrix("u")
    fg.add_input(u)
    fg.replace(y, u.T, import_missing=True)
    assert str(fg.llvm_layout(fg.outputs[0])) == "[?, P0]"


# ------------------------------------------------------------------ code follows the layout


def _blocks(node, lays):
    import pytensor_llvm.ir as pir

    from pytensor.link.llvm.codegen.function import build_node_function

    m = pir.Module("t")
    build_node_function(m, node, "node", in_layouts=lays)
    labels = [ln.split(":")[0] for ln in m.ir().splitlines() if ln and ln[0] not in " ;}" and ":" in ln]
    # (loops, run-time choices between loop variants: if_else blocks; checks are if_)
    return sum(lb.startswith(("loop", "fold")) and ".exit" not in lb for lb in labels), sum(
        lb.startswith("else") for lb in labels
    )


T2 = Layout.make((None, None), (1, 0), (PACKED, PACKED))


def test_known_layouts_decide_loops_at_compile_time():
    import pytensor.scalar as ps
    from pytensor.tensor.elemwise import Elemwise

    x = pt.dmatrix("x")
    # a full reduction is one flat loop for any contiguous layout, transposed too
    assert _blocks(x.sum().owner, [T2]) == (1, 0)
    loops, branches = _blocks(x.sum().owner, [None])
    assert branches == 1 and loops > 1  # unknown: flat or nest, chosen at run time
    # an elementwise op with many inputs: only inputs of unknown layout are checked, so
    # the code has one variant more, not one per combination
    s = [ps.float64(f"s{i}") for i in range(6)]
    comp = ps.Composite(s, [s[0] * s[1] + ps.exp(s[2]) - s[3] * s[4] + s[5]])
    node = Elemwise(comp)(*[pt.dmatrix(f"a{i}") for i in range(6)]).owner
    known = _blocks(node, [C(2)] * 6)
    one_unknown = _blocks(node, [C(2)] * 5 + [None])
    all_unknown = _blocks(node, [None] * 6)
    assert known[1] == 0 and one_unknown[1] == 1 and all_unknown[1] == 1
    # transposed inputs: no flat loop with a C output, but no run-time check either
    assert _blocks(node, [T2] * 6)[1] == 0


def test_code_follows_memory_order():
    x = pt.dmatrix("x")
    outs = [x.T.sum(axis=0), x.T.sum(axis=1), pt.exp(x.T) * x.T, x.T.sum()]
    v = rng.normal(size=(7, 5))
    check_against_numpy([x], outs, [v], mode=CHECKED.clone(link_kwargs={"input_layout": "c"}))
    check_against_numpy([x], outs, [v[:, ::2]], mode=CHECKED)


def test_elemwise_output_follows_its_inputs():
    x = pt.dmatrix("x")
    y = pt.dmatrix("y")
    mode = CHECKED.clone(link_kwargs={"input_layout": "c"})
    # allocated output: transposed like its inputs (one flat loop), not C with a strided loop
    f = pytensor.function([x, y], pt.exp(x.T) * y.T, mode=mode)
    xv, yv = rng.normal(size=(4, 6)), rng.normal(size=(4, 6))
    r = f(xv, yv)
    np.testing.assert_allclose(r, np.exp(xv.T) * yv.T)
    assert r.flags.f_contiguous and not r.flags.c_contiguous
    # static shapes: the temporary is an arena value in the inputs' order
    s = pt.tensor("s", shape=(4, 6))
    f = pytensor.function([s], (pt.exp(s.T) + 1).sum(axis=0), mode=mode.clone(link_kwargs={"input_layout": "c"}))
    np.testing.assert_allclose(f(xv), (np.exp(xv.T) + 1).sum(axis=0))
    # mixed: the majority decides, the result is still right
    z = pt.dmatrix("z")
    g = pytensor.function([x, y, z], pt.exp(x.T) + y.T - z, mode=mode)
    zv = rng.normal(size=(6, 4))
    np.testing.assert_allclose(g(xv, yv, zv), np.exp(xv.T) + yv.T - zv)


def test_declared_outputs():
    from pytensor.link.llvm.codegen.function import OutputMode

    x = pt.dmatrix("x")
    (o,) = node_outputs((pt.exp(x.T)).owner, [T2])
    assert o.mode is OutputMode.ALLOCATED and str(o.layout) == "[P0, P1]"
    (o,) = node_outputs((pt.exp(x.T)).owner, [None])
    assert o.layout == C(2)
    (o,) = node_outputs(x.T.owner, [C(2)])
    assert o.mode is OutputMode.VIEW and o.view_of == 0 and o.layout == T2


def test_provided_layouts_are_binding():
    s = pt.tensor("s", shape=(4, 6))
    t = pt.tensor("t", shape=(4, 6))
    mode = CHECKED.clone(link_kwargs={"input_layout": "c", "numba_fallback": False})
    out = pt.exp(s.T) * t.T  # declared: provided memory in the inputs' (transposed) order
    # as tasks of the program (inside a compiled unit the unit does the same)
    f = pytensor.function([s, t], out, mode=mode.excluding("llvm_units"))
    g = pytensor.function([s, t], out, mode=mode)
    p = f.maker.linker.program
    dump = p.dump()
    # a graph output must be C-contiguous: the node writes an arena temporary in its
    # declared order, which is copied into the output
    assert "arena +0, order [1, 0]" in dump and ") -> (%5)  copy" in dump, dump
    sv, tv = rng.normal(size=(4, 6)), rng.normal(size=(4, 6))
    for fn in (f, g):
        r = fn(sv, tv)
        np.testing.assert_allclose(r, np.exp(sv.T) * tv.T)
        assert r.flags.c_contiguous


def test_view_layout_is_checked_when_generating():
    from pytensor.graph.basic import Apply
    from pytensor.graph.op import Op
    from pytensor.link.llvm.codegen.emit import Array
    from pytensor.link.llvm.codegen.function import build_node_function, llvm_emit, llvm_outputs, view_output

    class BadTranspose(Op):
        """Claims its view is C-contiguous, but returns the transpose."""

        view_map = {0: [0]}

        def make_node(self, x):
            return Apply(self, [x], [x.type()])

        def perform(self, node, inputs, out):
            out[0][0] = inputs[0].T

    @llvm_outputs.register(BadTranspose)
    def _(op, node, in_layouts):
        return [view_output(node.outputs[0], C(2))]

    @llvm_emit.register(BadTranspose)
    def _(op, node, cx, inputs, outputs):
        x = inputs[0]
        lay = Layout.make([None, None], (1, 0), (PACKED, PACKED))
        outputs[0].set_view(Array(x.dtype, x.data, x.shape[::-1], x.strides[::-1], lay))

    x = pt.dmatrix("x")
    node = BadTranspose()(x).owner
    with pytest.raises(ValueError, match="declared"):
        build_node_function(ir.Module("bad"), node, "node", in_layouts=[C(2)])


def test_kernels_declare_their_outputs():
    from pytensor.link.llvm.codegen.function import OutputMode
    from pytensor.link.llvm.rust_kernels import rust_kernel_outputs, rust_kernel

    x = pt.dmatrix("x")
    node = x.T.owner
    k = rust_kernel(node.op, node, [C(2)])
    (o,) = rust_kernel_outputs(k, node, [C(2)])
    assert o.mode is OutputMode.VIEW and o.view_of == 0 and o.layout == T2
    # the Rust declaration agrees with the one for generated code
    assert o.layout == node_outputs(node, [C(2)])[0].layout
    a = pt.tensor("a", shape=(3, 4))
    for out, mode in [(pt.dot(x, x), OutputMode.ALLOCATED), (pt.dot(a, a.T), OutputMode.PROVIDED)]:
        node = out.owner
        (o,) = rust_kernel_outputs(rust_kernel(node.op, node, None), node, None)
        assert o.mode is mode and o.layout == C(2)
