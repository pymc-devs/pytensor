"""Which memory outputs may share: the op's view_map decides."""

from types import SimpleNamespace

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import get_mode
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op


ir = pytest.importorskip("pytensor_llvm.ir")

from pytensor.link.llvm.codegen.emit import Array, load, unknown_layout  # noqa: E402
from pytensor.link.llvm.codegen.interface import llvm_emit, llvm_layout  # noqa: E402
from pytensor.link.llvm.codegen.signature import OutputMode, node_outputs  # noqa: E402
from pytensor.link.llvm.layout import Layout  # noqa: E402
from pytensor.link.llvm.plan.memory import (  # noqa: E402
    first_fit,
    lifetimes,
    place_temporaries,
)


STRICT = get_mode("LLVM").clone(link_kwargs={"numba_fallback": False})
P, R = OutputMode.PROVIDED, OutputMode.RETURNED


# ------------------------------------------------------------------ what the op says


def test_how_an_output_comes_to_be_follows_from_the_op_and_the_type():
    x = pt.matrix("x")
    static = pt.tensor("s", shape=(3, 4))
    (o,) = node_outputs(pt.exp(static).owner)
    assert o.mode is P and o.aliases == ()  # new memory of a known size: provided
    (o,) = node_outputs(pt.exp(x).owner)
    assert o.mode is R and o.aliases == ()  # new memory: the kernel allocates
    (o,) = node_outputs(x.T.owner, [Layout.c(2)])
    assert (
        o.mode is R and o.aliases == (0,) and str(o.layout) == "[P0, P1]"
    )  # may be a view
    (o,) = node_outputs(static.T.owner)
    assert o.mode is R and o.aliases == (0,)  # ... also with a static shape


# ------------------------------------------------------------------ memory planning


def item(inputs, outputs, modes, aliases=None):
    """An item as `plan.memory` sees it: arguments, outputs, their modes and aliases
    (argument positions)."""
    aliases = aliases or [()] * len(outputs)
    outs = [SimpleNamespace(mode=m, aliases=a) for m, a in zip(modes, aliases)]
    return SimpleNamespace(
        arguments=inputs,
        item=SimpleNamespace(outputs=outputs),
        signature=SimpleNamespace(outputs=outs),
    )


def test_a_value_lives_as_long_as_what_may_alias_it():
    items = [
        item([0], [1], [P]),  # 1: new
        item([1], [2], [R], [(0,)]),  # 2 may be a view of 1
        item([2], [3], [R], [(0,)]),  # 3 may be a view of 2 (and so of 1)
        item([0], [4], [P]),  # 4: new, while 3 is still needed
        item([3, 4], [5], [P]),
    ]
    _, end = lifetimes(items, kept=[5])
    assert end[1] == end[2] == end[3] == 4  # the chain keeps 1 alive until 3 is read
    offsets, _ = place_temporaries(items, {1: 64, 4: 64}, kept=[5])
    assert offsets[1] != offsets[4]  # so 4 can't take 1's place


def test_several_candidates_are_all_kept_alive():
    items = [
        item([0], [1], [P]),
        item([0], [2], [P]),
        item([1, 2], [3], [R], [(0, 1)]),  # a view of 1 or of 2
        item([0], [4], [P]),
        item([3, 4], [5], [P]),
    ]
    _, end = lifetimes(items, kept=[5])
    assert end[1] == end[2] == 4
    offsets, _ = place_temporaries(items, {1: 64, 2: 64, 4: 64}, kept=[5])
    assert len(set(offsets.values())) == 3


def test_arena_space_is_reused_when_nothing_needs_it():
    items = [item([0], [1], [P]), item([1], [2], [P]), item([2], [3], [P])]
    # (3 is the output: not a temporary)
    assert place_temporaries(items, {1: 8, 2: 8}, kept=[3]) == ({1: 0, 2: 64}, 128)
    items.append(item([3], [4], [P]))
    assert place_temporaries(items, {1: 8, 2: 8, 3: 8}, kept=[4]) == (
        {1: 0, 2: 64, 3: 0},
        128,
    )
    offsets, total = first_fit(
        [(0, 1, 100, "a"), (1, 2, 10, "b"), (2, 3, 100, "c"), (3, 3, 10, "d")]
    )
    assert offsets == {"a": 0, "b": 128, "c": 0, "d": 128} and total == 192


# ------------------------------------------------------------------ an op that may alias two inputs


class Pick(Op):
    """`a` if the flag is set, else `b`: a view of either."""

    __props__ = ()
    view_map = {0: [1, 2]}

    def make_node(self, flag, a, b):
        flag, a, b = (
            pt.as_tensor_variable(flag),
            pt.as_tensor_variable(a),
            pt.as_tensor_variable(b),
        )
        return Apply(self, [flag, a, b], [a.type()])

    def perform(self, node, inputs, output_storage):
        flag, a, b = inputs
        output_storage[0][0] = a if flag else b


@llvm_layout.register(Pick)
def _(op, node, in_layouts):
    _, a, b = in_layouts
    return [a.join(b, (None,), 8)]  # what both views have


@llvm_emit.register(Pick)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    flag, x, y = inputs
    layout = x.layout.join(y.layout, (None,), 8)
    with b.if_(b.ne(load(b, flag.dtype, flag.data), 0)):
        outputs[0].view(x.view(x.data, x.shape, x.strides, layout))
    with b.else_():
        outputs[0].view(y.view(y.data, y.shape, y.strides, layout))


pick = Pick()


def test_an_output_may_alias_several_inputs():
    flag = pt.scalar("flag", dtype="int8")
    x, y = pt.vector("x"), pt.vector("y")
    (o,) = node_outputs(pick(flag, x, y).owner)
    assert o.mode is R and o.aliases == (1, 2)

    xv, yv = np.arange(5.0), -np.arange(3.0)
    # As the graph's output: a copy (the caller's arrays aren't handed back). PyTensor's
    # own function compilation doesn't accept such an op yet, so the program is built
    # from the graph directly.
    from pytensor.graph.fg import FunctionGraph
    from pytensor.link.llvm.program import Options, build_program

    program, _ = build_program(
        FunctionGraph([flag, x, y], [pick(flag, x, y)], clone=False),
        Options(numba_fallback=False),
    )
    for on, want in [(1, xv), (0, yv)]:
        (r,) = program(np.int8(on), xv, yv)
        np.testing.assert_array_equal(r, want)
        assert not np.shares_memory(r, want)
    # of temporaries, read after both of them were last used otherwise
    picked = pick(flag, pt.exp(x), pt.cumsum(y))
    # (PyTensor's DestroyHandler, which the inplace rewrites bring, doesn't take an
    # output that views several inputs)
    mode = STRICT.excluding("inplace")
    g = pytensor.function([flag, x, y], [picked * 2, picked[::-1] + 1], mode=mode)
    for on, want in [(1, np.exp(xv)), (0, np.cumsum(yv))]:
        r1, r2 = g(on, xv, yv)
        np.testing.assert_allclose(r1, want * 2)
        np.testing.assert_allclose(r2, want[::-1] + 1)
    dump = g.maker.linker.program.dump()
    assert "may be (a view of) %" in dump and dump.count("Pick") == 1


def test_a_view_is_copied_where_the_output_cant_be_one():
    class Passes(Op):
        def __init__(self, view):
            if view:
                self.view_map = {0: [0]}

        def make_node(self, a):
            return Apply(self, [a], [pt.tensor(dtype=a.type.dtype, shape=(2,))])

        def perform(self, node, inputs, output_storage):
            output_storage[0][0] = inputs[0][::2]

    @llvm_emit.register(Passes)
    def _(op, node, cg, inputs, outputs):
        (x,) = inputs
        outputs[0].view(x.view(x.data, [cg.b.i64(2)], [cg.b.mul(x.strides[0], 2)]))

    x = pt.vector("x", shape=(4,))
    v = np.arange(4.0)
    for view in (True, False):
        op = Passes(view)
        (o,) = node_outputs(op(x).owner)
        assert (o.aliases == (0,)) is view
        f = pytensor.function([x], op(x) + 1, mode=STRICT)
        np.testing.assert_array_equal(f(v), v[::2] + 1)


def test_inputs_are_read_only():
    m = ir.Module("m")
    f = m.function("f", [ir.ptr], ir.i32)
    b = f.builder()
    shape, strides = [b.i64(4)], [b.i64(8)]
    layout = unknown_layout(shape)
    x = Array("float64", f.params[0], shape, strides, layout, base=0, writable=False)
    with pytest.raises(ValueError, match="storing into an input"):
        x.store(b, 1.0, 0)
    with pytest.raises(ValueError, match="storing into an input"):
        x.at(b, 0).store(b, 1.0)  # views of an input are read-only too
    assert x.view(x.data, x.shape, x.strides).base == 0
    # other memory is fine
    Array("float64", f.params[0], shape, strides, layout).store(b, 1.0, 0)
