"""Which memory outputs may share: the op's view_map and destroy_map decide."""

from types import SimpleNamespace

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import get_mode
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op


ir = pytest.importorskip("pytensor_llvm.ir")

from pytensor.link.llvm.codegen.emit import Array, load, set_output  # noqa: E402
from pytensor.link.llvm.codegen.function import (  # noqa: E402
    OutputMode,
    build_node_function,
    llvm_emit,
    llvm_layout,
    node_outputs,
    output_aliases,
)
from pytensor.link.llvm.layout import Layout  # noqa: E402
from pytensor.link.llvm.plan.memory import first_fit, plan_memory  # noqa: E402
from pytensor.link.llvm.plan.values import Values  # noqa: E402


STRICT = get_mode("LLVM").clone(link_kwargs={"numba_fallback": False})
P, R = OutputMode.PROVIDED, OutputMode.RETURNED


# ------------------------------------------------------------------ what the op says


def test_aliases_come_from_the_ops_maps():
    op = SimpleNamespace(view_map={0: [1]}, destroy_map={})
    assert output_aliases(op, 0) == (1,) and output_aliases(op, 1) == ()
    # the output of an op that works in place is the destroyed input's memory
    op = SimpleNamespace(view_map={}, destroy_map={0: [0]})
    assert output_aliases(op, 0) == (0,)
    op = SimpleNamespace(view_map={0: [0, 2]}, destroy_map={0: [0], 1: [1]})
    assert output_aliases(op, 0) == (0, 2) and output_aliases(op, 1) == (1,)
    assert output_aliases(SimpleNamespace(), 0) == ()


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


def task(inputs, outputs, modes, aliases=None):
    return SimpleNamespace(
        inputs=inputs,
        outputs=outputs,
        modes=modes,
        aliases=aliases or [()] * len(outputs),
    )


def values_of(*shapes):
    v = Values()
    for shape in shapes:
        v.new(pt.tensor(dtype="float64", shape=shape).type)
    return v


def test_a_value_lives_as_long_as_what_may_alias_it():
    v = values_of((8,), (8,), (8,), (8,), (8,))
    tasks = [
        task([0], [1], [P]),  # 1: new
        task([1], [2], [R], [(0,)]),  # 2 may be a view of 1
        task([2], [3], [R], [(0,)]),  # 3 may be a view of 2 (and so of 1)
        task([0], [4], [P]),  # 4: new, while 3 is still needed
        task([3, 4], [5], [P]),
    ]
    v.new(pt.tensor(dtype="float64", shape=(8,)).type)
    m = plan_memory(v, tasks, in_ids=[0], out_ids=[5])
    assert m.kind == {
        0: "input",
        1: "provided",
        2: "returned",
        3: "returned",
        4: "provided",
        5: "provided",
    }
    assert m.aliases == {2: [1], 3: [2]}
    assert (
        m.end[1] == m.end[2] == m.end[3] == 4
    )  # the chain keeps 1 alive until 3 is read
    assert m.offsets[1] != m.offsets[4]  # so 4 can't take 1's place
    assert m.release_points(len(tasks))[4] == [1, 2, 3, 4]


def test_several_candidates_are_all_kept_alive():
    v = values_of((8,), (8,), (8,), (8,), (8,), (8,))
    tasks = [
        task([0], [1], [P]),
        task([0], [2], [P]),
        task([1, 2], [3], [R], [(0, 1)]),  # a view of 1 or of 2
        task([0], [4], [P]),
        task([3, 4], [5], [P]),
    ]
    m = plan_memory(v, tasks, in_ids=[0], out_ids=[5])
    assert m.aliases[3] == [1, 2]
    assert m.end[1] == m.end[2] == 4
    assert len({m.offsets[1], m.offsets[2], m.offsets[4]}) == 3


def test_arena_space_is_reused_when_nothing_needs_it():
    v = values_of((8,), (8,), (8,), (8,))
    tasks = [task([0], [1], [P]), task([1], [2], [P]), task([2], [3], [P])]
    m = plan_memory(v, tasks, in_ids=[0], out_ids=[3])
    assert m.offsets == {1: 0, 2: 64}  # (3 is the output: a fresh buffer)
    assert m.arena_bytes == 128
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
        outputs[0].set_view(x.view(x.data, x.shape, x.strides, layout))
    with b.else_():
        outputs[0].set_view(y.view(y.data, y.shape, y.strides, layout))


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
    g = pytensor.function([flag, x, y], [picked * 2, picked[::-1] + 1], mode=STRICT)
    for on, want in [(1, np.exp(xv)), (0, np.cumsum(yv))]:
        r1, r2 = g(on, xv, yv)
        np.testing.assert_allclose(r1, want * 2)
        np.testing.assert_allclose(r2, want[::-1] + 1)
    dump = g.maker.linker.program.dump()
    assert "may be a view of %" in dump and dump.count("Pick") == 1


def test_views_are_checked_against_the_view_map():
    x, y = pt.vector("x"), pt.vector("y")

    class Wrong(Pick):
        view_map = {0: [1]}

    @llvm_emit.register(Wrong)
    def _(op, node, cg, inputs, outputs):
        outputs[0].set_view(inputs[2])  # input 2 isn't in the view_map

    with pytest.raises(
        ValueError,
        match=r"a view of input 2, but the op's view_map only lets this output alias \[1\]",
    ):
        build_node_function(
            ir.Module("m"), Wrong()(pt.scalar("f", dtype="int8"), x, y).owner, "node"
        )

    class NoView(Op):
        def make_node(self, a):
            return Apply(self, [a], [a.type()])

        def perform(self, node, inputs, output_storage):
            output_storage[0][0] = inputs[0].copy()

    @llvm_emit.register(NoView)
    def _(op, node, cg, inputs, outputs):
        outputs[0].set_view(inputs[0])

    with pytest.raises(ValueError, match="this output can't be a view"):
        build_node_function(ir.Module("m"), NoView()(x).owner, "node")


def test_set_output_copies_where_a_view_isnt_allowed():
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
        set_output(
            cg, outputs[0], x.view(x.data, [cg.b.i64(2)], [cg.b.mul(x.strides[0], 2)])
        )

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
    x = Array("float64", f.params[0], [b.i64(4)], [b.i64(8)], base=0, writable=False)
    with pytest.raises(ValueError, match="storing into an input"):
        x.store(b, 1.0, 0)
    with pytest.raises(ValueError, match="storing into an input"):
        x.at(b, 0).store(b, 1.0)  # views of an input are read-only too
    assert x.view(x.data, x.shape, x.strides).base == 0
    Array("float64", f.params[0], [b.i64(4)], [b.i64(8)]).store(
        b, 1.0, 0
    )  # other memory is fine
