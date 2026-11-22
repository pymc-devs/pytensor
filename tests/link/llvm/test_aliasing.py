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

import pytensor.scalar as ps  # noqa: E402
from pytensor.link.llvm.analysis.layout import Layout  # noqa: E402
from pytensor.link.llvm.analysis.liveness import (  # noqa: E402
    first_fit,
    lifetimes,
    place_temporaries,
)
from pytensor.link.llvm.codegen.emit import Array, load, unknown_layout  # noqa: E402
from pytensor.link.llvm.codegen.interface import llvm_emit, llvm_layout  # noqa: E402
from pytensor.link.llvm.codegen.signature import (  # noqa: E402
    Output,
    Passing,
    Signature,
    node_outputs,
)
from pytensor.link.llvm.codegen.types import TensorRepr  # noqa: E402


STRICT = get_mode("LLVM").clone(link_kwargs={"numba_fallback": False})
P, R = Passing.CALLER_BUFFER, Passing.CALLEE_SLOT


# ------------------------------------------------------------------ what the op says


def test_how_an_output_comes_to_be_follows_from_the_op_and_the_type():
    x = pt.matrix("x")
    static = pt.tensor("s", shape=(3, 4))
    (o,) = node_outputs(pt.exp(static).owner)
    assert o.passing is P and o.aliased_inputs == ()  # new memory of a known size
    (o,) = node_outputs(pt.exp(x).owner)
    assert o.passing is R and o.aliased_inputs == ()  # new memory: the node allocates
    (o,) = node_outputs(x.T.owner, [Layout.c(2)])
    assert (
        o.passing is R and o.aliased_inputs == (0,) and str(o.layout) == "[P0, P1]"
    )  # a view
    (o,) = node_outputs(static.T.owner)
    assert o.passing is R and o.aliased_inputs == (0,)  # ... also with a static shape


# ------------------------------------------------------------------ memory planning


def step(inputs, outputs, passings, aliases=None):
    """A step as `analysis.liveness` sees it: arguments, outputs, how they are passed
    and the arguments they may be views of."""
    aliases = aliases or [()] * len(outputs)
    representation = TensorRepr(ps.float64, (8,), Layout.c(1))
    outs = [
        Output(representation, passing, a)
        for passing, a in zip(passings, aliases, strict=True)
    ]
    return SimpleNamespace(
        arguments=inputs,
        outputs=outputs,
        signature=Signature((), tuple(outs)),
    )


def test_a_value_lives_as_long_as_what_may_alias_it():
    steps = [
        step([0], [1], [P]),  # 1: new
        step([1], [2], [R], [(0,)]),  # 2 may be a view of 1
        step([2], [3], [R], [(0,)]),  # 3 may be a view of 2 (and so of 1)
        step([0], [4], [P]),  # 4: new, while 3 is still needed
        step([3, 4], [5], [P]),
    ]
    _, end = lifetimes(steps, kept=[5])
    assert end[1] == end[2] == end[3] == 4  # the chain keeps 1 alive until 3 is read
    offsets, _ = place_temporaries(steps, {1: 64, 4: 64}, kept=[5])
    assert offsets[1] != offsets[4]  # so 4 can't take 1's place


def test_several_candidates_are_all_kept_alive():
    steps = [
        step([0], [1], [P]),
        step([0], [2], [P]),
        step([1, 2], [3], [R], [(0, 1)]),  # a view of 1 or of 2
        step([0], [4], [P]),
        step([3, 4], [5], [P]),
    ]
    _, end = lifetimes(steps, kept=[5])
    assert end[1] == end[2] == 4
    offsets, _ = place_temporaries(steps, {1: 64, 2: 64, 4: 64}, kept=[5])
    assert len(set(offsets.values())) == 3


def test_arena_space_is_reused_when_nothing_needs_it():
    steps = [step([0], [1], [P]), step([1], [2], [P]), step([2], [3], [P])]
    # (3 is the output: not a temporary)
    assert place_temporaries(steps, {1: 8, 2: 8}, kept=[3]) == ({1: 0, 2: 64}, 128)
    steps.append(step([3], [4], [P]))
    assert place_temporaries(steps, {1: 8, 2: 8, 3: 8}, kept=[4]) == (
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
def _(op, node, input_layouts):
    _, a, b = input_layouts
    return [a.join(b, (None,), 8)]  # what both views have


@llvm_emit.register(Pick)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    flag, x, y = inputs
    with b.if_(b.ne(load(b, flag.scalar_type, flag.data), 0)):
        outputs[0].set_view(x)
    with b.else_():
        outputs[0].set_view(y)


pick = Pick()


def test_an_output_may_alias_several_inputs():
    flag = pt.scalar("flag", dtype="int8")
    x, y = pt.vector("x"), pt.vector("y")
    (o,) = node_outputs(pick(flag, x, y).owner)
    assert o.passing is R and o.aliased_inputs == (1, 2)

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


def test_an_emitter_only_returns_views_the_op_declares():
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
        stride = cg.b.mul(x.strides[0], 2)
        outputs[0].set_view(x.view(x.data, [cg.b.i64(2)], [stride]))

    x = pt.vector("x", shape=(4,))
    v = np.arange(4.0)
    # with a view_map, the output may be a view
    (o,) = node_outputs(Passes(True)(x).owner)
    assert o.aliased_inputs == (0,)
    f = pytensor.function([x], Passes(True)(x) + 1, mode=STRICT)
    np.testing.assert_array_equal(f(v), v[::2] + 1)
    # without one, it is new memory, and the emitter has no view to return
    (o,) = node_outputs(Passes(False)(x).owner)
    assert o.aliased_inputs == ()
    with pytest.raises(ValueError, match="view_map doesn't list it"):
        pytensor.function([x], Passes(False)(x) + 1, mode=STRICT)


def test_inputs_are_read_only():
    m = ir.Module("m")
    f = m.function("f", [ir.ptr], ir.i32)
    b = f.builder()
    shape, strides = [b.i64(4)], [b.i64(8)]
    layout = unknown_layout(shape)
    x = Array(ps.float64, f.params[0], shape, strides, layout, base=0, writable=False)
    with pytest.raises(ValueError, match="storing into an input"):
        x.store(b, 1.0, 0)
    with pytest.raises(ValueError, match="storing into an input"):
        x.at(b, 0).store(b, 1.0)  # views of an input are read-only too
    assert x.view(x.data, x.shape, x.strides).base == 0
    # other memory is fine
    Array(ps.float64, f.params[0], shape, strides, layout).store(b, 1.0, 0)
