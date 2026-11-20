"""Tasks that overwrite an input: what the op destroys (`destroy_map`), where the
planner sees that nothing else reads the memory (the input is `writable`) and the
emitter does."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import In
from pytensor.compile.builders import Execution, OpFromGraph
from pytensor.compile.mode import get_mode
from pytensor.graph.basic import Apply
from pytensor.graph.destroyhandler import DestroyHandler
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op


pytest.importorskip("pytensor_llvm.ir")

from pytensor_llvm import ir

from pytensor.link.llvm.codegen.emit import for_each, load, store
from pytensor.link.llvm.codegen.interface import (
    llvm_emit,
    overwrite_or_copy,
)
from pytensor.link.llvm.codegen.node import build_node_function
from pytensor.link.llvm.codegen.signature import (
    OutputMode,
    destroyed_input,
    node_outputs,
)
from pytensor.link.llvm.plan.memory import lifetimes, place_temporaries
from pytensor.link.llvm.program import Options, build_program
from pytensor.link.numba.dispatch.basic import numba_funcify, numba_njit
from tests.link.llvm.test_aliasing import item


# (fused, a chain of Elemwise is one loop and has nothing to overwrite)
MODE = get_mode("LLVM").excluding("fusion")
P, R = OutputMode.PROVIDED, OutputMode.RETURNED


def compiled(inputs, outputs, mode=MODE):
    f = pytensor.function(inputs, outputs, mode=mode)
    return f, f.maker.linker.stats


def expected(inputs, outputs, *args):
    inputs = [getattr(i, "variable", i) for i in inputs]
    return pytensor.function(inputs, outputs, mode="FAST_COMPILE")(*args)


def validated(inputs, outputs):
    """The program of a graph that already has ops that destroy inputs (`function`
    doesn't take those), with the DestroyHandler that makes its order safe."""
    fgraph = FunctionGraph(inputs, outputs, clone=False)
    fgraph.attach_feature(DestroyHandler())
    program, stats = build_program(fgraph, Options())
    return (lambda *args: program(*args)[0]), stats


X = np.arange(12.0).reshape(3, 4)


def test_elemwise_overwrites_temporaries_but_not_inputs():
    x = pt.matrix("x")
    out = pt.exp(pt.sin(x) * 2) + 1
    f, stats = compiled([x], out)
    # sin reads the caller's array; the others overwrite what came before
    assert stats.inplace == ["Mul", "Exp", "Add"]
    xv = X.copy()
    np.testing.assert_allclose(f(xv), np.exp(np.sin(X) * 2) + 1)
    np.testing.assert_array_equal(xv, X)
    np.testing.assert_allclose(f(xv), np.exp(np.sin(X) * 2) + 1)  # and again


def test_an_input_is_overwritten_when_the_caller_allows_it():
    x = pt.matrix("x")
    f, stats = compiled([In(x, mutable=True)], pt.exp(x))
    assert stats.inplace == ["Exp"]
    xv = X.copy()
    r = f(xv)
    np.testing.assert_allclose(r, np.exp(X))
    np.testing.assert_allclose(xv, np.exp(X))


def test_set_subtensor_overwrites():
    x, y = pt.matrix("x"), pt.matrix("y")
    out = pt.set_subtensor((x * 2)[1:], y[1:])
    f, stats = compiled([x, y], out)
    assert len(stats.inplace) == 1 and "SetSubtensor" in stats.inplace[0]
    np.testing.assert_allclose(f(X, -X), expected([x, y], out, X, -X))

    out = pt.inc_subtensor((x * 2)[[0, 2, 0]], y[0])
    f, stats = compiled([x, y], out)
    assert stats.inplace == ["AdvancedIncSubtensor"]
    np.testing.assert_allclose(f(X, -X), expected([x, y], out, X, -X))


def test_no_overwriting_what_another_input_may_share_memory_with():
    x = pt.matrix("x")
    z = x * 2
    out = pt.set_subtensor(z[1:], z[:-1])  # the rows written are also read
    f, stats = compiled([x], out)
    assert stats.inplace == []
    np.testing.assert_allclose(f(X), expected([x], out, X))


def test_views_can_be_overwritten():
    x = pt.matrix("x")
    out = pt.exp((x * 2).T[1:])
    f, stats = compiled([x], out)
    assert stats.inplace == ["Exp"]
    r = f(X)
    np.testing.assert_allclose(r, np.exp((X * 2).T[1:]))
    assert r.flags.c_contiguous

    # two results from one temporary: only the last reader may overwrite it
    z = x * 2
    outs = [pt.exp(z), pt.sin(z)]
    f, stats = compiled([x], outs)
    assert len(stats.inplace) == 1
    for r, want in zip(f(X), [np.exp(X * 2), np.sin(X * 2)], strict=True):
        np.testing.assert_allclose(r, want)


def test_layouts_are_checked_with_overwriting():
    x = pt.matrix("x")
    out = pt.exp((x * 2).T[::2]) + pt.set_subtensor((x + 1)[:, 1:3], 0.0).T[::2]
    mode = MODE.including()
    mode = type(mode)(mode.linker.clone(), mode.optimizer)
    f = pytensor.function([x], out, mode=mode)
    np.testing.assert_allclose(f(X), expected([x], out, X))


# ------------------------------------------------------------------ what the op says


def test_an_output_may_be_the_input_it_overwrites():
    x = pt.matrix("x")
    assert destroyed_input(pt.exp(x).owner, 0) is None
    inplace = (
        pytensor.function([x], pt.exp(x * 2), mode=MODE).maker.fgraph.outputs[0].owner
    )
    assert inplace.op.destroy_map == {0: [0]}
    assert destroyed_input(inplace, 0) == 0
    (o,) = node_outputs(inplace, writable={0})
    assert o.mode is R and o.aliases == (0,)
    (o,) = node_outputs(inplace)
    assert o.mode is R and o.aliases == ()  # without the planner's word: new memory


class Halve(Op):
    """x / 2, written over x."""

    __props__ = ()
    destroy_map = {0: [0]}

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        (x,) = inputs
        x /= 2
        output_storage[0][0] = x


def _halve(cg, out):
    b = cg.b
    for_each(
        b,
        out.shape,
        [out],
        lambda p: store(b, out.dtype, b.fmul(load(b, out.dtype, p[0]), 0.5), p[0]),
    )


@llvm_emit.register(Halve)
def _(op, node, cg, inputs, outputs):
    _halve(cg, overwrite_or_copy(cg, outputs[0], inputs[0]))


class HalveCopying(Halve):
    """The same, into new memory: its emitter ignores that it may overwrite x."""


@llvm_emit.register(HalveCopying)
def _(op, node, cg, inputs, outputs):
    (x,) = inputs
    out = outputs[0].allocate(x.shape)
    b = cg.b
    for_each(
        b,
        x.shape,
        [x, out],
        lambda p: store(b, out.dtype, b.fmul(load(b, x.dtype, p[0]), 0.5), p[1]),
    )


def test_the_emitter_decides_whether_to_overwrite():
    x = pt.matrix("x")
    for op in [Halve(), HalveCopying()]:
        f, stats = validated([x], [op(x * 2)])
        assert stats.inplace == [str(op)]  # it may
        np.testing.assert_allclose(f(X), X)

    # a graph nobody validated (no DestroyHandler): nothing is overwritten
    program, stats = build_program(
        FunctionGraph([x], [Halve()(x)], clone=False), Options()
    )
    assert stats.inplace == []
    xv = X.copy()
    np.testing.assert_allclose(program(xv)[0], X / 2)
    np.testing.assert_array_equal(xv, X)


def test_only_the_inputs_the_node_may_overwrite_are_writable():
    x, y = pt.matrix("x"), pt.matrix("y")

    class Two(Halve):
        def make_node(self, x, y):
            return Apply(self, [x, y], [x.type()])

    seen = []

    @llvm_emit.register(Two)
    def _(op, node, cg, inputs, outputs):
        seen.append([a.writable for a in inputs])
        inputs[1].store(cg.b, 0.0, 0, 0)

    with pytest.raises(ValueError, match="storing into an input"):
        build_node_function(ir.Module("m"), Two()(x, y).owner, "f", writable={0})
    assert seen == [[True, False]]


# ------------------------------------------------------------------ numba kernels


class Twice(Op):
    """2 x, written over x; only numba implements it."""

    __props__ = ()
    destroy_map = {0: [0]}

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        (x,) = inputs
        x *= 2
        output_storage[0][0] = x


@numba_funcify.register(Twice)
def _(op, node, **kwargs):
    @numba_njit
    def twice(x):
        x *= 2
        return x

    return twice


def test_a_numba_kernel_overwrites_only_what_is_safe():
    x = pt.matrix("x")
    f, stats = validated([x], [Twice()(x + 1) + Twice()(x * 3)])
    assert len(stats.numba) == 2
    np.testing.assert_allclose(f(X), 2 * (X + 1) + 6 * X)
    np.testing.assert_allclose(f(X), 2 * (X + 1) + 6 * X)

    # a graph nobody validated: the kernel overwrites a copy
    program, stats = build_program(
        FunctionGraph([x], [Twice()(x)], clone=False), Options()
    )
    xv = X.copy()
    np.testing.assert_allclose(program(xv)[0], 2 * X)
    np.testing.assert_array_equal(xv, X)


def test_a_numba_kernel_inside_generated_code_overwrites_a_copy():
    # inside a compiled unit nothing grants it the unit's input
    v = pt.matrix("v")
    unit = OpFromGraph([v], [Twice()(v)], execution=Execution.COMPILED)
    x = pt.matrix("x")
    f, stats = compiled([x], unit(x))
    assert len(stats.numba) == 1
    xv = X.copy()
    np.testing.assert_allclose(f(xv), 2 * X)
    np.testing.assert_array_equal(xv, X)


def test_a_value_that_stands_for_several_variables_is_not_overwritten():
    # the inner graph hands its input on: once inlined, `a` and `b` are one value
    v = pt.matrix("v")
    same = OpFromGraph([v], [v], inline=False)
    x = pt.matrix("x")
    a = x * 2
    b = same(a)
    out = pt.exp(b) + a
    f, _ = compiled([x], out)
    np.testing.assert_allclose(f(X), np.exp(2 * X) + 2 * X)
    a = x * 2
    f, _ = validated([x], [Twice()(same(a)) + a])
    np.testing.assert_allclose(f(X), 4 * X + 2 * X)
    # nor one a view the DestroyHandler doesn't see may lie in
    transposed = OpFromGraph([v], [v.T], inline=False)
    a = x * 2
    f, _ = validated([x], [Twice()(a) + transposed(a).T])
    np.testing.assert_allclose(f(X), 4 * X + 2 * X)


# ------------------------------------------------------------------ memory


def test_an_overwritten_value_lives_on_under_another_name():
    items = [
        item([0], [1], [P]),
        item([1], [2], [R], [(0,)]),  # 2 is 1, overwritten
        item([0], [3], [P]),
        item([2, 3], [4], [P]),
    ]
    _, end = lifetimes(items, kept=[4])
    assert end[1] == 3  # alive until 2 was read
    offsets, _ = place_temporaries(items, {1: 64, 3: 64}, kept=[4])
    assert offsets[1] != offsets[3]

    # overwriting the function's output: alive until the end
    _, end = lifetimes([item([0], [1], [P]), item([1], [2], [R], [(0,)])], kept=[2])
    assert end[1] == 2
