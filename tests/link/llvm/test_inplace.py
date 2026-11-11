"""Tasks that overwrite an input: what the op destroys (`destroy_map`), where the
emitter can (`llvm_inplace`) and the planner sees that nothing else reads the memory."""

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import In
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.mode import get_mode
from pytensor.graph.basic import Apply
from pytensor.graph.destroyhandler import DestroyHandler
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op


pytest.importorskip("pytensor_llvm.ir")

from pytensor.link.llvm.codegen.emit import copy, for_each, load, store  # noqa: E402
from pytensor.link.llvm.codegen.function import (  # noqa: E402
    OutputMode,
    inplace_outputs,
    llvm_emit,
    llvm_inplace,
    node_outputs,
)
from pytensor.link.llvm.plan.memory import plan_memory  # noqa: E402
from pytensor.link.llvm.program import Options, build_program  # noqa: E402
from pytensor.link.numba.dispatch.basic import numba_funcify, numba_njit  # noqa: E402
from tests.link.llvm.test_aliasing import task, values_of  # noqa: E402


# (fused, a chain of Elemwise is one loop and has nothing to overwrite)
MODE = get_mode("LLVM").excluding("fusion")
P, R, I = OutputMode.PROVIDED, OutputMode.RETURNED, OutputMode.INPLACE


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


def test_only_what_the_op_destroys_and_the_emitter_can_overwrite():
    x = pt.matrix("x")
    plain = pt.exp(x).owner
    assert inplace_outputs(plain, {}) == {}
    inplace = (
        pytensor.function([x], pt.exp(x * 2), mode=MODE).maker.fgraph.outputs[0].owner
    )
    assert inplace.op.destroy_map == {0: [0]}
    assert llvm_inplace(inplace.op, inplace) == {0: 0}
    assert inplace_outputs(inplace, {}) == {0: 0}
    (o,) = node_outputs(inplace, inplace={0: 0})
    assert o.mode is I and o.aliases == (0,)
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


@llvm_emit.register(Halve)
def _(op, node, cg, inputs, outputs):
    (x,) = inputs
    out = outputs[0].overwrite(cg, x)  # x's memory if we may, else a copy of x
    b = cg.b
    for_each(
        b,
        out.shape,
        [out],
        lambda p: store(b, out.dtype, b.fmul(load(b, out.dtype, p[0]), 0.5), p[0]),
    )


class HalveInPlace(Halve):
    """The same, and the emitter says it can overwrite."""


@llvm_inplace.register(HalveInPlace)
def _(op, node):
    return {0: 0}


def test_an_emitter_that_doesnt_opt_in_gets_a_copy():
    x = pt.matrix("x")
    f, stats = validated([x], [Halve()(x * 2)])
    assert stats.inplace == []  # (no `llvm_inplace` for Halve)
    np.testing.assert_allclose(f(X), X)

    f, stats = validated([x], [HalveInPlace()(x * 2)])
    assert stats.inplace == ["HalveInPlace"]
    np.testing.assert_allclose(f(X), X)

    # a graph nobody validated (no DestroyHandler): nothing is overwritten
    program, stats = build_program(
        FunctionGraph([x], [HalveInPlace()(x)], clone=False), Options()
    )
    assert stats.inplace == []
    xv = X.copy()
    np.testing.assert_allclose(program(xv)[0], X / 2)
    np.testing.assert_array_equal(xv, X)


def test_overwrite_takes_the_destroyed_input_only():
    x, y = pt.matrix("x"), pt.matrix("y")

    class Wrong(Halve):
        def make_node(self, x, y):
            return Apply(self, [x, y], [x.type()])

    @llvm_emit.register(Wrong)
    def _(op, node, cg, inputs, outputs):
        copy(cg, inputs[0], outputs[0].overwrite(cg, inputs[1]))

    llvm_inplace.register(Wrong, lambda op, node: {0: 0})
    from pytensor.link.llvm.codegen.function import build_node_function
    from pytensor_llvm import ir

    with pytest.raises(ValueError, match="overwrites input 0"):
        build_node_function(ir.Module("m"), Wrong()(x, y).owner, "f", inplace={0: 0})


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


# ------------------------------------------------------------------ memory


def test_an_overwritten_value_lives_on_under_another_name():
    v = values_of((8,), (8,), (8,), (8,), (8,))
    tasks = [
        task([0], [1], [P]),
        task([1], [2], [I], [(0,)]),  # 2 is 1, overwritten
        task([0], [3], [P]),
        task([2, 3], [4], [P]),
    ]
    m = plan_memory(v, tasks, in_ids=[0], out_ids=[4])
    assert m.same == {2: 1} and m.value(2) == 1 and m.value(3) == 3
    assert 2 not in m.kind and m.kind[1] == "provided"
    assert (m.defined[1], m.end[1]) == (0, 3)  # alive until 2 was read
    assert m.offsets[1] != m.offsets[3]

    # overwriting the program's output: the buffer handed back
    tasks = [task([0], [1], [P]), task([1], [2], [I], [(0,)])]
    m = plan_memory(v, tasks, in_ids=[0], out_ids=[2])
    assert m.same == {2: 1} and 1 not in m.offsets and m.arena_bytes == 0
