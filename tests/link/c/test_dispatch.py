import numpy as np
import pytest

import pytensor
import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor.compile.debug.debugmode import BadThunkOutput, DebugMode
from pytensor.compile.mode import Mode
from pytensor.configdefaults import config
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.graph.utils import MethodNotDefined
from pytensor.link.c.basic import CLinker
from pytensor.link.c.dispatch.basic import (
    CImpl,
    c_funcify,
    c_thunk_from_dispatch,
)
from pytensor.link.vm import VMLinker
from pytensor.tensor.shape import Shape, Shape_i


pytestmark = pytest.mark.skipif(
    not config.cxx, reason="A C compiler is required to test the C dispatch"
)

CVM_MODE = Mode(linker="cvm", optimizer=None)
PY_MODE = Mode(linker="py", optimizer=None)


class ScalarOpBase(Op):
    """A pure scalar op: only `make_node` and `perform`."""

    __props__ = ()
    increment = 1.0

    def make_node(self, x):
        x = ps.as_scalar(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        (x,) = inputs
        output_storage[0][0] = np.dtype(node.outputs[0].dtype).type(x + self.increment)


class IncOne(ScalarOpBase):
    pass


class IncOneNoImpl(ScalarOpBase):
    pass


class IncOneDeclining(ScalarOpBase):
    pass


class IncOneImpl(CImpl):
    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return f"{z} = {x} + 1;"

    def c_code_cache_version(self):
        return (1,)


class DecliningImpl(CImpl):
    def c_code(self, node, name, inputs, outputs, sub):
        raise MethodNotDefined("c_code")

    def c_code_cache_version(self):
        return (1,)


@c_funcify.register(IncOne)
def c_funcify_inc_one(op, node=None, **kwargs):
    return IncOneImpl(op)


@c_funcify.register(IncOneDeclining)
def c_funcify_declining(op, node=None, **kwargs):
    return DecliningImpl(op)


def make_thunk_for(op, x_value=2.0, dtype="float64"):
    x = ps.ScalarType(dtype)("x")
    out = op(x)
    node = out.owner
    storage_map = {x: [np.dtype(dtype).type(x_value)], out: [None]}
    compute_map = {x: [True], out: [False]}
    thunk = c_thunk_from_dispatch(node, storage_map, compute_map, [])
    return thunk, storage_map, compute_map, out


def test_pure_op_gains_c_thunk():
    thunk, storage_map, compute_map, out = make_thunk_for(IncOne())

    assert hasattr(thunk, "cthunk")
    assert thunk.lazy is False
    assert thunk.inputs == [storage_map[out.owner.inputs[0]]]
    assert thunk.outputs == [storage_map[out]]

    thunk()
    assert storage_map[out][0] == 3.0
    assert compute_map[out][0] is True


def test_pure_op_cvm_function_matches_perform():
    x = ps.float64("x")
    out = IncOne()(x)

    f_c = pytensor.function([x], out, mode=CVM_MODE)
    f_py = pytensor.function([x], out, mode=PY_MODE)
    assert f_c(2.0) == f_py(2.0) == 3.0


def test_unregistered_pure_op_falls_back():
    op = IncOneNoImpl()
    with pytest.raises(NotImplementedError, match="No C implementation registered"):
        c_funcify(op)

    x = ps.float64("x")
    f = pytensor.function([x], op(x), mode=CVM_MODE)
    assert f(2.0) == 3.0


def test_declining_impl_falls_back():
    op = IncOneDeclining()
    with pytest.raises(MethodNotDefined):
        make_thunk_for(op)

    x = ps.float64("x")
    f = pytensor.function([x], op(x), mode=CVM_MODE)
    assert f(2.0) == 3.0


def test_cop_is_its_own_impl():
    op = Shape()
    assert c_funcify(op) is op


def test_float16_guard_falls_back():
    # The guard raises NotImplementedError either way; the warning is only
    # emitted when the impl's C code builds for f16, but ScalarType's own C
    # support rejects f16 first.
    op = IncOne()
    with pytest.raises(NotImplementedError):
        make_thunk_for(op, dtype="float16")

    x = ps.ScalarType("float16")("x")
    f = pytensor.function([x], op(x), mode=CVM_MODE)
    assert f(np.float16(2.0)) == np.float16(3.0)


def test_vm_without_c_thunks_skips_dispatch(monkeypatch):
    def fail_dispatch(*args, **kwargs):
        raise AssertionError("dispatch should not run when c_thunks=False")

    monkeypatch.setattr(
        "pytensor.link.c.dispatch.basic.c_thunk_from_dispatch", fail_dispatch
    )

    x = ps.float64("x")
    mode = Mode(linker=VMLinker(use_cloop=False, c_thunks=False), optimizer=None)
    f = pytensor.function([x], IncOne()(x), mode=mode)
    assert f(2.0) == 3.0


@pytest.mark.parametrize("linker", ["c", "c|py"])
def test_whole_graph_linkers_use_dispatch(linker):
    x = ps.float64("x")
    f = pytensor.function([x], IncOne()(x), mode=Mode(linker=linker, optimizer=None))
    assert f(2.0) == 3.0


def test_whole_graph_c_linker_unregistered_raises():
    x = ps.float64("x")
    with pytest.raises(NotImplementedError, match="cannot produce C code"):
        pytensor.function([x], IncOneNoImpl()(x), mode=Mode(linker="c", optimizer=None))


def test_cmodule_key_stable_and_versioned():
    def key_for_fresh_graph():
        x = ps.float64("x")
        out = IncOne()(x)
        fgraph = FunctionGraph([x], [out])
        return CLinker().accept(fgraph).cmodule_key()

    key_a = key_for_fresh_graph()
    key_b = key_for_fresh_graph()
    assert key_a == key_b

    version, _sig = key_a
    # The registered impl's cache version makes the module versioned (cacheable
    # across processes), even though the graph op itself has no C methods.
    assert version != ()
    assert IncOneImpl(IncOne()).c_code_cache_version() in version


def test_params_constants_deduplicated_across_nodes():
    x = pt.matrix("x")
    y = pt.matrix("y")
    out = Shape_i(0)(x) + Shape_i(0)(y)

    fgraph = FunctionGraph([x, y], [out])
    cl = CLinker().accept(fgraph)
    shape_i_nodes = [n for n in cl.node_order if isinstance(n.op, Shape_i)]
    assert len(shape_i_nodes) == 2
    # Both Shape_i(0) nodes share one params Constant.
    assert len(cl.node_params) == 1

    f = pytensor.function([x, y], out, mode=Mode(linker="c", optimizer=None))
    assert f(np.ones((3, 2)), np.ones((5, 2))) == 8


class IncOneWrongImpl(ScalarOpBase):
    pass


class WrongImpl(CImpl):
    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return f"{z} = {x} + 2;"  # disagrees with perform on purpose

    def c_code_cache_version(self):
        return (1,)


@c_funcify.register(IncOneWrongImpl)
def c_funcify_wrong(op, node=None, **kwargs):
    return WrongImpl(op)


def test_cop_graph_resolves_to_identity():
    # An unregistered COp node resolves to itself, so CLinker calls the op's own
    # c_code/cache-version methods. A registered op resolves to its detached impl.
    from pytensor.link.c.dispatch.elemwise import DimShuffleImpl, ElemwiseImpl
    from pytensor.tensor.elemwise import DimShuffle, Elemwise

    x = pt.matrix("x")
    out = (x.T + 1.0).sum(axis=0)
    fgraph = FunctionGraph([x], [out])
    cl = CLinker().accept(fgraph)

    for node in cl.node_order:
        impl = cl._impl_for(node)
        if isinstance(node.op, DimShuffle):
            assert isinstance(impl, DimShuffleImpl)
        elif isinstance(node.op, Elemwise):
            assert isinstance(impl, ElemwiseImpl)
        else:
            # A genuine COp (e.g. the CAReduce sum) resolves to itself.
            assert impl is node.op

    # Source generation works and the module is versioned (cacheable).
    assert isinstance(cl.get_src_code(), str)
    version, _sig = cl.cmodule_key()
    assert version != ()


def test_debugmode_cross_checks_dispatch_impl():
    x = ps.float64("x")
    f = pytensor.function(
        [x], IncOne()(x), mode=DebugMode(optimizer=None, check_py_code=True)
    )
    assert f(2.0) == 3.0

    f_wrong = pytensor.function(
        [x], IncOneWrongImpl()(x), mode=DebugMode(optimizer=None, check_py_code=True)
    )
    with pytest.raises(BadThunkOutput):
        f_wrong(2.0)
