"""Backend-agnostic tests for cross-backend shared-value coherence.

A minimal backend (`CounterLinker`) whose native representation of a divergent
`CounterType` differs from the host stands in for a real backend such as JAX.
Compiling and calling real functions on it exercises the actual path -- the
reconcile/mark bracket in `Function.__call__` and the storage binding in
`Linker._bind_shared_backend_storage` -- so the behaviour we care about (a write
by either side reaching the next read on the other) is tested end-to-end without
depending on JAX. `CounterLinker` is a plain (non-JIT) `PerformLinker`, which
also shows the divergence machinery is not tied to JIT backends.
"""

import copy
import warnings

import numpy as np
import pytest

import pytensor
from pytensor.compile.mode import Mode
from pytensor.compile.sharedvalue import SharedVariable, shared
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.graph.type import Type
from pytensor.link.backend_conversion import (
    _CONVERSIONS,
    BackendConversion,
    register_backend_conversion,
)
from pytensor.link.basic import PerformLinker
from pytensor.tensor.type import TensorType


class CounterType(Type):
    """Divergent scalar type: the host holds a python int, the backend a dict."""

    __props__ = ()
    is_backend_divergent = True

    def filter(self, data, strict=False, allow_downcast=None):
        return int(data)


counter_type = CounterType()
_int_scalar = TensorType("int64", ())


class Step(Op):
    """(counter) -> (next_counter, value): read the counter, return it, advance it."""

    __props__ = ()

    def make_node(self, c):
        return Apply(self, [c], [counter_type(), _int_scalar()])

    def perform(self, node, inputs, outputs):
        (c,) = inputs
        # The op runs on the backend-native representation (bound in by the
        # linker), not the host int -- proof the storage binding took effect.
        assert isinstance(c, dict)
        n = c["count"]
        outputs[0][0] = {"count": n + 1}
        outputs[1][0] = np.int64(n)


class CounterLinker(PerformLinker):
    """An op-by-op backend that runs `perform` on the native representation."""

    def __init__(self, tag, **kwargs):
        self.backend_tag = tag
        super().__init__(**kwargs)


def _make_backend(tag, *, lossy=False):
    register_backend_conversion(
        BackendConversion(
            tag=tag,
            handles=lambda t: isinstance(t, CounterType),
            to_native=lambda h: {"count": int(h)},
            from_native=lambda n: n["count"],
            lossy=lossy,
        )
    )
    return Mode(linker=CounterLinker(tag), optimizer=RewriteDatabaseQuery(include=[]))


@pytest.fixture
def counter_backend():
    tag = "counter"
    yield _make_backend(tag)
    _CONVERSIONS.pop(tag, None)


@pytest.fixture
def lossy_backend():
    tag = "counter_lossy"
    yield _make_backend(tag, lossy=True)
    _CONVERSIONS.pop(tag, None)


def _counter_function(mode):
    s = SharedVariable(type=counter_type, value=0, strict=False, name="c")
    next_c, value = Step()(s)
    return s, pytensor.function([], value, updates={s: next_c}, mode=mode)


def test_host_and_backend_stay_coherent(counter_backend):
    s, fn = _counter_function(counter_backend)

    # A host set_value is seen by the compiled read.
    s.set_value(5)
    assert fn() == 5  # reads 5, advances the native copy to 6

    # The compiled update is seen by host get_value.
    assert s.get_value() == 6

    # The shared stream continues across interleaved calls, not restarted.
    assert fn() == 6
    assert fn() == 7

    # A later host set_value overrides the backend stream.
    s.set_value(0)
    assert fn() == 0  # reads 0, advances the native copy to 1

    # A copy taken while the backend is ahead captures the advanced value, not the
    # stale host snapshot (regression for the deepcopy reconcile), and is a fully
    # independent shared afterwards -- the two diverge, neither aliases the other.
    s_copy = copy.deepcopy(s)
    assert s_copy.get_value() == 1
    assert fn() == 1  # advance the original via the backend...
    assert s.get_value() == 2
    assert s_copy.get_value() == 1  # ...the copy is untouched
    s_copy.set_value(50)  # mutate the copy on the host...
    assert s_copy.get_value() == 50
    assert s.get_value() == 2  # ...the original is untouched


def test_lossy_reconcile_warns_once(lossy_backend):
    s, fn = _counter_function(lossy_backend)
    fn()  # backend advances; host now stale
    with pytest.warns(UserWarning, match="lossy"):
        s.get_value()

    # Once warned for a source, a later stale read from it does not warn again.
    fn()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s.get_value()


def test_no_state_without_backend():
    # An ordinary shared no backend ever touches allocates no coherence state.
    s = shared(np.array(1.0))
    assert s.container._backend_state is None
    s.get_value()
    s.set_value(np.array(2.0))
    assert s.container._backend_state is None
