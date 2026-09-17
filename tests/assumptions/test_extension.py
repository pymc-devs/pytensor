import pickle

import pytest

import pytensor.tensor as pt
from pytensor.assumptions import (
    ALL_KEYS,
    KEY_REGISTRY,
    SYMMETRIC,
    AssumptionKey,
    FactState,
    register_assumption,
    register_universal_assumption,
)
from pytensor.assumptions.core import ASSUMPTION_INFER_REGISTRY
from pytensor.assumptions.specify import SpecifyAssumptions, assume
from pytensor.printing import debugprint
from pytensor.tensor.basic import AllocDiag, alloc_diag
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.rewriting.assumptions import DrainSpecifyAssumptions
from tests.assumptions.conftest import make_fgraph


def test_key_registers_itself():
    key = AssumptionKey("time_varying", short_name="tv")
    assert KEY_REGISTRY["time_varying"] is key
    assert key in ALL_KEYS


def test_identical_key_redefinition_is_idempotent():
    """A module imported twice must not double-register its key or its rules."""
    first = AssumptionKey("time_varying", short_name="tv")
    n_keys = len(KEY_REGISTRY)
    second = AssumptionKey("time_varying", short_name="tv")

    assert second == first
    assert len(KEY_REGISTRY) == n_keys
    assert KEY_REGISTRY["time_varying"] is first


def test_name_collision_with_different_metadata_raises():
    AssumptionKey("time_varying", short_name="tv")
    with pytest.raises(ValueError, match="already registered"):
        AssumptionKey("time_varying", short_name="clashing")


def test_assume_accepts_extension_key():
    TIME_VARYING = AssumptionKey("time_varying", short_name="tv")
    x = pt.tensor3("x")
    x_tv = assume(x, time_varying=True)
    _, af = make_fgraph(x_tv)
    assert af.check(x_tv, TIME_VARYING)


def test_assume_mixes_core_and_extension_keys():
    TIME_VARYING = AssumptionKey("time_varying", short_name="tv")
    x = pt.tensor("x", shape=(10, 3, 3))
    x_both = assume(x, symmetric=True, time_varying=True)
    _, af = make_fgraph(x_both)
    assert af.check(x_both, SYMMETRIC)
    assert af.check(x_both, TIME_VARYING)


def test_assume_records_false_for_extension_key():
    TIME_VARYING = AssumptionKey("time_varying", short_name="tv")
    x = pt.tensor3("x")
    x_static = assume(x, time_varying=False)
    _, af = make_fgraph(x_static)
    assert af.get(x_static, TIME_VARYING) is FactState.FALSE


def test_assume_rejects_unregistered_name():
    """A typo must not silently become a no-op, and the error must aid discovery."""
    AssumptionKey("time_varying", short_name="tv")
    x = pt.matrix("x")
    with pytest.raises(ValueError, match="Unknown assumption\\(s\\): symmetrik"):
        assume(x, symmetrik=True)
    with pytest.raises(ValueError, match="are: time_varying"):
        assume(x, symmetrik=True)


def test_key_assume_and_holds():
    TIME_VARYING = AssumptionKey("time_varying", short_name="tv")
    x = pt.tensor3("x")

    assert TIME_VARYING.holds(TIME_VARYING.assume(x))
    assert not TIME_VARYING.holds(TIME_VARYING.assume(x, state=False))
    assert not TIME_VARYING.holds(x)


def test_holds_reuses_a_supplied_fgraph():
    TIME_VARYING = AssumptionKey("time_varying", short_name="tv")
    x_tv = TIME_VARYING.assume(pt.tensor3("x"))
    y = pt.matrix("y")
    fgraph, _ = make_fgraph(x_tv, y)

    assert TIME_VARYING.holds(x_tv, fgraph)
    assert not TIME_VARYING.holds(y, fgraph)


def test_universal_rules_reach_a_later_key():
    """A key created after ``blockwise`` was imported still gets its delegate."""
    SPARSE = AssumptionKey("sparse")
    register_assumption(SPARSE, AllocDiag)(
        lambda key, op, feature, fgraph, node, input_states: [FactState.TRUE]
    )

    v_core = pt.vector("v", shape=(3,))
    core_op = alloc_diag(v_core, offset=0, axis1=0, axis2=1).owner.op
    v_batch = pt.matrix("v_batch", shape=(5, 3))
    batched = Blockwise(core_op, signature="(n)->(n,n)")(v_batch)

    _, af = make_fgraph(batched)
    assert af.check(batched, SPARSE)


def test_universal_rule_reaches_existing_keys():
    """The decorator installs onto keys registered before it ran, not just after."""
    EARLY = AssumptionKey("early")

    @register_universal_assumption(AllocDiag)
    def _always_true(key, op, feature, fgraph, node, input_states):
        return [FactState.TRUE]

    LATE = AssumptionKey("late")

    diag = alloc_diag(pt.vector("v", shape=(3,)), offset=0, axis1=0, axis2=1)
    _, af = make_fgraph(diag)

    assert af.check(diag, EARLY)
    assert af.check(diag, LATE)


def test_membership_rejects_a_non_key_sharing_a_name():
    """``in`` compares keys, not names -- a variable named after one is not a key."""
    assert "symmetric" not in ALL_KEYS
    assert pt.matrix("symmetric") not in ALL_KEYS


def test_extension_key_appears_in_debugprint():
    AssumptionKey("time_varying", short_name="tv")
    x_tv = assume(pt.tensor3("x"), time_varying=True)
    printed = debugprint(x_tv, print_assumptions=True, file="str")
    assert "a={tv}" in printed


def test_drain_resolves_extension_key():
    TIME_VARYING = AssumptionKey("time_varying", short_name="tv")
    x = pt.tensor3("x")
    fgraph, af = make_fgraph(assume(x, time_varying=True) + 1, inputs=[x])

    DrainSpecifyAssumptions().apply(fgraph)

    assert not any(
        isinstance(node.op, SpecifyAssumptions) for node in fgraph.apply_nodes
    )
    assert af.check(x, TIME_VARYING)


def test_graph_carries_keys_not_names():
    """The declaration holds the key itself, so it cannot name an unregistered fact."""
    TIME_VARYING = AssumptionKey("time_varying", short_name="tv")
    x_tv = assume(pt.tensor3("x"), time_varying=True)
    assert x_tv.owner.op.assumptions == ((TIME_VARYING, FactState.TRUE),)


def test_declaring_by_name_is_rejected():
    AssumptionKey("time_varying", short_name="tv")
    with pytest.raises(TypeError, match="not by name"):
        SpecifyAssumptions({"time_varying": FactState.TRUE})


def test_key_survives_a_pickle_round_trip():
    """A key restored from a cached graph re-registers and keeps its universal rules."""
    key = AssumptionKey("time_varying", short_name="tv")
    blob = pickle.dumps(key)
    del KEY_REGISTRY["time_varying"]

    restored = pickle.loads(blob)

    assert KEY_REGISTRY["time_varying"] == restored
    assert (restored, SpecifyAssumptions) in ASSUMPTION_INFER_REGISTRY
    assert restored.holds(restored.assume(pt.tensor3("x")))


def test_pickled_graph_keeps_its_declaration():
    """A graph outliving the library that declared its key still drains the fact."""
    key = AssumptionKey("time_varying", short_name="tv")
    blob = pickle.dumps(key.assume(pt.tensor3("x")))
    del KEY_REGISTRY["time_varying"]

    restored_graph = pickle.loads(blob)
    [(restored_key, state)] = restored_graph.owner.op.assumptions

    assert state is FactState.TRUE
    assert restored_key.holds(restored_graph)
