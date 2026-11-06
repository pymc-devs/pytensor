import pytest

import pytensor.tensor as pt
from pytensor.assumptions import (
    DIAGONAL,
    LOWER_TRIANGULAR,
    POSITIVE_DEFINITE,
    UNIQUE_INDICES,
    UNIT_MAGNITUDE,
    FactState,
)
from pytensor.assumptions.specify import assume
from pytensor.graph.rewriting.basic import dfs_rewriter
from pytensor.graph.traversal import ancestors
from pytensor.scan.basic import scan
from pytensor.scan.op import Scan
from pytensor.scan.rewriting.push_out import scan_push_out_non_seq
from tests.assumptions.conftest import make_fgraph


def test_map_preserving_body_forwards_property():
    """map (nit-sot): ``s @ s`` of a diagonal ``s`` is diagonal, so the stack is."""
    x = pt.tensor3("seq")
    seq = assume(x, diagonal=True)
    out = scan(lambda s: s @ s, sequences=[seq], return_updates=False)
    _, af = make_fgraph(out, inputs=[x])
    assert af.check(out, DIAGONAL)


def test_map_breaking_body_is_unknown():
    """map: ``exp`` does not preserve the zero pattern, so the property is lost."""
    x = pt.tensor3("seq")
    seq = assume(x, diagonal=True)
    out = scan(lambda s: pt.exp(s), sequences=[seq], return_updates=False)
    _, af = make_fgraph(out, inputs=[x])
    assert af.get(out, DIAGONAL) == FactState.UNKNOWN


def test_non_sequence_forwards_property():
    """Transposing a diagonal non-sequence keeps every stacked step diagonal."""
    x = pt.matrix("m", shape=(4, 4))
    m = assume(x, diagonal=True)
    out = scan(lambda mat: mat.T, non_sequences=[m], n_steps=5, return_updates=False)
    _, af = make_fgraph(out, inputs=[x])
    assert af.check(out, DIAGONAL)


def test_recurrence_preserving_body_forwards_property():
    """recurrence (sit-sot): ``2 * prev`` keeps a diagonal carried state diagonal
    at every step."""
    x = pt.matrix("init", shape=(3, 3))
    init = assume(x, diagonal=True)
    out = scan(
        lambda prev: 2.0 * prev,
        outputs_info=[init],
        n_steps=5,
        return_updates=False,
    )
    _, af = make_fgraph(out, inputs=[x])
    assert af.check(out, DIAGONAL)


def test_recurrence_breaking_body_is_unknown():
    """recurrence: adding a non-diagonal term each step breaks the property."""
    x = pt.matrix("init", shape=(3, 3))
    init = assume(x, diagonal=True)
    out = scan(
        lambda prev: prev + pt.ones_like(prev),
        outputs_info=[init],
        n_steps=5,
        return_updates=False,
    )
    _, af = make_fgraph(out, inputs=[x])
    assert af.get(out, DIAGONAL) == FactState.UNKNOWN


def test_output_from_breaking_recurrence_is_unknown():
    """A traced output read from a recurrence whose body breaks the property is
    only diagonal at the first step, so the stack is not."""
    x = pt.matrix("init", shape=(3, 3))
    _, out = scan(
        lambda prev: [prev + pt.ones_like(prev), 2.0 * prev],
        outputs_info=[assume(x, diagonal=True), None],
        n_steps=5,
        return_updates=False,
    )
    _, af = make_fgraph(out, inputs=[x])
    assert af.get(out, DIAGONAL) == FactState.UNKNOWN


def test_recurrence_relying_on_breaking_recurrence_is_unknown():
    """``p1 @ p2`` keeps ``p1`` diagonal only while ``p2`` is, and ``p2`` breaks
    after the first step, so neither recurrence keeps the property."""
    a = pt.matrix("a", shape=(3, 3))
    b = pt.matrix("b", shape=(3, 3))
    out, _ = scan(
        lambda p1, p2: [p1 @ p2, p2 + pt.ones_like(p2)],
        outputs_info=[assume(a, diagonal=True), assume(b, diagonal=True)],
        n_steps=5,
        return_updates=False,
    )
    _, af = make_fgraph(out, inputs=[a, b])
    assert af.get(out, DIAGONAL) == FactState.UNKNOWN


def test_replacing_outer_input_updates_scan_fact():
    """A rewrite that gives a Scan's sequence a new fact reaches its outputs."""
    x = pt.tensor3("seq", shape=(4, 3, 3))
    seq = pt.exp(x)
    out = scan(lambda s: s * 2.0, sequences=[seq], return_updates=False)
    fg, af = make_fgraph(out, inputs=[x])
    assert af.get(out, DIAGONAL) == FactState.UNKNOWN

    fg.replace(seq, assume(x, diagonal=True))
    assert af.check(out, DIAGONAL)


def test_registered_elementwise_key_forwards_property():
    """``UNIT_MAGNITUDE`` is not a matrix property but is registered for Scan, so
    negating a unit-magnitude non-sequence keeps the stack unit-magnitude."""
    x = pt.vector("x", shape=(5,))
    out = scan(
        lambda m: -m,
        non_sequences=[assume(x, unit_magnitude=True)],
        n_steps=3,
        return_updates=False,
    )
    _, af = make_fgraph(out, inputs=[x])
    assert af.check(out, UNIT_MAGNITUDE)


def test_cross_key_rule_sees_sequence_fact():
    """``LOWER_TRIANGULAR`` of a transpose asks for ``UPPER_TRIANGULAR`` of its
    input, so the inner input must carry every key, not only the queried one."""
    x = pt.tensor3("seq")
    seq = assume(x, upper_triangular=True)
    out = scan(lambda s: s.T, sequences=[seq], return_updates=False)
    _, af = make_fgraph(out, inputs=[x])
    assert af.check(out, LOWER_TRIANGULAR)


@pytest.mark.parametrize(
    "assume_a, expected",
    [("pos", FactState.TRUE), ("gen", FactState.UNKNOWN)],
)
def test_inner_client_promise_reaches_outer_output(assume_a, expected):
    """An inner ``solve`` promises the fact of its ``a``, which the body also returns."""
    x = pt.tensor3("seq")
    b = pt.vector("b")
    out, _ = scan(
        lambda s: [s.T, pt.linalg.solve(s, b, assume_a=assume_a)],
        sequences=[x],
        return_updates=False,
    )
    _, af = make_fgraph(out, inputs=[x, b])
    assert af.get(out, POSITIVE_DEFINITE) == expected


def test_conflict_on_unrelated_key_does_not_escape():
    """A CONFLICT on ``orthogonal`` must not break a ``diagonal`` query through a
    Scan whose body never asks about orthogonality."""
    x = pt.tensor3("seq", shape=(4, 3, 3))
    seq = assume(assume(x, orthogonal=True), orthogonal=False)
    out = scan(lambda s: s * 2.0, sequences=[seq], return_updates=False)
    _, af = make_fgraph(out, inputs=[x])
    assert af.get(out, DIAGONAL) == FactState.UNKNOWN


def test_traced_output_drops_false():
    """A traced output may be empty, and an empty stack has every property, so a
    FALSE inner fact is not carried out."""
    x = pt.matrix("m", shape=(3, 3))
    m = assume(x, diagonal=False)
    out = scan(lambda mat: mat, non_sequences=[m], n_steps=0, return_updates=False)
    _, af = make_fgraph(out, inputs=[x])
    assert af.get(out, DIAGONAL) == FactState.UNKNOWN


def test_stacking_drops_whole_tensor_property():
    """Stacking copies of a unique index vector repeats its entries, so a key that is
    not registered for Scan stays UNKNOWN on the stacked output."""
    x = pt.lvector("idx")
    out = scan(
        lambda idx: idx,
        non_sequences=[assume(x, unique_indices=True)],
        n_steps=3,
        return_updates=False,
    )
    _, af = make_fgraph(out, inputs=[x])
    assert af.get(out, UNIQUE_INDICES) == FactState.UNKNOWN


@pytest.mark.parametrize(
    "declared, expected",
    [(True, FactState.TRUE), (False, FactState.UNKNOWN)],
    ids=["true", "false"],
)
def test_unrecognized_buffer_passes_only_true(declared, expected):
    """A buffer that is not ``SetSubtensor(AllocEmpty, init)`` gives its own fact to
    the initial value only when TRUE: FALSE says just that some row lacks it."""
    x = pt.matrix("init", shape=(3, 3))
    traced = scan(lambda prev: prev, outputs_info=[x], n_steps=5, return_updates=False)
    [node] = [v.owner for v in ancestors([traced]) if isinstance(v.owner_op, Scan)]

    n_steps, buffer = node.inputs
    fresh_buffer = buffer.type("buffer")
    out = node.op.make_node(n_steps, assume(fresh_buffer, diagonal=declared)).outputs[0]
    _, af = make_fgraph(out, inputs=[fresh_buffer])
    assert af.get(out, DIAGONAL) is expected


def test_mit_sot_multi_tap_recurrence_forwards_property():
    """2-tap recurrence: ``p1 + p2`` needs *both* taps diagonal, pinning that the
    delegate seeds both from the multi-tap init buffer."""
    x = pt.tensor("init", shape=(2, 3, 3))
    init = assume(x, diagonal=True)
    out = scan(
        lambda p2, p1: p1 + p2,
        outputs_info=[dict(initial=init, taps=[-2, -1])],
        n_steps=5,
        return_updates=False,
    )
    _, af = make_fgraph(out, inputs=[x])
    assert af.check(out, DIAGONAL)


def test_map_and_recurrence_combined_forwards_property():
    """A scan with both a sequence and a recurrence -- ``prev @ s`` needs both the
    carried state and the sequence element diagonal, pinning that the delegate
    seeds the recurrent and sequence inner inputs from the right outer inputs."""
    sx = pt.tensor("seq", shape=(5, 3, 3))
    seq = assume(sx, diagonal=True)
    ix = pt.matrix("init", shape=(3, 3))
    init = assume(ix, diagonal=True)
    out = scan(
        lambda s, prev: prev @ s,
        sequences=[seq],
        outputs_info=[init],
        return_updates=False,
    )
    _, af = make_fgraph(out, inputs=[sx, ix])
    assert af.check(out, DIAGONAL)


def test_multi_output_scan_maps_outputs_independently():
    """A scan with two outputs -- one diagonal-preserving, one breaking -- the
    delegate maps each inner output to its own outer output independently."""
    x = pt.tensor3("seq")
    seq = assume(x, diagonal=True)
    doubled, exponential = scan(
        lambda s: [s + s, pt.exp(s)], sequences=[seq], return_updates=False
    )
    _, af = make_fgraph(doubled, exponential, inputs=[x])
    assert af.check(doubled, DIAGONAL)
    assert af.get(exponential, DIAGONAL) == FactState.UNKNOWN


def test_nested_scan_forwards_property():
    """A scan whose body runs an inner scan -- the delegate recurses into the
    inner Scan while still inferring the outer one."""
    x = pt.tensor3("seq")
    seq = assume(x, diagonal=True)

    def body(s):
        inner = scan(
            lambda m: 2.0 * m, non_sequences=[s], n_steps=1, return_updates=False
        )
        return inner[-1]

    out = scan(body, sequences=[seq], return_updates=False)
    _, af = make_fgraph(out, inputs=[x])
    assert af.check(out, DIAGONAL)


def test_congruence_recurrence_keeps_property_through_push_out():
    """``A @ P @ A.T + Q`` keeps a positive-definite ``P`` positive definite, and
    hoisting loop-invariant work out of the body must not hide the transpose."""
    A = pt.matrix("A", shape=(3, 3))
    Q = pt.matrix("Q", shape=(3, 3))
    P0 = pt.matrix("P0", shape=(3, 3))
    out = scan(
        lambda P, A, Q: A @ P @ A.T + Q,
        outputs_info=[assume(P0, positive_definite=True)],
        non_sequences=[A, assume(Q, positive_definite=True)],
        n_steps=5,
        return_updates=False,
    )
    fg, af = make_fgraph(out, inputs=[A, Q, P0])
    dfs_rewriter(scan_push_out_non_seq, ignore_newtrees=True).rewrite(fg)
    assert af.check(fg.outputs[0], POSITIVE_DEFINITE)
