import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config, function, grad
from pytensor.assumptions import (
    DIAGONAL,
    LOWER_TRIANGULAR,
    MATRIX_KEYS,
    POSITIVE_DEFINITE,
    SYMMETRIC,
    UNIQUE_INDICES,
    UNIT_MAGNITUDE,
    ConflictingAssumptionsError,
    FactState,
)
from pytensor.assumptions import scan as scan_assumptions
from pytensor.assumptions.scan import push_assumptions_into_scan
from pytensor.assumptions.specify import SpecifyAssumptions, assume
from pytensor.compile.mode import Mode
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import dfs_rewriter
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.graph.traversal import ancestors
from pytensor.scan.basic import scan
from pytensor.scan.op import Scan
from pytensor.scan.rewriting.push_out import scan_push_out_non_seq
from pytensor.scan.utils import until
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.linalg import Cholesky, MatrixInverse
from tests.assumptions.conftest import make_fgraph
from tests.unittest_tools import RewriteTester


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


def test_body_denying_seeded_fact_without_proof_is_not_a_conflict():
    """``p`` is seeded diagonal, and the body declares its next value not diagonal
    while nothing proves otherwise. The seed is dropped, with no conflict, since the
    declaration only rules the fact out from the second step."""
    p0 = pt.matrix("p0", shape=(3, 3))
    out = scan(
        lambda p: assume(p + pt.ones_like(p), diagonal=False),
        outputs_info=[assume(p0, diagonal=True)],
        n_steps=5,
        return_updates=False,
    )
    _, af = make_fgraph(out, inputs=[p0])
    assert af.get(out, DIAGONAL) == FactState.UNKNOWN


def test_contradiction_in_body_fails_the_push_like_any_rewrite_query():
    """A body that contradicts its own declaration surfaces through the rewrite that
    asks, as it does outside a Scan, so ``on_opt_error`` decides what happens."""
    p0 = pt.matrix("p0", shape=(3, 3))
    out = scan(
        lambda p: assume(assume(p, orthogonal=True), orthogonal=False) * 2.0,
        outputs_info=[p0],
        n_steps=5,
        return_updates=False,
    )
    fg = FunctionGraph([p0], [out], clone=True)
    push_in = dfs_rewriter(push_assumptions_into_scan, ignore_newtrees=True)
    with config.change_flags(on_opt_error="raise"):
        with pytest.raises(ConflictingAssumptionsError):
            push_in.rewrite(fg)


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


@pytest.mark.parametrize(
    "keys",
    [(LOWER_TRIANGULAR, SYMMETRIC), (SYMMETRIC, LOWER_TRIANGULAR)],
    ids=["broken-first", "kept-first"],
)
def test_keys_seeding_one_recurrence_do_not_share_inference(keys):
    """Two keys that seed the same recurrence give different leaves, so one key's
    inner inference must not answer the other, whichever is queried first."""
    p0 = pt.matrix("p0", shape=(3, 3))
    out = scan(
        lambda prev: prev + prev.T,
        outputs_info=[assume(p0, lower_triangular=True, symmetric=True)],
        n_steps=5,
        return_updates=False,
    )
    _, af = make_fgraph(out, inputs=[p0])

    states = {key: af.get(out, key) for key in keys}
    assert states == {
        LOWER_TRIANGULAR: FactState.UNKNOWN,
        SYMMETRIC: FactState.TRUE,
    }


def test_compile_cost_does_not_multiply_with_nesting(monkeypatch):
    """Each Scan builds a bounded number of inner inferences, however deeply nested."""

    def nested(depth, x):
        if depth == 0:
            return 2.0 * x
        return scan(
            lambda p, c: nested(depth - 1, p) @ c,
            outputs_info=[x],
            non_sequences=[x],
            n_steps=2,
            return_updates=False,
        )[-1]

    def inferences_built(depth):
        built = []
        original = scan_assumptions.AssumptionInference

        def counting(*args, **kwargs):
            built.append(1)
            return original(*args, **kwargs)

        x = pt.matrix("x", shape=(3, 3))
        out = nested(depth, assume(x, diagonal=True))
        _, af = make_fgraph(out, inputs=[x])
        with monkeypatch.context() as patch:
            patch.setattr(scan_assumptions, "AssumptionInference", counting)
            for key in MATRIX_KEYS:
                af.get(out, key)
        return len(built)

    assert inferences_built(4) <= 5 * inferences_built(2)


def test_outer_assumption_pushed_into_scan_inner_graph():
    """`push_assumptions_into_scan` restates a sequence's facts, implied ones
    included, on the matching inner input."""
    Xs = pt.tensor("Xs", shape=(4, 3, 3))
    ys = pt.tensor("ys", shape=(4, 3, 3))
    seq = assume(Xs, positive_definite=True)
    out = scan(
        lambda Xt, yt: pt.linalg.inv(Xt) @ yt, sequences=[seq, ys], return_updates=False
    )
    rewrite = RewriteTester(
        [Xs, ys], [out], include=None, custom_rewrite=push_assumptions_into_scan
    )

    expected = scan(
        lambda Xt, yt: (
            pt.linalg.inv(assume(Xt, positive_definite=True, symmetric=True)) @ yt
        ),
        sequences=[seq, ys],
        return_updates=False,
    )
    rewrite.assert_graph(expected)


def test_push_in_does_not_refire_on_its_own_output():
    """An inner input that already carries pushed facts is left alone, so the
    rewrite reaches a fixed point instead of nesting assertions forever."""
    Xs = pt.tensor("Xs", shape=(4, 3, 3))
    ys = pt.tensor("ys", shape=(4, 3, 3))
    out = scan(
        lambda Xt, yt: pt.linalg.inv(Xt) @ yt,
        sequences=[assume(Xs, positive_definite=True), ys],
        return_updates=False,
    )
    fg = FunctionGraph([Xs, ys], [out], clone=True)
    push_in = dfs_rewriter(push_assumptions_into_scan, ignore_newtrees=True)

    push_in.rewrite(fg)
    pushed = fg.outputs[0]
    push_in.rewrite(fg)
    assert fg.outputs[0] is pushed


def test_inner_marker_does_not_suppress_pushed_facts():
    """A user's marker in the body covers only its own keys, so the facts proven for
    the sequence are still pushed, and a key the user states is left to the user."""
    Xs = pt.tensor("Xs", shape=(4, 3, 3))
    out = scan(
        lambda Xt: pt.linalg.inv(assume(Xt, symmetric=True)),
        sequences=[assume(Xs, positive_definite=True)],
        return_updates=False,
    )
    rewrite = RewriteTester(
        [Xs], [out], include=None, custom_rewrite=push_assumptions_into_scan
    )

    expected = scan(
        lambda Xt: pt.linalg.inv(
            assume(assume(Xt, positive_definite=True), symmetric=True)
        ),
        sequences=[assume(Xs, positive_definite=True)],
        return_updates=False,
    )
    rewrite.assert_graph(expected)


def test_marker_on_one_branch_does_not_state_fact_for_the_input():
    """``inv(Xt)`` does not read the marker on the other branch, so the fact it states
    is pushed onto the input as well."""
    Xs = pt.tensor("Xs", shape=(4, 3, 3))

    def body(Xt):
        return pt.linalg.inv(Xt) + assume(Xt, positive_definite=True)

    out = scan(
        body,
        sequences=[assume(Xs, positive_definite=True, symmetric=True)],
        return_updates=False,
    )
    rewrite = RewriteTester(
        [Xs], [out], include=None, custom_rewrite=push_assumptions_into_scan
    )

    def expected_body(Xt):
        pushed = assume(Xt, positive_definite=True, symmetric=True)
        return pt.linalg.inv(pushed) + assume(pushed, positive_definite=True)

    expected = scan(
        expected_body,
        sequences=[assume(Xs, positive_definite=True, symmetric=True)],
        return_updates=False,
    )
    rewrite.assert_graph(expected)


def test_fact_proven_later_is_pushed_and_rewrite_settles():
    """A fact that appears after a first push joins the pushed ones, and repeating the
    rewrite then changes nothing, however many markers the input carries."""
    Xs = pt.tensor("Xs", shape=(4, 3, 3))
    ys = pt.tensor("ys", shape=(4, 3, 3))
    seq = assume(Xs, symmetric=True)
    out = scan(
        lambda Xt, yt: pt.linalg.inv(Xt) @ yt,
        sequences=[seq, ys],
        return_updates=False,
    )
    fg = FunctionGraph([Xs, ys], [out], clone=True)
    push_in = dfs_rewriter(push_assumptions_into_scan, ignore_newtrees=True)
    push_in.rewrite(fg)

    [assumed] = [v for v in fg.variables if isinstance(v.owner_op, SpecifyAssumptions)]
    fg.replace(assumed, assume(fg.inputs[0], symmetric=True, diagonal=True))
    push_in.rewrite(fg)
    pushed = fg.outputs[0]
    push_in.rewrite(fg)
    assert fg.outputs[0] is pushed

    [scan_node] = [n for n in fg.apply_nodes if isinstance(n.op, Scan)]
    inner_markers = [
        n.op
        for n in scan_node.op.fgraph.toposort()
        if isinstance(n.op, SpecifyAssumptions)
    ]
    stated = {key for marker in inner_markers for key, _ in marker.assumptions}
    assert {SYMMETRIC, DIAGONAL} <= stated
    assert len(inner_markers) == 2


def test_proven_recurrence_fact_pushed_onto_carried_state():
    """A recurrence whose body keeps ``P`` positive definite gets that fact on its
    carried state, so a per-step solve against ``P`` can use a Cholesky solve."""
    A = pt.matrix("A", shape=(3, 3))
    P0 = pt.matrix("P0", shape=(3, 3))
    init = assume(P0, positive_definite=True)
    out = scan(
        lambda P, A: A @ P @ A.T + pt.eye(3),
        outputs_info=[init],
        non_sequences=[A],
        n_steps=5,
        return_updates=False,
    )
    rewrite = RewriteTester(
        [A, P0], [out], include=None, custom_rewrite=push_assumptions_into_scan
    )

    expected = scan(
        lambda P, A: (
            A @ assume(P, positive_definite=True, symmetric=True) @ A.T + pt.eye(3)
        ),
        outputs_info=[init],
        non_sequences=[A],
        n_steps=5,
        return_updates=False,
    )
    rewrite.assert_graph(expected)


@pytest.mark.parametrize(
    "body, n_states",
    [
        (lambda prev, M: prev @ M, 1),
        (lambda p1, p2, M: [p1 @ p2, p2 @ M], 2),
    ],
    ids=["breaking-body", "relies-on-breaking-recurrence"],
)
def test_unproven_recurrence_fact_not_pushed(body, n_states):
    """A carried state gets no fact the loop body does not preserve, including one
    that holds only while another, breaking, recurrence keeps its fact."""
    M = pt.matrix("M", shape=(3, 3))
    inits = [pt.matrix(f"init{i}", shape=(3, 3)) for i in range(n_states)]
    out = scan(
        body,
        outputs_info=[assume(init, diagonal=True) for init in inits],
        non_sequences=[M],
        n_steps=5,
        return_updates=False,
    )
    outs = out if n_states > 1 else [out]
    rewrite = RewriteTester(
        [*inits, M], outs, include=None, custom_rewrite=push_assumptions_into_scan
    )
    rewrite.assert_graph(*outs)


def test_non_sequence_assumption_does_not_block_push_out():
    """The assertion pushed onto a positive-definite non-sequence ``X`` must not
    anchor the loop-invariant ``inv(X)`` inside the loop: it is computed once
    outside, and the loop body is left with the per-step product."""
    X = pt.matrix("X", shape=(3, 3))
    ys = pt.tensor("ys", shape=(4, 3, 3))
    out = scan(
        lambda yt, X: pt.linalg.inv(X) @ yt,
        sequences=[ys],
        non_sequences=[assume(X, positive_definite=True)],
        return_updates=False,
    )
    rewrite = RewriteTester([X, ys], [out], include="o3", exclude=["fusion", "inplace"])

    expected = scan(
        lambda yt, A: assume(A, positive_definite=True, symmetric=True) @ yt,
        sequences=[ys],
        non_sequences=[MatrixInverse()(X)],
        return_updates=False,
    )
    rewrite.assert_graph(rewrite_graph(expected, include=("canonicalize",)))


def test_non_sequence_assumption_on_invariant_body_removes_scan():
    """The assertion pushed onto ``X`` is all that would be left of a body that never
    reads a sequence, so it must not keep the Scan alive."""
    X = pt.matrix("X", shape=(3, 3))
    out = scan(
        lambda X: pt.linalg.inv(X),
        non_sequences=[assume(X, positive_definite=True)],
        n_steps=5,
        return_updates=False,
    )
    rewrite = RewriteTester([X], [out], include="o3", exclude=["fusion", "inplace"])

    inverse = MatrixInverse()(assume(X, positive_definite=True, symmetric=True))
    expected = pt.alloc(inverse, pt.constant(5), *inverse.shape)
    rewrite.assert_graph(rewrite_graph(expected, include=("canonicalize",)))


def test_sequence_assumption_does_not_block_push_out():
    """The assertion pushed onto a sequence must not stop an elementwise body from
    being vectorized outside the loop."""
    x = pt.tensor("x", shape=(4, 3, 3))
    y = pt.matrix("y", shape=(3, 3))
    out = scan(
        lambda s, y: pt.exp(s) + y,
        sequences=[assume(x, diagonal=True)],
        non_sequences=[y],
        return_updates=False,
    )
    rewrite = RewriteTester([x, y], [out], include="o3", exclude=["fusion", "inplace"])

    rewrite.assert_graph(rewrite_graph(pt.exp(x) + y, include=("canonicalize",)))


def test_hoisted_sequence_work_keeps_fact_for_inner_rewrites():
    """``2 * s`` is hoisted out of the loop, and the new sequence takes the fact, so
    ``inv`` of it can still specialize inside the loop body."""
    Xs = pt.tensor("Xs", shape=(4, 3, 3))
    y = pt.matrix("y", shape=(3, 3))
    out = scan(
        lambda s, y: pt.linalg.inv(2.0 * s) @ y,
        sequences=[assume(Xs, positive_definite=True)],
        non_sequences=[y],
        return_updates=False,
    )
    rewritten = rewrite_graph(out, include=("fast_run",), exclude=("fusion", "inplace"))
    [scan_var] = [v for v in ancestors([rewritten]) if isinstance(v.owner_op, Scan)]
    inner_ops = [n.op for n in scan_var.owner_op.fgraph.toposort()]

    assert not any(isinstance(op, Elemwise) for op in inner_ops)
    [marker] = [op for op in inner_ops if isinstance(op, SpecifyAssumptions)]
    assert dict(marker.assumptions)[POSITIVE_DEFINITE] is FactState.TRUE


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


def _covariance_step(P, A, y):
    return A @ P @ A.T + pt.eye(3), pt.linalg.inv(P) @ y


def _covariance_scan(step=_covariance_step, **kwargs):
    P0 = pt.matrix("P0", shape=(3, 3))
    A = pt.matrix("A", shape=(3, 3))
    y = pt.matrix("y", shape=(3, 3))
    outputs = scan(
        step,
        outputs_info=[assume(P0, positive_definite=True), None],
        non_sequences=[A, y],
        return_updates=False,
        **kwargs,
    )
    return [P0, A, y], outputs


def _assert_specialized_body_matches_reference(inputs, outputs, *input_sets):
    """Compare a compile that specialized a loop body to Cholesky with an unoptimized
    one. ``assert_eval`` cannot do this: it never runs the rewritten body."""
    optimized = function(inputs, outputs, mode="FAST_RUN")
    reference = function(inputs, outputs, mode=Mode(linker="py", optimizer=None))

    scan_nodes = [
        n for n in optimized.maker.fgraph.apply_nodes if isinstance(n.op, Scan)
    ]
    assert any(
        isinstance(inner.op, Cholesky)
        for node in scan_nodes
        for inner in node.op.fgraph.apply_nodes
    )
    for values in input_sets:
        for result, expected in zip(
            optimized(*values), reference(*values), strict=True
        ):
            np.testing.assert_allclose(result, expected, rtol=1e-6)
    return optimized


COVARIANCE_VALUES = (2.0 * np.eye(3) + 0.5, 0.5 * np.eye(3) + 0.1, np.eye(3) + 0.2)


def test_specialized_body_with_until_condition():
    limit = pt.scalar("limit")

    def step(P, A, y):
        next_P, solved = _covariance_step(P, A, y)
        return (next_P, solved), until(pt.trace(next_P) > limit)

    inputs, outputs = _covariance_scan(step, n_steps=6)
    _assert_specialized_body_matches_reference(
        [*inputs, limit],
        outputs,
        (*COVARIANCE_VALUES, np.array(0.0, dtype=limit.dtype)),
        (*COVARIANCE_VALUES, np.array(1e9, dtype=limit.dtype)),
    )


def test_specialized_recurrence_with_zero_steps():
    n_steps = pt.iscalar("n_steps")
    inputs, outputs = _covariance_scan(n_steps=n_steps)
    optimized = _assert_specialized_body_matches_reference(
        [*inputs, n_steps],
        outputs,
        (*COVARIANCE_VALUES, np.array(3, dtype="int32")),
    )

    covariances, solved = optimized(*COVARIANCE_VALUES, np.array(0, dtype="int32"))
    assert covariances.shape == solved.shape == (0, 3, 3)


def test_gradient_through_specialized_body():
    inputs, (covariances, solved) = _covariance_scan(n_steps=4)
    cost = covariances.sum() + solved.sum()
    _assert_specialized_body_matches_reference(
        inputs, grad(cost, inputs[:2]), COVARIANCE_VALUES
    )


@pytest.mark.parametrize(
    "body",
    [lambda r: r[None, :], lambda r: pt.stack([r, r, r])],
    ids=["one-row", "repeated-row"],
)
def test_matrix_fact_does_not_reach_vector_slices(body):
    """Slicing a diagonal matrix gives vectors, which a matrix property says nothing
    about, even on a Scan whose outer sequence keeps the fact."""
    x = pt.matrix("x", shape=(3, 3))
    traced = scan(body, sequences=[x], return_updates=False)
    [node] = [v.owner for v in ancestors([traced]) if isinstance(v.owner_op, Scan)]

    n_steps, _, *rest = node.inputs
    out = node.op.make_node(n_steps, assume(x, diagonal=True), *rest).outputs[0]
    _, af = make_fgraph(out, inputs=[x])
    assert af.get(out, DIAGONAL) == FactState.UNKNOWN
