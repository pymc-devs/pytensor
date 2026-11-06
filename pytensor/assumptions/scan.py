from pytensor.assumptions.core import (
    MATRIX_KEYS,
    UNIT_MAGNITUDE,
    AssumptionInference,
    AssumptionKey,
    FactState,
    check_assumption,
    register_assumption,
)
from pytensor.assumptions.specify import SpecifyAssumptions
from pytensor.graph.rewriting.basic import copy_stack_trace, dfs_rewriter, node_rewriter
from pytensor.scan.op import Scan
from pytensor.scan.rewriting import scan_seqopt1
from pytensor.tensor.basic import AllocEmpty
from pytensor.tensor.subtensor import IncSubtensor


# Keys registered with ``register_scan_assumption``. Insertion-ordered, so the facts
# pushed into an inner graph are built deterministically.
_SCAN_KEYS: dict[AssumptionKey, None] = {}


def _recurrent_init_fact(buffer_var, key, feature, buffer_fact):
    """Return the *key* fact of a recurrence's initial value.

    Scan stores a ``sit-sot`` recurrence's initial value as
    ``SetSubtensor{:n_taps}(AllocEmpty(...), init)``. The buffer's own fact is
    UNKNOWN -- the rows scan has yet to fill are uninitialized -- so read the
    fact of the written ``init`` instead. For a buffer of any other shape, fall back
    to *buffer_fact*, the buffer's own fact, when it is TRUE: TRUE on the buffer holds
    for its initial rows, while FALSE only says some row lacks the property.
    """
    match buffer_var.owner_op_and_inputs:
        case (IncSubtensor(set_instead_of_inc=True), base, init, *_) if isinstance(
            base.owner_op, AllocEmpty
        ):
            return feature.get(init, key)
    return buffer_fact if buffer_fact is FactState.TRUE else FactState.UNKNOWN


def scan_delegate(key, op, feature, fgraph, node, input_states):
    """Infer *key* for a :class:`Scan`'s outer outputs, once per node and key."""
    memo_key = (node, key)
    out_states = feature.rule_memo.get(memo_key)
    if out_states is None:
        out_states = _infer_scan_outputs(key, op, feature, node, input_states)
        feature.rule_memo[memo_key] = out_states
    return out_states


def _infer_scan_outputs(key, op, feature, node, input_states):
    """Infer *key* for a :class:`Scan`'s outer outputs from its inner graph.

    A sequence or non-sequence inner input takes, for any key registered with
    :func:`register_scan_assumption`, the fact of the outer input feeding it, read only
    when inference reaches it. Each outer output takes the fact of its inner output.
    Sequences take TRUE facts only, and traced outputs keep TRUE only: FALSE on a stack
    says just that some slice lacks the property, and an empty stack has every
    property.

    A recurrence keeps its initial value's fact only when the loop body reproduces it
    with the other recurrences seeded the same way. Recurrences that fail are demoted
    to UNKNOWN and the body is re-inferred until the remaining seeds are consistent.
    Multi-output recurrences (``mit-mot``) stay UNKNOWN.
    """
    mappings = op.get_oinp_iinp_iout_oout_mappings()
    inner_inputs = op.inner_inputs
    inner_outputs = op.inner_outputs

    inner_seqs = set(op.inner_seqs(inner_inputs))
    outer_input_of = {
        inner_input: node.inputs[mappings["outer_inp_from_inner_inp"][inner_idx]]
        for inner_idx, inner_input in enumerate(inner_inputs)
    }

    # Recurrences keyed by outer output. Their outer input is the state buffer,
    # whose own fact is UNKNOWN, so the seed comes from the initial value.
    init_facts = {}
    recurrence_of = {}
    for outer_oidx, inner_iidxs in mappings["inner_inp_from_outer_out"].items():
        if not inner_iidxs:
            continue
        outer_iidx = mappings["outer_inp_from_outer_out"][outer_oidx]
        if len(mappings["inner_out_from_outer_out"][outer_oidx]) == 1:
            init_facts[outer_oidx] = _recurrent_init_fact(
                node.inputs[outer_iidx], key, feature, input_states[outer_iidx]
            )
        else:
            init_facts[outer_oidx] = FactState.UNKNOWN
        for inner_iidx in inner_iidxs:
            recurrence_of[inner_inputs[inner_iidx]] = outer_oidx

    def inner_output_fact(inner_inference, outer_oidx):
        [inner_oidx] = mappings["inner_out_from_outer_out"][outer_oidx]
        return inner_inference.get(inner_outputs[inner_oidx], key)

    seeded = {
        outer_oidx
        for outer_oidx, init_fact in init_facts.items()
        if init_fact is not FactState.UNKNOWN
    }

    def inner_inference_for(seeds):
        def leaf_fact(inner_input, leaf_key):
            if inner_input in recurrence_of:
                outer_oidx = recurrence_of[inner_input]
                if leaf_key == key and outer_oidx in seeds:
                    return init_facts[outer_oidx]
                return FactState.UNKNOWN
            if inner_input not in outer_input_of or leaf_key not in _SCAN_KEYS:
                return FactState.UNKNOWN
            state = feature.get(outer_input_of[inner_input], leaf_key)
            if inner_input in inner_seqs and state is not FactState.TRUE:
                return FactState.UNKNOWN
            return state

        if seeds:
            return AssumptionInference(op.fgraph, leaf_fact=leaf_fact)

        # With no seeds the leaves do not depend on *key*, so every key shares one
        # inference. Without that, nested Scans multiply the keys asked at each depth.
        memo_key = (node, None)
        inference = feature.rule_memo.get(memo_key)
        if inference is None:
            inference = AssumptionInference(op.fgraph, leaf_fact=leaf_fact)
            feature.rule_memo[memo_key] = inference
        return inference

    while True:
        inner_inference = inner_inference_for(frozenset(seeded))
        broken = {
            outer_oidx
            for outer_oidx in seeded
            if inner_output_fact(inner_inference, outer_oidx)
            is not init_facts[outer_oidx]
        }
        if not broken:
            break
        seeded -= broken

    out_states = [FactState.UNKNOWN] * len(node.outputs)
    for outer_oidx in range(len(node.outputs)):
        if len(mappings["inner_out_from_outer_out"].get(outer_oidx, [])) != 1:
            continue
        fact = inner_output_fact(inner_inference, outer_oidx)
        # A recurrent output's buffer holds the initial value too, so its fact must
        # agree with the initial value's. A traced output keeps TRUE only.
        if fact is init_facts.get(outer_oidx, FactState.TRUE):
            out_states[outer_oidx] = fact
    return out_states


def register_scan_assumption(key: AssumptionKey) -> None:
    """Let *key* propagate through :class:`~pytensor.scan.op.Scan`.

    Register only a key that holds for a tensor exactly when it holds for every
    slice along a leading axis, such as a property of the trailing two axes or an
    elementwise property. A Scan stacks its per-step results and slices its
    sequences, so any other key would be carried where it does not hold. An
    unregistered key is UNKNOWN on every Scan output.
    """
    _SCAN_KEYS[key] = None
    register_assumption(key, Scan)(scan_delegate)


for _key in (*MATRIX_KEYS, UNIT_MAGNITUDE):
    register_scan_assumption(_key)


@node_rewriter([Scan])
def push_assumptions_into_scan(fgraph, node):
    """Restate the facts a Scan's inner inputs are known to have on those inputs.

    An inner input is a bare leaf, so an ``assume`` on the outer variable is
    invisible to rewrites of the inner graph. This re-asserts it with a
    :class:`SpecifyAssumptions` node inside, so those rewrites can fire -- e.g.
    ``inv(X) @ y`` of a positive-definite :math:`X` specializes to a Cholesky
    solve within the loop body. Only keys registered with
    :func:`register_scan_assumption` are pushed, because only they hold for every
    slice of a tensor that has them.

    A sequence or non-sequence takes the facts of the outer input feeding it. A
    recurrence's carried state takes the facts of the recurrence's outer output: that
    output is the buffer of every state, so a TRUE fact on it is the proof that the
    loop body preserves the fact from the initial value on.
    """
    scan_op = node.op
    inner_inputs = scan_op.inner_inputs
    mappings = scan_op.get_oinp_iinp_iout_oout_mappings()

    source_of = {
        inner_inp: node.inputs[mappings["outer_inp_from_inner_inp"][inner_idx]]
        for inner_idx, inner_inp in enumerate(inner_inputs)
    }
    for outer_oidx, inner_iidxs in mappings["inner_inp_from_outer_out"].items():
        for inner_iidx in inner_iidxs:
            source_of[inner_inputs[inner_iidx]] = node.outputs[outer_oidx]

    new_facts = {}
    for inner_inp, source in source_of.items():
        clients = scan_op.fgraph.clients.get(inner_inp, ())
        if any(isinstance(client.op, SpecifyAssumptions) for client, _ in clients):
            # Already carries an inner assertion -- skip to avoid re-firing.
            continue
        facts = {
            key: FactState.TRUE
            for key in _SCAN_KEYS
            if check_assumption(fgraph, source, key)
        }
        if facts:
            new_facts[inner_inp] = facts

    if not new_facts:
        return None

    # Rebuild the inner graph over fresh leaves, splicing the assertions on.
    replace = {}
    new_inner_inputs = []
    for inner_inp in inner_inputs:
        dummy = inner_inp.type()
        new_inner_inputs.append(dummy)
        facts = new_facts.get(inner_inp)
        replace[inner_inp] = SpecifyAssumptions(facts)(dummy) if facts else dummy
    new_inner_outputs = scan_op.fgraph.bind(replace)

    new_scan_op = Scan(
        new_inner_inputs,
        new_inner_outputs,
        scan_op.info,
        mode=scan_op.mode,
        profile=scan_op.profile,
        truncate_gradient=scan_op.truncate_gradient,
        name=scan_op.name,
        allow_gc=scan_op.allow_gc,
    )
    new_outs = new_scan_op.make_node(*node.inputs).outputs
    copy_stack_trace(node.outputs, new_outs)
    return new_outs


scan_seqopt1.register(
    push_assumptions_into_scan.__name__,
    dfs_rewriter(push_assumptions_into_scan, ignore_newtrees=True),
    "fast_run",
    "scan",
    position=1,
)
