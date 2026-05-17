from copy import copy

from pytensor.assumptions.core import (
    ALL_KEYS,
    AssumptionFeature,
    FactState,
    check_assumption,
    register_assumption,
)
from pytensor.assumptions.specify import SpecifyAssumptions
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.basic import copy_stack_trace, dfs_rewriter, node_rewriter
from pytensor.scan.op import Scan
from pytensor.scan.rewriting import scan_seqopt1
from pytensor.tensor.subtensor import IncSubtensor


def _recurrent_init_fact(buffer_var, key, feature, fallback):
    """Return the *key* fact of a recurrence's initial value.

    Scan stores a ``sit-sot`` recurrence's initial value as
    ``SetSubtensor{:n_taps}(AllocEmpty(...), init)``. The buffer's own fact is
    UNKNOWN -- the rows scan has yet to fill are uninitialised -- so read the
    fact of the written ``init`` instead. Return *fallback* when the buffer is
    not that shape.
    """
    owner = buffer_var.owner
    if (
        owner is not None
        and isinstance(owner.op, IncSubtensor)
        and owner.op.set_instead_of_inc
    ):
        return feature.get(owner.inputs[1], key)
    return fallback


def scan_delegate(key, op, feature, fgraph, node, input_states):
    """Infer *key* for a :class:`Scan`'s outer outputs by delegating into its inner graph.

    The outer-input facts seed the matching inner inputs; the inner graph is then
    inferred and the inner-output facts are mapped back onto the outer outputs.

    For a non-recurrent (``nit-sot``) output the per-step inner output is stacked
    along a new leading axis, so the fact carries straight through. For a
    recurrent (``sit-sot``) output the carried state is seeded from the
    recurrence's initial value, and the fact is kept only when the loop body
    reproduces it -- a one-step fixpoint, exact because the per-key lattice
    (``UNKNOWN`` < ``TRUE``) leaves no room to iterate. Multi-output recurrences
    (``mit-mot``) are left UNKNOWN.
    """
    mappings = op.get_oinp_iinp_iout_oout_mappings()
    inner_inputs = op.inner_inputs
    inner_outputs = op.inner_outputs

    # Inner inputs that carry recurrent state. Their outer input is the sit-sot
    # buffer; the buffer's own fact is UNKNOWN, so seed from the initial value.
    recurrent_iinps = {
        iidx
        for iidxs in mappings["inner_inp_from_outer_out"].values()
        for iidx in iidxs
    }

    inner_feature = AssumptionFeature()
    op.fgraph.attach_feature(inner_feature)
    try:
        # Seed each inner input with the fact of the outer input feeding it.
        # The seed is written straight into the cache: an inner input is a
        # graph leaf, so this is the only way to inject a fact onto it.
        for iinp_idx, iinp in enumerate(inner_inputs):
            outer_iidx = mappings["outer_inp_from_inner_inp"][iinp_idx]
            seed = input_states[outer_iidx]
            if iinp_idx in recurrent_iinps:
                seed = _recurrent_init_fact(node.inputs[outer_iidx], key, feature, seed)
            if seed is not FactState.UNKNOWN:
                inner_feature.cache[(iinp, key)] = seed
                inner_feature._var_to_keys.setdefault(iinp, set()).add(key)

        out_states = [FactState.UNKNOWN] * len(node.outputs)
        for outer_oidx in range(len(node.outputs)):
            inner_oidxs = mappings["inner_out_from_outer_out"].get(outer_oidx, [])
            if len(inner_oidxs) != 1:
                # Multi-output (mit-mot) recurrences are left UNKNOWN for now.
                continue
            fact = inner_feature.get(inner_outputs[inner_oidxs[0]], key)

            if mappings["inner_inp_from_outer_out"].get(outer_oidx):
                # Recurrent: the fact survives only if the loop body reproduces
                # the initial value's fact.
                outer_iidx = mappings["outer_inp_from_outer_out"][outer_oidx]
                init_fact = _recurrent_init_fact(
                    node.inputs[outer_iidx], key, feature, input_states[outer_iidx]
                )
                if fact is not init_fact:
                    fact = FactState.UNKNOWN
            out_states[outer_oidx] = fact
        return out_states
    finally:
        op.fgraph.remove_feature(inner_feature)


for _key in ALL_KEYS:
    register_assumption(_key, Scan)(scan_delegate)


@node_rewriter([Scan])
def lift_assumptions_into_scan(fgraph, node):
    """Lift structural assumptions from a Scan's sequence and non-sequence inputs
    onto the matching inner inputs.

    An inner input is a bare leaf, so an ``assume`` on the outer variable is
    invisible to rewrites of the inner graph. This re-asserts it with a
    :class:`SpecifyAssumptions` node inside, so those rewrites can fire -- e.g.
    ``inv(X) @ y`` of a positive-definite :math:`X` specializes to a Cholesky
    solve within the loop body. Matrix properties are invariant to batch axes,
    so the assertion is valid for every per-step slice.

    Recurrent inner inputs are excluded: the loop body need not preserve the
    initial value's properties, so the carried state cannot be assumed to keep
    them past the first step.
    """
    scan_op = node.op
    inner_inputs = scan_op.inner_inputs
    non_recurrent = set(scan_op.inner_seqs(inner_inputs))
    non_recurrent.update(scan_op.inner_non_seqs(inner_inputs))
    outer_from_inner = scan_op.get_oinp_iinp_iout_oout_mappings()[
        "outer_inp_from_inner_inp"
    ]

    new_facts = {}
    for inner_idx, inner_inp in enumerate(inner_inputs):
        if inner_inp not in non_recurrent:
            continue
        clients = scan_op.fgraph.clients.get(inner_inp, ())
        if any(
            not isinstance(client, str) and isinstance(client.op, SpecifyAssumptions)
            for client, _ in clients
        ):
            # Already carries an inner assertion -- skip to avoid re-firing.
            continue
        outer_inp = node.inputs[outer_from_inner[inner_idx]]
        facts = {
            key.name: FactState.TRUE
            for key in ALL_KEYS
            if check_assumption(fgraph, outer_inp, key)
        }
        if facts:
            new_facts[inner_inp] = facts

    if not new_facts:
        return None

    # Rebuild the inner graph over fresh leaves, splicing the assertions on.
    replace = {}
    input_clones = []
    for inner_inp in inner_inputs:
        clone = inner_inp.clone()
        input_clones.append(clone)
        facts = new_facts.get(inner_inp)
        replace[inner_inp] = SpecifyAssumptions(facts)(clone) if facts else clone
    new_inner_outputs = clone_replace(scan_op.inner_outputs, replace=replace)

    new_scan_op = copy(scan_op)
    new_scan_op.fgraph = FunctionGraph(input_clones, new_inner_outputs, clone=False)
    new_outs = new_scan_op.make_node(*node.inputs).outputs
    copy_stack_trace(node.outputs, new_outs)
    return new_outs


scan_seqopt1.register(
    lift_assumptions_into_scan.__name__,
    dfs_rewriter(lift_assumptions_into_scan, ignore_newtrees=True),
    "fast_run",
    "scan",
    position=1,
)
