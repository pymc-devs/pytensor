"""Cross-Scan and intra-Scan merging.

* :class:`ScanMerge` — merge sibling ``Scan`` nodes that share the same
  driving ``n_steps`` and signature into a single op.
* :func:`scan_merge_inouts` — within one ``Scan``, merge inner / outer
  inputs and outputs that compute the same value.
"""

from itertools import chain

from pytensor.graph.basic import NominalVariable, equal_computations
from pytensor.graph.features import ReplaceValidate
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.basic import GraphRewriter, node_rewriter
from pytensor.graph.traversal import ancestors, apply_depends_on
from pytensor.scan.op import Scan, ScanInfo
from pytensor.scan.utils import ScanArgs, reconstruct_graph
from pytensor.tensor.basic import get_scalar_constant_value
from pytensor.tensor.exceptions import NotScalarConstantError


class ScanMerge(GraphRewriter):
    r"""Graph optimizer that merges different scan ops.

    This optimization attempts to fuse distinct `Scan` `Op`s into a single `Scan` `Op`
    that performs all the computation. The main advantage of merging `Scan` `Op`\s
    together comes from the possibility of both original `Op`\s having some
    computation in common. In such a setting, this computation ends up being done
    twice. The fused `Scan` `Op`, however, would only need to do it once and could
    therefore be more computationally efficient. Also, since every `Scan` node
    involves a certain overhead, at runtime, reducing the number of `Scan` nodes in
    the graph can improve performance.

    """

    def add_requirements(self, fgraph):
        fgraph.attach_feature(ReplaceValidate())

    def merge(self, nodes):
        if nodes[0].op.info.as_while:
            as_while = True
            condition = nodes[0].op.inner_outputs[-1]
        else:
            as_while = False

        # We keep the inner_ins and inner_outs of each original node separated.
        # To be able to recombine them in the right order after the clone,
        # we also need to split them by types (seq, mitmot, ...).
        # On the other hand, outer_ins, outer_outs and info are held together.
        inner_ins = [[] for nd in nodes]
        outer_ins = []
        inner_outs = [[] for nd in nodes]
        outer_outs = []

        for idx, nd in enumerate(nodes):
            inner_ins[idx].append(nd.op.inner_seqs(nd.op.inner_inputs))
            outer_ins += nd.op.outer_seqs(nd.inputs)

        mit_mot_out_slices = ()

        mit_mot_in_slices = ()
        for idx, nd in enumerate(nodes):
            inner_ins[idx].append(nd.op.inner_mitmot(nd.op.inner_inputs))
            inner_outs[idx].append(nd.op.inner_mitmot_outs(nd.op.inner_outputs))
            mit_mot_in_slices += nd.op.info.mit_mot_in_slices
            mit_mot_out_slices += nd.op.info.mit_mot_out_slices[: nd.op.info.n_mit_mot]
            outer_ins += nd.op.outer_mitmot(nd.inputs)
            outer_outs += nd.op.outer_mitmot_outs(nd.outputs)

        mit_sot_in_slices = ()
        for idx, nd in enumerate(nodes):
            inner_ins[idx].append(nd.op.inner_mitsot(nd.op.inner_inputs))
            inner_outs[idx].append(nd.op.inner_mitsot_outs(nd.op.inner_outputs))
            mit_sot_in_slices += nd.op.info.mit_sot_in_slices
            outer_ins += nd.op.outer_mitsot(nd.inputs)
            outer_outs += nd.op.outer_mitsot_outs(nd.outputs)

        sit_sot_in_slices = ()
        for idx, nd in enumerate(nodes):
            inner_ins[idx].append(nd.op.inner_sitsot(nd.op.inner_inputs))
            sit_sot_in_slices += tuple((-1,) for x in range(nd.op.info.n_sit_sot))
            inner_outs[idx].append(nd.op.inner_sitsot_outs(nd.op.inner_outputs))
            outer_ins += nd.op.outer_sitsot(nd.inputs)
            outer_outs += nd.op.outer_sitsot_outs(nd.outputs)

        for idx, nd in enumerate(nodes):
            # Shared
            inner_ins[idx].append(nd.op.inner_untraced_sit_sot(nd.op.inner_inputs))
            outer_ins += nd.op.outer_untraced_sit_sot(nd.inputs)

        for idx, nd in enumerate(nodes):
            # NitSot
            inner_outs[idx].append(nd.op.inner_nitsot_outs(nd.op.inner_outputs))
            outer_ins += nd.op.outer_nitsot(nd.inputs)
            outer_outs += nd.op.outer_nitsot_outs(nd.outputs)

        for idx, nd in enumerate(nodes):
            # Shared
            outer_outs += nd.op.outer_untraced_sit_sot_outs(nd.outputs)
            inner_outs[idx].append(
                nd.op.inner_untraced_sit_sot_outs(nd.op.inner_outputs)
            )

        n_non_seqs = 0
        for idx, nd in enumerate(nodes):
            # Non Seqs
            node_inner_non_seqs = nd.op.inner_non_seqs(nd.op.inner_inputs)
            n_non_seqs += len(node_inner_non_seqs)
            inner_ins[idx].append(node_inner_non_seqs)
            outer_ins += nd.op.outer_non_seqs(nd.inputs)

        # Add back the number of steps
        outer_ins = [nodes[0].inputs[0], *outer_ins]

        if as_while:
            # add the condition, which was the one of nodes[0]
            inner_outs[0].append([condition])

        # Clone the inner graph of each node independently
        for idx, nd in enumerate(nodes):
            # concatenate all inner_ins and inner_outs of nd
            flat_inner_ins = list(chain.from_iterable(inner_ins[idx]))
            flat_inner_outs = list(chain.from_iterable(inner_outs[idx]))
            # clone
            flat_inner_ins, flat_inner_outs = reconstruct_graph(
                flat_inner_ins, flat_inner_outs
            )
            # split the new inner variables again in seq, mitmot, etc.
            new_inner_ins = []
            count = 0
            for nl in inner_ins[idx]:
                seq_len = len(nl)
                new_inner_ins.append(flat_inner_ins[count : (count + seq_len)])
                count += seq_len

            new_inner_outs = []
            count = 0
            for nl in inner_outs[idx]:
                seq_len = len(nl)
                new_inner_outs.append(flat_inner_outs[count : (count + seq_len)])
                count += seq_len

            inner_ins[idx] = new_inner_ins
            inner_outs[idx] = new_inner_outs

        # Flatten inner_ins and inner_outs so that all seqs are first,
        # then mitmot, etc.
        new_inner_ins = []
        new_inner_outs = []
        nb_ins_groups = len(inner_ins[0])
        nb_outs_groups = len(inner_outs[0])
        for idx, nd in enumerate(nodes):
            # All inner_ins should have the same length
            assert len(inner_ins[idx]) == nb_ins_groups

            # All inner_outs should have the same length, except if as_while,
            # in which case the first one should have one more element
            if as_while and idx > 0:
                assert len(inner_outs[idx]) == nb_outs_groups - 1
            else:
                assert len(inner_outs[idx]) == nb_outs_groups

        for gr_idx in range(nb_ins_groups):
            for idx, nd in enumerate(nodes):
                new_inner_ins += inner_ins[idx][gr_idx]

        for gr_idx in range(nb_outs_groups):
            for idx, nd in enumerate(nodes):
                if as_while and idx > 0 and gr_idx == (nb_outs_groups - 1):
                    # There is no condition on that node, skip it
                    pass
                else:
                    new_inner_outs += inner_outs[idx][gr_idx]

        info = ScanInfo(
            n_seqs=sum(nd.op.info.n_seqs for nd in nodes),
            mit_mot_in_slices=mit_mot_in_slices,
            mit_mot_out_slices=mit_mot_out_slices,
            mit_sot_in_slices=mit_sot_in_slices,
            sit_sot_in_slices=sit_sot_in_slices,
            n_nit_sot=sum(nd.op.info.n_nit_sot for nd in nodes),
            n_untraced_sit_sot=sum(nd.op.info.n_untraced_sit_sot for nd in nodes),
            n_non_seqs=n_non_seqs,
            as_while=as_while,
        )

        old_op = nodes[0].op
        new_op = Scan(
            new_inner_ins,
            new_inner_outs,
            info,
            mode=old_op.mode,
            profile=old_op.profile,
            truncate_gradient=old_op.truncate_gradient,
            allow_gc=old_op.allow_gc,
            name="&".join(nd.op.name for nd in nodes),
        )
        new_outs = new_op(*outer_ins)

        if not isinstance(new_outs, list | tuple):
            new_outs = [new_outs]

        return list(zip(outer_outs, new_outs, strict=True))

    def belongs_to_set(self, node, set_nodes):
        """
        This function checks if node `node` belongs to `set_nodes`, in the
        sense that it can be merged together with every other node in
        `set_nodes`. In order for two nodes to be mergeable, they have to go
        over the same number of steps, have the same condition (if any),
        have the same value for truncate_gradient, and have the same mode.
        Questionable, we should also consider profile ?

        """
        op = node.op
        rep_node = set_nodes[0]
        rep_op = rep_node.op
        if (
            op.info.as_while != rep_op.info.as_while
            or op.truncate_gradient != rep_op.truncate_gradient
            or op.mode != rep_op.mode
        ):
            return False

        nsteps = node.inputs[0]
        try:
            nsteps = int(get_scalar_constant_value(nsteps))
        except NotScalarConstantError:
            pass

        rep_nsteps = rep_node.inputs[0]
        try:
            rep_nsteps = int(get_scalar_constant_value(rep_nsteps))
        except NotScalarConstantError:
            pass

        if nsteps != rep_nsteps:
            return False

        # Check to see if it is an input of a different node
        for nd in set_nodes:
            if apply_depends_on(node, nd) or apply_depends_on(nd, node):
                return False

        if not op.info.as_while:
            return True

        # We need to check the while conditions are identical
        conds = [op.inner_outputs[-1]]
        rep_conds = [rep_op.inner_outputs[-1]]
        if not equal_computations(
            conds, rep_conds, op.inner_inputs, rep_op.inner_inputs
        ):
            return False

        # If they depend on inner inputs we need to check for equivalence on the respective outer inputs
        nominal_inputs = [a for a in ancestors(conds) if isinstance(a, NominalVariable)]
        if not nominal_inputs:
            return True
        rep_nominal_inputs = [
            a for a in ancestors(rep_conds) if isinstance(a, NominalVariable)
        ]

        conds = []
        rep_conds = []
        mapping = op.get_oinp_iinp_iout_oout_mappings()["outer_inp_from_inner_inp"]
        rep_mapping = rep_op.get_oinp_iinp_iout_oout_mappings()[
            "outer_inp_from_inner_inp"
        ]
        inner_inputs = op.inner_inputs
        rep_inner_inputs = rep_op.inner_inputs
        for nominal_input, rep_nominal_input in zip(
            nominal_inputs, rep_nominal_inputs, strict=True
        ):
            conds.append(node.inputs[mapping[inner_inputs.index(nominal_input)]])
            rep_conds.append(
                rep_node.inputs[rep_mapping[rep_inner_inputs.index(rep_nominal_input)]]
            )

        return equal_computations(conds, rep_conds)

    def apply(self, fgraph):
        # Collect all scan nodes ordered according to toposort
        scan_nodes = [nd for nd in fgraph.toposort() if isinstance(nd.op, Scan)]

        # All sets of possibly mergeable nodes
        all_sets = []

        for nd in scan_nodes:
            belongs_to_set_idx = -1
            for pos, subset in enumerate(all_sets):
                if self.belongs_to_set(nd, subset):
                    belongs_to_set_idx = pos
                    # It is possible that nd belongs to more than one subset.
                    # For instance, if we have 3 Scan nodes X, Y and Z, if Z
                    # depends on the output of X, then X and Z are incompatible
                    # and would create different subsets, but Y could be
                    # compatible with both X and Z. We choose the first one.
                    break

            if belongs_to_set_idx == -1:
                all_sets.append([nd])
            else:
                all_sets[belongs_to_set_idx].append(nd)

        for subset in all_sets:
            if len(subset) > 1:
                proposal = self.merge(subset)
                fgraph.replace_all_validate_remove(
                    proposal, remove=subset, reason="scan_merge"
                )


def has_duplicates(l):
    """
    Returns true if l has any duplicates (according to __eq__).

    """
    return len(set(l)) < len(l)


def make_equiv(lo, li):
    """
    Builds a dictionary of equivalences between inner inputs based on
    the equivalence of their corresponding outer inputs.

    """
    seeno = {}
    left = []
    right = []
    for o, i in zip(lo, li, strict=True):
        if o in seeno:
            left += [i]
            right += [o]
        else:
            seeno[o] = i
    return left, right


@node_rewriter([Scan])
def scan_merge_inouts(fgraph, node):
    """
    This optimization attempts to merge a `Scan` `Op`'s identical outer inputs as well
    as merge its identical outer outputs (outputs that perform the same
    computation on the same inputs). This can reduce the amount of computation as
    well as result in a simpler graph for both the inner function and the outer
    function.
    """
    if not isinstance(node.op, Scan):
        return False

    # Do a first pass to merge identical external inputs.
    # Equivalent inputs will be stored in inp_equiv, then a new
    # scan node created without duplicates.
    a = ScanArgs(
        node.inputs,
        node.outputs,
        node.op.inner_inputs,
        node.op.inner_outputs,
        node.op.info,
    )

    inp_equiv = {}

    if has_duplicates(a.outer_in_seqs):
        new_outer_seqs = []
        new_inner_seqs = []
        for out_seq, in_seq in zip(a.outer_in_seqs, a.inner_in_seqs, strict=True):
            if out_seq in new_outer_seqs:
                i = new_outer_seqs.index(out_seq)
                inp_equiv[in_seq] = new_inner_seqs[i]
            else:
                new_outer_seqs.append(out_seq)
                new_inner_seqs.append(in_seq)
        a.outer_in_seqs = new_outer_seqs
        a.inner_in_seqs = new_inner_seqs

    if has_duplicates(a.outer_in_non_seqs):
        new_outer_nseqs = []
        new_inner_nseqs = []
        for out_nseq, in_nseq in zip(
            a.outer_in_non_seqs, a.inner_in_non_seqs, strict=True
        ):
            if out_nseq in new_outer_nseqs:
                i = new_outer_nseqs.index(out_nseq)
                inp_equiv[in_nseq] = new_inner_nseqs[i]
            else:
                new_outer_nseqs.append(out_nseq)
                new_inner_nseqs.append(in_nseq)
        a.outer_in_non_seqs = new_outer_nseqs
        a.inner_in_non_seqs = new_inner_nseqs

    if len(inp_equiv) > 0:
        # do the replacement now. The rest will be left to ScanSaveMem
        inner_inputs = a.inner_inputs
        outer_inputs = a.outer_inputs
        info = a.info
        a_inner_outs = a.inner_outputs
        inner_outputs = clone_replace(a_inner_outs, replace=inp_equiv)

        new_op = Scan(
            inner_inputs,
            inner_outputs,
            info,
            mode=node.op.mode,
            profile=node.op.profile,
            truncate_gradient=node.op.truncate_gradient,
            # TODO: This seems questionable
            name=node.op.name,
            allow_gc=node.op.allow_gc,
        )
        outputs = new_op(*outer_inputs)

        if not isinstance(outputs, list | tuple):
            outputs = [outputs]

        na = ScanArgs(
            outer_inputs,
            outputs,
            new_op.inner_inputs,
            new_op.inner_outputs,
            new_op.info,
        )
        remove = [node]
    else:
        na = a
        remove = []

    # Now that the identical external inputs have been merged, we do a new
    # loop in order to merge external outputs that compute the same things
    # from the same inputs.
    left = []
    right = []

    if has_duplicates(na.outer_in_shared):
        _left, _right = make_equiv(na.outer_in_shared, na.inner_in_shared)
        left += _left
        right += _right
    if has_duplicates(na.outer_in_sit_sot):
        _left, _right = make_equiv(na.outer_in_sit_sot, na.inner_in_sit_sot)
        left += _left
        right += _right
    if has_duplicates(na.outer_in_mit_mot):
        seen = {}
        for omm, imm, _sl in zip(
            na.outer_in_mit_mot, na.inner_in_mit_mot, na.mit_mot_in_slices, strict=True
        ):
            sl = tuple(_sl)
            if (omm, sl) in seen:
                simm = seen[(omm, sl)]
                left += imm
                right += simm
            else:
                seen[(omm, sl)] = imm

    if has_duplicates(na.outer_in_mit_sot):
        seen = {}
        for oms, ims, _sl in zip(
            na.outer_in_mit_sot, na.inner_in_mit_sot, na.mit_sot_in_slices, strict=True
        ):
            sl = tuple(_sl)
            if (oms, sl) in seen:
                sims = seen[(oms, sl)]
                left += ims
                right += sims
            else:
                seen[(oms, sl)] = ims

    def map_out(outer_i, inner_o, outer_o, seen):
        # Return the outer input corresponding to an
        # (outer input, inner output) pair. If we see that pair for the first
        # time, return the provided outer output. If an equivalent pair had
        # already been seen, return that one instead.
        # Note that we need to check that the outer input match as well,
        # because they could have different sizes, and the corresponding
        # outer outputs cannot be merged in that case.
        for s_outer_i, s_inner_o, s_outer_o in seen:
            if (
                equal_computations([inner_o], [s_inner_o], left, right)
                and outer_i == s_outer_i
            ):
                return s_outer_o
        seen.append((outer_i, inner_o, outer_o))
        return outer_o

    seen = []

    assert len(na.outer_in_nit_sot) == len(na.inner_out_nit_sot)
    assert len(na.inner_out_nit_sot) == len(na.outer_out_nit_sot)
    na.outer_out_nit_sot = [
        map_out(outer_i, inner_o, outer_o, seen)
        for outer_i, inner_o, outer_o in zip(
            na.outer_in_nit_sot, na.inner_out_nit_sot, na.outer_out_nit_sot, strict=True
        )
    ]

    seen = []
    assert len(na.outer_in_sit_sot) == len(na.inner_out_sit_sot)
    assert len(na.inner_out_sit_sot) == len(na.outer_out_sit_sot)
    na.outer_out_sit_sot = [
        map_out(outer_i, inner_o, outer_o, seen)
        for outer_i, inner_o, outer_o in zip(
            na.outer_in_sit_sot, na.inner_out_sit_sot, na.outer_out_sit_sot, strict=True
        )
    ]

    seen = []
    assert len(na.outer_in_mit_sot) == len(na.inner_out_mit_sot)
    assert len(na.inner_out_mit_sot) == len(na.outer_out_mit_sot)
    na.outer_out_mit_sot = [
        map_out(outer_i, inner_o, outer_o, seen)
        for outer_i, inner_o, outer_o in zip(
            na.outer_in_mit_sot, na.inner_out_mit_sot, na.outer_out_mit_sot, strict=True
        )
    ]

    seen = []
    new_outer_out_mit_mot = []
    assert len(na.outer_in_mit_mot) == len(na.inner_out_mit_mot)
    assert len(na.inner_out_mit_mot) == len(na.outer_out_mit_mot)
    assert len(na.outer_out_mit_mot) == len(na.mit_mot_out_slices)
    for outer_imm, inner_omm, outer_omm, osl in zip(
        na.outer_in_mit_mot,
        na.inner_out_mit_mot,
        na.outer_out_mit_mot,
        na.mit_mot_out_slices,
        strict=True,
    ):
        for s_outer_imm, s_inner_omm, s_outer_omm, sosl in seen:
            if (
                osl == sosl
                and equal_computations(inner_omm, s_inner_omm, left, right)
                and outer_imm == s_outer_imm
            ):
                new_outer_out_mit_mot.append(s_outer_omm)
                break
        else:
            seen.append((outer_imm, inner_omm, outer_omm, osl))
            new_outer_out_mit_mot.append(outer_omm)
    na.outer_out_mit_mot = new_outer_out_mit_mot
    if remove:
        return dict(
            [("remove", remove), *zip(node.outputs, na.outer_outputs, strict=True)]
        )
    return na.outer_outputs
