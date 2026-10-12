"""Rewrites for conditional graph evaluation."""

from pytensor.compile.mode import optdb
from pytensor.conditional.op import IfElse, ifelse
from pytensor.graph.features import ReplaceValidate
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.basic import (
    GraphRewriter,
    in2out,
    node_rewriter,
)
from pytensor.graph.traversal import apply_depends_on
from pytensor.tensor.basic import Alloc
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import Argmax, Dot, Max
from pytensor.tensor.shape import Reshape, Shape, SpecifyShape
from pytensor.tensor.subtensor import IncSubtensor, Subtensor


@node_rewriter([IfElse])
def cond_make_inplace(fgraph, node):
    """Allow a conditional to return a view of its selected true input.

    Pattern::

        ifelse(c, x, y, as_view=False)

    Replacement::

        ifelse(c, x, y, as_view=True)

    Already-viewing conditionals need no replacement. Large scalar graphs skip
    this rewrite because the cost of checking views outweighs the saved copies.
    """
    op = node.op
    if op.as_view:
        return False
    # For big graph, do not make inplace scalar to speed up optimization.
    if len(fgraph.apply_nodes) >= 500 and all(
        getattr(o.type, "ndim", -1) == 0 for o in node.outputs
    ):
        return False
    return IfElse(n_outs=op.n_outs, as_view=True, name=op.name)(
        *node.inputs, return_list=True
    )


optdb.register(
    "cond_make_inplace",
    in2out(cond_make_inplace, ignore_newtrees=True),
    "fast_run",
    "inplace",
    position=95,
)

# XXX: Optimizations commented pending further debugging (certain optimizations
# make computation less lazy than it should be currently).
#
# ifelse_equilibrium = graph.rewriting.db.EquilibriumDB()
# ifelse_seqopt = graph.rewriting.db.SequenceDB()
# ifelse_equilibrium.register('seq_ifelse', ifelse_seqopt, 'fast_run',
#                             'ifelse')
""" Comments:
I've wrote this comments to explain how the optimization of ifelse function
(for future developers that need to parse this part of code. Please try to
keep this comments in sync with whatever changes you add to the code.

ifelse optimization are registered before canonicalize !

The optimizations are called in sequence as follows:
    * equilibrium shell (runs until no change):
        * ifelse_lift
        * ifelse_merge_ifs
        * ifelse_merge_nodes
        * ifelse_remove_identical_inside
        * ifelse_sameCondTrue_inside
        * ifelse_sameCondFalse_inside
    * merge_nodes_1
    * ifelse_sameCondTrue
    * ifelse_sameCondFalse
    * ifelse_removeIdentical

where, each of the optimization do the following things:
    `ifelse_lift` (def cond_lift_single_if):

"""
# optdb.register('ifelse_equilibriumOpt', ifelse_equilibrium, 'fast_run',
#                'ifelse', position=.5)


acceptable_ops = (
    Shape,
    SpecifyShape,
    Reshape,
    Dot,
    Max,
    Argmax,
    Subtensor,
    IncSubtensor,
    Alloc,
    Elemwise,
    DimShuffle,
    Blockwise,
)


@node_rewriter(acceptable_ops)
def ifelse_lift_single_if_through_acceptable_ops(fgraph, main_node):
    """This optimization lifts up certain ifelse instances.

        op(ifelse(c, x, y)) -> ifelse(c, op(x), op(y))

    if `op` is in the `acceptable_ops` list, and there is no other if as
    input to that specific `op`, and the if has no other clients !?
    """
    if not (isinstance(main_node.op, acceptable_ops)):
        return False
    all_inp_nodes = set()
    for inp in main_node.inputs:
        all_inp_nodes.add(inp.owner)
    ifnodes = [x for x in list(all_inp_nodes) if x and isinstance(x.op, IfElse)]
    # if we have multiple ifs as inputs .. it all becomes quite complicated
    # :)
    if len(ifnodes) != 1:
        return False
    node = ifnodes[0]
    op = node.op

    aes = node.inputs[1:][: op.n_outs]
    fs = node.inputs[1:][op.n_outs :]

    # outs = main_node.outputs
    mop = main_node.op
    true_ins = []
    false_ins = []

    for x in main_node.inputs:
        if x in node.outputs:
            idx = node.outputs.index(x)
            true_ins.append(aes[idx])
            false_ins.append(fs[idx])
        else:
            true_ins.append(x)
            false_ins.append(x)
    true_eval = mop(*true_ins, return_list=True)
    false_eval = mop(*false_ins, return_list=True)
    # true_eval  = clone_replace(outs, replace = dict(zip(node.outputs, aes)))
    # false_eval = clone_replace(outs, replace = dict(zip(node.outputs, fs)))

    nw_outs = ifelse(node.inputs[0], true_eval, false_eval, return_list=True)
    return nw_outs


@node_rewriter([IfElse])
def cond_merge_ifs_true(fgraph, node):
    """Simplify a nested true-branch conditional with the same predicate.

    Pattern::

        ifelse(c, ifelse(c, x, y), z)

    Replacement::

        ifelse(c, x, z)

    The outer true branch already establishes that the shared predicate is true,
    so an inner conditional supplying that branch can use its true inputs.
    """
    op = node.op
    t_ins = node.inputs[1:][: op.n_outs]

    replace = {}
    for idx, tval in enumerate(t_ins):
        if (
            tval.owner
            and isinstance(tval.owner.op, IfElse)
            and tval.owner.inputs[0] == node.inputs[0]
        ):
            ins_op = tval.owner.op
            ins_t = tval.owner.inputs[1:][: ins_op.n_outs]
            replace[idx + 1] = ins_t[tval.owner.outputs.index(tval)]

    if len(replace) == 0:
        return False

    old_ins = list(node.inputs)
    for pos, var in replace.items():
        old_ins[pos] = var
    return op(*old_ins, return_list=True)


@node_rewriter([IfElse])
def cond_merge_ifs_false(fgraph, node):
    """Simplify a nested false-branch conditional with the same predicate.

    Pattern::

        ifelse(c, z, ifelse(c, x, y))

    Replacement::

        ifelse(c, z, y)

    The outer false branch already establishes that the shared predicate is false,
    so an inner conditional supplying that branch can use its false inputs.
    """
    op = node.op
    f_ins = node.inputs[1:][op.n_outs :]

    replace = {}
    for idx, fval in enumerate(f_ins):
        if (
            fval.owner
            and isinstance(fval.owner.op, IfElse)
            and fval.owner.inputs[0] == node.inputs[0]
        ):
            ins_op = fval.owner.op
            ins_t = fval.owner.inputs[1:][ins_op.n_outs :]
            replace[idx + 1 + op.n_outs] = ins_t[fval.owner.outputs.index(fval)]

    if len(replace) == 0:
        return False

    old_ins = list(node.inputs)
    for pos, var in replace.items():
        old_ins[pos] = var
    return op(*old_ins, return_list=True)


class CondMerge(GraphRewriter):
    """Graph Optimizer that merges different cond ops"""

    def add_requirements(self, fgraph):
        fgraph.add_feature(ReplaceValidate())

    def apply(self, fgraph):
        nodelist = list(fgraph.toposort())
        cond_nodes = [s for s in nodelist if isinstance(s.op, IfElse)]
        if len(cond_nodes) < 2:
            return False
        merging_node = cond_nodes[0]
        for proposal in cond_nodes[1:]:
            if proposal.inputs[0] == merging_node.inputs[0] and not apply_depends_on(
                proposal, merging_node
            ):
                # Create a list of replacements for proposal
                mn_ts = merging_node.inputs[1:][: merging_node.op.n_outs]
                mn_fs = merging_node.inputs[1:][merging_node.op.n_outs :]
                pl_ts = proposal.inputs[1:][: proposal.op.n_outs]
                pl_fs = proposal.inputs[1:][proposal.op.n_outs :]
                new_ins = [merging_node.inputs[0], *mn_ts, *pl_ts, *mn_fs, *pl_fs]
                mn_name = "?"
                if merging_node.op.name:
                    mn_name = merging_node.op.name
                pl_name = "?"
                # mn_n_ts = len(mn_ts)
                # mn_n_fs = len(mn_fs)
                if proposal.op.name:
                    pl_name = proposal.op.name
                new_ifelse = IfElse(
                    n_outs=len(mn_ts + pl_ts),
                    as_view=False,
                    name=mn_name + "&" + pl_name,
                )
                new_outs = new_ifelse(*new_ins, return_list=True)
                new_outs = [clone_replace(x) for x in new_outs]
                old_outs = []
                if not isinstance(merging_node.outputs, list | tuple):
                    old_outs += [merging_node.outputs]
                else:
                    old_outs += merging_node.outputs
                if not isinstance(proposal.outputs, list | tuple):
                    old_outs += [proposal.outputs]
                else:
                    old_outs += proposal.outputs
                pairs = list(zip(old_outs, new_outs, strict=True))
                fgraph.replace_all_validate(pairs, reason="cond_merge")


@node_rewriter([IfElse])
def cond_remove_identical(fgraph, node):
    op = node.op

    aes = node.inputs[1:][: op.n_outs]
    fs = node.inputs[1:][op.n_outs :]

    # sync outs
    out_map = {}
    for idx in range(len(node.outputs)):
        if idx not in out_map:
            for jdx in range(idx + 1, len(node.outputs)):
                if aes[idx] == aes[jdx] and fs[idx] == fs[jdx] and jdx not in out_map:
                    out_map[jdx] = idx

    if len(out_map) == 0:
        return False

    nw_ts = []
    nw_fs = []
    inv_map = {}
    pos = 0
    for idx in range(len(node.outputs)):
        if idx not in out_map:
            inv_map[idx] = pos
            pos = pos + 1
            nw_ts.append(aes[idx])
            nw_fs.append(fs[idx])

    new_ifelse = IfElse(n_outs=len(nw_ts), as_view=op.as_view, name=op.name)

    new_ins = [node.inputs[0], *nw_ts, *nw_fs]
    new_outs = new_ifelse(*new_ins, return_list=True)

    rval = []
    for idx in range(len(node.outputs)):
        if idx in out_map:
            rval += [new_outs[inv_map[out_map[idx]]]]
        else:
            rval += [new_outs[inv_map[idx]]]

    return rval


@node_rewriter([IfElse])
def cond_merge_random_op(fgraph, main_node):
    if isinstance(main_node.op, IfElse):
        return False

    all_inp_nodes = set()
    for inp in main_node.inputs:
        all_inp_nodes.add(inp.owner)
    cond_nodes = [x for x in list(all_inp_nodes) if x and isinstance(x.op, IfElse)]

    if len(cond_nodes) < 2:
        return False

    merging_node = cond_nodes[0]
    for proposal in cond_nodes[1:]:
        if (
            proposal.inputs[0] == merging_node.inputs[0]
            and not apply_depends_on(proposal, merging_node)
            and not apply_depends_on(merging_node, proposal)
        ):
            # Create a list of replacements for proposal
            mn_ts = merging_node.inputs[1:][: merging_node.op.n_outs]
            mn_fs = merging_node.inputs[1:][merging_node.op.n_outs :]
            pl_ts = proposal.inputs[1:][: proposal.op.n_outs]
            pl_fs = proposal.inputs[1:][proposal.op.n_outs :]
            new_ins = [merging_node.inputs[0], *mn_ts, *pl_ts, *mn_fs, *pl_fs]
            mn_name = "?"
            if merging_node.op.name:
                mn_name = merging_node.op.name
            pl_name = "?"
            # mn_n_ts = len(mn_ts)
            # mn_n_fs = len(mn_fs)
            if proposal.op.name:
                pl_name = proposal.op.name
            new_ifelse = IfElse(
                n_outs=len(mn_ts + pl_ts),
                as_view=False,
                name=mn_name + "&" + pl_name,
            )
            new_outs = new_ifelse(*new_ins, return_list=True)
            old_outs = []
            if not isinstance(merging_node.outputs, list | tuple):
                old_outs += [merging_node.outputs]
            else:
                old_outs += merging_node.outputs
            if not isinstance(proposal.outputs, list | tuple):
                old_outs += [proposal.outputs]
            else:
                old_outs += proposal.outputs
            pairs = list(zip(old_outs, new_outs, strict=True))
            main_outs = clone_replace(main_node.outputs, replace=pairs)
            return main_outs


# XXX: Optimizations commented pending further debugging (certain optimizations
# make computation less lazy than it should be currently).
#
# pushout_equilibrium = graph.rewriting.db.EquilibriumDB()
#
# XXX: This optimization doesn't seem to exist anymore?
# pushout_equilibrium.register("cond_lift_single_if",
#                              in2out(cond_lift_single_if,
#                                         ignore_newtrees=True),
#                              'fast_run', 'ifelse')
#
# pushout_equilibrium.register("cond_merge_random_op",
#                              in2out(cond_merge_random_op,
#                                         ignore_newtrees=True),
#                              'fast_run', 'ifelse')
#
#
# pushout_equilibrium.register("ifelse_merge",
#                              graph.opt.MergeOptimizer(skip_const_merge=False),
#                              'fast_run', 'ifelse')
#
# pushout_equilibrium.register("ifelse_remove_identical_inside",
#                              in2out(cond_remove_identical,
#                                         ignore_newtrees=True),
#                              'fast_run', 'ifelse')
#
# pushout_equilibrium.register('ifelse_sameCondTrue_inside',
#                              in2out(cond_merge_ifs_true,
#                                         ignore_newtrees=True),
#                              'fast_run', 'ifelse')
#
# pushout_equilibrium.register('ifelse_sameCondFalse_inside',
#                              in2out(cond_merge_ifs_false,
#                                         ignore_newtrees=True),
#                              'fast_run', 'ifelse')
#
# ifelse_seqopt.register('ifelse_condPushOut_equilibrium',
#                        pushout_equilibrium,
#                        'fast_run', 'ifelse', position=1)
#
# ifelse_seqopt.register('merge_nodes_1',
#                        graph.opt.MergeOptimizer(skip_const_merge=False),
#                        'fast_run', 'ifelse', position=2)
#
#
# ifelse_seqopt.register('ifelse_sameCondTrue',
#                        in2out(cond_merge_ifs_true,
#                                   ignore_newtrees=True),
#                        'fast_run', 'ifelse', position=3)
#
#
# ifelse_seqopt.register('ifelse_sameCondFalse',
#                        in2out(cond_merge_ifs_false,
#                                   ignore_newtrees=True),
#                        'fast_run', 'ifelse', position=4)
#
#
# ifelse_seqopt.register('ifelse_removeIdenetical',
#                        in2out(cond_remove_identical,
#                                   ignore_newtrees=True),
#                        'fast_run', 'ifelse', position=7)
