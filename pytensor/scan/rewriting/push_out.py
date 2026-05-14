"""Push-out rewrites: hoist computation out of the Scan inner graph.

* :func:`scan_push_out_non_seq` — pull node-only computation out into the
  outer graph.
* :func:`scan_push_out_seq` — pull elemwise / dimshuffle computation that
  depends on sequences out and apply it to the full outer sequence at
  once.
* :func:`scan_push_out_add` — push terminal nit_sot accumulators out
  into a single outer reduction.
* :func:`scan_push_out_dot1` — pull a trailing ``dot`` out of the inner
  graph.
"""

import copy
import dataclasses

import numpy as np

import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.fg import FunctionGraph, Output
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.graph.type import HasShape
from pytensor.scan.op import Scan
from pytensor.scan.utils import ScanArgs, reconstruct_graph, safe_new
from pytensor.tensor.basic import get_scalar_constant_value
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import Dot, dot
from pytensor.tensor.shape import shape
from pytensor.tensor.subtensor import Subtensor, get_idx_list, set_subtensor


@node_rewriter([Scan])
def scan_push_out_non_seq(fgraph, node):
    r"""Push out the variables inside the `Scan` that depend only on non-sequences.

    This optimizations pushes, out of `Scan`'s inner function and into the outer
    function, computation that depends only on non-sequence inputs. Such
    computation ends up being done every iteration on the same values so moving
    it to the outer function to be executed only once, before the `Scan` `Op`,
    reduces the amount of computation that needs to be performed.
    """
    node_inputs, node_outputs = node.op.inner_inputs, node.op.inner_outputs

    local_fgraph_topo = node.op.fgraph.toposort()
    local_fgraph_outs_set = set(node_outputs)
    local_fgraph_outs_map = {v: k for k, v in enumerate(node_outputs)}

    to_remove_set = set()
    to_replace_set = set()
    to_replace_map = {}

    def add_to_replace(y):
        to_replace_set.add(y)
        to_replace_map[y] = add_to_replace.n
        add_to_replace.n += 1

    add_to_replace.n = 0

    # The variables that will replace the variables pushed-out of the
    # inner-graph
    replace_with_in = []
    # The variables that have been pushed-out of the graph
    replace_with_out = []

    op = node.op
    # Construct the list of non_sequences to simplify a few things
    inner_non_seqs = op.inner_non_seqs(node_inputs)
    inner_non_seqs_set = set(inner_non_seqs)
    inner_non_seqs_map = {v: k for k, v in enumerate(inner_non_seqs)}

    outer_non_seqs = op.outer_non_seqs(node.inputs)

    inner_seqs = op.inner_seqs(node_inputs)
    outer_seqs = op.outer_seqs(node.inputs)

    assert len(inner_non_seqs) == len(outer_non_seqs)
    assert len(inner_seqs) == len(outer_seqs)

    for nd in local_fgraph_topo:
        if (  # we haven't already looked at this node
            nd not in to_remove_set
            and all(
                (
                    (x in inner_non_seqs_set)
                    or (x.owner in to_remove_set)
                    or isinstance(x, Constant)
                )
                for x in nd.inputs
            )
            # We can (supposedly) do this because the assumption is that a
            # `ViewOp` or `DeepCopyOp` will be just at the end of the
            # function and not somewhere in the middle
            and not isinstance(nd.op, ViewOp)
            and not isinstance(nd.op, DeepCopyOp)
        ):
            # We have a candidate node to remove from the inner-graph

            # Step 1. Reconstruct the node using the relevant outer-inputs.
            #
            # More specifically, the node's current inputs are either
            # a) inner-graph input place-holders for non-sequences,
            # b) the outputs of other nodes being pushed out of the inner-graph,
            # c) or constants.
            to_remove_set.add(nd)
            new_inputs = []
            for old_input in nd.inputs:
                if old_input in inner_non_seqs_set:
                    # This is case a), so we want to use the corresponding
                    # outer-graph input as the input to our new pushed-out node
                    _idx = inner_non_seqs_map[old_input]
                    new_input = outer_non_seqs[_idx]
                elif old_input in to_replace_set:
                    # This is case b), so we want to use the new pushed-out node
                    # as the input to this new pushed-out node
                    new_input = replace_with_out[to_replace_map[old_input]]
                else:
                    assert isinstance(old_input, Constant)
                    new_input = old_input

                new_input = old_input.type.filter_variable(new_input)
                new_inputs.append(new_input)

            pushed_out_node = nd.op.make_node(*new_inputs)

            # Step 2. Create variables to replace the old outputs of the node
            # that we're pushing out of the inner-graph
            for idx, y in enumerate(nd.outputs):
                y_place_holder = y.clone()
                # y_place_holder = safe_new(y, "_replace")
                add_to_replace(y)
                replace_with_in.append(y_place_holder)
                assert isinstance(y, type(pushed_out_node.outputs[idx]))
                replace_with_out.append(pushed_out_node.outputs[idx])

    # We need to check all candidate replacements and choose those that
    # make sense for us
    # Step 1. which elements of `to_replace` are used by remaining
    # components of the inner function
    clean_to_replace = []
    clean_replace_with_in = []
    clean_replace_with_out = []
    existent_nodes = [nd for nd in local_fgraph_topo if nd not in to_remove_set]
    existent_nodes_set = set(existent_nodes)

    to_keep_set = set()
    for nd in existent_nodes:
        to_keep_set.update(nd.inputs)

    for out, idx in to_replace_map.items():
        if (  # If types are different, conversion Op will be inserted,
            # and it may trigger an infinite loop.
            out.type.is_super(replace_with_in[idx].type)
            and out in to_keep_set
            and out.owner not in existent_nodes_set
        ):
            clean_to_replace.append(out)
            clean_replace_with_in.append(replace_with_in[idx])
            clean_replace_with_out.append(replace_with_out[idx])

    if len(clean_to_replace) > 0:
        # We can finally put an end to all this madness
        givens = {}
        nw_outer = []
        nw_inner = []
        for to_repl, repl_in, repl_out in zip(
            clean_to_replace, clean_replace_with_in, clean_replace_with_out, strict=True
        ):
            if isinstance(repl_out, Constant):
                repl_in = repl_out
            else:
                nw_inner.append(repl_in)
                nw_outer.append(repl_out)
            givens[to_repl] = repl_in

        op_outs = clone_replace(node_outputs, replace=givens)
        op_ins = node_inputs + nw_inner

        new_info = dataclasses.replace(
            op.info, n_non_seqs=op.info.n_non_seqs + len(nw_outer)
        )

        # Reconstruct node
        nwScan = Scan(
            op_ins,
            op_outs,
            new_info,
            mode=op.mode,
            profile=op.profile,
            truncate_gradient=op.truncate_gradient,
            # TODO: This seems questionable
            name=op.name,
            allow_gc=op.allow_gc,
        )

        nw_node = nwScan(*(node.inputs + nw_outer), return_list=True)[0].owner

        replacements = dict(zip(node.outputs, nw_node.outputs, strict=True))
        replacements["remove"] = [node]
        return replacements
    elif not to_keep_set:
        # Nothing in the inner graph should be kept
        replace_with = {}
        for out, idx in to_replace_map.items():
            if out in local_fgraph_outs_set:
                x = node.outputs[local_fgraph_outs_map[out]]
                y = replace_with_out[idx]
                y_shape = list(y.shape)
                replace_with[x] = pt.alloc(y, node.inputs[0], *y_shape)

        # We need to add one extra dimension to the outputs
        # because the scan op expects for a tensor3, to which an
        # subtensor is applied that takes only the last element
        if replace_with:
            if len(node.outputs) == len(replace_with):
                # Every output of the node has a replacement, the Scan
                # node can be removed from the graph
                replace_with["remove"] = [node]
                return replace_with
            else:
                # The node has some outputs for which no replacement has
                # been established. This can occur for outputs that are
                # not produced by apply nodes (since the optimizations
                # only visits apply nodes) such as constants or inputs
                # passed directly as outputs. The replacements can be
                # performed but the Scan node can't be removed at this
                # point.
                return replace_with

    else:
        return False


@node_rewriter([Scan])
def scan_push_out_seq(fgraph, node):
    r"""Push out the variables inside the `Scan` that depend only on constants and sequences.

    This optimization resembles `scan_push_out_non_seq` but it tries to push--out of
    the inner function--the computation that only relies on sequence and
    non-sequence inputs. The idea behind this optimization is that, when it is
    possible to do so, it is generally more computationally efficient to perform
    a single operation on a large tensor rather then perform that same operation
    many times on many smaller tensors. In many cases, this optimization can
    increase memory usage but, in some specific cases, it can also decrease it.
    """
    node_inputs, node_outputs = node.op.inner_inputs, node.op.inner_outputs

    local_fgraph_topo = node.op.fgraph.toposort()
    local_fgraph_outs_set = set(node_outputs)
    local_fgraph_outs_map = {v: k for k, v in enumerate(node_outputs)}

    to_remove_set = set()
    to_replace_set = set()
    to_replace_map = {}

    def add_to_replace(y):
        to_replace_set.add(y)
        to_replace_map[y] = add_to_replace.n
        add_to_replace.n += 1

    add_to_replace.n = 0

    replace_with_in = []
    replace_with_out = []

    op = node.op
    # Construct the list of non_sequences to simplify a few things
    inner_non_seqs = op.inner_non_seqs(node_inputs)
    inner_non_seqs_set = set(inner_non_seqs)
    inner_non_seqs_map = {v: k for k, v in enumerate(inner_non_seqs)}

    outer_non_seqs = op.outer_non_seqs(node.inputs)
    inner_seqs = op.inner_seqs(node_inputs)
    inner_seqs_set = set(inner_seqs)
    inner_seqs_map = {v: k for k, v in enumerate(inner_seqs)}

    outer_seqs = op.outer_seqs(node.inputs)
    assert len(inner_non_seqs) == len(outer_non_seqs)
    assert len(inner_seqs) == len(outer_seqs)

    for nd in local_fgraph_topo:
        if (
            nd not in to_remove_set
            and all(
                (x in inner_non_seqs_set)
                or (x.owner in to_remove_set)
                or isinstance(x, Constant)
                or (x in inner_seqs_set)
                for x in nd.inputs
            )
            and isinstance(nd.op, Elemwise)
        ):
            outside_ins = []
            depends_on_seqs = False

            for x in nd.inputs:
                if x in inner_non_seqs_set:
                    _idx = inner_non_seqs_map[x]
                    new_input = outer_non_seqs[_idx]
                elif x in inner_seqs_set:
                    new_input = outer_seqs[inner_seqs_map[x]]
                    depends_on_seqs = True
                elif x in to_replace_set:
                    new_input = replace_with_out[to_replace_map[x]]
                    depends_on_seqs = True
                else:
                    assert isinstance(x, Constant)
                    new_input = x

                outside_ins.append(new_input)

            if not depends_on_seqs:
                # Removing this node from the inner graph of scan
                # should be handled by the PushOutNonSeqScan
                # optimization. The current optimization only tries
                # to pull sequence-dependant computation out of
                # scan.
                continue

            to_remove_set.add(nd)

            nw_outer_node = nd.op.make_node(*outside_ins)

            # Step 2. Create variables for replacements
            for idx, y in enumerate(nd.outputs):
                y_place_holder = safe_new(y, "_replace")
                add_to_replace(y)
                replace_with_in.append(y_place_holder)
                replace_with_out.append(nw_outer_node.outputs[idx])

        elif (
            nd not in to_remove_set
            and isinstance(nd.op, DimShuffle)
            and (nd.inputs[0] in inner_seqs_set or nd.inputs[0].owner in to_remove_set)
        ):
            to_remove_set.add(nd)
            x = nd.inputs[0]
            if x in inner_seqs_set:
                outside_ins = outer_seqs[inner_seqs_map[x]]
            elif x in to_replace_set:
                outside_ins = replace_with_out[to_replace_map[x]]
            new_ord = (0,)
            for old_ord in nd.op.new_order:
                if old_ord == "x":
                    new_ord += (old_ord,)
                else:
                    new_ord += (old_ord + 1,)
            new_outer = outside_ins.dimshuffle(new_ord)
            y = nd.outputs[0]
            y_place_holder = safe_new(y, "_replace")
            add_to_replace(y)
            replace_with_in.append(y_place_holder)
            replace_with_out.append(new_outer)

    # We need to check all candidate replacements and choose those that
    # make sense for us
    # Step 1. which elements of `to_replace` are used by remaining
    # components of the inner function
    clean_to_replace = []
    clean_replace_with_in = []
    clean_replace_with_out = []

    existent_nodes = [nd for nd in local_fgraph_topo if nd not in to_remove_set]
    existent_nodes_set = set(existent_nodes)

    to_keep_set = set()
    for nd in existent_nodes:
        to_keep_set.update(nd.inputs)

    for out, idx in to_replace_map.items():
        if (
            out in to_keep_set
            and out.owner not in existent_nodes_set
            and
            # If types are different, conversion Op will be inserted,
            # and it may trigger an infinite loop.
            out.type.is_super(replace_with_in[idx].type)
        ):
            clean_to_replace.append(out)
            clean_replace_with_in.append(replace_with_in[idx])
            clean_replace_with_out.append(replace_with_out[idx])

    if len(clean_to_replace) > 0:
        # We can finally put an end to all this madness
        givens = {}
        nw_outer = []
        nw_inner = []
        for to_repl, repl_in, repl_out in zip(
            clean_to_replace, clean_replace_with_in, clean_replace_with_out, strict=True
        ):
            if isinstance(repl_out, Constant):
                repl_in = repl_out
            else:
                nw_inner.append(repl_in)
                nw_outer.append(repl_out)

            givens[to_repl] = repl_in

        op_outs = clone_replace(node_outputs, replace=givens)
        op_ins = nw_inner + node_inputs

        # Reconstruct node
        nw_info = dataclasses.replace(op.info, n_seqs=op.info.n_seqs + len(nw_inner))
        nwScan = Scan(
            op_ins,
            op_outs,
            nw_info,
            mode=op.mode,
            profile=op.profile,
            truncate_gradient=op.truncate_gradient,
            # TODO: This seems questionable
            name=op.name,
            allow_gc=op.allow_gc,
        )
        nw_node = nwScan(
            *(node.inputs[:1] + nw_outer + node.inputs[1:]),
            return_list=True,
        )[0].owner

        replacements = dict(zip(node.outputs, nw_node.outputs, strict=True))
        replacements["remove"] = [node]
        return replacements

    elif not to_keep_set and not op.info.as_while and not op.outer_mitmot(node.inputs):
        # Nothing in the inner graph should be kept.
        n_steps = node.inputs[0]

        replace_with = {}
        for out, idx in to_replace_map.items():
            if out in local_fgraph_outs_set:
                x = node.outputs[local_fgraph_outs_map[out]]
                _y = replace_with_out[idx]
                ls = node_outputs
                if out in op.inner_mitsot_outs(ls):
                    odx = op.inner_mitsot_outs(ls).index(out)
                    inp = op.outer_mitsot(node.inputs)[odx]
                    st = abs(np.min(op.info.mit_sot_in_slices))
                    y = set_subtensor(inp[st:], _y)
                elif out in op.inner_sitsot_outs(ls):
                    odx = op.inner_sitsot_outs(ls).index(out)
                    inp = op.outer_sitsot(node.inputs)[odx]
                    y = set_subtensor(inp[1:], _y)
                elif out in op.inner_nitsot_outs(ls):
                    # The pushed-out Elemwise has length == n_steps, but the
                    # nit_sot's outer buffer size (`outer_nitsot` input) may
                    # be larger. When it is, folding directly would silently
                    # drop the trailing zero-initialized slots that any
                    # downstream consumer reading the full buffer expects --
                    # e.g., `Scan.pullback` emits grad scans where the
                    # nit_sot size is the forward's step count while
                    # `n_steps == grad_steps == min(forward_steps, truncate)`
                    # and its concat padding depends on those trailing zeros.
                    # Fold into the direct Elemwise only when the two sizes
                    # are the same Variable; otherwise pad with zeros via
                    # set_subtensor.
                    odx = op.inner_nitsot_outs(ls).index(out)
                    nit_sot_size = op.outer_nitsot(node.inputs)[odx]
                    if nit_sot_size is n_steps:
                        y = _y
                    else:
                        extra_dims = [_y.shape[i] for i in range(1, _y.ndim)]
                        zero_buf = pt.zeros((nit_sot_size, *extra_dims), dtype=_y.dtype)
                        y = set_subtensor(zero_buf[:n_steps], _y)
                else:
                    y = _y[-1]
                replace_with[x] = y

        # We need to add one extra dimension to the outputs
        if replace_with and len(replace_with) == len(node.outputs):
            replacements = dict(replace_with.items())
            replacements["remove"] = [node]
            return replacements
    else:
        return False


def inner_sitsot_only_last_step_used(
    fgraph: FunctionGraph, var: Variable, scan_args: ScanArgs
) -> bool:
    """
    Given a inner sit-sot output of `Scan`, return ``True`` iff the outer
    sit-sot output has only one client and that client is a `Subtensor`
    instance that takes only the last step (last element along the first axis).
    """
    idx = scan_args.inner_out_sit_sot.index(var)
    outer_var = scan_args.outer_out_sit_sot[idx]

    if len(fgraph.clients[outer_var]) == 1:
        client = fgraph.clients[outer_var][0][0]
        if isinstance(client, Apply) and isinstance(client.op, Subtensor):
            lst = get_idx_list(client.inputs, client.op.idx_list)
            return (
                len(lst) == 1
                and get_scalar_constant_value(lst[0], raise_not_constant=False) == -1
            )

    return False


def get_outer_ndim(var: Variable, scan_args: ScanArgs) -> int:
    """Determine the number of dimension a variable would have if it was pushed out of a `Scan`."""
    assert isinstance(var.type, HasShape)

    if var in scan_args.inner_in_non_seqs or isinstance(var, Constant):
        outer_ndim = var.type.ndim
    else:
        outer_ndim = var.type.ndim + 1

    return outer_ndim


def push_out_inner_vars(
    fgraph: FunctionGraph,
    inner_vars: list[Variable],
    old_scan_node: Apply,
    old_scan_args: ScanArgs,
) -> tuple[list[Variable], ScanArgs, dict[Variable, Variable]]:
    tmp_outer_vars: list[Variable | None] = []
    new_scan_args = old_scan_args
    replacements: dict[Variable, Variable] = {}

    # For the inner_vars that already exist in the outer graph,
    # simply obtain a reference to them
    for idx in range(len(inner_vars)):
        var = inner_vars[idx]

        new_outer_var: Variable | None = None

        if var in old_scan_args.inner_in_seqs:
            idx_seq = old_scan_args.inner_in_seqs.index(var)
            new_outer_var = old_scan_args.outer_in_seqs[idx_seq]

        elif var in old_scan_args.inner_in_non_seqs:
            idx_non_seq = old_scan_args.inner_in_non_seqs.index(var)
            new_outer_var = old_scan_args.outer_in_non_seqs[idx_non_seq]

        elif isinstance(var, Constant):
            new_outer_var = var

        elif var in old_scan_args.inner_out_nit_sot:
            idx_nitsot = old_scan_args.inner_out_nit_sot.index(var)
            new_outer_var = old_scan_args.outer_out_nit_sot[idx_nitsot]

        tmp_outer_vars.append(new_outer_var)

    # For the inner_vars that don't already exist in the outer graph, add
    # them as new nitsot outputs to the scan node.
    idx_add_as_nitsots = [i for i, v in enumerate(tmp_outer_vars) if v is None]
    add_as_nitsots = [inner_vars[idx] for idx in idx_add_as_nitsots]

    new_outs: list[Variable] = []

    if len(add_as_nitsots) > 0:
        new_scan_node, replacements = add_nitsot_outputs(
            fgraph, old_scan_node, old_scan_args, add_as_nitsots
        )

        assert isinstance(new_scan_node.op, Scan)

        new_scan_args = ScanArgs(
            new_scan_node.inputs,
            new_scan_node.outputs,
            new_scan_node.op.inner_inputs,
            new_scan_node.op.inner_outputs,
            new_scan_node.op.info,
        )

        new_outs = new_scan_args.outer_out_nit_sot[-len(add_as_nitsots) :]

    outer_vars: list[Variable] = []

    for i, v in enumerate(tmp_outer_vars):
        if i in idx_add_as_nitsots:
            outer_vars.append(new_outs.pop(0))
        else:
            assert v is not None
            outer_vars.append(v)

    return outer_vars, new_scan_args, replacements


def add_nitsot_outputs(
    fgraph: FunctionGraph,
    old_scan_node: Apply,
    old_scan_args: ScanArgs,
    new_outputs_inner,
) -> tuple[Apply, dict[Variable, Variable]]:
    assert isinstance(old_scan_node.op, Scan)

    nb_new_outs = len(new_outputs_inner)

    # Create the initial values for the new nitsot outputs
    # (the initial value is the nb of steps to store. For a nistot,
    # it should be the number of steps performed by scan)
    new_nitsots_initial_value = [old_scan_node.inputs[0] for i in range(nb_new_outs)]

    # Create the `ScanArgs` corresponding to the new `Scan` `Op` to create
    new_scan_args = copy.copy(old_scan_args)
    new_scan_args.inner_out_nit_sot.extend(new_outputs_inner)
    new_scan_args.outer_in_nit_sot.extend(new_nitsots_initial_value)

    assert isinstance(old_scan_node.op, Scan)

    # Create the `Scan` `Op` from the `ScanArgs`
    new_scan_op = Scan(
        new_scan_args.inner_inputs,
        new_scan_args.inner_outputs,
        new_scan_args.info,
        mode=old_scan_node.op.mode,
        profile=old_scan_node.op.profile,
        truncate_gradient=old_scan_node.op.truncate_gradient,
        # TODO: This seems questionable
        name=old_scan_node.op.name,
        allow_gc=old_scan_node.op.allow_gc,
    )

    # Create the Apply node for the scan op
    new_scan_outs = new_scan_op(*new_scan_args.outer_inputs, return_list=True)
    assert isinstance(new_scan_outs, list)
    new_scan_node = new_scan_outs[0].owner
    assert new_scan_node is not None

    # Modify the outer graph to make sure the outputs of the new scan are
    # used instead of the outputs of the old scan
    new_node_new_outputs_idx = len(old_scan_args.outer_outputs) - len(
        old_scan_args.outer_out_shared
    )

    new_node_old_outputs = (
        new_scan_node.outputs[:new_node_new_outputs_idx]
        + new_scan_node.outputs[new_node_new_outputs_idx + nb_new_outs :]
    )

    # TODO FIXME:
    # replacements = dict(zip(old_scan_node.outputs, new_node_old_outputs))
    # replacements["remove"] = [old_scan_node]
    # return new_scan_node, replacements
    fgraph.replace_all_validate_remove(
        list(zip(old_scan_node.outputs, new_node_old_outputs, strict=True)),
        remove=[old_scan_node],
        reason="scan_pushout_add",
    )
    return new_scan_node, {}


@node_rewriter([Scan])
def scan_push_out_add(fgraph, node):
    r"""Push `Add` operations performed at the end of the inner graph to the outside.

    Like `scan_push_out_seq`, this optimization aims to replace many operations
    on small tensors by few operations on large tensors. It can also lead to
    increased memory usage.

    FIXME: This rewrite doesn't cover user defined graphs,
      since it doesn't account for the intermediate slice
      returned by the scan constructor for sit-sot (i.e., something like output[1:]).
      It only looks for `outputs[-1]` but the user will only ever write `outputs[1:][-1]`
      The relevant helper function is `inner_sitsot_only_last_step_used` which is only used by this rewrite
      Note this rewrite is registered before subtensor_merge, but even if it were after subtensor_merge is a mess
      and doesn't simplify to x[1:][-1] to x[-1] unless x length is statically known
    """
    # Don't perform the optimization on `as_while` `Scan`s. Because these
    # `Scan`s don't run for a predetermined number of steps, handling them is
    # more complicated and this optimization doesn't support it at the moment.
    op = node.op
    if op.info.as_while:
        return False

    # apply_ancestors(args.inner_outputs)

    add_of_dot_nodes = [
        n
        for n in op.fgraph.apply_nodes
        if (
            # We have an Add
            isinstance(n.op, Elemwise)
            and isinstance(n.op.scalar_op, ps.Add)
            and any(
                (
                    # With a Dot input that's only used in the Add
                    n_inp.owner is not None
                    and isinstance(n_inp.owner.op, Dot)
                    and len(op.fgraph.clients[n_inp]) == 1
                )
                for n_inp in n.inputs
            )
        )
    ]

    if not add_of_dot_nodes:
        return False

    # Use `ScanArgs` to parse the inputs and outputs of scan for ease of access
    args = ScanArgs(
        node.inputs,
        node.outputs,
        op.inner_inputs,
        op.inner_outputs,
        op.info,
        clone=False,
    )

    for nd in add_of_dot_nodes:
        if (
            nd.out in args.inner_out_sit_sot
            # FIXME: This function doesn't handle `sitsot_out[1:][-1]` pattern
            and inner_sitsot_only_last_step_used(fgraph, nd.out, args)
        ):
            # Ensure that one of the input to the add is the output of
            # the add from a previous iteration of the inner function
            sitsot_idx = args.inner_out_sit_sot.index(nd.out)
            if args.inner_in_sit_sot[sitsot_idx] in nd.inputs:
                sitsot_in_idx = nd.inputs.index(args.inner_in_sit_sot[sitsot_idx])

                # 0 if sitsot_in_idx==1, 1 if sitsot_in_idx==0
                dot_in_idx = 1 - sitsot_in_idx
                dot_input = nd.inputs[dot_in_idx]
                assert dot_input.owner is not None and isinstance(
                    dot_input.owner.op, Dot
                )

                if (
                    get_outer_ndim(dot_input.owner.inputs[0], args) == 3
                    and get_outer_ndim(dot_input.owner.inputs[1], args) == 3
                ):
                    # The optimization can be be applied in this case.

                    # Move out of scan the two inputs to the Dot and
                    # perform a dot outside of scan on these two inputs
                    inner_dot_inputs = nd.inputs[dot_in_idx].owner.inputs
                    (
                        outer_dot_inputs,
                        new_scan_args,
                        replacements,
                    ) = push_out_inner_vars(fgraph, inner_dot_inputs, node, args)

                    # Collapse some of the dimensions of the tensors
                    # so that they become matrices. This is because a
                    # dot is usually faster on two large matrices than
                    # a bunch of small ones
                    outer_dot_inputs[0] = pt.flatten(
                        outer_dot_inputs[0].dimshuffle(1, 0, 2), ndim=2
                    )

                    shape_input1 = shape(outer_dot_inputs[1])
                    outer_dot_inputs[1] = outer_dot_inputs[1].reshape(
                        (shape_input1[0] * shape_input1[1], shape_input1[2])
                    )

                    # Perform the dot on the newly obtained matrices and
                    # add the initial value
                    outer_dot_output = dot(*outer_dot_inputs)
                    init_value = new_scan_args.outer_in_sit_sot[sitsot_idx][0]
                    replacement = outer_dot_output + init_value

                    # Alter the outer graph to use the output of the
                    # external Dot instead of the output of scan
                    # Modify the outer graph to add the outer Dot
                    outer_sitsot = new_scan_args.outer_out_sit_sot[sitsot_idx]
                    # TODO: If we fix the FIXME above, we have to make sure we replace the last subtensor, not the immediate one
                    subtensor_node = fgraph.clients[outer_sitsot][0][0]
                    outer_sitsot_last_step = subtensor_node.outputs[0]

                    replacements[outer_sitsot_last_step] = replacement
                    return replacements

    return False


@node_rewriter([Scan])
def scan_push_out_dot1(fgraph, node):
    r"""
    This is another optimization that attempts to detect certain patterns of
    computation in a `Scan` `Op`'s inner function and move this computation to the
    outer graph.
    """
    if not isinstance(node.op, Scan):
        return False

    # Replace pattern of the form
    # x[t] = x[t-1] + dot(seq[t], value)
    # with Sequence.reshape((-1, seq.shape[2])) \dot Value
    # When seq[t] is a vector/matrix  and `value` is a matrix
    # Note that this works when only you need X[-1] in the end
    # and assumes dimshuffle are applied to vectors before calling dot
    op: Scan = node.op
    sitsot_ins = op.inner_sitsot(op.inner_inputs)
    sitsot_outs = op.inner_sitsot_outs(op.inner_outputs)
    outer_sitsot = op.outer_sitsot_outs(node.outputs)
    seqs = op.inner_seqs(op.inner_inputs)
    for inp, out, outer_out in zip(sitsot_ins, sitsot_outs, outer_sitsot, strict=True):
        if (
            out.owner
            and isinstance(out.owner.op, Elemwise)
            and isinstance(out.owner.op.scalar_op, ps.Add)
            and inp in out.owner.inputs
            and len(fgraph.clients[outer_out]) == 1
            and not isinstance(fgraph.clients[outer_out][0][0], Output)
            and isinstance(fgraph.clients[outer_out][0][0].op, Subtensor)
            and fgraph.clients[outer_out][0][0].op.idx_list == (-1,)
        ):
            x = out.owner.inputs[0]
            if x == inp:
                x = out.owner.inputs[1]
            # We need to check if x is the result of an outer product
            if (
                x.owner
                and isinstance(x.owner.op, Dot)
                and x.owner.inputs[0].ndim == 2
                and x.owner.inputs[1].ndim == 2
            ):
                # We need to check if any of the inputs are a sequence
                inp1 = x.owner.inputs[0]
                inp2 = x.owner.inputs[1]

                if inp1 in seqs or inp2 in seqs:
                    new_scan_out = inp1

                    if inp1 in seqs:
                        new_scan_out = inp2
                    idx = sitsot_outs.index(out)
                    # We've found our pattern and need to construct a new
                    # scan node to replace this one. For this we need to
                    # replace the sit_sot output with a nit_sot output

                    # First let us split all arguments according to their
                    # corresponding categories

                    inner_seqs = op.inner_seqs(op.inner_inputs)
                    outer_seqs = op.outer_seqs(node.inputs)
                    inner_mitmot = op.inner_mitmot(op.inner_inputs)
                    outer_mitmot = op.outer_mitmot(node.inputs)
                    inner_mitmot_outs = op.inner_mitmot_outs(op.inner_outputs)
                    inner_mitsot = op.inner_mitsot(op.inner_inputs)
                    outer_mitsot = op.outer_mitsot(node.inputs)
                    inner_mitsot_outs = op.inner_mitsot_outs(op.inner_outputs)
                    inner_sitsot = op.inner_sitsot(op.inner_inputs)
                    outer_sitsot = op.outer_sitsot(node.inputs)
                    inner_sitsot_outs = op.inner_sitsot_outs(op.inner_outputs)
                    outer_nitsot = op.outer_nitsot(node.inputs)
                    inner_nitsot_outs = op.inner_nitsot_outs(op.inner_outputs)
                    inner_untraced_sitsot = op.inner_untraced_sitsot(op.inner_inputs)
                    outer_untraced_sitsot_outs = op.outer_untraced_sitsot_outs(
                        node.inputs
                    )
                    inner_untraced_sitsot_outs = op.inner_untraced_sitsot_outs(
                        op.inner_outputs
                    )
                    inner_non_seqs = op.inner_non_seqs(op.inner_inputs)
                    outer_non_seqs = op.outer_non_seqs(node.inputs)

                    new_info = dataclasses.replace(
                        op.info,
                        sit_sot_in_slices=op.info.sit_sot_in_slices[:idx]
                        + op.info.sit_sot_in_slices[idx + 1 :],
                        n_nit_sot=op.info.n_nit_sot + 1,
                    )
                    inner_sitsot = inner_sitsot[:idx] + inner_sitsot[idx + 1 :]
                    outer_sitsot = outer_sitsot[:idx] + outer_sitsot[idx + 1 :]
                    inner_sitsot_outs = (
                        inner_sitsot_outs[:idx] + inner_sitsot_outs[idx + 1 :]
                    )
                    # add n_steps as the length
                    inner_nitsot_outs.append(new_scan_out)

                    _new_inner_inps = (
                        inner_seqs
                        + inner_mitmot
                        + inner_mitsot
                        + inner_sitsot
                        + inner_untraced_sitsot
                        + inner_non_seqs
                    )
                    _new_inner_outs = (
                        inner_mitmot_outs
                        + inner_mitsot_outs
                        + inner_sitsot_outs
                        + inner_nitsot_outs
                        + inner_untraced_sitsot_outs
                    )
                    new_inner_inps, new_inner_outs = reconstruct_graph(
                        _new_inner_inps, _new_inner_outs
                    )
                    new_op = Scan(
                        new_inner_inps,
                        new_inner_outs,
                        new_info,
                        mode=op.mode,
                        profile=op.profile,
                        truncate_gradient=op.truncate_gradient,
                        # TODO: This seems questionable
                        name=op.name,
                        allow_gc=op.allow_gc,
                    )
                    _scan_inputs = [
                        node.inputs[0],
                        *outer_seqs,
                        *outer_mitmot,
                        *outer_mitsot,
                        *outer_sitsot,
                        *outer_untraced_sitsot_outs,
                        *outer_nitsot,
                        node.inputs[0],
                        *outer_non_seqs,
                    ]

                    new_outs = new_op(*_scan_inputs)
                    if not isinstance(new_outs, list | tuple):
                        new_outs = [new_outs]

                    # We need now to pair correctly the new outputs
                    # with the old ones

                    outer_nitsot_outs = new_op.outer_nitsot_outs(new_outs)

                    _val = outer_nitsot_outs[-1]
                    outer_nitsot_outs = outer_nitsot_outs[:-1]
                    if inp1 in seqs:
                        _out_seq = op.outer_seqs(node.inputs)[seqs.index(inp1)]
                        # We need to clip the seq to the number of steps
                        _out_seq = _out_seq[: node.inputs[0]]
                        sh0 = _out_seq.shape[0]
                        sh1 = _out_seq.shape[1]
                        sh2 = _out_seq.shape[2]
                        out_seq = _out_seq.dimshuffle(1, 0, 2)
                        out_seq = out_seq.reshape((sh1, sh0 * sh2))
                        sh0 = _val.shape[0]
                        sh1 = _val.shape[1]
                        sh2 = _val.shape[2]

                        val = _val.reshape((sh0 * sh1, sh2))
                        new_out = dot(out_seq, val)
                    else:
                        _out_seq = op.outer_seqs(node.inputs)[seqs.index(inp2)]
                        out_seq = _out_seq.reshape(
                            (
                                _out_seq.shape[0] * _out_seq.shape[1],
                                _out_seq.shape[2],
                            )
                        )

                        val = _val.dimshuffle(1, 0, 2).reshape(
                            (_val.shape[1], _val.shape[0] * _val.shape[2])
                        )
                        new_out = dot(val, out_seq)

                    pos = node.outputs.index(outer_out)
                    old_new = list(zip(node.outputs[:pos], new_outs[:pos], strict=True))
                    old = fgraph.clients[node.outputs[pos]][0][0].outputs[0]
                    old_new.append((old, new_out))
                    old_new += list(
                        zip(node.outputs[pos + 1 :], new_outs[pos:], strict=True)
                    )
                    replacements = dict(old_new)
                    replacements["remove"] = [node]
                    return replacements

    return False
