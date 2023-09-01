import sys
from collections import defaultdict, deque
from functools import lru_cache
from typing import DefaultDict, Generator, List, Set, Tuple, TypeVar
from warnings import warn

import pytensor
import pytensor.scalar.basic as aes
from pytensor import clone_replace, compile
from pytensor.compile.mode import get_target_language
from pytensor.configdefaults import config
from pytensor.graph import FunctionGraph
from pytensor.graph.basic import Apply, Constant, Variable, ancestors, io_toposort
from pytensor.graph.features import ReplaceValidate
from pytensor.graph.fg import ApplyOrOutput
from pytensor.graph.rewriting.basic import (
    EquilibriumGraphRewriter,
    GraphRewriter,
    copy_stack_trace,
    in2out,
    node_rewriter,
)
from pytensor.graph.rewriting.db import SequenceDB
from pytensor.graph.utils import InconsistencyError, MethodNotDefined
from pytensor.scalar.loop import ScalarLoop
from pytensor.scalar.math import Grad2F1Loop, _grad_2f1_loop
from pytensor.tensor.basic import (
    MakeVector,
    alloc,
    cast,
    get_underlying_scalar_constant_value,
)
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.math import exp
from pytensor.tensor.rewriting.basic import (
    alloc_like,
    register_canonicalize,
    register_specialize,
)
from pytensor.tensor.shape import shape_padleft
from pytensor.tensor.variable import TensorConstant, get_unique_constant_value


class InplaceElemwiseOptimizer(GraphRewriter):
    r"""
    This is parameterized so that it works for `Elemwise` `Op`\s.
    """

    def __init__(self, OP):
        self.op = OP

    def add_requirements(self, fgraph):
        from pytensor.graph.destroyhandler import DestroyHandler

        fgraph.attach_feature(DestroyHandler())

    @classmethod
    def print_profile(cls, stream, prof, level=0):
        blanc = "    " * level
        print(blanc, cls.__name__, prof["opt"].op, file=stream)
        for k in [
            "node_before",
            "nb_call_replace",
            "nb_call_validate",
            "nb_inconsistent",
        ]:
            print(blanc, k, prof[k], file=stream)
        ndim = prof["ndim"]
        if ndim:
            print(blanc, "ndim", "nb", file=stream)
            for n in sorted(ndim.keys()):
                print(blanc, n, ndim[n], file=stream)

    def candidate_input_idxs(self, node):
        # TODO: Implement specialized InplaceCompositeOptimizer with logic
        #  needed to correctly assign inplace for multi-output Composites
        #  and ScalarLoops
        if isinstance(node.op.scalar_op, ScalarLoop):
            return []
        if isinstance(node.op.scalar_op, aes.Composite) and (len(node.outputs) > 1):
            return []
        else:
            return range(len(node.outputs))

    def apply(self, fgraph):
        r"""

        Attempts to replace all `Elemwise`\s by versions of them that operate
        inplace. It operates greedily: for each `Elemwise` that is encountered,
        for each output, it tries each input to see if it can operate inplace
        on that input. If so, it makes the change and goes to the next output
        or `Elemwise`.

        Examples
        --------

            x + y + z -> x += y += z
            (x + y) * (x * y) -> (x += y) *= (x * y) or (x + y) *= (x *= y)

        """
        # We should not validate too often as this takes too much time to
        # execute!
        # It is the _dfs_toposort() fct in pytensor/graph/destroyhandler.py
        # that takes so much time.
        # Should we try to use another lib that does toposort?
        #   igraph: http://igraph.sourceforge.net/
        #   networkx: https://networkx.lanl.gov/
        # Should we try to use cython?
        #   Compiling only that fct is not enough, should we try to add the
        #   deque class too?
        #   And init the deque and other list to an upper bound number of
        #   elements?
        # Maybe PyTensor should do online toposort as in
        #   http://code.google.com/p/acyclic
        #
        # The next longest rewriter is the canonizer phase.
        # Then I think it is the [io_?]toposort (need to validate) so check if
        # the solution is also applicable there.

        # We execute `validate` after this number of change.
        prof = {
            "opt": self,
            "node_before": len(fgraph.apply_nodes),
            "nb_call_replace": 0,
            "nb_call_validate": 0,
            "nb_inconsistent": 0,
            "ndim": defaultdict(lambda: 0),
        }

        check_each_change = config.tensor__insert_inplace_optimizer_validate_nb
        if check_each_change == -1:
            if len(fgraph.apply_nodes) > 500:
                check_each_change = 10
            else:
                check_each_change = 1

        nb_change_no_validate = 0
        chk = fgraph.checkpoint()

        if fgraph.update_mapping:
            update_outs = [fgraph.outputs[i] for i in fgraph.update_mapping]
        else:
            update_outs = []

        protected_inputs = [
            f.protected
            for f in fgraph._features
            if isinstance(f, pytensor.compile.function.types.Supervisor)
        ]
        protected_inputs = sum(protected_inputs, [])  # flatten the list
        protected_inputs.extend(fgraph.outputs)
        for node in list(io_toposort(fgraph.inputs, fgraph.outputs)):
            op = node.op
            if not isinstance(op, self.op):
                continue
            # If big graph and the outputs are scalar, do not make it
            # inplace.
            if (
                check_each_change != 1
                and
                # If multiple outputs, they must all have the same size,
                # so only check the first.
                getattr(node.outputs[0].type, "ndim", -1) == 0
            ):
                continue

            if op.inplace_pattern:
                # Maybe this isn't needed anymore, but I don't want to
                # rish regression now. This case only happen if the
                # original node add already some inplace patter and we
                # still try to add more pattern.

                baseline = op.inplace_pattern
                candidate_outputs = [
                    i for i in self.candidate_input_idxs(node) if i not in baseline
                ]
                # node inputs that are Constant, already destroyed,
                # or fgraph protected inputs and fgraph outputs can't be used as
                # inplace target.
                # Remove here as faster.
                candidate_inputs = [
                    i
                    for i in range(len(node.inputs))
                    if i not in baseline.values()
                    and not isinstance(node.inputs[i], Constant)
                    and
                    # the next line should not be costly most of the time.
                    not fgraph.has_destroyers([node.inputs[i]])
                    and node.inputs[i] not in protected_inputs
                ]
            else:
                baseline = []
                candidate_outputs = self.candidate_input_idxs(node)
                # node inputs that are Constant, already destroyed,
                # fgraph protected inputs and fgraph outputs can't be used as inplace
                # target.
                # Remove here as faster.
                candidate_inputs = [
                    i
                    for i in range(len(node.inputs))
                    if not isinstance(node.inputs[i], Constant)
                    and not fgraph.has_destroyers([node.inputs[i]])
                    and node.inputs[i] not in protected_inputs
                ]

            verbose = False

            raised_warning = not verbose

            for candidate_output in candidate_outputs:
                # If the output of the node can be established as an update
                # output of the fgraph, visit the candidate_inputs in an order
                # that will improve the chances of making the node operate
                # inplace on the input it's meant to update
                candidate_out_var = node.outputs[candidate_output]
                sorted_candidate_inputs = candidate_inputs

                if candidate_out_var in update_outs:
                    # The candidate output is an update. Sort the
                    # variables in candidate_inputs in the following order:
                    # - Vars corresponding to the actual updated input
                    #   (best case scenario is for the node that procudes
                    #   an update to operate inplace on the variable to
                    #   update)
                    # - Vars computed inplace on the updates input (second
                    #   best scenario if for the node to work inplace on
                    #   a variable obtained by a chain of inplace on the
                    #   variable to update. In some cases, this will be
                    #   equivalent to operating inplace on the variable to
                    #   update)
                    # - Remaining variables
                    updated_inputs = []
                    for i, f_out in enumerate(fgraph.outputs):
                        if f_out is candidate_out_var and i in fgraph.update_mapping:
                            updated_inp_idx = fgraph.update_mapping[i]
                            updated_inputs.append(fgraph.inputs[updated_inp_idx])

                    updated_vars = []
                    vars_from_inplace = []
                    other_vars = []
                    for inp_idx in candidate_inputs:
                        inp = node.inputs[inp_idx]
                        if inp in updated_inputs:
                            # the candidate input is the actual updated input
                            updated_vars.append(inp_idx)
                        elif (
                            hasattr(fgraph, "destroy_handler")
                            and inp.owner
                            and any(
                                fgraph.destroy_handler.root_destroyer.get(up_inp, None)
                                is inp.owner
                                for up_inp in updated_inputs
                            )
                        ):
                            # the candidate input is a variable computed
                            # inplace on the updated input via a sequence of
                            # one or more inplace operations
                            vars_from_inplace.append(inp_idx)
                        else:
                            other_vars.append(inp_idx)

                    sorted_candidate_inputs = (
                        updated_vars + vars_from_inplace + other_vars
                    )

                for candidate_input in sorted_candidate_inputs:
                    # remove inputs that don't have the same dtype as the output
                    if (
                        node.inputs[candidate_input].type
                        != node.outputs[candidate_output].type
                    ):
                        continue

                    inplace_pattern = dict(baseline)
                    inplace_pattern[candidate_output] = candidate_input
                    try:
                        if hasattr(op.scalar_op, "make_new_inplace"):
                            new_scal = op.scalar_op.make_new_inplace(
                                aes.transfer_type(
                                    *[
                                        inplace_pattern.get(i, o.dtype)
                                        for i, o in enumerate(node.outputs)
                                    ]
                                )
                            )
                        else:
                            new_scal = op.scalar_op.__class__(
                                aes.transfer_type(
                                    *[
                                        inplace_pattern.get(i, None)
                                        for i in range(len(node.outputs))
                                    ]
                                )
                            )
                        new_outputs = self.op(new_scal, inplace_pattern)(
                            *node.inputs, return_list=True
                        )
                        new_node = new_outputs[0].owner

                        for r, new_r in zip(node.outputs, new_outputs):
                            prof["nb_call_replace"] += 1
                            fgraph.replace(
                                r, new_r, reason="inplace_elemwise_optimizer"
                            )
                        nb_change_no_validate += 1
                        prof["ndim"][candidate_out_var.ndim] += 1
                        if nb_change_no_validate >= check_each_change:
                            prof["nb_call_validate"] += 1
                            fgraph.validate()
                            chk = fgraph.checkpoint()
                            nb_change_no_validate = 0
                    except (ValueError, InconsistencyError) as e:
                        prof["nb_inconsistent"] += 1
                        if check_each_change != 1 and not raised_warning:
                            print(
                                (
                                    "Some inplace rewriting was not "
                                    "performed due to an unexpected error:"
                                ),
                                file=sys.stderr,
                            )
                            print(e, file=sys.stderr)
                            raised_warning = True
                        fgraph.revert(chk)
                        continue
                    candidate_inputs.remove(candidate_input)
                    node = new_node
                    baseline = inplace_pattern
                    break

        if nb_change_no_validate > 0:
            try:
                fgraph.validate()
            except Exception:
                if not raised_warning:
                    print(
                        (
                            "Some inplace rewriting was not "
                            "performed due to an unexpected error"
                        ),
                        file=sys.stderr,
                    )
                fgraph.revert(chk)
        return prof

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(
            f"{' ' * level}{self.__class__.__name__} ({self.op})",
            file=stream,
        )
        return inplace_elemwise_optimizer


inplace_elemwise_optimizer = InplaceElemwiseOptimizer(Elemwise)
compile.optdb.register(
    "inplace_elemwise_opt",
    inplace_elemwise_optimizer,
    "inplace_opt",  # for historic reason
    "inplace_elemwise_optimizer",
    "fast_run",
    "inplace",
    position=75,
)


def apply_local_dimshuffle_lift(fgraph, var):
    """
    lift recursively
    """
    if not var.owner:
        return var
    new = local_dimshuffle_lift.transform(fgraph, var.owner)
    if new:
        return new[0]
    return var


def is_dimshuffle_useless(new_order, input):
    """
    Checks for two types of useless dimshuffles:
      1 - dimshuffle all dimensions in order.
      2 - dimshuffle a broadcastable dimension.
    """
    is_useless = True
    if len(new_order) == input.type.ndim:
        all_broadcastable_dims = [
            i
            for (i, is_broadcastable) in enumerate(input.type.broadcastable)
            if is_broadcastable
        ] + ["x"]
        for i in range(input.type.ndim):
            if new_order[i] == i or (
                i in all_broadcastable_dims and new_order[i] in all_broadcastable_dims
            ):
                is_useless = True
            else:
                is_useless = False
                break
    else:
        is_useless = False
    return is_useless


@register_canonicalize
@register_specialize
@node_rewriter([DimShuffle])
def local_dimshuffle_lift(fgraph, node):
    """
    "Lifts" DimShuffle through Elemwise operations and merges
    consecutive DimShuffles. Basically, applies the following
    transformations on the whole graph:

    DimShuffle(Elemwise(x, y)) => Elemwise(DimShuffle(x), DimShuffle(y))
    DimShuffle(DimShuffle(x)) => DimShuffle(x)
    DimShuffle{0,1,...}(x) => x (when the dimshuffle do nothing)

    After this transform, clusters of Elemwise operations are
    void of DimShuffle operations.

    """
    op = node.op
    if not isinstance(op, DimShuffle):
        return False

    inp = node.inputs[0]
    inode = inp.owner
    new_order = op.new_order
    if (
        inode
        and isinstance(inode.op, Elemwise)
        and len(inode.outputs) == 1
        and (len(fgraph.clients[inp]) == 1)
    ):
        # Don't use make_node to have tag.test_value set.
        new_inputs = []
        for inp in inode.inputs:
            new_inp = op.__class__(inp.type.broadcastable, op.new_order)(inp)
            new_inputs.append(apply_local_dimshuffle_lift(fgraph, new_inp))
        copy_stack_trace(node.outputs[0], new_inputs)
        ret = inode.op(*new_inputs, return_list=True)
        return ret
    if inode and isinstance(inode.op, DimShuffle):
        new_order = [x == "x" and "x" or inode.op.new_order[x] for x in new_order]
        inp = inode.inputs[0]

    if is_dimshuffle_useless(new_order, inp):
        return [inp]
    elif inode and isinstance(inode.op, DimShuffle):
        ret = op.__class__(inp.type.broadcastable, new_order)(inp)
        ret = apply_local_dimshuffle_lift(fgraph, ret)
        copy_stack_trace(node.outputs[0], ret)
        return [ret]


@register_canonicalize
@register_specialize
@node_rewriter([DimShuffle])
def local_useless_dimshuffle_makevector(fgraph, node):
    r"""Remove `DimShuffle`\s that drop one dimensional broadcastable `MakeVector`s.

    This rewrite is needed in order to clean up after
    `local_subtensor_remove_broadcastable_index`, which produces a
    not-so-intuitive canonical form for `x[0]` when `x.shape == (1,)`
    (i.e. one broadcastable dimension): i.e. `x.dimshuffle(())`.
    """

    # The `DimShuffle` should be removing the single broadcastable dimension
    if node.op.new_order != ():
        return

    makevector_out = node.inputs[0]

    if (
        not makevector_out.owner
        or not isinstance(makevector_out.owner.op, MakeVector)
        or not makevector_out.broadcastable == (True,)
    ):
        return

    assert len(makevector_out.owner.inputs) == 1

    return [makevector_out.owner.inputs[0]]


@register_canonicalize
@node_rewriter([Elemwise])
def local_upcast_elemwise_constant_inputs(fgraph, node):
    """This explicitly upcasts constant inputs to elemwise Ops, when
    those Ops do implicit upcasting anyway.

    Rationale: it helps merge things like (1-x) and (1.0 - x).

    """
    if len(node.outputs) > 1:
        return
    try:
        shape_i = fgraph.shape_feature.shape_i
    except AttributeError:
        shape_i = None
    if isinstance(node.op, Elemwise):
        scalar_op = node.op.scalar_op
        # print "aa", scalar_op.output_types_preference
        if getattr(scalar_op, "output_types_preference", None) in (
            aes.upgrade_to_float,
            aes.upcast_out,
        ):
            # this is the kind of op that we can screw with the input
            # dtypes by upcasting explicitly
            output_dtype = node.outputs[0].type.dtype
            new_inputs = []
            for i in node.inputs:
                if i.type.dtype == output_dtype:
                    new_inputs.append(i)
                else:
                    try:
                        # works only for scalars
                        cval_i = get_underlying_scalar_constant_value(
                            i, only_process_constants=True
                        )
                        if all(i.broadcastable):
                            new_inputs.append(
                                shape_padleft(cast(cval_i, output_dtype), i.ndim)
                            )
                        else:
                            if shape_i is None:
                                return
                            new_inputs.append(
                                alloc(
                                    cast(cval_i, output_dtype),
                                    *[shape_i(d)(i) for d in range(i.ndim)],
                                )
                            )
                            # print >> sys.stderr, "AAA",
                            # *[Shape_i(d)(i) for d in range(i.ndim)]
                    except NotScalarConstantError:
                        # for the case of a non-scalar
                        if isinstance(i, TensorConstant):
                            new_inputs.append(cast(i, output_dtype))
                        else:
                            new_inputs.append(i)

            if new_inputs != node.inputs:
                rval = [node.op(*new_inputs)]
                if not node.outputs[0].type.is_super(rval[0].type):
                    # This can happen for example when floatX=float32
                    # and we do the true division between and int64
                    # and a constant that will get typed as int8.

                    # As this is just to allow merging more case, if
                    # the upcast don't work, we can just skip it.
                    return

                # Copy over output stacktrace from before upcasting
                copy_stack_trace(node.outputs[0], rval)
                return rval


@node_rewriter([Elemwise])
def local_add_mul_fusion(fgraph, node):
    """Fuse consecutive add or mul in one such node with more inputs.

    It is better to fuse add/mul that way then in a Composite node as
    this make the inner graph of the Composite smaller. This allows to
    put more computation in a Composite before hitting the max
    recursion limit when pickling Composite.

    This rewrite is almost useless after the AlgebraicCanonizer is used,
    but it catches a few edge cases that are not canonicalized by it
    """
    if not isinstance(node.op, Elemwise) or not isinstance(
        node.op.scalar_op, (aes.Add, aes.Mul)
    ):
        return False

    s_op = node.op.scalar_op.__class__
    new_inp = []
    fused = False
    nb_inputs = len(node.inputs)
    max_inputs = float("inf")
    if hasattr(node.op, "max_inputs"):
        max_inputs = node.op.max_inputs(node)
    for inp in node.inputs:
        if (
            inp.owner
            and isinstance(inp.owner.op, Elemwise)
            and isinstance(inp.owner.op.scalar_op, s_op)
            and
            # Do not duplicate the operation.
            len(fgraph.clients[inp]) == 1
            and (nb_inputs + len(inp.owner.inputs) - 1) <= max_inputs
        ):
            new_inp.extend(inp.owner.inputs)
            fused = True
        else:
            new_inp.append(inp)

    # We can not compare the number of inputs as Mul and Add could have
    # 0 or 1 inputs in some corner cases.
    if fused:
        output = node.op(*new_inp)
        copy_stack_trace(node.outputs[0], output)

        # Do the recursion here to help lower the number of
        # FusionOptimizer iteration.
        if output.owner:
            output2 = local_add_mul_fusion.transform(fgraph, output.owner)
            if output2:
                return output2
        return [output]


def elemwise_max_operands_fct(node) -> int:
    # `Elemwise.perform` uses NumPy ufuncs and they are limited to 32 operands (inputs and outputs)
    if not config.cxx:
        return 32
    return 1024


class FusionOptimizer(GraphRewriter):
    """Graph optimizer that fuses consecutive Elemwise operations."""

    def add_requirements(self, fgraph):
        fgraph.attach_feature(ReplaceValidate())

    @staticmethod
    def elemwise_to_scalar(inputs, outputs):
        replace_inputs = [(inp, inp.clone()) for inp in inputs]
        outputs = clone_replace(outputs, replace=replace_inputs)

        inputs = [inp for _, inp in replace_inputs]
        fg = FunctionGraph(inputs=inputs, outputs=outputs, clone=False)
        middle_inputs = []

        scalar_inputs = [
            aes.get_scalar_type(inp.type.dtype).make_variable() for inp in inputs
        ]
        middle_scalar_inputs = []

        for node in fg.toposort():
            node_scalar_inputs = []
            for inp in node.inputs:
                if inp in inputs:
                    node_scalar_inputs.append(scalar_inputs[inputs.index(inp)])
                elif inp in middle_inputs:
                    node_scalar_inputs.append(
                        middle_scalar_inputs[middle_inputs.index(inp)]
                    )
                else:
                    new_scalar_input = aes.get_scalar_type(
                        inp.type.dtype
                    ).make_variable()
                    node_scalar_inputs.append(new_scalar_input)
                    middle_scalar_inputs.append(new_scalar_input)
                    middle_inputs.append(inp)

            new_scalar_node = node.op.scalar_op.make_node(*node_scalar_inputs)
            middle_scalar_inputs.append(new_scalar_node.outputs[0])
            middle_inputs.append(node.outputs[0])

        scalar_outputs = [
            middle_scalar_inputs[middle_inputs.index(out)] for out in fg.outputs
        ]
        return scalar_inputs, scalar_outputs

    def apply(self, fgraph):
        nb_replacement = 0

        if fgraph.profile:
            validate_before = fgraph.profile.validate_time
            callbacks_before = fgraph.execute_callbacks_times.copy()
            callback_before = fgraph.execute_callbacks_time

        max_operands = elemwise_max_operands_fct(None)

        def find_next_fuseable_subgraph(
            fg: FunctionGraph,
        ) -> Generator[Tuple[List[Variable], List[Variable]], None, None]:
            """Find all subgraphs in a FunctionGraph that can be fused together

            Yields
            -------
            List of inputs and outputs that determine subgraphs which can be fused.
            This generator assumes that such subgraph is replaced by a single
            Elemwise Composite before being accessed again in the next iteration.
            """

            FUSEABLE_MAPPING = DefaultDict[Variable, List[Apply]]
            UNFUSEABLE_MAPPING = DefaultDict[Variable, Set[ApplyOrOutput]]

            def initialize_fuseable_mappings(
                *, fg: FunctionGraph
            ) -> Tuple[FUSEABLE_MAPPING, UNFUSEABLE_MAPPING]:
                @lru_cache(maxsize=None)
                def elemwise_scalar_op_has_c_code(node: Apply) -> bool:
                    # TODO: This should not play a role in non-c backends!
                    if node.op.scalar_op.supports_c_code(node.inputs, node.outputs):
                        return True
                    else:
                        warn(
                            "Optimization Warning: "
                            f"The Op {node.op.scalar_op} does not provide a C implementation."
                            " As well as being potentially slow, this also disables "
                            "loop fusion."
                        )
                        return False

                # Fuseable nodes have to be accessed in a deterministic manner
                # to ensure the rewrite remains deterministic.
                # This is not a problem from unfuseable ones, as they can never
                # become part of the graph.
                fuseable_clients: FUSEABLE_MAPPING = defaultdict(list)
                unfuseable_clients: UNFUSEABLE_MAPPING = defaultdict(set)
                for out, clients in fg.clients.items():
                    # Old FunctionGraph nodes remain in the clients dictionary
                    # even after they are removed by rewrites
                    if not clients:
                        continue

                    out_maybe_fuseable = (
                        out.owner
                        and isinstance(out.owner.op, Elemwise)
                        # and not isinstance(out.owner.op.scalar_op, aes.Composite)
                        and len(out.owner.outputs) == 1
                        and elemwise_scalar_op_has_c_code(out.owner)
                    )
                    for client, _ in clients:
                        if (
                            out_maybe_fuseable
                            and not isinstance(client, str)  # "output"
                            and isinstance(client.op, Elemwise)
                            # and not isinstance(client.op.scalar_op, aes.Composite)
                            and len(client.outputs) == 1
                            and out.type.broadcastable
                            == client.outputs[0].type.broadcastable
                            and elemwise_scalar_op_has_c_code(client)
                        ):
                            if client not in fuseable_clients[out]:
                                fuseable_clients[out].append(client)
                        else:
                            unfuseable_clients[out].add(client)

                return fuseable_clients, unfuseable_clients

            def find_fuseable_subgraph(
                *,
                fg: FunctionGraph,
                visited_nodes: Set[Apply],
                fuseable_clients: FUSEABLE_MAPPING,
                unfuseable_clients: UNFUSEABLE_MAPPING,
            ) -> Tuple[List[Variable], List[Variable]]:
                KT = TypeVar("KT")
                VT = TypeVar("VT", list, set)

                def shallow_clone_defaultdict(
                    d: DefaultDict[KT, VT]
                ) -> DefaultDict[KT, VT]:
                    new_dict: DefaultDict[KT, VT] = defaultdict(d.default_factory)
                    new_dict.update({k: v.copy() for k, v in d.items()})
                    return new_dict

                def variables_depend_on(
                    variables, depend_on, stop_search_at=None
                ) -> bool:
                    return any(
                        a in depend_on
                        for a in ancestors(variables, blockers=stop_search_at)
                    )

                toposort = fg.toposort()
                for starting_node in toposort:
                    if starting_node in visited_nodes:
                        continue

                    starting_out = starting_node.outputs[0]
                    if not fuseable_clients.get(starting_out):
                        visited_nodes.add(starting_node)
                        continue

                    subgraph_inputs: List[Variable] = []
                    subgraph_outputs: List[Variable] = []
                    unfuseable_clients_subgraph: Set[Variable] = set()

                    # Shallow cloning of maps so that they can be manipulated in place
                    fuseable_clients_temp = shallow_clone_defaultdict(fuseable_clients)
                    unfuseable_clients_clone = shallow_clone_defaultdict(
                        unfuseable_clients
                    )

                    fuseable_nodes_to_visit = deque([starting_node])

                    # We now try to expand as much as possible towards the potentially
                    # fuseable clients and ancestors to detect the largest possible
                    # subgraph that can be Composed together into a single `Op`. The
                    # largest issue to watch out is for cyclical dependencies, where
                    # some inputs or clients may depend on other nodes of the same
                    # subgraph via a path that cannot be included in the Composite
                    # (unfuseable)
                    while fuseable_nodes_to_visit:
                        next_node = fuseable_nodes_to_visit.popleft()
                        visited_nodes.add(next_node)
                        next_out = next_node.outputs[0]

                        # If the output variable of next_node has no fuseable clients
                        # or has unfuseable clients, then next_node must become an output
                        # if it is to be fused.
                        must_become_output = (
                            next_out not in fuseable_clients_temp
                            or next_out in unfuseable_clients_clone
                        )

                        # We have backtracked to this node, and it may no longer be a viable output,
                        # so we remove it and check again as if we had never seen this node
                        if must_become_output and next_out in subgraph_outputs:
                            subgraph_outputs.remove(next_out)

                        required_unfuseable_inputs = [
                            inp
                            for inp in next_node.inputs
                            if next_node in unfuseable_clients_clone.get(inp, ())
                        ]
                        new_required_unfuseable_inputs = [
                            inp
                            for inp in required_unfuseable_inputs
                            if inp not in subgraph_inputs
                        ]

                        must_backtrack = False
                        if new_required_unfuseable_inputs and subgraph_outputs:
                            # We need to check that any new inputs required by this node
                            # do not depend on other outputs of the current subgraph,
                            # via an unfuseable path.
                            if variables_depend_on(
                                [next_out],
                                depend_on=unfuseable_clients_subgraph,
                                stop_search_at=subgraph_outputs,
                            ):
                                must_backtrack = True

                        if not must_backtrack:
                            implied_unfuseable_clients = {
                                c
                                for client in unfuseable_clients_clone.get(next_out, ())
                                if not isinstance(client, str)  # "output"
                                for c in client.outputs
                            }

                            new_implied_unfuseable_clients = (
                                implied_unfuseable_clients - unfuseable_clients_subgraph
                            )

                            if new_implied_unfuseable_clients and subgraph_inputs:
                                # We need to check that any inputs of the current subgraph
                                # do not depend on other clients of this node,
                                # via an unfuseable path.
                                if variables_depend_on(
                                    subgraph_inputs,
                                    depend_on=new_implied_unfuseable_clients,
                                ):
                                    must_backtrack = True

                        if must_backtrack:
                            for inp in next_node.inputs:
                                if (
                                    inp.owner in visited_nodes
                                    # next_node could have the same input repeated
                                    and next_node in fuseable_clients_temp[inp]
                                ):
                                    fuseable_clients_temp[inp].remove(next_node)
                                    unfuseable_clients_clone[inp].add(next_node)
                                    # This input must become an output of the subgraph,
                                    # because it can't be merged with next_node.
                                    # We will revisit it to make sure this is safe.
                                    fuseable_nodes_to_visit.appendleft(inp.owner)

                            for client in fuseable_clients_temp[next_out]:
                                if client in visited_nodes:
                                    fuseable_clients_temp[next_out].remove(client)
                                    unfuseable_clients_clone[next_out].add(client)
                                    # next_out must become an input of the subgraph.
                                    # We will revisit any of its clients currently
                                    # in the subgraph to make sure this is safe.
                                    fuseable_nodes_to_visit.appendleft(client)

                            # Revisit node at a later time
                            visited_nodes.remove(next_node)
                            continue

                        # Adding next_node to subgraph does not result in any
                        # immediate dependency problems. Update subgraph
                        # mappings as if it next_node was part of it.
                        # Useless inputs will be removed by the useless Composite rewrite
                        for inp in new_required_unfuseable_inputs:
                            if inp not in subgraph_inputs:
                                subgraph_inputs.append(inp)

                        if must_become_output:
                            subgraph_outputs.append(next_out)
                            unfuseable_clients_subgraph.update(
                                new_implied_unfuseable_clients
                            )

                        # Expand through unvisited fuseable ancestors
                        for inp in sorted(
                            (
                                inp
                                for inp in next_node.inputs
                                if (
                                    inp not in required_unfuseable_inputs
                                    and inp.owner not in visited_nodes
                                )
                            ),
                            key=lambda inp: toposort.index(inp.owner),
                            reverse=True,
                        ):
                            fuseable_nodes_to_visit.appendleft(inp.owner)

                        # Expand through unvisited fuseable clients
                        for next_node in sorted(
                            (
                                node
                                for node in fuseable_clients_temp.get(next_out, ())
                                if node not in visited_nodes
                            ),
                            key=lambda node: toposort.index(node),
                        ):
                            fuseable_nodes_to_visit.append(next_node)

                    # Don't return if final subgraph is just the original Elemwise
                    if len(subgraph_outputs) == 1 and set(
                        subgraph_outputs[0].owner.inputs
                    ) == set(subgraph_inputs):
                        # Update global fuseable mappings
                        # No input was actually fuseable
                        for inp in starting_node.inputs:
                            if starting_node in fuseable_clients.get(inp, ()):
                                fuseable_clients[inp].remove(starting_node)
                                unfuseable_clients[inp].add(starting_node)
                        # No client was actually fuseable
                        unfuseable_clients[starting_out].update(
                            fuseable_clients.pop(starting_out, ())
                        )
                        continue

                    return subgraph_inputs, subgraph_outputs
                raise ValueError

            def update_fuseable_mappings_after_fg_replace(
                *,
                fg: FunctionGraph,
                visited_nodes: Set[Apply],
                fuseable_clients: FUSEABLE_MAPPING,
                unfuseable_clients: UNFUSEABLE_MAPPING,
                starting_nodes: Set[Apply],
            ) -> None:
                # Find new composite node and dropped intermediate nodes
                # by comparing the current fg.apply nodes with the cached
                # original nodes
                next_nodes = fg.apply_nodes
                (new_composite_node,) = next_nodes - starting_nodes
                dropped_nodes = starting_nodes - next_nodes

                # Remove intermediate Composite nodes from mappings
                for dropped_node in dropped_nodes:
                    (dropped_out,) = dropped_node.outputs
                    fuseable_clients.pop(dropped_out, None)
                    unfuseable_clients.pop(dropped_out, None)
                    visited_nodes.remove(dropped_node)

                # Update fuseable information for subgraph inputs
                for inp in subgraph_inputs:
                    if inp in fuseable_clients:
                        new_fuseable_clients = [
                            client
                            for client in fuseable_clients[inp]
                            if client not in dropped_nodes
                        ]
                        if new_fuseable_clients:
                            fuseable_clients[inp] = new_fuseable_clients
                        else:
                            fuseable_clients.pop(inp)
                    unfuseable_clients[inp] = (
                        unfuseable_clients[inp] - dropped_nodes
                    ) | {new_composite_node}

                # Update fuseable information for subgraph outputs
                for out in new_composite_node.outputs:
                    unfuseable_clients[out] = {client for client, _ in fg.clients[out]}

                visited_nodes.add(new_composite_node)
                return

            # We start by creating two maps, 1) from each node to each potentially
            # fuseable client (both nodes must be single output Elemwise with same
            # broadcast type) and 2) from each node to each certainly unfuseable
            # client (those that don't fit into 1))
            fuseable_clients, unfuseable_clients = initialize_fuseable_mappings(fg=fg)
            visited_nodes: Set[Apply] = set()
            while True:
                starting_nodes = fg.apply_nodes.copy()
                try:
                    subgraph_inputs, subgraph_outputs = find_fuseable_subgraph(
                        fg=fg,
                        visited_nodes=visited_nodes,
                        fuseable_clients=fuseable_clients,
                        unfuseable_clients=unfuseable_clients,
                    )
                except ValueError:
                    return
                else:
                    # The caller is now expected to update fg in place,
                    # by replacing the subgraph with a Composite Op
                    yield subgraph_inputs, subgraph_outputs

                    # This is where we avoid repeated work by using a stateful
                    # generator. For large models (as in `TestFusion.test_big_fusion`)
                    # this can provide huge speedups
                    update_fuseable_mappings_after_fg_replace(
                        fg=fg,
                        visited_nodes=visited_nodes,
                        fuseable_clients=fuseable_clients,
                        unfuseable_clients=unfuseable_clients,
                        starting_nodes=starting_nodes,
                    )

        for inputs, outputs in find_next_fuseable_subgraph(fgraph):
            if (len(inputs) + len(outputs)) > max_operands:
                warn(
                    "Loop fusion failed because the resulting node would exceed "
                    "the kernel argument limit."
                )
                break

            scalar_inputs, scalar_outputs = self.elemwise_to_scalar(inputs, outputs)
            composite_outputs = Elemwise(aes.Composite(scalar_inputs, scalar_outputs))(
                *inputs
            )
            if not isinstance(composite_outputs, list):
                composite_outputs = [composite_outputs]
            for old_out, composite_out in zip(outputs, composite_outputs):
                if old_out.name:
                    composite_out.name = old_out.name

            fgraph.replace_all_validate(
                list(zip(outputs, composite_outputs)),
                reason=self.__class__.__name__,
            )
            nb_replacement += 1

        if fgraph.profile:
            validate_time = fgraph.profile.validate_time - validate_before
            callback_time = fgraph.execute_callbacks_time - callback_before
            callbacks_time = {}
            for k, v in fgraph.execute_callbacks_times.items():
                if k in callbacks_before:
                    callbacks_time[k] = v - callbacks_before[k]
                else:
                    callbacks_time[k] = v
        else:
            validate_time = None
            callback_time = None
            callbacks_time = {}

        return (
            self,
            1,  # nb_iter
            nb_replacement,
            0,  # nb_inconsintency_replace
            validate_time,
            callback_time,
            callbacks_time,
            -1,  # toposort_time
        )

    @staticmethod
    def print_profile(stream, prof, level=0):
        blanc = "    " * level
        print(blanc, "FusionOptimizer", file=stream)
        print(blanc, " nb_iter", prof[1], file=stream)
        print(blanc, " nb_replacement", prof[2], file=stream)
        print(blanc, " nb_inconsistency_replace", prof[3], file=stream)
        print(blanc, " validate_time", prof[4], file=stream)
        print(blanc, " callback_time", prof[5], file=stream)
        if prof[5] is not None and prof[5] > 1:
            print(blanc, " callbacks_time", file=stream)
            for i in sorted(prof[6].items(), key=lambda a: a[1])[::-1]:
                if i[1] > 0:
                    print(blanc, "     ", i)
        print(blanc, " time_toposort", prof[7], file=stream)


@register_canonicalize
@register_specialize
@node_rewriter([Elemwise])
def local_useless_composite_outputs(fgraph, node):
    """Remove inputs and outputs of Composite Ops that are not used anywhere."""
    if not isinstance(node.op, Elemwise) or not isinstance(
        node.op.scalar_op, aes.Composite
    ):
        return
    comp = node.op.scalar_op
    used_outputs_idxs = [
        i for i, o_extern in enumerate(node.outputs) if fgraph.clients[o_extern]
    ]
    used_inner_outputs = [comp.outputs[i] for i in used_outputs_idxs]
    comp_fgraph = FunctionGraph(
        inputs=comp.inputs, outputs=used_inner_outputs, clone=False
    )
    used_inputs_idxs = [
        i
        for i, i_intern in enumerate(comp_fgraph.inputs)
        if comp_fgraph.clients[i_intern]
    ]
    used_inner_inputs = [comp.inputs[i] for i in used_inputs_idxs]
    if len(used_inner_inputs) < len(node.inputs) or len(used_inner_outputs) < len(
        node.outputs
    ):
        used_inputs = [node.inputs[i] for i in used_inputs_idxs]
        c = aes.Composite(inputs=used_inner_inputs, outputs=used_inner_outputs)
        e = Elemwise(scalar_op=c)(*used_inputs, return_list=True)
        return dict(zip([node.outputs[i] for i in used_outputs_idxs], e))


@node_rewriter([CAReduce])
def local_careduce_fusion(fgraph, node):
    """Fuse a `CAReduce` applied to an `Elemwise`."""

    (car_input,) = node.inputs
    car_scalar_op = node.op.scalar_op

    # FIXME: This check is needed because of the faulty logic in the FIXME below!
    # Right now, rewrite only works for `Sum`/`Prod`
    if not isinstance(car_scalar_op, (aes.Add, aes.Mul)):
        return None

    elm_node = car_input.owner

    if elm_node is None or not isinstance(elm_node.op, Elemwise):
        return False

    elm_scalar_op = elm_node.op.scalar_op

    elm_inputs = elm_node.inputs
    elm_outputs = elm_node.outputs

    if len(elm_inputs) > 1 or len(elm_outputs) > 1:
        # TODO: Implement the multiple inputs case
        return False

    if len(fgraph.clients[elm_outputs[0]]) > 1:
        return False

    # Don't form the fusion when the target language is Python
    if get_target_language() == ("py",):
        return False

    if not elm_scalar_op.supports_c_code(elm_inputs, elm_outputs):
        return None

    # FIXME: This fails with Ops like `Max` whose `c_code` always expects two inputs!
    #  Should implement a `CAReduce.supports_c_code`?
    try:
        car_scalar_op.c_code(
            node,
            "test_presence_of_c_code",
            ["x" for x in node.inputs],
            ["z" for z in node.outputs],
            {"fail": "%(fail)s"},
        )
    except (NotImplementedError, MethodNotDefined):
        return False

    car_op = node.op
    car_acc_dtype = node.op.acc_dtype

    scalar_elm_inputs = [
        aes.get_scalar_type(inp.type.dtype).make_variable() for inp in elm_inputs
    ]

    elm_output = elm_scalar_op(*scalar_elm_inputs)

    # This input represents the previous value in the `CAReduce` binary reduction
    carried_car_input = aes.get_scalar_type(car_acc_dtype).make_variable()

    scalar_fused_output = car_scalar_op(carried_car_input, elm_output)
    if scalar_fused_output.type.dtype != car_acc_dtype:
        scalar_fused_output = aes.cast(scalar_fused_output, car_acc_dtype)

    fused_scalar_op = aes.Composite(
        inputs=[carried_car_input] + scalar_elm_inputs, outputs=[scalar_fused_output]
    )

    # The fused `Op` needs to look and behave like a `BinaryScalarOp`
    # TODO: Generate a new `type` and make this relationship official?
    fused_scalar_op.identity = car_scalar_op.identity
    fused_scalar_op.nin = 2
    fused_scalar_op.nout = 1

    new_car_op = CAReduce(
        scalar_op=fused_scalar_op,
        axis=car_op.axis,
        acc_dtype=car_acc_dtype,
        dtype=car_op.dtype,
        upcast_discrete_output=car_op.upcast_discrete_output,
    )

    return [new_car_op(*elm_inputs)]


@node_rewriter([Elemwise])
def local_inline_composite_constants(fgraph, node):
    """Inline scalar constants in Composite graphs."""
    composite_op = node.op.scalar_op

    if not isinstance(composite_op, aes.Composite):
        return None

    new_outer_inputs = []
    new_inner_inputs = []
    inner_replacements = {}
    for outer_inp, inner_inp in zip(node.inputs, composite_op.fgraph.inputs):
        # Complex variables don't have a `c_literal` that can be inlined
        if "complex" not in outer_inp.type.dtype:
            unique_value = get_unique_constant_value(outer_inp)
            if unique_value is not None:
                inner_replacements[inner_inp] = aes.constant(
                    unique_value, dtype=inner_inp.dtype
                )
                continue
        new_outer_inputs.append(outer_inp)
        new_inner_inputs.append(inner_inp)

    if not inner_replacements:
        return None

    new_inner_outs = clone_replace(
        composite_op.fgraph.outputs, replace=inner_replacements
    )
    new_composite_op = aes.Composite(new_inner_inputs, new_inner_outs)
    new_outputs = Elemwise(new_composite_op).make_node(*new_outer_inputs).outputs

    # Some of the inlined constants were broadcasting the output shape
    if node.outputs[0].type.broadcastable != new_outputs[0].type.broadcastable:
        new_outputs = [
            alloc_like(new_out, template=node.outputs[0], fgraph=fgraph)
            for new_out in new_outputs
        ]

    copy_stack_trace(node.outputs, new_outputs)
    return new_outputs


# Register fusion database just before AddDestroyHandler(49.5) (inplace rewrites)
fuse_seqopt = SequenceDB()
compile.optdb.register(
    "elemwise_fusion",
    fuse_seqopt,
    "fast_run",
    "fusion",
    "local_elemwise_fusion",
    "FusionOptimizer",
    position=49,
)

fuse_seqopt.register(
    "local_add_mul_fusion",
    EquilibriumGraphRewriter(rewriters=[local_add_mul_fusion], max_use_ratio=1000),
    "fast_run",
    "fusion",
    position=0,
)
fuse_seqopt.register(
    "composite_elemwise_fusion",
    FusionOptimizer(),
    "fast_run",
    "fusion",
    position=1,
)
fuse_seqopt.register(
    "local_useless_composite_outputs",
    in2out(local_useless_composite_outputs),
    "fast_run",
    "fusion",
    position=2,
)
fuse_seqopt.register(
    "local_careduce_fusion",
    in2out(local_careduce_fusion),
    "fast_run",
    "fusion",
    position=10,
)
fuse_seqopt.register(
    "local_inline_composite_constants",
    in2out(local_inline_composite_constants),
    "fast_run",
    "fusion",
    position=20,
)


def _rebuild_partial_2f1grad_loop(node, wrt):
    a, b, c, log_z, sign_z = node.inputs[-5:]
    z = exp(log_z) * sign_z

    # Reconstruct scalar loop with relevant outputs
    a_, b_, c_, z_ = (x.type.to_scalar_type()() for x in (a, b, c, z))
    new_loop_op = _grad_2f1_loop(
        a_, b_, c_, z_, skip_loop=False, wrt=wrt, dtype=a_.type.dtype
    )[0].owner.op

    # Reconstruct elemwise loop
    new_elemwise_op = Elemwise(scalar_op=new_loop_op)
    n_steps = node.inputs[0]
    init_grad_vars = node.inputs[1:10]
    other_inputs = node.inputs[10:]

    init_grads = init_grad_vars[: len(wrt)]
    init_gs = init_grad_vars[3 : 3 + len(wrt)]
    init_gs_signs = init_grad_vars[6 : 6 + len(wrt)]
    subset_init_grad_vars = init_grads + init_gs + init_gs_signs

    return new_elemwise_op(n_steps, *subset_init_grad_vars, *other_inputs)


@register_specialize
@node_rewriter([Elemwise])
def local_useless_2f1grad_loop(fgraph, node):
    # Remove unused terms from the hyp2f1 grad loop

    loop_op = node.op.scalar_op
    if not isinstance(loop_op, Grad2F1Loop):
        return

    grad_related_vars = node.outputs[:-4]
    # Rewrite was already applied
    if len(grad_related_vars) // 3 != 3:
        return None

    grad_vars = grad_related_vars[:3]
    grad_var_is_used = [bool(fgraph.clients.get(v)) for v in grad_vars]

    # Nothing to do here
    if sum(grad_var_is_used) == 3:
        return None

    *other_vars, converges = node.outputs[3:]

    # Check that None of the remaining vars (except the converge flag) is used anywhere
    if any(bool(fgraph.clients.get(v)) for v in other_vars):
        return None

    wrt = [i for i, used in enumerate(grad_var_is_used) if used]
    *new_outs, new_converges = _rebuild_partial_2f1grad_loop(node, wrt=wrt)

    replacements = {converges: new_converges}
    i = 0
    for grad_var, is_used in zip(grad_vars, grad_var_is_used):
        if not is_used:
            continue
        replacements[grad_var] = new_outs[i]
        i += 1
    return replacements


@node_rewriter([Elemwise])
def split_2f1grad_loop(fgraph, node):
    """
    2f1grad loop has too many operands for Numpy frompyfunc code used by Elemwise nodes on python mode.

    This rewrite splits it across 3 different operations. It is not needed if `local_useless_2f1grad_loop` was applied
    """
    loop_op = node.op.scalar_op

    if not isinstance(loop_op, Grad2F1Loop):
        return None

    grad_related_vars = node.outputs[:-4]
    # local_useless_2f1grad_loop was used, we should be safe
    if len(grad_related_vars) // 3 != 3:
        return None

    grad_vars = grad_related_vars[:3]
    *other_vars, converges = node.outputs[3:]

    # Check that None of the remaining vars is used anywhere
    if any(bool(fgraph.clients.get(v)) for v in other_vars):
        return None

    new_grad0, new_grad1, *_, new_converges01 = _rebuild_partial_2f1grad_loop(
        node, wrt=[0, 1]
    )
    new_grad2, *_, new_converges2 = _rebuild_partial_2f1grad_loop(node, wrt=[2])

    replacements = {
        converges: new_converges01 & new_converges2,
        grad_vars[0]: new_grad0,
        grad_vars[1]: new_grad1,
        grad_vars[2]: new_grad2,
    }
    return replacements


compile.optdb["py_only"].register(
    "split_2f1grad_loop",
    split_2f1grad_loop,
    "fast_compile",
)
