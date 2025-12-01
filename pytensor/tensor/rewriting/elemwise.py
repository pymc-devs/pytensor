import abc
import itertools
import operator
import sys
from collections.abc import Generator, Sequence
from functools import cache, reduce
from heapq import heapify, heappop, heappush
from operator import or_
from warnings import warn

from pytensor.compile.function.types import Supervisor
from pytensor.compile.mode import get_target_language, optdb
from pytensor.configdefaults import config
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.destroyhandler import DestroyHandler, inplace_candidates
from pytensor.graph.features import ReplaceValidate
from pytensor.graph.fg import FunctionGraph, Output
from pytensor.graph.op import Op
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.basic import (
    GraphRewriter,
    copy_stack_trace,
    dfs_rewriter,
    in2out,
    node_rewriter,
    out2in,
)
from pytensor.graph.rewriting.db import SequenceDB
from pytensor.graph.rewriting.unify import OpPattern
from pytensor.graph.traversal import toposort
from pytensor.graph.utils import InconsistencyError, MethodNotDefined
from pytensor.scalar import (
    Add,
    Composite,
    Mul,
    ScalarOp,
    get_scalar_type,
    upcast_out,
    upgrade_to_float,
)
from pytensor.scalar import cast as scalar_cast
from pytensor.scalar import constant as scalar_constant
from pytensor.scalar.math import Grad2F1Loop, _grad_2f1_loop
from pytensor.tensor.basic import MakeVector
from pytensor.tensor.basic import constant as tensor_constant
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.math import add, exp, mul
from pytensor.tensor.rewriting.basic import (
    alloc_like,
    broadcasted_by,
    elemwise_of,
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.variable import TensorConstant, TensorVariable


class InplaceGraphOptimizer(GraphRewriter):
    op: type[Op]

    def add_requirements(self, fgraph):
        fgraph.attach_feature(DestroyHandler())

    @abc.abstractmethod
    def filter_candidate_pairs(
        self, fgraph: FunctionGraph, node: Apply, protected_inputs: Sequence[Variable]
    ) -> Sequence[tuple[tuple[int, Variable], tuple[int, Variable]]]:
        pass

    @abc.abstractmethod
    def create_inplace_node(
        self, node: Apply, inplace_pattern: dict[int, Sequence[int]]
    ) -> Apply:
        pass

    def apply(self, fgraph):
        r"""

        Attempts to replace all `Op`\s by versions of them that operate
        inplace. It operates greedily: for each `Op` that is encountered,
        it tries to inplace all the valid inputs at once (if the Op supports it),
        if that fails, it tries to inplace one input at a time.

        Examples
        --------

            x + y + z -> x += y += z
            (x + y) * (x * y) -> (x += y) *= (x * y) or (x + y) *= (x *= y)

        """
        # We should not validate too often as this takes too much time to execute!
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

        # 2025: The above comment is not specific to Elemwise, if we have concerns about this approach, we should
        # tackle them in a more general way. The whole try/except approach is probably suboptimal.
        # We can consider restricting inputs with static shapes that are large enough.

        if config.tensor__insert_inplace_optimizer_validate_nb != -1:
            warn(
                "tensor__insert_inplace_optimizer_validate_nb config is deprecated. Setting it will fail in a future release.",
                FutureWarning,
            )

        reason = f"{self.op}_inplace_optimizer"
        prof = {
            "opt": self,
            "node_before": len(fgraph.apply_nodes),
            "nb_eager_inconsistent": 0,
            "nb_inconsistent": 0,
            "nb_replaced": 0,
        }
        large_graph = len(fgraph.apply_nodes) > 500

        protected_inputs = set(
            itertools.chain.from_iterable(
                f.protected for f in fgraph._features if isinstance(f, Supervisor)
            )
        )
        protected_inputs.update(fgraph.outputs)
        root_destroyer = fgraph.destroy_handler.root_destroyer

        self_op = self.op
        update_mapping = fgraph.update_mapping or {}
        op_updates: dict[TensorVariable, TensorVariable] = {
            out: fgraph.inputs[update_mapping[out_idx]]
            for out_idx, out in enumerate(fgraph.outputs)
            if (
                out_idx in update_mapping
                and out.owner
                and isinstance(out.owner.op, self_op)
            )
        }
        set_op_updates = set(op_updates.keys())

        for node in fgraph.toposort():
            if not isinstance(node.op, self_op) or node.op.destroy_map:
                continue

            # If big graph and the outputs are scalar, do not make it inplace.
            if large_graph and all(node.outputs[0].type.broadcastable):
                continue

            candidate_pairs = self.filter_candidate_pairs(
                fgraph, node, protected_inputs
            )

            if not candidate_pairs:
                continue

            sorted_candidate_pairs = candidate_pairs
            if op_updates and (node_updates := set(node.outputs) & set_op_updates):
                # If the fgraph has updates, we try to prioritize in-placing on the pairs that correspond to the update
                direct_update_pairs = []
                indirect_update_pairs = []
                other_update_pairs = []
                for pair in candidate_pairs:
                    ((o, out), (i, inp)) = pair
                    if out in node_updates:
                        direct_update_inp = op_updates[out]
                        if direct_update_inp is inp:
                            # This pair is the whole graph update
                            direct_update_pairs.append(pair)
                            continue
                        elif (inp_node := inp.owner) is not None and any(
                            root_destroyer.get(up_inp, None) is inp_node
                            for up_inp in op_updates.values()
                        ):
                            # This pair connects to an updated input
                            indirect_update_pairs.append(pair)
                            continue
                    other_update_pairs.append(pair)

                sorted_candidate_pairs = (
                    direct_update_pairs + indirect_update_pairs + other_update_pairs
                )

            # Try in-placing all outputs at once
            tried_inputs = set()
            inplace_pattern = {}
            for (o, _), (i, _) in sorted_candidate_pairs:
                if o not in inplace_pattern and i not in tried_inputs:
                    inplace_pattern[o] = [i]
                    tried_inputs.add(i)

            inplace_node = self.create_inplace_node(node, inplace_pattern)
            if inplace_node.op.destroy_map == inplace_pattern:
                replacements = tuple(zip(node.outputs, inplace_node.outputs))
                try:
                    fgraph.replace_all_validate(replacements, reason=reason)
                except InconsistencyError:
                    prof["nb_eager_inconsistent"] += 1
                else:
                    prof["nb_replaced"] += 1
                    copy_stack_trace(node.outputs, inplace_node.outputs)
                    continue

            # If it fails or doesn't match the desired inplace pattern, try one output/input at a time
            tried_inputs = set()
            inplace_pattern = {}
            replaced = False
            original_node = node
            for (o, _), (i, _) in sorted_candidate_pairs:
                if o not in inplace_pattern and i not in tried_inputs:
                    inplace_pattern[o] = [i]
                    tried_inputs.add(i)

                    inplace_node = self.create_inplace_node(node, inplace_pattern)
                    if inplace_node.op.destroy_map != inplace_pattern:
                        # This Op can't respect this partial inplace pattern,
                        # We assume it can't support any other cases
                        break
                    else:
                        replacements = tuple(zip(node.outputs, inplace_node.outputs))
                        try:
                            fgraph.replace_all_validate(replacements, reason=reason)
                            node = inplace_node
                            replaced = True
                        except InconsistencyError:
                            prof["nb_inconsistent"] += 1
                            # The input, not the output caused inconsistencies
                            inplace_pattern.pop(o)
            if replaced:
                copy_stack_trace(original_node.outputs, node.outputs)
                prof["nb_replaced"] += replaced

        return prof

    @classmethod
    def print_profile(cls, stream, prof, level=0):
        blanc = "    " * level
        print(blanc, cls.__name__, file=stream)
        for k in [
            "node_before",
            "nb_eager_inconsistent",
            "nb_inconsistent",
            "nb_replaced",
        ]:
            print(blanc, k, prof[k], file=stream)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(
            f"{' ' * level}{self.__class__.__name__}",
            file=stream,
        )


class InplaceElemwiseOptimizer(InplaceGraphOptimizer):
    op = Elemwise

    def filter_candidate_pairs(self, fgraph, node, protected_inputs):
        candidate_inputs = [
            (node.inputs.index(inp), inp)
            for inp in inplace_candidates(
                fgraph,
                node.inputs,
                protected_inputs=protected_inputs,
            )
        ]
        if not candidate_inputs:
            return []

        return [
            ((o, out), (i, inp))
            for o, out in enumerate(node.outputs)
            for i, inp in candidate_inputs
            if inp.type == out.type
        ]

    def create_inplace_node(self, node, inplace_pattern):
        op = node.op
        scalar_op = op.scalar_op
        inplace_pattern = {i: o for i, [o] in inplace_pattern.items()}
        try:
            return type(op)(scalar_op, inplace_pattern).make_node(*node.inputs)
        except TypeError:
            # Elemwise raises TypeError if we try to inplace an output on an input of a different dtype
            if config.optimizer_verbose:
                print(  # noqa: T201
                    f"InplaceElemwise failed because the output dtype of {node} changed when rebuilt. "
                    "Perhaps due to a change in config.floatX or config.cast_policy"
                )
            # InplaceGraphOptimizer will chug along fine if we return the original node
            return node


optdb.register(
    "inplace_elemwise",
    InplaceElemwiseOptimizer(),
    "inplace_elemwise_opt",  # for historic reason
    "inplace_elemwise_optimizer",
    "fast_run",
    "inplace",
    position=50.5,
)


def apply_local_dimshuffle_lift(fgraph, var):
    """
    lift recursively
    """
    if var.owner is None:
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
@register_stabilize
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
            new_inp = inp.dimshuffle(op.new_order)
            new_inputs.append(apply_local_dimshuffle_lift(fgraph, new_inp))
        copy_stack_trace(node.outputs[0], new_inputs)
        ret = inode.op(*new_inputs, return_list=True)
        return ret
    if inode and isinstance(inode.op, DimShuffle):
        new_order = [(x == "x" and "x") or inode.op.new_order[x] for x in new_order]
        inp = inode.inputs[0]

    if is_dimshuffle_useless(new_order, inp):
        return [inp]
    elif inode and isinstance(inode.op, DimShuffle):
        ret = inp.dimshuffle(new_order)
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

    if not (
        makevector_out.owner
        and isinstance(makevector_out.owner.op, MakeVector)
        and makevector_out.broadcastable == (True,)
    ):
        return

    assert len(makevector_out.owner.inputs) == 1

    return [makevector_out.owner.inputs[0]]


@register_canonicalize
@node_rewriter(
    [
        elemwise_of(OpPattern(ScalarOp, output_types_preference=upgrade_to_float)),
        elemwise_of(OpPattern(ScalarOp, output_types_preference=upcast_out)),
    ]
)
def local_upcast_elemwise_constant_inputs(fgraph, node):
    """This explicitly upcasts constant inputs to elemwise Ops, when
    those Ops do implicit upcasting anyway.

    Rationale: it helps merge things like (1-x) and (1.0 - x).

    """
    if len(node.outputs) > 1:
        return None

    # this is the kind of op that we can screw with the input
    # dtypes by upcasting explicitly
    [old_out] = node.outputs
    output_dtype = old_out.type.dtype
    new_inputs = list(node.inputs)
    changed = False
    for i, inp in enumerate(node.inputs):
        if inp.type.dtype != output_dtype and isinstance(inp, TensorConstant):
            new_inputs[i] = tensor_constant(inp.data.astype(output_dtype))
            changed = True

    if not changed:
        return None

    rval = node.op(*new_inputs)
    if not old_out.type.is_super(rval.type):
        # This can happen for example when floatX=float32
        # and we do the true division between and int64
        # and a constant that will get typed as int8.
        # As this is just to allow merging more case, if
        # the upcast don't work, we can just skip it.
        return None

    # Copy over output stacktrace from before upcasting
    copy_stack_trace(old_out, rval)
    return [rval]


@node_rewriter([add, mul])
def flatten_nested_add_mul(fgraph, node):
    """Fuse consecutive add or mul in one such node with more inputs.

    It is better to fuse add/mul that way then in a Composite node as
    this make the inner graph of the Composite smaller. This allows to
    put more computation in a Composite before hitting the max
    recursion limit when pickling Composite.

    This rewrite is almost useless after the AlgebraicCanonizer is used,
    but it catches a few edge cases that are not canonicalized by it
    """
    s_op = node.op.scalar_op
    new_inp = []
    fused = False
    for inp in node.inputs:
        if (
            inp.owner
            and isinstance(inp.owner.op, Elemwise)
            and inp.owner.op.scalar_op == s_op
            # Do not duplicate the operation.
            and len(fgraph.clients[inp]) == 1
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
            output2 = flatten_nested_add_mul.transform(fgraph, output.owner)
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
        replacement = {
            inp: get_scalar_type(inp.type.dtype).make_variable() for inp in inputs
        }
        for node in toposort(outputs, blockers=inputs):
            scalar_inputs = [replacement[inp] for inp in node.inputs]
            replacement.update(
                dict(
                    zip(
                        node.outputs,
                        node.op.scalar_op.make_node(*scalar_inputs).outputs,
                    )
                )
            )

        return (
            [replacement[inp] for inp in inputs],
            [replacement[out] for out in outputs],
        )

    def apply(self, fgraph):
        if fgraph.profile:
            validate_before = fgraph.profile.validate_time
            callbacks_before = fgraph.execute_callbacks_times.copy()
            callback_before = fgraph.execute_callbacks_time

        def find_fuseable_subgraphs(
            fg: FunctionGraph,
        ) -> Generator[tuple[tuple[Variable], tuple[Variable]], None, None]:
            """Find subgraphs of Elemwise nodes that can be fused together.

            In general, there is no single solution. We try to find large subgraphs eagerly

            Any two consecutive Elemwise nodes that have the same broadcasting pattern,
            and a C-implementation (historical accident that should be revisited), are potentially fuseable.

            However, not all collections of fuseable pairs make a valid fused subgraph.
            A valid fused subgraph must be "convex", meaning that no two nodes in the subgraph
            are connected via a path that goes outside the subgraph, either because they
            are connected via unfuseable nodes, or nodes that have been claimed by another fused subgraph.

            For example the subgraph add(sin(exp(x)), sum(exp(x)) cannot be fused together,
            because the sum node breaks the convexity of the subgraph {exp, sin, add}.
            However, we can fuse {exp, sin}, and perhaps fuse add with something else.

            This function yields subgraph in reverse topological order so they can be safely replaced one at a time
            """

            @cache
            def elemwise_scalar_op_has_c_code(
                node: Apply, optimizer_verbose=config.optimizer_verbose
            ) -> bool:
                # TODO: This should not play a role in non-c backends!
                if node.op.scalar_op.supports_c_code(node.inputs, node.outputs):
                    return True
                elif optimizer_verbose:
                    warn(
                        f"Loop fusion interrupted because {node.op.scalar_op} does not provide a C implementation."
                    )
                return False

            # Create a map from node to a set of fuseable client (successor) nodes
            # A node and a client are fuseable if they are both single output Elemwise
            # (with C-implementation) and have the same output broadcastable pattern
            # Nodes that have no fuseable clients are not included
            fuseable_clients: dict[Apply, set[Apply]] = {}
            # We also create a set with candidate nodes from which to start a subgraph expansion
            # These are Single output Elemwise nodes (with C-implementation) that may or not
            # have fuseable ancestors/clients at the start.
            candidate_starting_nodes = set()
            fg_clients = fg.clients
            for out, clients_and_indices in fg_clients.items():
                out_node = out.owner

                if not (
                    out_node is not None
                    and len(out_node.outputs) == 1
                    and isinstance(out_node.op, Elemwise)
                    and elemwise_scalar_op_has_c_code(out_node)
                ):
                    continue

                candidate_starting_nodes.add(out_node)
                out_bcast = out.type.broadcastable
                out_fuseable_clients = {
                    client
                    for client, _ in clients_and_indices
                    if (
                        len(client.outputs) == 1
                        and isinstance(client.op, Elemwise)
                        and out_bcast == client.outputs[0].type.broadcastable
                        and elemwise_scalar_op_has_c_code(client)
                    )
                }
                if out_fuseable_clients:
                    fuseable_clients[out_node] = out_fuseable_clients

            if not candidate_starting_nodes:
                return None

            # To enable fast dependency queries, we create a bitset of ancestors for each node.
            # Each node is first represented by a bit flag of it's position in the toposort
            # This can be achieved with python integers, via 1 << toposort_idx (equivalent to slower 2 ** toposort_idx)
            # The ancestors bitsets of each node are obtained by bitwise OR of the ancestor bitsets
            # of each of the nodes' inputs, and the bit flag of the node itself.
            #
            # Example: With three variables {a, b, c} owned by nodes {A, B, C}, where a is an input of b, and b an input of c,
            # the nodes bit flags would be {A: 0b001, B: 0b010, C: 0b100} (integers {A: 1, B: 2, C: 4})
            # and the ancestors bitset would be {A: 0b001, B: 0b011, C: 0b111} (integers {A: 1, B: 3, C: 7})
            #
            # This allows us to quickly ask if one or more variables are ancestors of a node by a simple bitwise AND
            # For example, to ask if A is an ancestor of C we can do `ancestors_bitset[C] & node_bitset[A] != 0`
            # We can also easily handle multiple nodes at once, for example to ask if A or B are ancestors of C we can do
            # `ancestors_bitset[C] & (node_bitset[A] | node_bitset[B]) != 0`
            nodes_bitflags = {node: 1 << i for i, node in enumerate(fgraph.toposort())}
            # Root variables have `None` as owner, which we can handle with a bitset of 0
            ancestors_bitsets: dict[Apply | None, int] = {None: 0}
            for node, node_bitflag in nodes_bitflags.items():
                # The bitset of each node is the union of the bitsets of its inputs, plus its own bit flag
                ancestors_bitsets[node] = reduce(
                    or_,
                    (ancestors_bitsets[inp.owner] for inp in node.inputs),
                    node_bitflag,
                )
            # Handle root and leaf nodes gracefully
            # We do it after the ancestors_bitset are built to simplify the previous loop.
            # Root variables have `None` as owner, which we can handle with a bitflag of 0
            nodes_bitflags[None] = 0
            # Nothing ever depends on the special Output nodes, so just use a new bit for all of them
            out_bitflag = 1 << len(nodes_bitflags)
            nodes_bitflags |= (
                (client, out_bitflag)
                for out in fg.outputs
                for client, _ in fg_clients[out]
                if isinstance(client.op, Output)
            )

            # Start main loop to find collection of fuseable subgraphs
            # We store the collection in `sorted_subgraphs`, in reverse topological order
            sorted_subgraphs: list[
                tuple[int, tuple[tuple[Variable], tuple[Variable]]]
            ] = []
            # Keep a bitset of nodes that have been claimed by subgraphs
            all_subgraphs_bitset = 0
            # Start exploring in reverse topological order from candidate sink nodes
            # Sink nodes, are nodes that don't have any potential fuseable clients
            for starting_node, starting_bitflag in reversed(nodes_bitflags.items()):
                if (
                    starting_bitflag & all_subgraphs_bitset
                    or starting_node not in candidate_starting_nodes
                    or starting_node in fuseable_clients
                ):
                    continue

                # We use an ordered queue to control the direction in which we expand the subgraph
                # For simplicity, we always want to visit ancestors before clients
                # For ancestors, we want to visit the later nodes first (those that have more dependencies)
                # whereas for clients we want to visit earlier nodes first (those that have fewer dependencies)
                # To achieve this we use the ancestors_bitset as the sorting key (which encodes the topological order)
                # and negate it for ancestors. We use the ancestors_bitset instead of the node bitflag because we
                # update the former when we find a fuseable subgraph, emulating the effect of recomputing the
                # topological order on the remaining nodes.
                fuseables_nodes_queue = [
                    (-ancestors_bitsets[starting_node], starting_bitflag, starting_node)
                ]
                heapify(fuseables_nodes_queue)

                # We keep 3 bitsets during the exploration of a new subgraph:
                #  - the nodes that are part of the subgraph
                #  - the unfuseable ancestors of the subgraph (i.e., ancestors that are not fuseable with a node in the subgraph)
                #  - the unfuseable clients of the subgraph (i.e., clients that are not fuseable with a node in the subgraph)
                # Whenever we visit a candidate node, we check if the subgraph's unfuseable ancestors depend on it,
                # or if it depends on one of the subgraphs' unfuseable client, in which case we can't add it.
                # If we can add it, we then add its unfuseable ancestors/clients to the respective bitsets
                # and add its fuseable ancestors/clients to the queue to explore later.
                # To work correctly, we must visit candidate subgraph nodes in the order described by the queue above.
                # Otherwise, we would need to perform more complex dependency checks in every iteration and/or backtrack.
                subgraph_nodes = []
                subgraph_bitset = 0
                unfuseable_ancestors_bitset = 0
                unfuseable_clients_bitset = 0

                while fuseables_nodes_queue:
                    node_ancestors_bitset, node_bitflag, node = heappop(
                        fuseables_nodes_queue
                    )
                    is_ancestor = node_ancestors_bitset < 0
                    if is_ancestor:
                        node_ancestors_bitset = -node_ancestors_bitset

                    if node_bitflag & subgraph_bitset:
                        # Already part of the subgraph
                        continue

                    if is_ancestor:
                        if node_bitflag & unfuseable_ancestors_bitset:
                            # An unfuseable ancestor of the subgraph depends on this node, can't fuse
                            continue
                    elif node_ancestors_bitset & unfuseable_clients_bitset:
                        # This node depends on an unfuseable client of the subgraph, can't fuse
                        continue

                    # Add node to subgraph
                    subgraph_nodes.append(node)
                    subgraph_bitset |= node_bitflag

                    # Expand through ancestors and client nodes
                    # A node can either be:
                    #  - already part of the subgraph (skip)
                    #  - fuseable (add to queue)
                    #  - unfuseable (add to respective unfuseable bitset)
                    for inp in node.inputs:
                        ancestor_node = inp.owner
                        ancestor_bitflag = nodes_bitflags[ancestor_node]
                        if (not is_ancestor) and (ancestor_bitflag & subgraph_bitset):
                            continue
                        if node in fuseable_clients.get(ancestor_node, ()):
                            heappush(
                                fuseables_nodes_queue,
                                (
                                    -ancestors_bitsets[ancestor_node],
                                    ancestor_bitflag,
                                    ancestor_node,
                                ),
                            )
                        else:
                            # If the node is not in the ancestor's fuseable clients set, it's not fuseable with it,
                            # nor with any of the ancestor's ancestors
                            unfuseable_ancestors_bitset |= ancestors_bitsets[
                                ancestor_node
                            ]

                    next_fuseable_clients = fuseable_clients.get(node, ())
                    for client, _ in fg_clients[node.outputs[0]]:
                        client_bitflag = nodes_bitflags[client]
                        if is_ancestor and (client_bitflag & subgraph_bitset):
                            continue
                        if client in next_fuseable_clients:
                            heappush(
                                fuseables_nodes_queue,
                                (ancestors_bitsets[client], client_bitflag, client),
                            )
                        else:
                            # If a client is not in the node's fuseable clients set, it's nto fuseable with it,
                            # nor any of its clients. But we don't need to keep track of those as any downstream
                            # client we may consider later will also depend on this unfuseable client and be rejected
                            unfuseable_clients_bitset |= client_bitflag

                # Finished expansion of subgraph
                if subgraph_bitset == starting_bitflag:
                    # We ended were we started, no fusion possible
                    continue

                # Find out the actual inputs/outputs variables of the subgraph
                not_subgraph_bitset = ~subgraph_bitset
                # Inputs are variables whose nodes are not part of the subgraph (including root variables without nodes)
                # Use a dict to deduplicate while preserving order
                subgraph_inputs = tuple(
                    dict.fromkeys(
                        inp
                        for node in subgraph_nodes
                        for inp in node.inputs
                        if (inp_node := inp.owner) is None
                        or nodes_bitflags[inp_node] & not_subgraph_bitset
                    )
                )
                # Outputs are variables with client nodes that are not part of the subgraph (including special fgraph output nodes)
                # Outputs are unique, no need to deduplicate
                subgraph_outputs = tuple(
                    node.outputs[0]
                    for node in subgraph_nodes
                    if any(
                        nodes_bitflags[client] & not_subgraph_bitset
                        for client, _ in fg_clients[node.outputs[0]]
                    )
                )

                # Update fuseable clients mapping for subgraph inputs and outputs
                # Inputs cannot be fused with nodes in the subgraph
                for inp in subgraph_inputs:
                    if (inp_node := inp.owner) is not None and (
                        inp_fuseable_clients := fuseable_clients.get(inp_node)
                    ):
                        inp_fuseable_clients.difference_update(subgraph_nodes)
                        # If there are no fuseable_clients left for this input delete it's entry
                        if not inp_fuseable_clients:
                            del fuseable_clients[inp_node]
                # Outputs cannot be fused with anything else
                for out in subgraph_outputs:
                    fuseable_clients.pop(out.owner, None)

                # When we fuse multi-output subgraphs, we also need to fuse the dependencies of successor nodes.
                # Nodes that previously depended on a subset of the fused outputs, now depend on all of them.
                if len(subgraph_outputs) > 1:
                    subgraph_and_ancestors = (
                        subgraph_bitset | unfuseable_ancestors_bitset
                    )
                    ancestors_bitsets |= (
                        (node, node_ancestors_bitset | subgraph_and_ancestors)
                        for node, node_ancestors_bitset in ancestors_bitsets.items()
                        if node_ancestors_bitset & subgraph_bitset
                    )

                # Add new subgraph to sorted_subgraphs
                # Because we start from sink nodes in reverse topological order, most times new subgraphs
                # don't depend on previous subgraphs, so we can just append them at the end.
                if not (unfuseable_ancestors_bitset & all_subgraphs_bitset):
                    # That's the case here
                    # None of the unfuseable_ancestors (i.e, the ancestors) are present in the previous collected subgraphs
                    sorted_subgraphs.append(
                        (subgraph_bitset, (subgraph_inputs, subgraph_outputs))
                    )
                else:
                    # But not here, so we need to find the right position for insertion.
                    # We iterate through the previous subgraphs in topological order (reverse of the stored order).
                    # We cumulatively exclude each subgraph_bitset and perform the same dependency check again, until it passes.
                    remaining_subgraphs_bitset = all_subgraphs_bitset
                    for index, (other_subgraph_bitset, _) in enumerate(
                        reversed(sorted_subgraphs)
                    ):
                        # Exclude subgraph bitset
                        remaining_subgraphs_bitset &= ~other_subgraph_bitset
                        if not (
                            unfuseable_ancestors_bitset & remaining_subgraphs_bitset
                        ):
                            break  # bingo
                    else:  # no-break
                        raise RuntimeError(
                            "Failed to find insertion point for fused subgraph"
                        )
                    sorted_subgraphs.insert(
                        -(index + 1),
                        (subgraph_bitset, (subgraph_inputs, subgraph_outputs)),
                    )

                # Add subgraph to all_subgraphs_bitset
                all_subgraphs_bitset |= subgraph_bitset

            # Finished exploring the whole graph
            # Yield from sorted_subgraphs, discarding the subgraph_bitset
            yield from (io for _, io in sorted_subgraphs)

        max_operands = elemwise_max_operands_fct(None)
        reason = self.__class__.__name__
        nb_fused = 0
        nb_replacement = 0
        for inputs, outputs in find_fuseable_subgraphs(fgraph):
            if (len(inputs) + len(outputs)) > max_operands:
                warn(
                    "Loop fusion failed because the resulting node would exceed the kernel argument limit."
                )
                continue

            scalar_inputs, scalar_outputs = self.elemwise_to_scalar(inputs, outputs)
            composite_outputs = Elemwise(
                # No need to clone Composite graph, because `self.elemwise_to_scalar` creates fresh variables
                Composite(scalar_inputs, scalar_outputs, clone_graph=False)
            )(*inputs, return_list=True)
            assert len(outputs) == len(composite_outputs)
            for old_out, composite_out in zip(outputs, composite_outputs):
                # Preserve any names on the original outputs
                if old_name := old_out.name:
                    composite_out.name = old_name

            starting_nodes = len(fgraph.apply_nodes)
            fgraph.replace_all_validate(
                tuple(zip(outputs, composite_outputs)),
                reason=reason,
            )
            nb_fused += 1
            nb_replacement += (starting_nodes - len(fgraph.apply_nodes)) + 1

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
            nb_fused,
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
        print(blanc, " nb_fused", prof[1], file=stream)
        print(blanc, " nb_replacement", prof[2], file=stream)
        print(blanc, " nb_inconsistency_replace", prof[3], file=stream)
        print(blanc, " validate_time", prof[4], file=stream)
        print(blanc, " callback_time", prof[5], file=stream)
        if prof[5] is not None and prof[5] > 1:
            print(blanc, " callbacks_time", file=stream)
            for i in sorted(prof[6].items(), key=lambda a: a[1])[::-1]:
                if i[1] > 0:
                    print(blanc, "     ", i)  # noqa: T201
        print(blanc, " time_toposort", prof[7], file=stream)


@register_canonicalize
@register_specialize
@node_rewriter([elemwise_of(Composite)])
def local_useless_composite_outputs(fgraph, node):
    """Remove inputs and outputs of Composite Ops that are not used anywhere."""
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
        c = Composite(inputs=used_inner_inputs, outputs=used_inner_outputs)
        e = Elemwise(scalar_op=c)(*used_inputs, return_list=True)
        return dict(zip([node.outputs[i] for i in used_outputs_idxs], e, strict=True))


@node_rewriter([CAReduce])
def local_careduce_fusion(fgraph, node):
    """Fuse a `CAReduce` applied to an `Elemwise`."""

    (car_input,) = node.inputs
    car_scalar_op = node.op.scalar_op

    # FIXME: This check is needed because of the faulty logic in the FIXME below!
    # Right now, rewrite only works for `Sum`/`Prod`
    if not isinstance(car_scalar_op, Add | Mul):
        return None

    elm_node = car_input.owner

    if not (elm_node and isinstance(elm_node.op, Elemwise)):
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
        get_scalar_type(inp.type.dtype).make_variable() for inp in elm_inputs
    ]

    elm_output = elm_scalar_op(*scalar_elm_inputs)

    # This input represents the previous value in the `CAReduce` binary reduction
    carried_car_input = get_scalar_type(car_acc_dtype).make_variable()

    scalar_fused_output = car_scalar_op(carried_car_input, elm_output)
    if scalar_fused_output.type.dtype != car_acc_dtype:
        scalar_fused_output = scalar_cast(scalar_fused_output, car_acc_dtype)

    fused_scalar_op = Composite(
        inputs=[carried_car_input, *scalar_elm_inputs], outputs=[scalar_fused_output]
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


@node_rewriter([elemwise_of(Composite)])
def local_inline_composite_constants(fgraph, node):
    """Inline scalar constants in Composite graphs."""
    composite_op = node.op.scalar_op
    new_outer_inputs = []
    new_inner_inputs = []
    inner_replacements = {}
    for outer_inp, inner_inp in zip(
        node.inputs, composite_op.fgraph.inputs, strict=True
    ):
        # Complex variables don't have a `c_literal` that can be inlined
        if (
            isinstance(outer_inp, TensorConstant)
            and "complex" not in outer_inp.type.dtype
        ):
            if outer_inp.unique_value is not None:
                inner_replacements[inner_inp] = scalar_constant(
                    outer_inp.unique_value, dtype=inner_inp.dtype
                )
                continue
        new_outer_inputs.append(outer_inp)
        new_inner_inputs.append(inner_inp)

    if not inner_replacements:
        return None

    new_inner_outs = clone_replace(
        composite_op.fgraph.outputs, replace=inner_replacements
    )
    new_composite_op = Composite(new_inner_inputs, new_inner_outs)
    new_outputs = Elemwise(new_composite_op).make_node(*new_outer_inputs).outputs

    # Some of the inlined constants were broadcasting the output shape
    if node.outputs[0].type.broadcastable != new_outputs[0].type.broadcastable:
        new_outputs = [
            alloc_like(new_out, template=node.outputs[0], fgraph=fgraph)
            for new_out in new_outputs
        ]

    copy_stack_trace(node.outputs, new_outputs)
    return new_outputs


@node_rewriter(tracks=[add, mul])
def constant_fold_branches_of_add_mul(fgraph, node):
    old_constants = [inp for inp in node.inputs if isinstance(inp, TensorConstant)]

    if len(old_constants) <= 1:
        return None

    new_constants = old_constants.copy()

    # Multiply constants if it doesn't result in higher intermediate memory
    while True:
        n_constants = len(new_constants)
        if n_constants <= 1:
            break

        for i in range(n_constants):
            reference_inp = new_constants[i]
            other_inps = []
            for j in range(n_constants):
                if i == j:
                    continue
                other_inp = new_constants[j]
                if not broadcasted_by(reference_inp, other_inp):
                    other_inps.append(other_inp)
            if other_inps:
                python_op = operator.mul if node.op == mul else operator.add
                folded_inputs = [reference_inp, *other_inps]
                new_inp = tensor_constant(
                    reduce(python_op, (const.data for const in folded_inputs))
                )
                new_constants = [
                    new_inp,
                    *(inp for inp in new_constants if inp not in folded_inputs),
                ]
                break
        else:  # no-break
            break

    if len(new_constants) == len(old_constants):
        return None

    non_constants = [inp for inp in node.inputs if not isinstance(inp, TensorConstant)]
    new_out = node.op(
        *new_constants,
        *non_constants,
    )
    copy_stack_trace(node.outputs[0], new_out)
    return [new_out]


add_mul_fusion_seqopt = SequenceDB()
optdb.register(
    "add_mul_fusion",
    add_mul_fusion_seqopt,
    "fast_run",
    position=48,  # Before Elemwise fusion
)
add_mul_fusion_seqopt.register(
    flatten_nested_add_mul.__name__,
    out2in(flatten_nested_add_mul, ignore_newtrees=False),
    "fast_run",
    position=0,
)
add_mul_fusion_seqopt.register(
    constant_fold_branches_of_add_mul.__name__,
    in2out(constant_fold_branches_of_add_mul, ignore_newtrees=True),
    "fast_run",
    position=1,
)

# Register fusion database just before AddDestroyHandler(49.5) (inplace rewrites)
fuse_seqopt = SequenceDB()
optdb.register(
    "elemwise_fusion",
    fuse_seqopt,
    "fast_run",
    "fusion",
    "local_elemwise_fusion",
    "FusionOptimizer",
    position=49,
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
    dfs_rewriter(local_useless_composite_outputs),
    "fast_run",
    "fusion",
    position=2,
)
fuse_seqopt.register(
    "local_careduce_fusion",
    dfs_rewriter(local_careduce_fusion),
    "fast_run",
    "fusion",
    position=10,
)
fuse_seqopt.register(
    "local_inline_composite_constants",
    dfs_rewriter(local_inline_composite_constants, ignore_newtrees=True),
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
@node_rewriter([elemwise_of(Grad2F1Loop)])
def local_useless_2f1grad_loop(fgraph, node):
    # Remove unused terms from the hyp2f1 grad loop
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
    for grad_var, is_used in zip(grad_vars, grad_var_is_used, strict=True):
        if not is_used:
            continue
        replacements[grad_var] = new_outs[i]
        i += 1
    return replacements


@node_rewriter([elemwise_of(Grad2F1Loop)])
def split_2f1grad_loop(fgraph, node):
    """
    2f1grad loop has too many operands for Numpy frompyfunc code used by Elemwise nodes on python mode.

    This rewrite splits it across 3 different operations. It is not needed if `local_useless_2f1grad_loop` was applied
    """
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


optdb["py_only"].register(
    "split_2f1grad_loop",
    split_2f1grad_loop,
    "fast_compile",
)
