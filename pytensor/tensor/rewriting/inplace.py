import abc
import itertools
import sys
from collections import defaultdict
from collections.abc import Sequence

from pytensor.compile.aliasing import Supervisor
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.destroyhandler import DestroyHandler
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.graph.rewriting.basic import (
    GraphRewriter,
    copy_stack_trace,
)
from pytensor.graph.rewriting.reachability import (
    ancestor_bitsets,
)
from pytensor.graph.utils import InconsistencyError
from pytensor.tensor.variable import TensorVariable


def _root_and_readers(fgraph, var):
    """The variable `var` views (its root) and the nodes that read the root or a view of it."""
    view_i, view_o = fgraph.destroy_handler.view_i, fgraph.destroy_handler.view_o
    while var in view_i:
        var = view_i[var]
    tree, readers = [var], set()
    while tree:
        v = tree.pop()
        readers.update(c for c, _ in fgraph.clients[v])
        tree.extend(view_o.get(v, ()))
    return var, readers


def _descendants(fgraph, node):
    """Nodes that run after `node`, through data edges or the destroy orderings."""
    after = defaultdict(list)
    for n, before in fgraph.orderings().items():
        for b in before:
            after[b].append(n)
    seen, stack = set(), [node]
    while stack:
        n = stack.pop()
        for c in itertools.chain(
            (c for out in n.outputs for c, _ in fgraph.clients[out]), after[n]
        ):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


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

    def _get_protected_inputs(self, fgraph):
        """Collect inputs protected from in-place destruction."""
        protected = set(
            itertools.chain.from_iterable(
                f.protected for f in fgraph._features if isinstance(f, Supervisor)
            )
        )
        protected.update(fgraph.outputs)
        return protected

    def try_inplace_on_node(
        self,
        fgraph,
        node,
        candidate_pairs=None,
        reason="inplace_optimizer",
        extra_protected_inputs=frozenset(),
    ):
        """Try to make a single node operate in-place.

        First attempts all candidate pairs at once, then falls back to
        one-at-a-time. Returns the (possibly replaced) node.

        Parameters
        ----------
        candidate_pairs
            Pre-filtered and sorted list of ``((out_idx, out), (in_idx, inp))``
            pairs. If None, computed from ``filter_candidate_pairs`` using
            the standard protections plus ``extra_protected_inputs``.
        extra_protected_inputs
            Additional variables that must not be used as in-place targets,
            only used when ``candidate_pairs`` is None.
        """
        if candidate_pairs is None:
            protected_inputs = self._get_protected_inputs(fgraph)
            protected_inputs.update(extra_protected_inputs)
            candidate_pairs = self.filter_candidate_pairs(
                fgraph, node, protected_inputs
            )
        candidate_pairs = [
            p for p in candidate_pairs if not fgraph.has_destroyers([p[1][1]])
        ]
        if len(candidate_pairs) > 1:
            # Drop pairs whose destroyed memory is read after the node (they'd form a cycle),
            # walking the graph only if the memory has other readers
            readers = [
                _root_and_readers(fgraph, p[1][1])[1] - {node} for p in candidate_pairs
            ]
            if any(readers):
                after = _descendants(fgraph, node)
                candidate_pairs = [
                    p for p, r in zip(candidate_pairs, readers) if after.isdisjoint(r)
                ]
        if not candidate_pairs:
            return node

        failed_pattern = None
        # Try in-placing all outputs at once
        tried_inputs = set()
        inplace_pattern = {}
        for (o, _), (i, _) in candidate_pairs:
            if o not in inplace_pattern and i not in tried_inputs:
                inplace_pattern[o] = [i]
                tried_inputs.add(i)

        inplace_node = self.create_inplace_node(node, inplace_pattern)
        if inplace_node.op.destroy_map == inplace_pattern:
            replacements = tuple(zip(node.outputs, inplace_node.outputs))
            try:
                fgraph.replace_all_validate(replacements, reason=reason)
            except InconsistencyError:
                failed_pattern = inplace_pattern
            else:
                copy_stack_trace(node.outputs, inplace_node.outputs)
                return inplace_node

        # Fall back to one output/input at a time
        tried_inputs = set()
        inplace_pattern = {}
        original_node = node
        for (o, _), (i, _) in candidate_pairs:
            if o not in inplace_pattern and i not in tried_inputs:
                inplace_pattern[o] = [i]
                tried_inputs.add(i)

                if node is original_node and inplace_pattern == failed_pattern:
                    inplace_pattern.pop(o)
                    tried_inputs.discard(i)
                    continue

                inplace_node = self.create_inplace_node(node, inplace_pattern)
                if inplace_node.op.destroy_map != inplace_pattern:
                    # This Op can't respect this partial inplace pattern,
                    # We assume it can't support any other cases
                    break
                replacements = tuple(zip(node.outputs, inplace_node.outputs))
                try:
                    fgraph.replace_all_validate(replacements, reason=reason)
                    node = inplace_node
                except InconsistencyError:
                    inplace_pattern.pop(o)
                    # The input wasn't consumed, so let another output try to
                    # reuse it in place.
                    tried_inputs.discard(i)
        if node is not original_node:
            copy_stack_trace(original_node.outputs, node.outputs)
        return node

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
        reason = f"{self.op}_inplace_optimizer"
        prof = {
            "opt": self,
            "node_before": len(fgraph.apply_nodes),
            "nb_replaced": 0,
        }

        protected_inputs = self._get_protected_inputs(fgraph)
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

        nodes = fgraph.toposort()
        ancestors, bitflags = ancestor_bitsets(fgraph, nodes)

        # Validating each in-placing on its own costs a whole-graph cycle check each.
        # Pairs that would surely form a cycle (another reader of the destroyed memory
        # depends on the node) are dropped up front, the rest validated in batches.
        claimed, plan = set(), []
        for node in nodes:
            if not isinstance(node.op, self_op) or node.op.destroy_map:
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
                    ((_o, out), (_i, inp)) = pair
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

            flag, viable, pattern, roots, contended = (
                bitflags[node],
                [],
                {},
                set(),
                False,
            )
            for pair in sorted_candidate_pairs:
                (o, _), (i, inp) = pair
                root, readers = _root_and_readers(fgraph, inp)
                if any(c is not node and ancestors.get(c, 0) & flag for c in readers):
                    continue
                viable.append(pair)
                contended |= root in claimed
                if (
                    root not in claimed | roots
                    and o not in pattern
                    and i not in pattern.values()
                ):
                    pattern[o] = i
                    roots.add(root)
            # a node contending for memory an earlier node claimed is tried on its own, in order
            if viable:
                plan.append((node, viable, None if contended else pattern))
                claimed |= roots if not contended else set()

        def try_batch(batch):
            new_nodes = [
                self.create_inplace_node(n, {o: [i] for o, i in p.items()})
                for n, _, p in batch
            ]
            if any(
                new.op.destroy_map != {o: [i] for o, i in p.items()}
                for new, (_, _, p) in zip(new_nodes, batch)
            ):
                return False
            # In reverse topological order, so a node is replaced after its clients
            replacements = [
                pair
                for new, (n, _, _) in reversed(list(zip(new_nodes, batch)))
                for pair in zip(n.outputs, new.outputs)
            ]
            try:
                fgraph.replace_all_validate(replacements, reason=reason)
            except InconsistencyError:
                return False
            for new, (n, _, _) in zip(new_nodes, batch):
                copy_stack_trace(n.outputs, new.outputs)
            prof["nb_replaced"] += len(batch)
            return True

        def try_one(node, viable):
            # node.inputs, not the pairs' variables: earlier in-placings replace producers
            viable = [((o, out), (i, node.inputs[i])) for (o, out), (i, _) in viable]
            if self.try_inplace_on_node(fgraph, node, viable, reason) is not node:
                prof["nb_replaced"] += 1

        def run(batch):
            if len(batch) == 1:
                try_one(*batch[0][:2])
            elif batch and not try_batch(batch):
                run(batch[: len(batch) // 2])
                run(batch[len(batch) // 2 :])

        batch = []
        for node, viable, pattern in plan:
            if pattern:
                batch.append((node, viable, pattern))
            else:
                run(batch)
                batch = []
                try_one(node, viable)
        run(batch)
        return prof

    @classmethod
    def print_profile(cls, stream, prof, level=0):
        blanc = "    " * level
        print(blanc, cls.__name__, file=stream)
        for k in [
            "node_before",
            "nb_replaced",
        ]:
            print(blanc, k, prof[k], file=stream)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(
            f"{' ' * level}{self.__class__.__name__}",
            file=stream,
        )
