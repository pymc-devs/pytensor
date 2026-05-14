"""In-place pass for Scan: rewrite outer inputs to be destroyed in place.

The :class:`ScanInplaceOptimizer` walks every ``Scan`` in the graph and
marks the eligible mit_mot / mit_sot / sit_sot buffer inputs as
destroyable, mirroring the behavior any other in-place rewriter would
have if it understood Scan's input categories.
"""

from itertools import chain

from pytensor.compile.ops import deep_copy_op
from pytensor.graph.basic import Apply
from pytensor.graph.destroyhandler import DestroyHandler
from pytensor.graph.features import ReplaceValidate
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import GraphRewriter
from pytensor.graph.utils import InconsistencyError
from pytensor.scan.op import Scan
from pytensor.tensor.basic import Alloc, AllocEmpty


class ScanInplaceOptimizer(GraphRewriter):
    """Make `Scan`s perform in-place.

    This optimization attempts to make `Scan` compute its recurrent outputs inplace
    on the input tensors that contain their initial states. This optimization can
    improve runtime performance as well as reduce memory usage.

    """

    alloc_ops = (Alloc, AllocEmpty)
    """
    Classes that represent operation that allocate new memory and that the
    optimization should duplicate so it can operate inplace on them.
    """

    def add_requirements(self, fgraph):
        fgraph.attach_feature(ReplaceValidate())
        fgraph.attach_feature(DestroyHandler())

    def attempt_scan_inplace(
        self, fgraph: FunctionGraph, node: Apply[Scan], output_indices: list[int]
    ) -> Apply | None:
        """Attempt to replace a `Scan` node by one which computes the specified outputs inplace.

        Parameters
        ----------
        fgraph
            Function graph in which to attempt the replacement
        node
            Scan node to replace by an inplace version
        output_indices
            Indices of the outputs to attempt to compute inplace.
        """

        op = node.op
        n_tap_outs = op.info.n_mit_mot + op.info.n_mit_sot + op.info.n_sit_sot
        untraced_out_start = n_tap_outs + op.info.n_nit_sot

        # inputs corresponding to sequences and n_steps
        ls_begin = node.inputs[: 1 + op.info.n_seqs]
        ls = op.outer_mitmot(node.inputs)
        ls += op.outer_mitsot(node.inputs)
        ls += op.outer_sitsot(node.inputs)
        ls_end = op.outer_untraced_sit_sot(node.inputs)
        ls_end += op.outer_nitsot(node.inputs)
        ls_end += op.outer_non_seqs(node.inputs)

        # In `ls`, duplicate any input which has more than one client and is
        # the output of an eligible allocation op
        for i in range(len(ls)):
            inp = ls[i]
            if (
                len(fgraph.clients[inp]) > 1
                and inp.owner is not None
                and isinstance(inp.owner.op, self.alloc_ops)
            ):
                [new_lsi_out] = inp.owner.op.make_node(*inp.owner.inputs).outputs
                ls[i] = new_lsi_out

        n_outs = len(ls)
        for idx in range(n_outs):
            if ls[idx] in ls[:idx]:
                ls[idx] = deep_copy_op(ls[idx])

        inputs = ls_begin + ls + ls_end

        new_op = op.clone()

        destroy_map = op.destroy_map.copy()
        for out_idx in output_indices:
            if out_idx < n_tap_outs:
                # Recurrent output: input is at position out_idx + 1 + n_seqs
                destroy_map[out_idx] = [out_idx + 1 + op.info.n_seqs]
            else:
                # Untraced sit_sot output: input is at untraced_sit_sot_arg_offset + j
                j = out_idx - untraced_out_start
                destroy_map[out_idx] = [op.untraced_sit_sot_arg_offset + j]

        new_op.destroy_map = destroy_map

        # Remove view_map entries for outputs that are now in destroy_map
        if hasattr(new_op, "view_map"):
            new_op.view_map = {
                k: v for k, v in new_op.view_map.items() if k not in destroy_map
            }

        new_node: Apply = new_op.make_node(*inputs)

        try:
            fgraph.replace_all_validate_remove(
                list(zip(node.outputs, new_node.outputs, strict=True)),
                remove=[node],
                reason="scan_make_inplace",
            )
            return new_node
        except InconsistencyError:
            # Failed moving output to be computed inplace
            return None

    def apply(self, fgraph):
        for original_node in reversed(fgraph.toposort()):
            if not isinstance(original_node.op, Scan):
                continue

            # First attempt to make the Scan eagerly compute inplace every output
            #  that seems like it could be computed inplace.
            # If that fails, go through these outputs individually, trying each one at a time.
            op = original_node.op
            n_tap_outs = op.info.n_mit_mot + op.info.n_mit_sot + op.info.n_sit_sot
            untraced_out_start = n_tap_outs + op.info.n_nit_sot

            # Generate a list of outputs on which the node could potentially
            # operate inplace: recurrent outputs and untraced_sit_sot outputs.
            candidate_out_indices = list(range(n_tap_outs)) + list(
                range(
                    untraced_out_start, untraced_out_start + op.info.n_untraced_sit_sot
                )
            )

            out_indices = []
            for out_idx in candidate_out_indices:
                if out_idx < n_tap_outs:
                    inp_idx = 1 + op.info.n_seqs + out_idx
                else:
                    j = out_idx - untraced_out_start
                    inp_idx = op.untraced_sit_sot_arg_offset + j

                inp = original_node.inputs[inp_idx]

                # If the input is from an eligible allocation node, attempt to
                # be inplace on it, even if other nodes are modifying it inplace.
                if inp.owner and isinstance(inp.owner.op, self.alloc_ops):
                    out_indices.append(out_idx)
                    continue

                # If the input is not from an eligible allocation node, only
                # attempt to be inplace on it if nothing else is currently inplace on it.
                input_used_inplace = False
                for c in fgraph.clients[inp]:
                    client = c[0]

                    # Get the indices of this client's inputs on which it operates inplace
                    if client.op.destroy_map:
                        # This flattens the content of destroy_map.values()
                        # which is a list of lists
                        inplace_inp_indices = chain.from_iterable(
                            client.op.destroy_map.values()
                        )

                        if inp in (client.inputs[i] for i in inplace_inp_indices):
                            input_used_inplace = True
                            break

                if not input_used_inplace:
                    out_indices.append(out_idx)

            if len(out_indices) == 0:
                continue

            new_node = self.attempt_scan_inplace(fgraph, original_node, out_indices)

            if new_node is None:
                # Making the scan compute all plausible recurrent outputs
                # inplace has failed. Attempt all plausible recurrent outputs individually.
                new_node = original_node
                for pos in out_indices:
                    new_node = (
                        self.attempt_scan_inplace(fgraph, new_node, [pos]) or new_node
                    )
