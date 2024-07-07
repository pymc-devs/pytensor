from typing import Optional

from pytensor.compile.builders import OpFromGraph
from pytensor.graph.basic import Apply, io_toposort
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.basic import NodeRewriter, WalkingGraphRewriter
from pytensor.scan.op import Scan


class WalkingNestedGraphRewriter(WalkingGraphRewriter):
    def process_node(
        self,
        fgraph: FunctionGraph,
        node: Apply,
        node_rewriter: Optional[NodeRewriter] = None,
    ):
        self.node_rewriter = node_rewriter or self.node_rewriter
        if isinstance(node.op, (Scan, OpFromGraph)):
            return self.process_scan_node(fgraph, node)
        else:
            return super().process_node(fgraph, node)

    def process_scan_node(self, fgraph: FunctionGraph, node: Apply):
        try:
            replacements = self.transform_scan_node(fgraph, node)
        except Exception as e:
            if self.failure_callback is not None:
                self.failure_callback(
                    e,
                    self,
                    [(x, None) for x in node.outputs],
                    self.node_rewriter,  # type: ignore
                    node,
                )
                return False
            else:
                raise
        if replacements is False or replacements is None:
            return False

        repl_pairs = zip(node.outputs, replacements)
        try:
            fgraph.replace_all_validate_remove(  # type: ignore
                repl_pairs,
                reason=self.node_rewriter,
                remove=[],
            )
            return True
        except Exception as e:
            # This means the replacements were rejected by the fgraph.
            #
            # This is not supposed to happen.  The default failure_callback
            # will print a traceback as a warning.
            if self.failure_callback is not None:
                self.failure_callback(
                    e,
                    self,
                    repl_pairs,  # type: ignore
                    self.node_rewriter,  # type: ignore
                    node,
                )
                return False
            else:
                raise

    def transform_scan_node(self, fgraph, node):
        node_inputs, node_outputs = node.op.inner_inputs, node.op.inner_outputs
        local_fgraph_topo = io_toposort(node_inputs, node_outputs)
        op = node.op

        givens = dict()
        to_remove_set = set()
        for nd in local_fgraph_topo:
            if nd not in to_remove_set:
                if isinstance(nd.op, (Scan, OpFromGraph)):
                    [new_node] = self.transform_scan_node(node.op.fgraph, nd)
                    if new_node is not None:
                        givens.update(zip(nd.outputs, new_node.owner.outputs))
                        to_remove_set.add(nd)
                else:
                    replacements = self.node_rewriter.transform(node.op.fgraph, nd)
                    if replacements is False or replacements is None:
                        pass
                    elif not isinstance(replacements, (tuple, list, dict)):
                        raise TypeError(
                            f"Node rewriter {self.node_rewriter} gave wrong type of replacement. "
                            f"Expected list, tuple or dict; got {replacements}"
                        )
                    elif isinstance(replacements, (list, tuple)):
                        if len(nd.outputs) != len(replacements):
                            raise ValueError(
                                f"Node rewriter {self.node_rewriter} gave wrong number of replacements"
                            )
                        givens.update(zip(nd.outputs, replacements))
                        to_remove_set.add(nd)
                    elif isinstance(replacements, dict):
                        to_remove_set.add(nd)
                        for key, value in replacements.items():
                            if key == "remove":
                                for item in value:
                                    givens[item] = None
                            else:
                                givens[key] = value

        if len(to_remove_set) == 0:
            return None
        op_outs = clone_replace(node_outputs, replace=givens)
        if isinstance(op, Scan):
            nwScan = Scan(
                node_inputs,
                op_outs,
                op.info,
                mode=op.mode,
                profile=op.profile,
                truncate_gradient=op.truncate_gradient,
                name=op.name,
                allow_gc=op.allow_gc,
            )
            nw_node = nwScan(*(node.inputs), return_list=True)

        else:
            nwOpFromGraph = OpFromGraph(
                node_inputs,
                op_outs,
                op.is_inline,
                op.lop_overrides,
                op.grad_overrides,
                op.rop_overrides,
                connection_pattern=op._connection_pattern,
                name=op.name,
                **op.kwargs,
            )
            nw_node = nwOpFromGraph(*(node.inputs), return_list=True)
        return nw_node
