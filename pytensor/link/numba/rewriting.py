"""Numba graph rewrites that run after output ownership copies are inserted."""

from pytensor.compile.ops import DeepCopyOp
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.scan.op import Scan
from pytensor.tensor.type import DenseTensorType


@node_rewriter([DeepCopyOp])
def numba_remove_scalar_scan_output_copy(fgraph, node):
    [source] = node.inputs
    scan = source.owner_op
    if (
        isinstance(scan, Scan)
        and isinstance(source.type, DenseTensorType)
        and source.ndim == 0
        and source in scan.outer_untraced_sit_sot_outs(source.owner.outputs)
        and fgraph.clients[source] == [(node, 0)]
    ):
        # Numba boxes this scalar afresh even at zero steps. Reusing that box
        # is safe only when no other client can observe the removed copy.
        return [source]
