"""optdb registry for the Scan rewrites.

This module is the single source of truth for which rewrites run, in
which order, and at which optdb position. Importing it triggers the
``optdb.register`` calls that wire every rewrite into the global
``optdb``.
"""

from pytensor.compile import optdb
from pytensor.graph.rewriting.basic import EquilibriumGraphRewriter, dfs_rewriter
from pytensor.graph.rewriting.db import EquilibriumDB, SequenceDB
from pytensor.scan.op import Scan
from pytensor.scan.rewriting.inplace import ScanInplaceOptimizer
from pytensor.scan.rewriting.io import (
    scan_inline_invariant_constants,
    scan_merge_duplicate_inputs,
    scan_remove_unused,
)
from pytensor.scan.rewriting.merge import ScanMerge, scan_merge_inouts
from pytensor.scan.rewriting.push_out import (
    scan_push_out_add,
    scan_push_out_dot1,
    scan_push_out_non_seq,
    scan_push_out_seq,
)
from pytensor.scan.rewriting.trace import (
    scan_save_mem_no_prealloc,
    scan_save_mem_prealloc,
    scan_sit_sot_to_untraced,
)
from pytensor.tensor.rewriting.basic import constant_folding


class ScanEquilibriumGraphRewriter(EquilibriumGraphRewriter):
    """Subclass of EquilibriumGraphRewriter that aborts early if there are no Scan Ops in the graph"""

    def apply(self, fgraph, start_from=None):
        if not any(isinstance(node.op, Scan) for node in fgraph.apply_nodes):
            return
        super().apply(fgraph=fgraph, start_from=start_from)


# An equilibrium pass is used because later Scan rewrites in the sequence
# can re-enable earlier ones. In practice the sequence rarely runs more
# than once.
scan_eqopt1 = EquilibriumDB(eq_rewriter_class=ScanEquilibriumGraphRewriter)
scan_seqopt1 = SequenceDB()
scan_eqopt2 = EquilibriumDB(eq_rewriter_class=ScanEquilibriumGraphRewriter)

# scan_eqopt1 before ShapeOpt at 0.1
# This is needed to don't have ShapeFeature trac old Scan that we
# don't want to reintroduce.
optdb.register("scan_eqopt1", scan_eqopt1, "fast_run", "scan", position=0.05)
# We run before blas opt at 1.7 and specialize 2.0
# but after stabilize at 1.5. Should we put it before stabilize?
optdb.register("scan_eqopt2", scan_eqopt2, "fast_run", "scan", position=1.6)
# ScanSaveMem should execute only once per node.
optdb.register(
    "scan_save_mem_prealloc",
    dfs_rewriter(scan_save_mem_prealloc, ignore_newtrees=True),
    "fast_run",
    "scan",
    "scan_save_mem",
    position=1.61,
)
optdb.register(
    "scan_save_mem_no_prealloc",
    dfs_rewriter(scan_save_mem_no_prealloc, ignore_newtrees=True),
    "numba",
    "jax",
    "pytorch",
    use_db_name_as_tag=False,
    position=1.61,
)
# After scan_save_mem (it could be merged with it, but that rewrite is already a beast as is)
optdb.register(
    "scan_remove_unused_top",
    dfs_rewriter(scan_remove_unused, ignore_newtrees=True),
    "fast_run",
    "scan",
    "scan_remove_unused",
    position=1.605,
)
optdb.register(
    "scan_sit_sot_to_untraced",
    dfs_rewriter(scan_sit_sot_to_untraced, ignore_newtrees=True),
    "fast_run",
    "scan",
    position=1.62,
)
optdb.register(
    "scan_make_inplace",
    ScanInplaceOptimizer(),
    "fast_run",
    "inplace",
    "scan",
    position=50.5,
)

scan_eqopt1.register("all_pushout_opt", scan_seqopt1, "fast_run", "scan")


scan_seqopt1.register(
    "scan_input_and_output_cleanup0",
    dfs_rewriter(
        scan_remove_unused,
        scan_inline_invariant_constants,
        scan_merge_duplicate_inputs,
    ),
    "scan_remove_unused",
    "scan_inline_invariant_constants",
    "scan_merge_duplicate_inputs",
    "fast_run",
    "scan",
    position=1,
)

scan_seqopt1.register(
    "scan_push_out_non_seq",
    dfs_rewriter(scan_push_out_non_seq, ignore_newtrees=True),
    "scan_pushout_nonseqs_ops",  # For backcompat: so it can be tagged with old name
    "fast_run",
    "scan",
    "scan_pushout",
    position=3,
)

scan_seqopt1.register(
    "scan_push_out_seq",
    dfs_rewriter(scan_push_out_seq, ignore_newtrees=True),
    "scan_pushout_seqs_ops",  # For backcompat: so it can be tagged with old name
    "fast_run",
    "scan",
    "scan_pushout",
    position=4,
)


scan_seqopt1.register(
    "scan_push_out_dot1",
    dfs_rewriter(scan_push_out_dot1, ignore_newtrees=True),
    "scan_pushout_dot1",  # For backcompat: so it can be tagged with old name
    "fast_run",
    "more_mem",
    "scan",
    "scan_pushout",
    position=5,
)


scan_seqopt1.register(
    "scan_push_out_add",
    # TODO: Perhaps this should be an `EquilibriumGraphRewriter`?
    dfs_rewriter(scan_push_out_add, ignore_newtrees=False),
    "scan_pushout_add",  # For backcompat: so it can be tagged with old name
    "fast_run",
    "more_mem",
    "scan",
    "scan_pushout",
    position=6,
)

scan_eqopt2.register(
    "constant_folding_for_scan2",
    dfs_rewriter(constant_folding, ignore_newtrees=True),
    "fast_run",
    "scan",
)


scan_eqopt2.register(
    "scan_input_and_output_cleanup1",
    dfs_rewriter(
        scan_remove_unused,
        scan_inline_invariant_constants,
        scan_merge_duplicate_inputs,
    ),
    "scan_remove_unused",
    "scan_inline_invariant_constants",
    "scan_merge_duplicate_inputs",
    "fast_run",
    "scan",
)


# after const merge but before stabilize so that we can have identity
# for equivalent nodes but we still have the chance to hoist stuff out
# of the scan later.
scan_eqopt2.register("scan_merge", ScanMerge(), "fast_run", "scan")

# After Merge optimization
scan_eqopt2.register(
    "scan_input_and_output_cleanup2",
    dfs_rewriter(
        scan_remove_unused,
        scan_inline_invariant_constants,
        scan_merge_duplicate_inputs,
    ),
    "scan_remove_unused",
    "scan_inline_invariant_constants",
    "scan_merge_duplicate_inputs",
    "fast_run",
    "scan",
)

scan_eqopt2.register(
    "scan_merge_inouts",
    dfs_rewriter(scan_merge_inouts, ignore_newtrees=True),
    "fast_run",
    "scan",
)

# After everything else
scan_eqopt2.register(
    "scan_input_and_output_cleanup3",
    dfs_rewriter(
        scan_remove_unused,
        scan_inline_invariant_constants,
        scan_merge_duplicate_inputs,
    ),
    "scan_remove_unused",
    "scan_inline_invariant_constants",
    "scan_merge_duplicate_inputs",
    "fast_run",
    "scan",
)
