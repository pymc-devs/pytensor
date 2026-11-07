import os

from pytensor.link.basic import JITLinker


# Fusion of indexed reads/writes and reductions into Elemwise loops (FusedElemwise).
# Included by name: their tag ("numba") would pull in other numba-only rewrites.
FUSION_REWRITES = ("fuse_indexed_into_elemwise", "fuse_elemwise", "wrap_reduced_gather_in_elemwise")
# Rewrites the LLVM mode includes on top of fast_run: FusedElemwise, and the
# save-memory scan rewrite for linkers without preallocated outputs.
LLVM_REWRITES = (*FUSION_REWRITES, "scan_save_mem_no_prealloc")


class LLVMLinker(JITLinker):
    """Compile a FunctionGraph to machine code through LLVM.

    Code generation, buffer planning and the runtime live in the native `pytensor_llvm`
    extension (``rust/pytensor-llvm``). In-place rewrites are excluded on purpose: buffer
    reuse, including in-place Elemwise, is decided by the native planner from liveness.
    """

    required_rewrites = ("minimum_compile",)
    incompatible_rewrites = (
        "cxx_only",
        "BlasOpt",
        "local_careduce_fusion",
        "scan_reduce_trace_prealloc",
        "inplace",
    )

    def __init__(
        self,
        fastmath=True,
        opt_level=3,
        cache=True,
        dump_ir=None,
        segments=True,
        segment_max_elems=64,
        numba_fallback=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fastmath = fastmath
        # merge runs of statically small nodes into one compilation unit (inlined)
        self.segments = segments
        self.segment_max_elems = segment_max_elems
        # run Ops without an LLVM implementation through numba
        self.numba_fallback = numba_fallback
        self.opt_level = opt_level
        self.cache = cache
        self.dump_ir = dump_ir or os.environ.get("PYTENSOR_LLVM_DUMP_IR")
        self.compiled = None

    def fgraph_convert(self, fgraph, **kwargs):
        from pytensor.link.llvm.dispatch import fgraph_to_llvm

        self.compiled = fgraph_to_llvm(
            fgraph,
            fastmath=self.fastmath,
            opt_level=self.opt_level,
            cache=self.cache,
            dump_ir=self.dump_ir,
            segments=self.segments,
            segment_max_elems=self.segment_max_elems,
            numba_fallback=self.numba_fallback,
        )
        return self.compiled

    def jit_compile(self, fn):
        return fn

    def create_thunk_inputs(self, storage_map):
        return [storage_map[n] for n in self.fgraph.inputs]
