from pytensor.link.basic import JITLinker
from pytensor.link.llvm.linker import FUSION_REWRITES


# Rewrites the LLVM_V2 mode includes on top of fast_run: FusedElemwise, the save-memory
# scan rewrite for linkers without preallocated outputs, and grouping into units.
LLVM_V2_REWRITES = (*FUSION_REWRITES, "scan_save_mem_no_prealloc", "llvm_units")


class LLVMV2Linker(JITLinker):
    """Compile a FunctionGraph with the function-based native backend.

    Nodes run their precompiled Rust kernels or generated code; OpFromGraphs with
    ``unit=UnitKind.COMPILED`` (see `grouping.py`) are generated as one function.
    """

    required_rewrites = ("minimum_compile",)
    incompatible_rewrites = (
        "cxx_only",
        "BlasOpt",
        "local_careduce_fusion",
        "scan_reduce_trace_prealloc",
        "inplace",
    )

    def __init__(self, fastmath=True, opt_level=3, cache=True, numba_fallback=True, **kwargs):
        super().__init__(**kwargs)
        self.fastmath = fastmath
        self.opt_level = opt_level
        self.cache = cache
        self.numba_fallback = numba_fallback
        self.compiled = None
        self.timings = None

    def fgraph_convert(self, fgraph, **kwargs):
        from pytensor.link.llvm_v2.dispatch import fgraph_to_llvm

        self.compiled, self.timings = fgraph_to_llvm(
            fgraph,
            fastmath=self.fastmath,
            opt_level=self.opt_level,
            cache=self.cache,
            numba_fallback=self.numba_fallback,
        )
        return self.compiled

    def jit_compile(self, fn):
        return fn

    def create_thunk_inputs(self, storage_map):
        return [storage_map[n] for n in self.fgraph.inputs]
