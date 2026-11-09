from pytensor.link.basic import JITLinker
from pytensor.link.llvm.linker import FUSION_REWRITES


# Rewrites the LLVM_V2 mode includes on top of fast_run: FusedElemwise, the save-memory
# scan rewrite for linkers without preallocated outputs, and grouping into units.
LLVM_V2_REWRITES = (*FUSION_REWRITES, "scan_save_mem_no_prealloc", "llvm_units")


class LLVMV2Linker(JITLinker):
    """Compile a FunctionGraph with the LLVM backend (second design, see DESIGN.md).

    The graph is planned in Python (`program.py`) into steps that run Rust kernels,
    code generated from Python, or numba implementations; a Rust interpreter runs them.
    """

    required_rewrites = ("minimum_compile",)
    incompatible_rewrites = (
        "cxx_only",
        "BlasOpt",
        "local_careduce_fusion",
        "scan_reduce_trace_prealloc",
        "inplace",
    )

    #: options passed on to the planner (program.Options)
    OPTIONS = ("fastmath", "opt_level", "cache", "numba_fallback", "input_layout", "check_layouts")

    def __init__(
        self,
        fastmath=True,
        opt_level=3,
        cache=True,
        numba_fallback=True,
        input_layout="any",
        check_layouts=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fastmath = fastmath
        self.opt_level = opt_level
        self.cache = cache
        self.numba_fallback = numba_fallback
        self.input_layout = input_layout
        self.check_layouts = check_layouts
        self.compiled = None
        self.stats = None

    def clone(self, allow_gc=None, **options):
        """A copy with some options replaced, e.g.
        `mode.clone(link_kwargs={"numba_fallback": False})`."""
        new = super().clone(allow_gc=allow_gc)
        for k, v in options.items():
            if k not in self.OPTIONS:
                raise TypeError(f"Unknown LLVMV2Linker option {k!r}")
            setattr(new, k, v)
        new.compiled = new.stats = None
        return new

    def fgraph_convert(self, fgraph, **kwargs):
        from pytensor.link.llvm_v2.program import Options, build_program

        opts = Options(**{k: getattr(self, k) for k in self.OPTIONS})
        fn, self.stats = build_program(fgraph, opts)
        # the rt.Program (fn wraps it when generators are passed through)
        self.compiled = getattr(fn, "program", fn)
        return fn

    def jit_compile(self, fn):
        return fn

    def create_thunk_inputs(self, storage_map):
        return [storage_map[n] for n in self.fgraph.inputs]
