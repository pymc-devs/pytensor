from pytensor.link.basic import JITLinker


class LLVMLinker(JITLinker):
    """Compile a FunctionGraph with the LLVM backend.

    The graph is planned in Python (`program.py`) into items that run Rust kernels,
    code generated from Python, or numba implementations; a generated function, the
    program's, runs them.
    """

    # llvm_layouts attaches the LayoutFeature (the layouts the planner gives the
    # values); llvm_units keeps ScalarType values inside compiled units: they don't
    # cross the kernel ABI
    required_rewrites = ("minimum_compile", "llvm_layouts", "llvm_units")
    incompatible_rewrites = (
        "cxx_only",
        "BlasOpt",
        # Numba-only: scalar-typed ops neither group into units nor plan
        "scalarize",
        "remove_scalar_tensor_boundaries",
        "local_careduce_fusion",
        "scan_reduce_trace_prealloc",
    )

    #: options passed on to the planner (program.Options)
    OPTIONS = (
        "fastmath",
        "opt_level",
        "numba_fallback",
        "input_layout",
        "check_layouts",
        "parallel",
        "linalg_parallel",
        "debug_info",
    )

    def __init__(
        self,
        fastmath=True,
        opt_level=3,
        numba_fallback=True,
        input_layout="any",
        check_layouts=False,
        parallel=True,
        linalg_parallel=None,
        debug_info=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fastmath = fastmath
        self.opt_level = opt_level
        self.numba_fallback = numba_fallback
        self.input_layout = input_layout
        self.check_layouts = check_layouts
        self.parallel = parallel
        if linalg_parallel is None:
            linalg_parallel = parallel
        self.linalg_parallel = linalg_parallel
        self.debug_info = debug_info
        self.program = None
        self.stats = None

    def clone(self, allow_gc=None, **options):
        """A copy with some options replaced, e.g.
        `mode.clone(link_kwargs={"numba_fallback": False})`."""
        new = super().clone(allow_gc=allow_gc)
        for k, v in options.items():
            if k not in self.OPTIONS:
                raise TypeError(f"Unknown LLVMLinker option {k!r}")
            setattr(new, k, v)
        new.program = new.stats = None
        return new

    def fgraph_convert(self, fgraph, **kwargs):
        from pytensor.link.llvm.program import Options, compile_program

        opts = Options(**{k: getattr(self, k) for k in self.OPTIONS})
        self.program, fn, self.stats = compile_program(fgraph, opts)
        return fn

    def jit_compile(self, fn):
        return fn

    def ir(
        self, optimized=False, opt_level=None, remarks=False, locations=False
    ) -> str:
        """The LLVM IR of the code generated for the compiled graph: the module of
        every item compiled on its own and the program's function, each under a comment
        naming it (see `program.dump()`).
        With `optimized`, the IR after the optimization pipeline, i.e. what the machine
        code is generated from (at `opt_level`, default the one compiled with); with
        `remarks` too, LLVM's optimization remarks first (what the loop vectorizer did
        or why not, inlining, unrolling...). `locations`: with the `debug_info` option,
        the Python code and node every instruction comes from, as comments.

        Ops that run Rust or numba kernels have no module here."""
        if self.stats is None:
            raise RuntimeError("nothing compiled yet")
        return self.stats.ir(optimized, opt_level, remarks, locations)

    def create_thunk_inputs(self, storage_map):
        return [storage_map[n] for n in self.fgraph.inputs]
