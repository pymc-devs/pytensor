from pytensor.link.basic import JITLinker


class MLXLinker(JITLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using Apple's MLX."""

    incompatible_rewrites = (
        "cxx_only",
        "BlasOpt",
        "fusion",
        "inplace",
        "scan_save_mem_prealloc",
    )

    def __init__(self, use_compile=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_functors = []
        self.use_compile = use_compile

    def fgraph_convert(self, fgraph, **kwargs):
        """Convert a PyTensor FunctionGraph to an MLX-compatible function.

        Parameters
        ----------
        fgraph : FunctionGraph
            The function graph to convert

        Returns
        -------
        callable
            An MLX-compatible function
        """
        from pytensor.link.mlx.dispatch import mlx_funcify

        return mlx_funcify(
            fgraph,
            **kwargs,
        )

    def jit_compile(self, fn):
        import mlx.core as mx

        from pytensor.link.mlx.dispatch import mlx_typify

        if not self.use_compile:
            # Skip compilation and just return the function with MLX typification
            def fn_no_compile(*inputs):
                return fn(*(mlx_typify(inp) for inp in inputs))

            return fn_no_compile

        inner_fn = mx.compile(fn)

        def fn(*inputs, inner_fn=inner_fn):
            return inner_fn(*(mlx_typify(inp) for inp in inputs))

        return fn

    def create_thunk_inputs(self, storage_map):
        """Create inputs for the MLX thunk.

        Parameters
        ----------
        storage_map : dict
            Map from variables to their storage

        Returns
        -------
        list
            The inputs for the thunk
        """
        thunk_inputs = []
        for n in self.fgraph.inputs:
            sinput = storage_map[n]
            thunk_inputs.append(sinput)

        return thunk_inputs
