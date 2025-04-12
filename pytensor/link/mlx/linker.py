from pytensor.link.basic import JITLinker


class MLXLinker(JITLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using Apple's MLX."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_functors = []

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

        class wrapper:
            def __init__(self, fn, gen_functors):
                self.fn = mx.compile(fn)
                self.gen_functors = gen_functors.copy()

            def __call__(self, *inputs, **kwargs):
                import pytensor.link.utils

                # set attrs
                for n, fn in self.gen_functors:
                    setattr(pytensor.link.utils, n[1:], fn)

                # MLX doesn't support np.ndarray as input
                outs = self.fn(*(mlx_typify(inp) for inp in inputs), **kwargs)

                # unset attrs
                for n, _ in self.gen_functors:
                    if getattr(pytensor.link.utils, n[1:], False):
                        delattr(pytensor.link.utils, n[1:])

                return outs

            def __del__(self):
                del self.gen_functors

        inner_fn = wrapper(fn, self.gen_functors)
        self.gen_functors = []

        return inner_fn

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
        from numpy.random import Generator, RandomState

        from pytensor.link.mlx.dispatch import mlx_typify

        thunk_inputs = []
        for n in self.fgraph.inputs:
            sinput = storage_map[n]
            # Handle random number generators specially
            if isinstance(sinput[0], RandomState | Generator):
                new_value = mlx_typify(
                    sinput[0], dtype=getattr(sinput[0], "dtype", None)
                )
                sinput[0] = new_value
            thunk_inputs.append(sinput)

        return thunk_inputs
