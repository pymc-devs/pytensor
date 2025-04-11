from pytensor.link.basic import JITLinker
from pytensor.link.utils import unique_name_generator


class MLXLinker(JITLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using Apple's MLX."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_functors = []

    def fgraph_convert(
        self,
        fgraph,
        order,
        input_storage,
        output_storage,
        storage_map,
        **kwargs,
    ):
        """Convert a PyTensor FunctionGraph to an MLX-compatible function.

        Parameters
        ----------
        fgraph : FunctionGraph
            The function graph to convert
        order : list
            The order in which to compute the nodes
        input_storage : list
            Storage for the input variables
        output_storage : list
            Storage for the output variables
        storage_map : dict
            Map from variables to their storage

        Returns
        -------
        callable
            An MLX-compatible function
        """
        from pytensor.link.mlx.dispatch import mlx_funcify

        # We want to have globally unique names
        # across the entire pytensor graph, not
        # just the subgraph
        generator = unique_name_generator(["mlx_linker"])

        # Ensure that torch is aware of the generated
        # code so we can compile without graph breaks
        def conversion_func_register(*args, **kwargs):
            functor = mlx_funcify(*args, **kwargs)
            name = kwargs["unique_name"](functor)
            self.gen_functors.append((f"_{name}", functor))
            return functor

        built_kwargs = {
            "unique_name": generator,
            "conversion_func": conversion_func_register,
            **kwargs,
        }
        return mlx_funcify(
            fgraph,
            input_storage=input_storage,
            storage_map=storage_map,
            **built_kwargs,
        )

    def jit_compile(self, fn):
        """JIT compile an MLX function.

        Parameters
        ----------
        fn : callable
            The function to compile

        Returns
        -------
        callable
            The compiled function
        """
        import mlx.core as mx

        return mx.compile(fn)

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
