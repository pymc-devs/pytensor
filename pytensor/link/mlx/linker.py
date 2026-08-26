import warnings

from pytensor.compile.sharedvalue import SharedVariable, shared
from pytensor.link.basic import JITLinker


UNCOMPILABLE_SHAPE_WARNING = (
    "This graph contains an `Op` whose output shape depends on its input "
    "values, which `mx.compile` cannot trace, so the whole graph will run "
    "uncompiled. To get compilation back, give the graph static shapes: a "
    "boolean mask that is constant becomes integer indices, and a mask only "
    "known at runtime can often be replaced by shape-preserving arithmetic, "
    "e.g. `(x * mask).sum()` rather than `x[mask].sum()`."
)


def _has_data_dependent_shape(fgraph):
    """Whether any `Op` in ``fgraph`` has an output shape that depends on data.

    `mx.compile` traces a graph once per input *shape*, so an output whose shape
    is only known from the values cannot be traced at all.
    """
    # Imported here because `pytensor.compile` imports this module while
    # `pytensor.tensor` is still initializing.
    from pytensor.graph.op import HasInnerGraph
    from pytensor.tensor.basic import Nonzero

    for node in fgraph.apply_nodes:
        if isinstance(node.op, Nonzero):
            return True

        if isinstance(node.op, HasInnerGraph):
            inner_fgraph = getattr(node.op, "fgraph", None)
            if inner_fgraph is not None and _has_data_dependent_shape(inner_fgraph):
                return True

    return False


class MLXLinker(JITLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using Apple's MLX."""

    required_rewrites = ("minimum_compile",)
    incompatible_rewrites = (
        "cxx_only",
        "BlasOpt",
        "local_careduce_fusion",
        "inplace",
        "scan_reduce_trace_prealloc",
        "inline_einsum",
    )

    def __init__(self, use_compile=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_functors = []
        self.use_compile = use_compile

    def fgraph_convert(self, fgraph, input_storage, storage_map, **kwargs):
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
        from pytensor.tensor.random.type import RandomType

        shared_rng_inputs = [
            inp
            for inp in fgraph.inputs
            if (isinstance(inp, SharedVariable) and isinstance(inp.type, RandomType))
        ]

        # Replace any shared RNG inputs so that their values can be updated in place
        # without affecting the original RNG container. This is necessary because
        # MLX does not accept Generators as inputs, and they will have to
        # be typified
        if shared_rng_inputs:
            warnings.warn(
                f"The RandomType SharedVariables {shared_rng_inputs} will not be used "
                f"in the compiled MLX graph. Instead a copy will be used.",
                UserWarning,
            )
            new_shared_rng_inputs = [
                shared(inp.get_value(borrow=False)) for inp in shared_rng_inputs
            ]

            fgraph.replace_all(
                zip(shared_rng_inputs, new_shared_rng_inputs, strict=True),
                import_missing=True,
                reason="MLXLinker.fgraph_convert",
            )

            for old_inp, new_inp in zip(
                shared_rng_inputs, new_shared_rng_inputs, strict=True
            ):
                new_inp_storage = [new_inp.get_value(borrow=True)]
                storage_map[new_inp] = new_inp_storage
                old_inp_storage = storage_map.pop(old_inp)
                # Find index of old_inp_storage in input_storage
                for input_storage_idx, input_storage_item in enumerate(input_storage):
                    # We have to establish equality based on identity because input_storage may contain numpy arrays
                    if input_storage_item is old_inp_storage:
                        break
                else:  # no break
                    raise ValueError()
                input_storage[input_storage_idx] = new_inp_storage
                # We need to change the order of the inputs of the FunctionGraph
                # so that the new input is in the same position as to old one,
                # to align with the storage_map. We hope this is safe!
                old_inp_fgraph_index = fgraph.inputs.index(old_inp)
                fgraph.remove_input(
                    old_inp_fgraph_index,
                    reason="MLXLinker.fgraph_convert",
                )
                fgraph.inputs.remove(new_inp)
                fgraph.inputs.insert(old_inp_fgraph_index, new_inp)

        return mlx_funcify(
            fgraph,
            input_storage=input_storage,
            storage_map=storage_map,
            **kwargs,
        )

    def jit_compile(self, fn):
        import mlx.core as mx

        from pytensor.link.mlx.dispatch import mlx_typify

        use_compile = self.use_compile
        if use_compile and _has_data_dependent_shape(self.fgraph):
            warnings.warn(UNCOMPILABLE_SHAPE_WARNING, UserWarning, stacklevel=2)
            use_compile = False

        if not use_compile:
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
        from numpy.random import Generator

        from pytensor.link.mlx.dispatch import mlx_typify

        thunk_inputs = []
        for n in self.fgraph.inputs:
            sinput = storage_map[n]
            if isinstance(sinput[0], Generator):
                # Convert Generator into MLX PRNG key
                sinput[0] = mlx_typify(sinput[0])
            thunk_inputs.append(sinput)

        return thunk_inputs
