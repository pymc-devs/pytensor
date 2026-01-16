import warnings

from numpy.random import Generator

from pytensor.compile.sharedvalue import SharedVariable, shared
from pytensor.link.basic import JITLinker


class JAXLinker(JITLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using JAX."""

    required_rewrites = (
        "minimum_compile",
        "jax",
    )  # TODO: Distinguish between optional "jax" and "minimum_compile_jax"
    incompatible_rewrites = (
        "cxx_only",
        "BlasOpt",
        "local_careduce_fusion",
        "scan_save_mem_prealloc",
        # JAX does it his own inplace optimization
        "inplace",
        # There are specific variants for the LU decompositions supported by JAX
        "reuse_lu_decomposition_multiple_solves",
        "scan_split_non_sequence_lu_decomposition_solve",
    )

    scalar_shape_inputs: tuple[int, ...]

    def __init__(self, *args, **kwargs):
        self.scalar_shape_inputs = ()
        super().__init__(*args, **kwargs)

    def fgraph_convert(self, fgraph, input_storage, storage_map, **kwargs):
        from pytensor.link.jax.dispatch import jax_funcify
        from pytensor.link.jax.dispatch.shape import JAXShapeTuple
        from pytensor.tensor.random.type import RandomType

        shared_rng_inputs = [
            inp
            for inp in fgraph.inputs
            if (isinstance(inp, SharedVariable) and isinstance(inp.type, RandomType))
        ]

        # Replace any shared RNG inputs so that their values can be updated in place
        # without affecting the original RNG container. This is necessary because
        # JAX does not accept Generators as inputs, and they will have to
        # be tipyfied
        if shared_rng_inputs:
            warnings.warn(
                f"The RandomType SharedVariables {shared_rng_inputs} will not be used "
                f"in the compiled JAX graph. Instead a copy will be used.",
                UserWarning,
            )
            new_shared_rng_inputs = [
                shared(inp.get_value(borrow=False)) for inp in shared_rng_inputs
            ]

            fgraph.replace_all(
                zip(shared_rng_inputs, new_shared_rng_inputs, strict=True),
                import_missing=True,
                reason="JAXLinker.fgraph_convert",
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
                old_inp_fgrap_index = fgraph.inputs.index(old_inp)
                fgraph.remove_input(
                    old_inp_fgrap_index,
                    reason="JAXLinker.fgraph_convert",
                )
                fgraph.inputs.remove(new_inp)
                fgraph.inputs.insert(old_inp_fgrap_index, new_inp)

        fgraph_inputs = fgraph.inputs
        clients = fgraph.clients
        # Detect scalar shape inputs that are used only in JAXShapeTuple nodes
        scalar_shape_inputs = [
            inp
            for node in fgraph.apply_nodes
            if isinstance(node.op, JAXShapeTuple)
            for inp in node.inputs
            if inp in fgraph_inputs
            and all(
                isinstance(cl_node.op, JAXShapeTuple) for cl_node, _ in clients[inp]
            )
        ]
        self.scalar_shape_inputs = tuple(
            fgraph_inputs.index(inp) for inp in scalar_shape_inputs
        )

        return jax_funcify(
            fgraph, input_storage=input_storage, storage_map=storage_map, **kwargs
        )

    def jit_compile(self, fn):
        import jax

        jit_fn = jax.jit(fn, static_argnums=self.scalar_shape_inputs)

        if not self.scalar_shape_inputs:
            return jit_fn

        def convert_scalar_shape_inputs(
            *args, scalar_shape_inputs=set(self.scalar_shape_inputs)
        ):
            return jit_fn(
                *(
                    int(arg) if i in scalar_shape_inputs else arg
                    for i, arg in enumerate(args)
                )
            )

        return convert_scalar_shape_inputs

    def create_thunk_inputs(self, storage_map):
        from pytensor.link.jax.dispatch import jax_typify

        thunk_inputs = []
        for n in self.fgraph.inputs:
            sinput = storage_map[n]
            if isinstance(sinput[0], Generator):
                # Neet to convert Generator into JAX PRNGkey
                sinput[0] = jax_typify(sinput[0])
            thunk_inputs.append(sinput)

        return thunk_inputs
