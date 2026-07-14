from pytensor.link.basic import JITLinker


class JAXLinker(JITLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using JAX."""

    backend_tag = "jax"

    required_rewrites = (
        "minimum_compile",
        "jax",
    )  # TODO: Distinguish between optional "jax" and "minimum_compile_jax"
    incompatible_rewrites = (
        "cxx_only",
        "BlasOpt",
        "local_careduce_fusion",
        "scan_reduce_trace_prealloc",
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
        return [storage_map[n] for n in self.fgraph.inputs]
