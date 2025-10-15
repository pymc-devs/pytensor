from pytensor.link.basic import JITLinker


numba_njit_and_cache = (
    None  # To be imported inside the class to avoid eager numba import
)


class NumbaLinker(JITLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using Numba."""

    def fgraph_convert(self, fgraph, **kwargs):
        # Import numba_njit_and_cache lazily (as numba is an optional dependency)
        # and import dispatches to register them
        global numba_njit_and_cache

        if numba_njit_and_cache:
            import pytensor.link.numba.dispatch  # noqa
            from pytensor.link.numba.cache import numba_njit_and_cache

        return numba_njit_and_cache(fgraph, **kwargs)[0]

    def jit_compile(self, fn):
        return fn

    def create_thunk_inputs(self, storage_map):
        return [storage_map[n] for n in self.fgraph.inputs]
