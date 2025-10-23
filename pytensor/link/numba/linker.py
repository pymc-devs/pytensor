from pytensor.link.basic import JITLinker


# To be imported inside the class to avoid eager numba import
numba_funcify_ensure_cache = None


class NumbaLinker(JITLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using Numba."""

    def fgraph_convert(self, fgraph, **kwargs):
        # Import numba_njit_and_cache lazily (as numba is an optional dependency)
        # and import dispatches to register them
        global numba_funcify_ensure_cache

        if numba_funcify_ensure_cache is None:
            from pytensor.link.numba.dispatch.basic import numba_funcify_ensure_cache

        return numba_funcify_ensure_cache(fgraph, final_function=True, **kwargs)[0]

    def jit_compile(self, fn):
        # Already jitted by setting `final_function=True` in fgraph_convert
        return fn

    def create_thunk_inputs(self, storage_map):
        return [storage_map[n] for n in self.fgraph.inputs]
