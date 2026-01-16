from pytensor.link.basic import JITLinker


class NumbaLinker(JITLinker):
    required_rewrites = (
        "minimum_compile",
        "numba",
    )  # TODO: Distinguish between optional "numba" and "minimum_compile_numba"
    incompatible_rewrites = (
        "cxx_only",
        "BlasOpt",
        "local_careduce_fusion",
        "scan_save_mem_prealloc",
    )

    """A `Linker` that JIT-compiles NumPy-based operations using Numba."""

    def fgraph_convert(self, fgraph, **kwargs):
        # Import numba_njit_and_cache lazily (as numba is an optional dependency)
        # This is what triggers the registering of the dispatches as well
        from pytensor.link.numba.dispatch.basic import numba_funcify_ensure_cache

        return numba_funcify_ensure_cache(fgraph, **kwargs)

    def jit_compile(self, fn_and_cache):
        from pytensor.link.numba.dispatch.basic import numba_njit

        fn, cache_key = fn_and_cache
        return numba_njit(fn.py_func, final_function=True, cache=cache_key is not None)

    def create_thunk_inputs(self, storage_map):
        return [storage_map[n] for n in self.fgraph.inputs]
