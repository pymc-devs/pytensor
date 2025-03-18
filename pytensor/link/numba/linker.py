from pytensor.link.basic import JITLinker


class NumbaLinker(JITLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using Numba."""

    def fgraph_convert(self, fgraph, **kwargs):
        from pytensor.link.numba.dispatch import numba_funcify

        return numba_funcify(fgraph, **kwargs)

    def jit_compile(self, fn):
        from pytensor.link.numba.dispatch.basic import numba_njit

        jitted_fn = numba_njit(fn, no_cpython_wrapper=False, no_cfunc_wrapper=False)
        return jitted_fn

    def create_thunk_inputs(self, storage_map):
        return [storage_map[n] for n in self.fgraph.inputs]
