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
        from numpy.random import RandomState

        from pytensor.link.numba.dispatch import numba_typify

        thunk_inputs = []
        for n in self.fgraph.inputs:
            sinput = storage_map[n]
            if isinstance(sinput[0], RandomState):
                new_value = numba_typify(
                    sinput[0], dtype=getattr(sinput[0], "dtype", None)
                )
                # We need to remove the reference-based connection to the
                # original `RandomState`/shared variable's storage, because
                # subsequent attempts to use the same shared variable within
                # other non-Numba-fied graphs will have problems.
                sinput = [new_value]
            thunk_inputs.append(sinput)

        return thunk_inputs
