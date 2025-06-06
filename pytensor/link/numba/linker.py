from functools import singledispatch

from pytensor.link.basic import JITLinker


@singledispatch
def numba_typify(data, dtype=None, **kwargs):
    raise NotImplementedError(
        f"Numba funcify not implemented for data type {type(data)}"
    )


@singledispatch
def numba_funcify(obj, *args, **kwargs):
    raise NotImplementedError(f"Numba funcify not implemented for type {type(obj)}")


class NumbaLinker(JITLinker):
    """A `Linker` that JIT-compiles NumPy-based operations using Numba."""

    def fgraph_convert(self, fgraph, **kwargs):
        return numba_funcify(fgraph, **kwargs)

    def jit_compile(self, fn):
        from pytensor.link.numba.dispatch.basic import numba_njit

        jitted_fn = numba_njit(fn, no_cpython_wrapper=False, no_cfunc_wrapper=False)
        return jitted_fn

    def create_thunk_inputs(self, storage_map):
        return [storage_map[n] for n in self.fgraph.inputs]
