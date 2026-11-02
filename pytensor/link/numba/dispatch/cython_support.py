"""Numba adapter for the shared Cython function resolver."""

import numba

from pytensor.scalar.cython_special import (
    _CythonFunctionSpec,
    resolve_cython_function,
)


class _NumbaCythonFunctionSpec(_CythonFunctionSpec):
    def numba_signature(self):
        return numba.from_dtype(self.output_dtype)(
            *(numba.from_dtype(dtype) for dtype in self.input_dtypes)
        )


def wrap_cython_function(func, restype, arg_types):
    impl = resolve_cython_function(func, restype, arg_types)
    return _NumbaCythonFunctionSpec(impl.signature, impl.capi_name, impl.module_name)
