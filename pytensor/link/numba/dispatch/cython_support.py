import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass

import numba
import numpy as np
from numpy.typing import DTypeLike
from scipy import LowLevelCallable


_C_TO_NUMPY: dict[str, DTypeLike] = {
    "bool": np.bool_,
    "signed char": np.byte,
    "unsigned char": np.ubyte,
    "short": np.short,
    "unsigned short": np.ushort,
    "int": np.intc,
    "unsigned int": np.uintc,
    "long": np.int_,
    "unsigned long": np.uint,
    "long long": np.longlong,
    "float": np.single,
    "double": np.double,
    "long double": np.longdouble,
    "float complex": np.csingle,
    "double complex": np.cdouble,
    "Py_ssize_t": np.intp,
}


@dataclass
class Signature:
    res_dtype: DTypeLike
    arg_dtypes: list[DTypeLike]
    arg_names: list[str | None]

    @property
    def arg_numba_types(self) -> list[DTypeLike]:
        return [numba.from_dtype(dtype) for dtype in self.arg_dtypes]

    def can_cast_args(self, args: list[DTypeLike]) -> bool:
        ok = True
        count = 0
        for name, dtype in zip(self.arg_names, self.arg_dtypes, strict=True):
            if name == "__pyx_skip_dispatch":
                continue
            if len(args) <= count:
                raise ValueError("Incorrect number of arguments")
            ok &= np.can_cast(args[count], dtype)
            count += 1
        if count != len(args):
            return False
        return ok

    def provides(self, restype: DTypeLike, arg_dtypes: list[DTypeLike]) -> bool:
        args_ok = self.can_cast_args(arg_dtypes)
        if np.issubdtype(restype, np.inexact):
            result_ok = np.can_cast(self.res_dtype, restype, casting="same_kind")
            # We do not want to provide less accuracy than advertised
            result_ok &= np.dtype(self.res_dtype).itemsize >= np.dtype(restype).itemsize
        else:
            result_ok = np.can_cast(self.res_dtype, restype)
        return args_ok and result_ok

    @staticmethod
    def from_c_types(signature: bytes) -> "Signature":
        # Match strings like "double(int, double)"
        # and extract the return type and the joined arguments
        expr = re.compile(rb"\s*(?P<restype>[\w ]*\w+)\s*\((?P<args>[\w\s,]*)\)")
        re_match = re.fullmatch(expr, signature)

        if re_match is None:
            raise ValueError(f"Invalid signature: {signature.decode()}")

        groups = re_match.groupdict()
        res_c_type = groups["restype"].decode()
        res_dtype: DTypeLike = _C_TO_NUMPY[res_c_type]

        raw_args = groups["args"]

        decl_expr = re.compile(
            rb"\s*(?P<type>"
            rb"((long )|(unsigned )|(signed )|(double )|)"
            rb"((double)|(float)|(int)|(short)|(char)|(long)|(bool)|(complex))"
            rb"|Py_ssize_t)"
            rb"(\s(?P<name>[\w_]*))?\s*"
        )

        arg_dtypes = []
        arg_names: list[str | None] = []
        for raw_arg in raw_args.split(b","):
            re_match = re.fullmatch(decl_expr, raw_arg)
            if re_match is None:
                raise ValueError(f"Invalid signature: {signature.decode()}")
            groups = re_match.groupdict()
            arg_c_type = groups["type"].decode()
            try:
                arg_dtype = _C_TO_NUMPY[arg_c_type]
            except KeyError:
                raise ValueError(f"Unknown C type: {arg_c_type}")

            arg_dtypes.append(arg_dtype)
            name = groups["name"]
            if not name:
                arg_names.append(None)
            else:
                arg_names.append(name.decode())

        return Signature(res_dtype, arg_dtypes, arg_names)


def _available_impls(func: Callable) -> list[tuple[Signature, str]]:
    """Find all available implementations for a fused cython function.

    Each entry is ``(signature, capi_name)``, where ``capi_name`` is the key under
    which the implementation is exported in the module's ``__pyx_capi__`` table. That name is a
    stable, picklable handle for the C function, used to re-resolve its address at runtime.
    """
    impls = []
    mod = importlib.import_module(func.__module__)

    if getattr(func, "__signatures__", None) is not None:
        # Cython 3.3 exports typed names and retains the numbered names as aliases.
        # Python specialization names need not match either C API name.
        names = [
            name for name in mod.__pyx_capi__ if name.startswith(f"{func.__name__}[")
        ]
        if not names:
            pattern = re.compile(rf"__pyx_fuse_[0-9_]+{re.escape(func.__name__)}")
            names = [name for name in mod.__pyx_capi__ if pattern.fullmatch(name)]
    else:
        names = [func.__name__]
    for name in names:
        capsule = mod.__pyx_capi__[name]
        llc = LowLevelCallable(capsule)
        try:
            signature = Signature.from_c_types(llc.signature.encode())
        except KeyError:
            continue
        impls.append((signature, name))
    return impls


class _CythonFunctionSpec:
    """The cython implementation selected for a requested ``(restype, arg_types)`` signature.

    Holds the resolved C signature together with the module and ``__pyx_capi__`` name needed to
    resolve the function's address at call time via ``get_cython_function_address``. The address
    itself is deliberately not captured here, so kernels calling the function stay disk-cacheable.
    """

    def __init__(self, signature, capi_name, module_name):
        self._signature = signature
        self.capi_name = capi_name
        self.module_name = module_name
        self.input_dtypes = signature.arg_dtypes
        self.output_dtype = signature.res_dtype

    def signature(self):
        return numba.from_dtype(self._signature.res_dtype)(
            *self._signature.arg_numba_types
        )

    def has_pyx_skip_dispatch(self):
        if not self._signature.arg_names:
            return False
        if any(
            name == "__pyx_skip_dispatch" for name in self._signature.arg_names[:-1]
        ):
            raise ValueError("skip_dispatch parameter must be last")
        return self._signature.arg_names[-1] == "__pyx_skip_dispatch"


def wrap_cython_function(func, restype, arg_types):
    impls = _available_impls(func)
    compatible = []
    for sig, capi_name in impls:
        if sig.provides(restype, arg_types):
            compatible.append((sig, capi_name))

    def sort_key(args):
        sig, _ = args

        # Prefer functions with less inputs bytes
        argsize = sum(np.dtype(dtype).itemsize for dtype in sig.arg_dtypes)

        # Prefer functions with more exact (integer) arguments
        num_inexact = sum(np.issubdtype(dtype, np.inexact) for dtype in sig.arg_dtypes)
        return (num_inexact, argsize)

    compatible.sort(key=sort_key)

    if not compatible:
        raise NotImplementedError(f"Could not find a compatible impl of {func}")
    sig, capi_name = compatible[0]
    return _CythonFunctionSpec(sig, capi_name, func.__module__)
