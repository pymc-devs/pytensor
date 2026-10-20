"""Shared Cython signature resolution and C implementations of SciPy scalar Ops."""

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from textwrap import dedent

import numpy as np
from numpy.typing import DTypeLike

from pytensor.scalar.basic import ScalarOp, complex_types


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

    c_signature: str = ""
    res_c_type: str = ""
    arg_c_types: tuple[str, ...] = ()

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

        arg_c_types = []
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

            arg_c_types.append(arg_c_type)
            arg_dtypes.append(arg_dtype)
            name = groups["name"]
            if not name:
                arg_names.append(None)
            else:
                arg_names.append(name.decode())

        return Signature(
            res_dtype,
            arg_dtypes,
            arg_names,
            signature.decode(),
            res_c_type,
            tuple(arg_c_types),
        )


def _available_impls(func: Callable) -> list[tuple[Signature, str]]:
    """Find all available implementations for a fused cython function.

    Each entry is ``(signature, capi_name)``, where ``capi_name`` is the key under
    which the implementation is exported in the module's ``__pyx_capi__`` table. That name is a
    stable, picklable handle for the C function, used to re-resolve its address at runtime.
    """
    from scipy import LowLevelCallable

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
    resolve the function's address at runtime. The address itself is deliberately not
    captured here, so kernels calling the function stay disk-cacheable.
    """

    def __init__(self, signature, capi_name, module_name):
        self._signature = signature
        self.capi_name = capi_name
        self.module_name = module_name
        self.input_dtypes = signature.arg_dtypes
        self.output_dtype = signature.res_dtype

    def has_pyx_skip_dispatch(self):
        if not self._signature.arg_names:
            return False
        if any(
            name == "__pyx_skip_dispatch" for name in self._signature.arg_names[:-1]
        ):
            raise ValueError("skip_dispatch parameter must be last")
        return self._signature.arg_names[-1] == "__pyx_skip_dispatch"


def resolve_cython_function(func, restype, arg_types):
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


class SciPyScalarOp(ScalarOp):
    """Call a SciPy Cython function with double arguments and a double result.

    Subclasses supply ``nfunc_spec`` and a fixed ``nin``. Resolve one real,
    double-precision signature per Op, independent of the node dtype, so that
    declarations and initialization also work inside Composite and ScalarLoop.
    Fused exports and optional Cython dispatch flags use the shared resolver.
    As with SciPy's Cython interface, errors return NaN without the Python
    ufunc wrappers' warning/error-state handling.

    Resolve the function address when the generated module is imported, while holding the
    GIL, rather than in an elementwise (possibly OpenMP) loop. Only the export name
    and signature enter the compilation cache, never a process-local address.
    """

    @property
    def scipy_func_name(self):
        module, name = self.nfunc_spec[0].rsplit(".", 1)
        if module != "scipy.special" or self.nin < 1 or self.nout != 1:
            raise NotImplementedError("Expected a fixed-arity scipy.special function")
        return name

    @property
    def scipy_c_impl(self):
        from scipy.special import cython_special

        func = getattr(cython_special, self.scipy_func_name)
        return resolve_cython_function(func, np.float64, [np.float64] * self.nin)

    @property
    def scipy_c_symbol(self):
        impl = self.scipy_c_impl
        identity = (impl.module_name, impl.capi_name, impl._signature.c_signature)
        return "pytensor_scipy_" + sha256(repr(identity).encode()).hexdigest()

    def c_support_code(self, **kwargs):
        impl = self.scipy_c_impl
        symbol = self.scipy_c_symbol
        signature = impl._signature
        args = ", ".join(signature.arg_c_types)
        return dedent(
            f"""
            #ifndef {symbol}_DEFINED
            #define {symbol}_DEFINED
            static {signature.res_c_type} (*{symbol})({args}) = NULL;
            #endif
            """
        )

    def c_init_code(self, **kwargs):
        impl = self.scipy_c_impl
        name = impl.capi_name
        symbol = self.scipy_c_symbol
        signature = impl._signature.c_signature
        args = ", ".join(impl._signature.arg_c_types)
        return [
            dedent(
                f"""
                {{
                    PyObject *module = PyImport_ImportModule("scipy.special.cython_special");
                    if (module == NULL) return NULL;
                    PyObject *capi = PyObject_GetAttrString(module, "__pyx_capi__");
                    Py_DECREF(module);
                    if (capi == NULL) return NULL;
                    if (!PyDict_Check(capi)) {{
                        Py_DECREF(capi);
                        PyErr_SetString(PyExc_ImportError,
                            "scipy.special.cython_special.__pyx_capi__ is not a dictionary");
                        return NULL;
                    }}
                    PyObject *capsule = PyDict_GetItemString(capi, "{name}");
                    if (capsule == NULL) {{
                        Py_DECREF(capi);
                        PyErr_SetString(PyExc_ImportError,
                            "scipy.special.cython_special does not export {name}");
                        return NULL;
                    }}
                    void *pointer = PyCapsule_GetPointer(capsule, "{signature}");
                    Py_DECREF(capi);
                    if (pointer == NULL) {{
                        PyErr_SetString(PyExc_ImportError,
                            "Incompatible scipy.special.cython_special.{name} C API; "
                            "expected {signature}");
                        return NULL;
                    }}
                    {symbol} = ({impl._signature.res_c_type} (*)({args})) pointer;
                }}
                """
            )
        ]

    def c_code(self, node, name, inputs, outputs, sub):
        if any(var.dtype == "float16" for var in node.inputs + node.outputs):
            raise NotImplementedError("Float16 C arithmetic is not supported")
        if any(inp.type in complex_types for inp in node.inputs):
            raise NotImplementedError("Complex inputs are not supported")
        (output,) = outputs
        cast = node.outputs[0].type.dtype_specs()[1]
        impl = self.scipy_c_impl
        args = [f"(double){inp}" for inp in inputs]
        if impl.has_pyx_skip_dispatch():
            args.append("1")
        return f"{output} = ({cast}){self.scipy_c_symbol}({', '.join(args)});"

    def c_code_cache_version(self):
        version = super().c_code_cache_version()
        if not version:
            return ()
        impl = self.scipy_c_impl
        identity = (impl.module_name, impl.capi_name, impl._signature.c_signature)
        signature_version = tuple(sha256(repr(identity).encode()).digest())
        return (2, signature_version, *version)
