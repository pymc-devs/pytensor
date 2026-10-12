"""C implementations for real, non-fused SciPy Cython special functions."""

from textwrap import dedent

from pytensor.scalar.basic import ScalarOp, complex_types


class SciPyScalarOp(ScalarOp):
    """Call a SciPy Cython function with double arguments and a double result.

    Subclasses supply ``nfunc_spec`` and a fixed ``nin``. This deliberately only
    handles non-fused ``cpdef`` exports with the trailing Cython dispatch flag.
    Fused functions, complex values and other signatures need separate support.
    As with SciPy's Cython interface, errors return NaN without the Python
    ufunc wrappers' warning/error-state handling.

    Resolve the capsule when the generated module is imported, while holding the
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
    def scipy_c_signature(self):
        args = ", ".join(["double"] * self.nin + ["int __pyx_skip_dispatch"])
        return f"double ({args})"

    def c_support_code(self, **kwargs):
        name = self.scipy_func_name
        args = ", ".join(["double"] * self.nin + ["int"])
        return dedent(
            f"""
            #ifndef PYTENSOR_SCIPY_{name}
            #define PYTENSOR_SCIPY_{name}
            static double (*pytensor_scipy_{name})({args}) = NULL;
            #endif
            """
        )

    def c_init_code(self, **kwargs):
        name = self.scipy_func_name
        signature = self.scipy_c_signature
        args = ", ".join(["double"] * self.nin + ["int"])
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
                    pytensor_scipy_{name} = (double (*)({args})) pointer;
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
        args = ", ".join([f"(double){inp}" for inp in inputs] + ["1"])
        return f"{output} = ({cast})pytensor_scipy_{self.scipy_func_name}({args});"

    def c_code_cache_version(self):
        version = super().c_code_cache_version()
        return (1, 1, *version) if version else ()
