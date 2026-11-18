"""
.. warning::

This directory is for the internal of PyTensor.

You are strongly advised not to use it, except if you know
what you are doing!

If you want to use a scalar variable in an PyTensor graph,
you probably want to use pytensor.tensor.[c,z,f,d,b,w,i,l,]scalar!
"""

import builtins
import math
from itertools import chain
from textwrap import dedent
from typing import Any

import numpy as np

import pytensor
from pytensor import printing
from pytensor.configdefaults import config
from pytensor.gradient import disconnected_type, grad_undefined
from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.fg import FrozenFunctionGraph
from pytensor.graph.op import HasInnerGraph, Op
from pytensor.graph.replace import clone_replace
from pytensor.graph.traversal import applys_between
from pytensor.graph.type import HasDataType, HasShape
from pytensor.graph.utils import MethodNotDefined
from pytensor.link.c.op import COp
from pytensor.link.c.type import CType
from pytensor.printing import pprint
from pytensor.utils import (
    apply_across_args,
    difference,
    to_return_values,
)


_abs = builtins.abs


class ComplexError(NotImplementedError):
    """
    Raised if complex numbers are used in an unsupported operation.

    """


class IntegerDivisionError(Exception):
    """
    Raised if someone tries to divide integers with '/' instead of '//'.

    """


def upcast(dtype, *dtypes) -> str:
    # This tries to keep data in floatX or lower precision, unless we
    # explicitly request a higher precision datatype.
    keep_float32 = [
        (config.cast_policy == "numpy+floatX" and config.floatX == "float32")
    ]
    keep_float16 = [
        (config.cast_policy == "numpy+floatX" and config.floatX == "float16")
    ]

    def make_array(dt):
        if dt == "float64":
            # There is an explicit float64 dtype: we cannot keep float32.
            keep_float32[0] = False
            keep_float16[0] = False
        if dt == "float32":
            keep_float16[0] = False
        return np.zeros((), dtype=dt)

    z = make_array(dtype)
    for dt in dtypes:
        z = z + make_array(dt=dt)
    rval = str(z.dtype)
    if rval == "float64":
        if keep_float16[0]:
            return "float16"
        if keep_float32[0]:
            return "float32"
    elif rval == "float32":
        if keep_float16[0]:
            return "float16"
    return rval


class NumpyAutocaster:
    """
    This class is used to cast python ints and floats to numpy arrays.

    The behavior when called on scalar `x` depends on `config.cast_policy`:
        - 'numpy' will simply use the same type as found by `numpy.asarray(x)`.
        - 'numpy+floatX' will do the same, except it will use float32 instead
          of float64 if `x` is a Python float and `config.floatX` is set to
          'float32' (note that if `x` is a numpy scalar whose data type is
          float64, it is not modified since we assume the user is purposely
          using float64).
        - 'custom' lets one define a tuple of data types such that:
            - if `x` is already a numpy scalar and its data type is in this
              tuple, then it is returned unchanged;
            - otherwise, the first data type in this tuple that can represent
              `x` without loss of precision will be used, unless `x` is a float
              and 'float32' is in the tuple (in which case `x` is cast as a
              float32);
            - if no data type can represent `x` without loss of precision, then
              the last data type in the tuple will be used.


    Parameters
    ----------
    dtypes: tuple of strings
        The ordered list of preferred data types (only used when
        `config.cast_policy` is set to 'custom', see the `NumpyAutocaster`
        help for details).

    """

    def __init__(self, dtypes):
        self.dtypes = tuple(dtypes)

    def __call__(self, x):
        # Make sure we only deal with scalars.
        assert isinstance(x, int | builtins.float) or (
            isinstance(x, np.ndarray) and x.ndim == 0
        )

        if config.cast_policy == "numpy+floatX":
            rval = np.asarray(x)
            if (
                not hasattr(x, "dtype")
                and rval.dtype in ("float64", "float32")
                and rval.dtype != config.floatX
            ):
                rval = rval.astype(config.floatX)
            return rval

        # The following is the original code, corresponding to the 'custom'
        # option for `config.cast_policy`.
        assert config.cast_policy == "custom"

        try:
            # Pass through numpy scalars, since they are already typed on
            # purpose typically.
            if str(x.dtype) in self.dtypes:
                # No need to cast `x` into a new dtype. Note that we still
                # need to convert it into an array, because it may not be
                # one already (e.g. if x == numpy.float64(1.1)).
                return np.asarray(x)
        except AttributeError:
            # Means `x` has no 'dtype' attribute.
            pass

        # unsafe downcast of float64 variables when config.floatX == 'float32'
        # recall: float is numpy.float
        if (
            isinstance(x, float)
            and config.floatX in self.dtypes
            and config.floatX != "float64"
        ):
            return np.asarray(x, dtype=config.floatX)

        # Don't autocast to float16 unless config.floatX is float16
        try_dtypes = [
            d for d in self.dtypes if config.floatX == "float16" or d != "float16"
        ]

        for dtype in try_dtypes:
            x_ = np.asarray(x).astype(dtype=dtype)
            if np.all(
                np.asarray(x) == x_
            ):  # use np.asarray(x) to match TensorType.filter
                break
        # returns either an exact x_==x, or the last cast x_
        return x_


autocast_int = NumpyAutocaster(("int8", "int16", "int32", "int64"))
# autocast_float dtypes might be manipulated in tensor.*
autocast_float = NumpyAutocaster(("float16", "float32", "float64"))


class autocast_float_as:
    """
    Temporarily adjust autocasting behavior.

    This class makes it possible to temporarily and locally adjust autocasting
    behavior when `config.cast_policy` is set to 'custom'.
    If `config.cast_policy` is not 'custom', an exception is raised.
    This class might be convenient in some code, but it definitely
    helps to test the autocasting mechanism.

    Examples
    --------
    >>> from pytensor import config
    >>> from pytensor.tensor import fvector
    >>> with config.change_flags(cast_policy="custom"):
    ...     with autocast_float_as("float32"):
    ...         assert (fvector() + 1.1).dtype == "float32"  # temporary downcasting
    ...     assert (fvector() + 1.1).dtype == "float64"  # default 'custom' behaviour

    """

    def __init__(self, *dtypes):
        self.dtypes = dtypes
        assert config.cast_policy == "custom"

    def __enter__(self):
        assert config.cast_policy == "custom"
        self.old_dtypes = autocast_float.dtypes
        autocast_float.dtypes = self.dtypes

    def __exit__(self, *args):
        assert config.cast_policy == "custom"
        autocast_float.dtypes = self.old_dtypes


def convert(x, dtype=None):
    """Convert the input to a properly typed NumPy value according to the current casting policy.

    Parameters
    ----------
    x : Number, numpy.ndarray, or Sequence[Number]
        The value(s) to be converted
    dtype : str or numpy.dtype (optional)
        The dtype to use for the conversion of `x`.

    """
    if isinstance(x, np.ma.MaskedArray):
        raise NotImplementedError("MaskedArrays are not supported")

    if dtype is not None:
        # in this case, the semantics are that the caller is forcing the dtype
        if dtype == "floatX":
            dtype = config.floatX
        x_ = np.asarray(x).astype(dtype)
    else:
        # In this case, this function should infer the dtype according to the
        # autocasting rules. See autocasting above.
        x_ = None
        if isinstance(x, int):
            try:
                x_ = autocast_int(x)
            except OverflowError:
                # This is to imitate numpy behavior which tries to fit
                # bigger numbers into a uint64.
                x_ = np.asarray(x, dtype="uint64")
        elif isinstance(x, builtins.float):
            x_ = autocast_float(x)
        elif isinstance(x, np.ndarray):
            x_ = x
        else:
            # Here x is probably a list or a tuple. If it contains a
            # long, we will behave like the current NumPy version: it
            # will work if the long fits in int64 or uint64.
            x_ = np.asarray(x)
            if x_.size == 0 and not hasattr(x, "dtype"):
                x_ = np.asarray(x, dtype=config.floatX)
    assert issubclass(type(x_), np.ndarray | np.memmap)
    return x_


class ScalarType(CType, HasDataType, HasShape):
    """
    Internal class, should not be used by clients.

    Primarily used by tensor.elemwise and tensor.reduce.
    Analogous to TensorType, but for zero-dimensional objects.
    Maps directly to C primitives.

    TODO: refactor to be named ScalarType for consistency with TensorType.

    """

    __props__ = ("dtype",)
    ndim = 0
    shape = ()

    def __init__(self, dtype):
        if isinstance(dtype, str) and dtype == "floatX":
            dtype = config.floatX
        else:
            dtype = np.dtype(dtype).name

        self.dtype = dtype
        self.dtype_specs()  # error checking

    def clone(self, dtype=None, **kwargs):
        if dtype is None:
            dtype = self.dtype
        return type(self)(dtype)

    def filter(self, data, strict=False, allow_downcast=None):
        py_type = self.dtype_specs()[0]
        if strict and not isinstance(data, py_type):
            raise TypeError(
                f"{self} expected a {py_type}, got {data} of type {type(data)}",
                data,
            )
        try:
            converted_data = py_type(data)
            if (
                allow_downcast
                or (
                    allow_downcast is None
                    and isinstance(data, float | np.floating)
                    and self.dtype == config.floatX
                )
                or np.array_equal(data, converted_data, equal_nan=True)
            ):
                return py_type(data)
            else:
                raise TypeError(
                    "Value cannot accurately be converted to dtype"
                    f" ({self.dtype}) and allow_downcast is not True"
                )
        except Exception:
            raise TypeError(
                f"Could not convert {type(data)} (value={data}) to {self.dtype}",
            )

    def values_eq_approx(self, a, b, tolerance=1e-4):
        # The addition have risk of overflow especially with [u]int8
        if self.dtype == "bool":
            return a == b
        diff = a - b
        if diff == 0:
            return True
        return _abs(diff) <= (_abs(a) * tolerance) + (_abs(b) * tolerance)

    def c_element_type(self):
        return self.dtype_specs()[1]

    def c_headers(self, c_compiler=None, **kwargs):
        l = ["<math.h>"]
        # These includes are needed by ScalarType and TensorType,
        # we declare them here and they will be re-used by TensorType
        l.append("<numpy/arrayobject.h>")
        l.append("<numpy/arrayscalars.h>")
        l.append("<numpy/npy_math.h>")

        if config.lib__amdlibm and c_compiler.supports_amdlibm:
            l += ["<amdlibm.h>"]
        return l

    def c_libraries(self, c_compiler=None, **kwargs):
        l = []
        if config.lib__amdlibm and c_compiler and c_compiler.supports_amdlibm:
            l += ["amdlibm"]
        return l

    def c_compile_args(self, c_compiler=None, **kwargs):
        if config.lib__amdlibm and c_compiler and c_compiler.supports_amdlibm:
            return ["-DREPLACE_WITH_AMDLIBM"]
        else:
            return []

    def dtype_specs(self):
        try:
            # To help debug dtype/typenum problem, here is code to get
            # the list of numpy typenum.  This list change between 32
            # and 64 bit platform and probably also also between
            # Windows and Linux.
            # NOTE: equivalent type on a platform can have different typenum.
            #     This is the source of all dtype/typenum problem found up to
            #     now, as PyTensor always expect the exact typenum that
            #     correspond to our supported dtype.
            """
            for dtype in ['bool', 'int8', 'uint8', 'short', 'ushort', 'intc',
                          'uintc',
                          'longlong', 'ulonglong', 'single', 'double',
                          'longdouble', 'csingle', 'cdouble', 'clongdouble',
                          'float32', 'float64', 'int8', 'int16', 'int32',
                          'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                          'complex64', 'complex128', 'float', 'double',
                          'int', 'uint']:
                print(dtype, np.zeros(1, dtype=dtype).dtype.num)
            """
            return {  # dtype: (py_type, c_type, cls_name)
                "float16": (np.float16, "npy_float16", "Float16"),
                "float32": (np.float32, "npy_float32", "Float32"),
                "float64": (np.float64, "npy_float64", "Float64"),
                "complex128": (np.complex128, "pytensor_complex128", "Complex128"),
                "complex64": (np.complex64, "pytensor_complex64", "Complex64"),
                "bool": (np.bool_, "npy_bool", "Bool"),
                "uint8": (np.uint8, "npy_uint8", "UInt8"),
                "int8": (np.int8, "npy_int8", "Int8"),
                "uint16": (np.uint16, "npy_uint16", "UInt16"),
                "int16": (np.int16, "npy_int16", "Int16"),
                "uint32": (np.uint32, "npy_uint32", "UInt32"),
                "int32": (np.int32, "npy_int32", "Int32"),
                "uint64": (np.uint64, "npy_uint64", "UInt64"),
                "int64": (np.int64, "npy_int64", "Int64"),
            }[self.dtype]
        except KeyError:
            raise TypeError(
                f"Unsupported dtype for {self.__class__.__name__}: {self.dtype}"
            )

    def upcast(self, *others):
        return upcast(*[x.dtype for x in [self, *others]])

    def make_variable(self, name=None):
        return ScalarVariable(self, None, name=name)

    def __str__(self):
        return str(self.dtype)

    def __repr__(self):
        return f"ScalarType({self.dtype})"

    def c_literal(self, data):
        if "complex" in self.dtype:
            return None
        if self.dtype == "bool":
            return "1" if data else "0"
        if data == np.inf:
            return "INFINITY"
        if data == -np.inf:
            return "-INFINITY"
        if np.isnan(data):
            return "NAN"
        return str(data)

    def c_declare(self, name, sub, check_input=True):
        dtype = self.dtype_specs()[1]

        if check_input:
            pre = f"""
                typedef {dtype} dtype_{name};
            """
        else:
            pre = ""

        return f"""{pre}
        {dtype} {name};
        """

    def c_init(self, name, sub):
        return f"""
        {name} = 0;
        """

    def c_extract(self, name, sub, check_input=True, **kwargs):
        if self.dtype == "float16":
            # This doesn't work at the numpy level
            raise NotImplementedError("float16")
        specs = self.dtype_specs()
        if check_input:
            fail = sub["fail"]
            dtype = specs[1]
            pyarr_type = f"Py{specs[2]}ArrType_Type"
            pre = f"""
            if (!PyObject_TypeCheck(py_{name}, &{pyarr_type}))
            {{
                PyErr_Format(PyExc_ValueError,
                    "Scalar check failed ({dtype})");
                {fail}
            }}
            """
        else:
            pre = ""
        return (
            pre
            + f"""
        PyArray_ScalarAsCtype(py_{name}, &{name});
        """
        )

    def c_sync(self, name, sub):
        specs = self.dtype_specs()
        fail = sub["fail"]
        (np_dtype, _c_dtype, _cls_name) = specs
        np_dtype_num = np.dtype(np_dtype).num

        return f"""
        Py_XDECREF(py_{name});

        PyArray_Descr* {name}_descr = PyArray_DescrFromType({np_dtype_num});  // {np_dtype}
        if (!{name}_descr) {{
            PyErr_Format(PyExc_RuntimeError, "Could not get descriptor for {np_dtype_num}={np_dtype}");
            {fail}
        }}

        // PyArray_Scalar creates a new scalar object by copying data from the pointer &{name}
        py_{name} = PyArray_Scalar(&{name}, {name}_descr, NULL);

        // Clean up the descriptor reference (PyArray_DescrFromType returns a new ref)
        Py_DECREF({name}_descr);

        if (!py_{name})
        {{
            Py_XINCREF(Py_None);
            py_{name} = Py_None;
            PyErr_Format(PyExc_MemoryError,
                "Instantiation of new Python NumPy scalar failed ({np_dtype_num}={np_dtype})");
            {fail}
        }}
        """

    def c_cleanup(self, name, sub):
        return ""

    def c_support_code(self, **kwargs):
        if self.dtype.startswith("complex"):
            cplx_types = ["pytensor_complex64", "pytensor_complex128"]
            real_types = [
                "npy_int8",
                "npy_int16",
                "npy_int32",
                "npy_int64",
                "npy_uint8",  # also covers npy_bool
                "npy_uint16",
                "npy_uint32",
                "npy_uint64",
                "npy_float32",
                "npy_float64",
            ]
            # If the 'int' C type is not exactly the same as an existing
            # 'npy_intX', some C code may not compile, e.g. when assigning
            # the value 0 (cast to 'int' in C) to an PyTensor_complex64.
            if np.dtype("intc").num not in [np.dtype(d[4:]).num for d in real_types]:
                # In that case we add the 'int' type to the real types.
                real_types.append("int")

            def _make_get_set_real_imag(scalar_type: str) -> str:
                """Make overloaded getter/setter functions for real/imag parts of numpy complex types.

                The functions called by these getter/setter functions are defining in npy_math.h

                Args:
                    scalar_type: float, double, or longdouble

                Returns:
                    C++ code for defining set_real, set_imag, get_real, and get_imag, overloaded for the
                    given type.
                """
                complex_type = "npy_c" + scalar_type
                suffix = "" if scalar_type == "double" else scalar_type[0]

                if scalar_type == "longdouble":
                    scalar_type = "npy_" + scalar_type

                return_type = scalar_type

                template = f"""
                static inline {return_type} get_real(const {complex_type} z)
                {{
                    return npy_creal{suffix}(z);
                }}

                static inline void set_real({complex_type} *z, const {scalar_type} r)
                {{
                    NPY_CSETREAL{suffix.upper()}(z, r);
                }}

                static inline {return_type} get_imag(const {complex_type} z)
                {{
                    return npy_cimag{suffix}(z);
                }}

                static inline void set_imag({complex_type} *z, const {scalar_type} i)
                {{
                    NPY_CSETIMAG{suffix.upper()}(z, i);
                }}
                """
                return template

            get_set_aliases = "\n".join(
                _make_get_set_real_imag(stype)
                for stype in ["float", "double", "longdouble"]
            )

            # Template for defining pytensor_complex64 and pytensor_complex128 structs/classes
            #
            # The npy_complex64, npy_complex128 types are aliases defined at run time based on
            # the size of floats and doubles on the machine. This means that both types are
            # not necessarily defined on every machine, but a machine with 32-bit floats and
            # 64-bit doubles will have npy_complex64 as an alias of npy_cfloat and npy_complex128
            # as an alias of npy_complex128.
            #
            # In any case, the get/set real/imag functions defined above will always work for
            # npy_complex64 and npy_complex128.
            template = """
            struct pytensor_complex%(nbits)s : public npy_complex%(nbits)s {
              typedef pytensor_complex%(nbits)s complex_type;
              typedef npy_float%(half_nbits)s scalar_type;

              complex_type operator+(const complex_type &y) const {
                complex_type ret;
                set_real(&ret, get_real(*this) + get_real(y));
                set_imag(&ret, get_imag(*this) + get_imag(y));
                return ret;
              }

              complex_type operator-() const {
                complex_type ret;
                set_real(&ret, -get_real(*this));
                set_imag(&ret, -get_imag(*this));
                return ret;
              }
              bool operator==(const complex_type &y) const {
                return (get_real(*this) == get_real(y)) && (get_imag(*this) == get_imag(y));
              }
              bool operator==(const scalar_type &y) const {
                return (get_real(*this) == y) && (get_real(*this) == 0);
              }
              complex_type operator-(const complex_type &y) const {
                complex_type ret;
                set_real(&ret, get_real(*this) - get_real(y));
                set_imag(&ret, get_imag(*this) - get_imag(y));
                return ret;
              }
              complex_type operator*(const complex_type &y) const {
                complex_type ret;
                set_real(&ret, get_real(*this) * get_real(y) - get_imag(*this) * get_imag(y));
                set_imag(&ret, get_imag(*this) * get_real(y) + get_real(*this) * get_imag(y));
                return ret;
              }
              complex_type operator/(const complex_type &y) const {
                complex_type ret;
                scalar_type y_norm_square = get_real(y) * get_real(y) + get_imag(y) * get_imag(y);
                set_real(&ret, (get_real(*this) * get_real(y) + get_imag(*this) * get_imag(y)) / y_norm_square);
                set_imag(&ret, (get_imag(*this) * get_real(y) - get_real(*this) * get_imag(y)) / y_norm_square);
                return ret;
              }
              template <typename T> complex_type &operator=(const T &y);


              pytensor_complex%(nbits)s() {}

              template <typename T> pytensor_complex%(nbits)s(const T &y) { *this = y; }

              template <typename TR, typename TI>
              pytensor_complex%(nbits)s(const TR &r, const TI &i) {
                set_real(this, r);
                set_imag(this, i);
              }
            };
            """

            def operator_eq_real(mytype, othertype):
                return f"""
                template <> {mytype} & {mytype}::operator=<{othertype}>(const {othertype} & y)
                {{ set_real(this, y); set_imag(this, 0); return *this; }}
                """

            def operator_eq_cplx(mytype, othertype):
                return f"""
                template <> {mytype} & {mytype}::operator=<{othertype}>(const {othertype} & y)
                {{ set_real(this, get_real(y)); set_imag(this, get_imag(y)); return *this; }}
                """

            operator_eq = "".join(
                operator_eq_real(ctype, rtype)
                for ctype in cplx_types
                for rtype in real_types
            ) + "".join(
                operator_eq_cplx(ctype1, ctype2)
                for ctype1 in cplx_types
                for ctype2 in cplx_types
            )

            # We are not using C++ generic templating here, because this would
            # generate two different functions for adding a complex64 and a
            # complex128, one returning a complex64, the other a complex128,
            # and the compiler complains it is ambiguous.
            # Instead, we generate code for known and safe types only.

            def operator_plus_real(mytype, othertype):
                return f"""
                const {mytype} operator+(const {mytype} &x, const {othertype} &y)
                {{ return {mytype}(get_real(x) + y, get_imag(x)); }}

                const {mytype} operator+(const {othertype} &y, const {mytype} &x)
                {{ return {mytype}(get_real(x) + y, get_imag(x)); }}
                """

            operator_plus = "".join(
                operator_plus_real(ctype, rtype)
                for ctype in cplx_types
                for rtype in real_types
            )

            def operator_minus_real(mytype, othertype):
                return f"""
                const {mytype} operator-(const {mytype} &x, const {othertype} &y)
                {{ return {mytype}(get_real(x) - y, get_imag(x)); }}

                const {mytype} operator-(const {othertype} &y, const {mytype} &x)
                {{ return {mytype}(y - get_real(x), -get_imag(x)); }}
                """

            operator_minus = "".join(
                operator_minus_real(ctype, rtype)
                for ctype in cplx_types
                for rtype in real_types
            )

            def operator_mul_real(mytype, othertype):
                return f"""
                const {mytype} operator*(const {mytype} &x, const {othertype} &y)
                {{ return {mytype}(get_real(x) * y, get_imag(x) * y); }}

                const {mytype} operator*(const {othertype} &y, const {mytype} &x)
                {{ return {mytype}(get_real(x) * y, get_imag(x) * y); }}
                """

            operator_mul = "".join(
                operator_mul_real(ctype, rtype)
                for ctype in cplx_types
                for rtype in real_types
            )

            return (
                get_set_aliases
                + template % dict(nbits=64, half_nbits=32)
                + template % dict(nbits=128, half_nbits=64)
                + operator_eq
                + operator_plus
                + operator_minus
                + operator_mul
            )

        else:
            return ""

    def c_init_code(self, **kwargs):
        return ["import_array();"]

    def c_code_cache_version(self):
        return (15, np.__version__)

    def get_shape_info(self, obj):
        return obj.itemsize

    def get_size(self, shape_info):
        return shape_info


def get_scalar_type(dtype, cache: dict[str, ScalarType] = {}) -> ScalarType:
    """
    Return a ScalarType(dtype) object.

    This caches objects to save allocation and run time.

    """
    try:
        return cache[dtype]
    except KeyError:
        cache[dtype] = res = ScalarType(dtype=dtype)
    return res


# Register C code for ViewOp on Scalars.
pytensor.compile.ops.register_view_op_c_code(
    ScalarType,
    """
    %(oname)s = %(iname)s;
    """,
    1,
)


bool: ScalarType = get_scalar_type("bool")
int8: ScalarType = get_scalar_type("int8")
int16: ScalarType = get_scalar_type("int16")
int32: ScalarType = get_scalar_type("int32")
int64: ScalarType = get_scalar_type("int64")
uint8: ScalarType = get_scalar_type("uint8")
uint16: ScalarType = get_scalar_type("uint16")
uint32: ScalarType = get_scalar_type("uint32")
uint64: ScalarType = get_scalar_type("uint64")
float16: ScalarType = get_scalar_type("float16")
float32: ScalarType = get_scalar_type("float32")
float64: ScalarType = get_scalar_type("float64")
complex64: ScalarType = get_scalar_type("complex64")
complex128: ScalarType = get_scalar_type("complex128")

type _ScalarTypes = tuple[ScalarType, ...]
int_types: _ScalarTypes = (int8, int16, int32, int64)
uint_types: _ScalarTypes = (uint8, uint16, uint32, uint64)
float_types: _ScalarTypes = (float16, float32, float64)
complex_types: _ScalarTypes = (complex64, complex128)

integer_types: _ScalarTypes = int_types + uint_types
discrete_types: _ScalarTypes = (bool, *integer_types)
continuous_types: _ScalarTypes = float_types + complex_types
all_types: _ScalarTypes = discrete_types + continuous_types

discrete_dtypes = tuple(t.dtype for t in discrete_types)
integer_dtypes = tuple(t.dtype for t in integer_types)


class _scalar_py_operators:
    # These can't work because Python requires native output types
    def __bool__(self):
        raise TypeError(
            "ScalarVariable cannot be converted to Python boolean. "
            "Call `.astype(bool)` for the symbolic equivalent."
        )

    def __index__(self):
        raise TypeError(
            "ScalarVariable cannot be converted to Python integer. "
            "Call `.astype(int)` for the symbolic equivalent."
        )

    def __int__(self):
        raise TypeError(
            "ScalarVariable cannot be converted to Python integer. "
            "Call `.astype(int)` for the symbolic equivalent."
        )

    def __float__(self):
        raise TypeError(
            "ScalarVariable cannot be converted to Python float. "
            "Call `.astype(float)` for the symbolic equivalent."
        )

    def __complex__(self):
        raise TypeError(
            "ScalarVariable cannot be converted to Python complex number. "
            "Call `.astype(complex)` for the symbolic equivalent."
        )

    # So that we can simplify checking code when we have a mixture of ScalarType
    # variables and Tensor variables
    ndim = 0

    dtype = property(lambda self: self.type.dtype)
    """The dtype of this scalar."""

    @property
    def shape(self):
        from pytensor.tensor.basic import as_tensor_variable

        return as_tensor_variable([], ndim=1, dtype=np.int64)

    # UNARY
    def __abs__(self):
        return abs(self)

    def __neg__(self):
        return neg(self)

    # BITWISE
    def __invert__(self):
        return invert(self)

    def __and__(self, other):
        return and_(self, other)

    def __or__(self, other):
        return or_(self, other)

    def __xor__(self, other):
        return xor(self, other)

    def __rand__(self, other):
        return and_(other, self)

    def __ror__(self, other):
        return or_(other, self)

    def __rxor__(self, other):
        return xor(other, self)

    # COMPARISONS
    def __lt__(self, other):
        return lt(self, other)

    def __le__(self, other):
        return le(self, other)

    def __gt__(self, other):
        return gt(self, other)

    def __ge__(self, other):
        return ge(self, other)

    # ARITHMETIC - NORMAL
    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __truediv__(self, other):
        return true_div(self, other)

    def __floordiv__(self, other):
        return int_div(self, other)

    def __mod__(self, other):
        return mod_check(self, other)

    def __pow__(self, other):
        return pow(self, other)

    # ARITHMETIC - RIGHT-OPERAND
    def __radd__(self, other):
        return add(other, self)

    def __rsub__(self, other):
        return sub(other, self)

    def __rmul__(self, other):
        return mul(other, self)

    def __rtruediv__(self, other):
        return true_div(other, self)

    def __rfloordiv__(self, other):
        return int_div(other, self)

    def __rmod__(self, other):
        return mod(other, self)

    def __rpow__(self, other):
        return pow(other, self)

    def zeros_like(self, dtype=None):
        # The second is needed for Elemwise ops to work right
        if dtype is None:
            dtype = str(self.type.dtype)
        return second(self, ScalarConstant(get_scalar_type(dtype), 0))

    def ones_like(self, dtype=None):
        # The second is needed for Elemwise ops to work right
        if dtype is None:
            dtype = str(self.type.dtype)
        return second(self, ScalarConstant(get_scalar_type(dtype), 1))

    def astype(self, dtype):
        return cast(self, dtype)


class ScalarVariable(_scalar_py_operators, Variable):
    pass


ScalarType.variable_type = ScalarVariable


class ScalarConstantSignature(tuple):
    """Signature for ScalarConstant that handles NaN equality and hashing."""

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        (t0, d0), (t1, d1) = self, other
        if t0 != t1:
            return False
        return (d0 == d1) or (np.isnan(d0) and np.isnan(d1))

    def __hash__(self):
        t, d = self
        if np.isnan(d):
            return hash((type(self), t, "NaN"))
        return hash((type(self), t, d))


class ScalarConstant(ScalarVariable, Constant):
    def __init__(self, *args, **kwargs):
        Constant.__init__(self, *args, **kwargs)

    def signature(self):
        return ScalarConstantSignature((self.type, self.data))


# Register ScalarConstant as the type of Constant corresponding to ScalarType
ScalarType.constant_type = ScalarConstant


def constant(x, name=None, dtype=None) -> ScalarConstant:
    x = convert(x, dtype=dtype)
    assert x.ndim == 0
    return ScalarConstant(get_scalar_type(str(x.dtype)), x, name=name)


def as_scalar(x: Any, name: str | None = None) -> ScalarVariable:
    if isinstance(x, Variable):
        if isinstance(x.type, ScalarType):
            # NOTE: We may have a Variable with ScalarType that is not ScalarVariable/ScalarConstant/ScalarSharedVariable
            # if it bypasses those constructs (same issue with TensorVariables). We should make that impossible to happen!
            return x  # type: ignore[return-value]

        from pytensor.tensor.basic import scalar_from_tensor
        from pytensor.tensor.type import TensorType

        if isinstance(x.type, TensorType) and x.type.ndim == 0:
            return scalar_from_tensor(x)
        else:
            raise TypeError(f"Cannot convert {x} to a scalar type")

    if isinstance(x, Apply):
        # FIXME: Why do we support calling this with Apply?
        #  Also, if we do, why can't we support multiple outputs?
        if len(x.outputs) != 1:
            raise ValueError(
                "It is ambiguous which output of a multi-output Op has to be fetched.",
                x,
            )
        return as_scalar(x.outputs[0])

    return constant(x)


ints = apply_across_args(int64)
floats = apply_across_args(float64)
complexs = apply_across_args(complex128)
complexs64 = apply_across_args(complex64)
complexs128 = apply_across_args(complex128)


def _float_dtype(dtype):
    if dtype in ("bool", "int8", "int16", "uint8", "uint16"):
        return "float32"
    if dtype in ("int32", "int64", "uint32", "uint64"):
        return "float64"
    return dtype


def _real_dtype(dtype):
    if dtype == "complex64":
        return "float32"
    if dtype == "complex128":
        return "float64"
    return dtype


def upgrade_to_float(*types) -> tuple[ScalarType]:
    """Choose a floating dtype for tensor functions that accept integer inputs."""
    dtypes = tuple(_float_dtype(t.dtype) for t in types)
    dtype = dtypes[0] if len(dtypes) == 1 else upcast(*dtypes)
    return (get_scalar_type(dtype),)


def resolve_weak_dtypes(input_dtypes):
    """Resolve Python scalar type markers against explicitly typed operands."""
    if all(isinstance(dtype, str) for dtype in input_dtypes):
        return input_dtypes
    strong = [dtype for dtype in input_dtypes if isinstance(dtype, str)]
    if not strong:
        defaults = {
            int: "int64",
            builtins.float: config.floatX,
            builtins.complex: "complex128",
        }
        return tuple(defaults[dtype] for dtype in input_dtypes)
    dtype = strong[0] if len(strong) == 1 else upcast(*strong)
    if builtins.complex in input_dtypes:
        weak_dtype = (
            "complex64"
            if dtype in ("float16", "float32", "complex64")
            else "complex128"
        )
    elif builtins.float in input_dtypes and dtype in discrete_dtypes:
        weak_dtype = config.floatX
    elif dtype == "bool":
        weak_dtype = "int64"
    else:
        weak_dtype = dtype
    return tuple(
        dtype if isinstance(dtype, str) else weak_dtype for dtype in input_dtypes
    )


def as_variables(inputs, as_variable=as_scalar, resolve_dtypes=None):
    """Convert inputs together, preserving NumPy's weak Python scalar promotion.

    Explicitly typed inputs keep their dtype. Under ``numpy+floatX``, Python
    int, float and complex literals take their precision from the other inputs.
    Without a typed input, constants follow the usual casting policy.
    ``resolve_dtypes`` determines the operation's input conversions before any
    literal is materialized. Without it, ordinary numeric promotion is used.
    """
    # This module defines its own complex
    weak_types = (builtins.int, builtins.float, builtins.complex)
    if len(inputs) < 2 or not any(type(inp) in weak_types for inp in inputs):
        return [as_variable(inp) for inp in inputs]
    if config.cast_policy == "custom":
        return [as_variable(inp) for inp in inputs]

    weak_inputs = {}
    variables = []
    for i, inp in enumerate(inputs):
        if type(inp) in weak_types:
            weak_inputs[i] = inp
            variables.append(None)
        else:
            variables.append(as_variable(inp))

    dtypes = [inp.dtype for inp in variables if inp is not None]
    if not dtypes:
        return [as_variable(inp) for inp in inputs]

    input_dtypes = tuple(
        type(inp) if i in weak_inputs else variables[i].dtype
        for i, inp in enumerate(inputs)
    )
    if resolve_dtypes is None:
        resolved = resolve_weak_dtypes(input_dtypes)
    else:
        resolved, _ = resolve_dtypes(input_dtypes)
    for i, inp in weak_inputs.items():
        variables[i] = as_variable(np.asarray(inp, dtype=resolved[i]))
    return variables


def _resolve_numeric_dtypes(input_dtypes):
    inputs = resolve_weak_dtypes(input_dtypes)
    dtype = inputs[0] if len(inputs) == 1 else upcast(*inputs)
    return inputs, (dtype,)


def _resolve_numeric_bool_to_int8_dtypes(self, input_dtypes):
    inputs, (output,) = _resolve_numeric_dtypes(input_dtypes)
    return inputs, ("int8" if output == "bool" else output,)


def _resolve_real_dtypes(self, input_dtypes):
    inputs = resolve_weak_dtypes(input_dtypes)
    return inputs, (_real_dtype(inputs[0]),)


def _resolve_float_only_dtypes(self, input_dtypes):
    inputs = resolve_weak_dtypes(input_dtypes)
    if inputs[0] not in ("float16", "float32", "float64"):
        raise TypeError("only float types are supported")
    return inputs, inputs


def _resolve_nonbool_dtypes(self, input_dtypes):
    inputs = resolve_weak_dtypes(input_dtypes)
    if inputs[0] == "bool":
        raise TypeError("bool input not supported")
    return inputs, inputs


def _resolve_float_dtypes(self, input_dtypes):
    if len(input_dtypes) == 1 and input_dtypes[0] in (
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ):
        return input_dtypes, input_dtypes
    inputs = resolve_weak_dtypes(input_dtypes)
    floats = tuple(_float_dtype(dtype) for dtype in inputs)
    output = floats[0] if len(floats) == 1 else upcast(*floats)
    if inputs is not input_dtypes:
        inputs = tuple(
            dtype if isinstance(given, str) else output
            for given, dtype in zip(input_dtypes, inputs, strict=True)
        )
    return inputs, (output,)


def _resolve_float_no_complex_dtypes(self, input_dtypes):
    inputs, outputs = _resolve_float_dtypes(self, input_dtypes)
    if any(dtype.startswith("complex") for dtype in inputs):
        raise TypeError("complex argument not supported")
    return inputs, outputs


def _cast_to_promised_scalar_dtype(x, dtype):
    try:
        return x.astype(dtype)
    except AttributeError:
        if dtype == "bool":
            return np.bool_(x)
        else:
            return getattr(np, dtype)(x)


class ScalarOp(COp):
    nin = -1
    nout = 1

    name: str | None = None

    def resolve_dtypes(self, input_dtypes):
        """Return input and output dtype tuples, accepting weak Python scalar types.

        Concrete dtypes are strings. Python int, float and complex type objects
        describe weak literals. Explicitly typed operands retain their dtype.
        """
        return _resolve_numeric_dtypes(input_dtypes)

    def make_node(self, *inputs):
        if self.nin >= 0:
            if len(inputs) != self.nin:
                raise TypeError(
                    f"Wrong number of inputs for {self}.make_node "
                    f"(got {len(inputs)}({inputs}), expected {self.nin})"
                )
        inputs = as_variables(inputs, resolve_dtypes=self.resolve_dtypes)
        _, output_dtypes = self.resolve_dtypes(tuple(input.dtype for input in inputs))
        outputs = [get_scalar_type(dtype)() for dtype in output_dtypes]
        if len(outputs) != self.nout:
            inputs_str = (", ".join(str(input) for input in inputs),)
            raise TypeError(
                f"Not the right number of outputs produced for {self}({inputs_str}). "
                f"Expected {self.nout}, got {len(outputs)}."
            )
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, output_storage):
        if self.nout == 1:
            output_storage[0][0] = _cast_to_promised_scalar_dtype(
                self.impl(*inputs),
                node.outputs[0].dtype,
            )
        else:
            # strict=False because we are in a hot loop
            for out, storage, variable in zip(
                node.outputs, output_storage, self.impl(*inputs), strict=False
            ):
                storage[0] = _cast_to_promised_scalar_dtype(variable, out.dtype)

    def impl(self, *inputs):
        raise MethodNotDefined("impl", type(self), self.__class__.__name__)

    def pullback(self, inputs, outputs, output_gradients):
        # Fall back to deprecated L_op/grad if overridden by subclass
        if type(self).L_op is not Op.L_op:
            return type(self).L_op(self, inputs, outputs, output_gradients)
        if type(self).grad is not Op.grad:
            return self.grad(inputs, output_gradients)
        raise MethodNotDefined("pullback", type(self), self.__class__.__name__)

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        if hasattr(self, "name") and self.name:
            return self.name
        return self.__class__.__name__

    def c_code_cache_version(self):
        return (4,)

    def c_code_contiguous(self, node, name, inp, out, sub):
        """
        This function is called by Elemwise when all inputs and outputs are
        c_contiguous. This allows to use the SIMD version of this op.

        The inputs are the same as c_code except that:

            - inp and out must be the names of the variables associated to the
              ndarrays in the C code
            - node must be the elemwise node (this is needed to know
              the inputs/outputs types)

        """
        raise MethodNotDefined()

    def supports_c_code(self, inputs, outputs):
        """Returns True if the current op has functioning C code for
        the given Elemwise inputs, outputs.

        """
        tmp_s_input = []
        # To keep the same aliasing between inputs
        mapping = {}
        for ii in inputs:
            if ii in mapping:
                tmp_s_input.append(mapping[ii])
            else:
                tmp = mapping[ii] = get_scalar_type(ii.dtype).make_variable()
                tmp_s_input.append(tmp)

        try:
            self.c_code(
                self.make_node(*tmp_s_input),
                "test_presence_of_c_code",
                # FIXME: Shouldn't this be a unique name per unique variable?
                ["x" for x in inputs],
                ["z" for z in outputs],
                {"fail": "%(fail)s"},
            )
        except (NotImplementedError, MethodNotDefined):
            return False
        return True


class UnaryScalarOp(ScalarOp):
    nin = 1
    amd_float32: str | None = None
    amd_float64: str | None = None

    preserves_zero = False
    monotonic_increasing = False
    monotonic_decreasing = False

    # Set only when the output is >= 0 for every real input, including the sign of zero
    # (`sqrt` returns -0.0 for -0.0, so it does not qualify). Consumers must still check
    # the dtype: signed integers wrap on overflow and complex values are unordered.
    non_negative = False

    def c_code_contiguous(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if not config.lib__amdlibm or node.inputs[0].type != node.outputs[0].type:
            raise MethodNotDefined()

        dtype = node.inputs[0].type.dtype_specs()[1]
        fct_call = self.c_code_contiguous_raw(dtype, "n", "x", "z")
        return f"""
{{
        npy_intp n = PyArray_SIZE({z});
        {dtype} * x = ({dtype}*) PyArray_DATA({x});
        {dtype} * z = ({dtype}*) PyArray_DATA({z});
        {fct_call};
}}
        """

    def c_code_contiguous_raw(self, dtype, n, i, o):
        if not config.lib__amdlibm:
            raise MethodNotDefined()
        if dtype.startswith("npy_"):
            dtype = dtype[4:]
        if dtype == "float32" and self.amd_float32 is not None:
            dtype = "float"
            fct = self.amd_float32
        elif dtype == "float64" and self.amd_float64 is not None:
            dtype = "double"
            fct = self.amd_float64
        else:
            raise MethodNotDefined()
        return f"{fct}({n}, {i}, {o})"


class BinaryScalarOp(ScalarOp):
    # One may define in subclasses the following fields:
    #   - `commutative`: whether op(a, b) == op(b, a)
    #   - `associative`: whether op(op(a, b), c) == op(a, op(b, c))
    commutative: builtins.bool | None = None
    associative: builtins.bool | None = None
    identity: builtins.bool | builtins.float | builtins.int | None = None
    """
    For an associative operation, the identity object corresponds to the neutral
    element. For instance, it will be ``0`` for addition, ``1`` for multiplication,
    ``True`` for ``and``, ``False`` for ``or``.
    """
    nin = 2


class LogicalComparison(BinaryScalarOp):
    def resolve_dtypes(self, input_dtypes):
        return resolve_weak_dtypes(input_dtypes), ("bool",)

    def pullback(self, inputs, outputs, output_gradients):
        x, y = inputs
        assert outputs[0].type == bool
        return [
            x.zeros_like(dtype=config.floatX),
            y.zeros_like(dtype=config.floatX),
        ]

    def c_code_cache_version(self):
        super_version = super().c_code_cache_version()
        return (*super_version, 0)


class FixedLogicalComparison(UnaryScalarOp):
    """
    Comparison to a fixed value.

    """

    resolve_dtypes = LogicalComparison.resolve_dtypes

    def pullback(self, inputs, outputs, output_gradients):
        (x,) = inputs
        assert outputs[0].type == bool
        return [x.zeros_like(dtype=config.floatX)]

    def c_code_cache_version(self):
        super_version = super().c_code_cache_version()
        return (*super_version, 0)


class LT(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    nfunc_spec = ("less", 2, 1)

    def impl(self, x, y):
        # built-in < don't support complex
        return np.less(x, y)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return f"{z} = ({x} < {y});"


lt = LT()


class GT(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    nfunc_spec = ("greater", 2, 1)

    def impl(self, x, y):
        # built-in > don't support complex
        return np.greater(x, y)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return f"{z} = ({x} > {y});"


gt = GT()


class LE(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    nfunc_spec = ("less_equal", 2, 1)

    def impl(self, x, y):
        # built-in <= don't support complex
        return np.less_equal(x, y)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return f"{z} = ({x} <= {y});"


le = LE()


class GE(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    nfunc_spec = ("greater_equal", 2, 1)

    def impl(self, x, y):
        # built-in >= don't support complex
        return np.greater_equal(x, y)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return f"{z} = ({x} >= {y});"


ge = GE()


class EQ(LogicalComparison):
    identity = False
    commutative = True
    associative = False
    nfunc_spec = ("equal", 2, 1)

    def impl(self, x, y):
        return x == y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        return f"{z} = ({x} == {y});"


eq = EQ()


class NEQ(LogicalComparison):
    identity = False
    commutative = True
    associative = False
    nfunc_spec = ("not_equal", 2, 1)

    def impl(self, x, y):
        return x != y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return f"{z} = ({x} != {y});"


neq = NEQ()


class IsNan(FixedLogicalComparison):
    nfunc_spec = ("isnan", 1, 1)

    def impl(self, x):
        return np.isnan(x)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        # Discrete type can never be nan
        if node.inputs[0].type in discrete_types:
            return f"{z} = false;"

        # Windows tries to be different and sometimes return -1, but we want
        # to be consistent with numpy (which returns True), hence the "abs".
        return f"{z} = abs(isnan({x}));"

    def c_code_cache_version(self):
        scalarop_version = super().c_code_cache_version()
        return (*scalarop_version, 3)


isnan = IsNan()


class IsInf(FixedLogicalComparison):
    nfunc_spec = ("isinf", 1, 1)

    def impl(self, x):
        return np.isinf(x)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        # Discrete type can never be inf
        if node.inputs[0].type in discrete_types:
            return f"{z} = false;"

        # Note that the C isinf returns -1 for -Inf and +1 for +Inf, while
        # numpy simply returns True: we mimic numpy's behavior here, thus
        # the absolute value.
        return f"{z} = abs(isinf({x}));"

    def c_code_cache_version(self):
        scalarop_version = super().c_code_cache_version()
        return (*scalarop_version, 3)


isinf = IsInf()


class Switch(ScalarOp):
    nin = 3
    nfunc_spec = ("where", 3, 1)

    def impl(self, cond, ift, iff):
        return ift if cond else iff

    def c_code(self, node, name, inputs, outputs, sub):
        (cond, ift, iff) = inputs
        (z,) = outputs
        return f"{z} = {cond} ? {ift} : {iff};"

    def pullback(self, inputs, outputs, gout):
        (cond, ift, iff) = inputs
        (gz,) = gout
        first_part = switch(cond, gz, 0.0)
        second_part = switch(cond, 0.0, gz)

        if outputs[0].type in discrete_types:
            first_part = ift.zeros_like(dtype=config.floatX)
            second_part = iff.zeros_like(dtype=config.floatX)

        # cond does affect the elements of the output so it is connected.
        # For the sake of making the gradient convenient we assume that
        # condition + epsilon always triggers the same branch as condition
        condition_grad = cond.zeros_like(dtype=config.floatX)

        return (condition_grad, first_part, second_part)

    def resolve_dtypes(self, input_dtypes):
        cond, *branches = input_dtypes
        inputs, outputs = super().resolve_dtypes(tuple(branches))
        return (*resolve_weak_dtypes((cond,)), *inputs), outputs


switch = Switch()


class UnaryBitOp(UnaryScalarOp):
    def resolve_dtypes(self, input_dtypes):
        inputs, outputs = super().resolve_dtypes(input_dtypes)
        allowed = (
            discrete_dtypes
            if True or all(dtype == "bool" for dtype in inputs)
            else integer_dtypes
        )
        if any(dtype not in allowed for dtype in inputs):
            raise TypeError(f"Integer inputs required for {self}, got {inputs}")
        return inputs, outputs

    def pullback(self, inputs, outputs, output_gradients):
        return [inputs[0].zeros_like(dtype=config.floatX)]


class BinaryBitOp(BinaryScalarOp):
    def resolve_dtypes(self, input_dtypes):
        inputs, outputs = super().resolve_dtypes(input_dtypes)
        allowed = (
            discrete_dtypes
            if False or all(dtype == "bool" for dtype in inputs)
            else integer_dtypes
        )
        if any(dtype not in allowed for dtype in inputs):
            raise TypeError(f"Integer inputs required for {self}, got {inputs}")
        return inputs, outputs

    def pullback(self, inputs, outputs, output_gradients):
        a, b = inputs
        return [
            a.zeros_like(dtype=config.floatX),
            b.zeros_like(dtype=config.floatX),
        ]


class OR(BinaryBitOp):
    identity = 0
    commutative = True
    associative = True
    nfunc_spec = ("bitwise_or", 2, 1)

    def impl(self, x, y):
        return x | y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        return f"{z} = ({x} | {y});"


or_ = OR()


class XOR(BinaryBitOp):
    identity = 0
    commutative = True
    associative = True
    nfunc_spec = ("bitwise_xor", 2, 1)

    def impl(self, x, y):
        return x ^ y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        return f"{z} = ({x} ^ {y});"


xor = XOR()


class AND(BinaryBitOp):
    identity = -1
    commutative = True
    associative = True
    nfunc_spec = ("bitwise_and", 2, 1)

    def impl(self, x, y):
        return x & y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        return f"{z} = ({x} & {y});"

    def c_code_cache_version(self):
        super_version = super().c_code_cache_version()
        return (*super_version, 3)


and_ = AND()


class Invert(UnaryBitOp):
    nfunc_spec = ("invert", 1, 1)
    monotonic_decreasing = True

    def impl(self, x):
        return ~x

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.outputs[0].type == bool:
            return f"{z} = (!{x});"
        return f"{z} = (~{x});"


invert = Invert()


##############
# Arithmetic
##############
class Maximum(BinaryScalarOp):
    commutative = True
    associative = True
    nfunc_spec = ("maximum", 2, 1)
    identity = -np.inf

    def impl(self, *inputs):
        # The built-in max function don't support complex type
        return np.maximum(*inputs)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if any(i.type in complex_types for i in node.inputs):
            raise NotImplementedError()
        # Test for both y>x and x>=y to detect NaN
        return f'{z} = (({y})>({x})? ({y}): (({x})>=({y})? ({x}): nan("")));'

    def pullback(self, inputs, outputs, gout):
        (x, y) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            # max is currently defined for complex_types,
            # but the gradient for complex is not.
            raise NotImplementedError()

        if outputs[0].type in discrete_types:
            return [
                x.zeros_like(dtype=config.floatX),
                y.zeros_like(dtype=config.floatX),
            ]
        # maximum returns x when x >= y, ties included; testing eq(outputs[0], x)
        # would need x recomputed bit-for-bit, which fastmath reassoc can break.
        e = ge(x, y)
        gx = e * gz
        gy = (constant(1, dtype=gz.dtype) - e) * gz
        return (gx, gy)


maximum = Maximum()

# Backward compatibility
ScalarMaximum = Maximum
scalar_maximum = maximum


class Minimum(BinaryScalarOp):
    commutative = True
    associative = True
    nfunc_spec = ("minimum", 2, 1)
    identity = np.inf

    def impl(self, *inputs):
        # The built-in min function don't support complex type
        return np.minimum(*inputs)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if any(i.type in complex_types for i in node.inputs):
            raise NotImplementedError()
        return f'{z} = (({y})<({x})? ({y}): (({x})<=({y})? ({x}): nan("")));'

    def pullback(self, inputs, outputs, gout):
        (x, y) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            # min is currently defined for complex_types,
            # but the gradient for complex is not.
            raise NotImplementedError()

        if outputs[0].type in discrete_types:
            return [
                x.zeros_like(dtype=config.floatX),
                y.zeros_like(dtype=config.floatX),
            ]
        # minimum returns x when x <= y, ties included; testing eq(outputs[0], x)
        # would need x recomputed bit-for-bit, which fastmath reassoc can break.
        e = le(x, y)
        gx = e * gz
        gy = (constant(1, dtype=gz.dtype) - e) * gz
        return (gx, gy)


minimum = Minimum()

# Backward compatibility
ScalarMinimum = Minimum
scalar_minimum = minimum


class Add(ScalarOp):
    name = "add"

    identity = 0
    commutative = True
    associative = True
    nfunc_spec = ("add", 2, 1)

    def impl(self, *inputs):
        return sum(inputs)

    def c_code(self, node, name, inputs, outputs, sub):
        (z,) = outputs
        op = " + "
        if node.outputs[0].type == bool:
            op = " || "
        if not inputs:
            return z + " = 0;"
        else:
            return z + " = " + op.join(inputs) + ";"

    def pullback(self, inputs, outputs, gout):
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            assert gz is not None
            retval = []
            for ii, inp in enumerate(inputs):
                if hasattr(inp, "zeros_like"):
                    retval.append(inp.zeros_like(dtype=config.floatX))
                else:
                    retval.append(grad_undefined(self, ii, inp))
        else:
            retval = []
            for i in inputs:
                retval += [gz]
        return retval


add = Add()


class Mul(ScalarOp):
    name = "mul"

    identity = 1
    commutative = True
    associative = True
    nfunc_spec = ("multiply", 2, 1)

    def impl(self, *inputs):
        return np.prod(inputs)

    def c_code(self, node, name, inputs, outputs, sub):
        (z,) = outputs
        op = " * "
        if node.outputs[0].type == bool:
            op = " && "
        if not inputs:
            return z + " = 1;"
        else:
            return z + " = " + op.join(inputs) + ";"

    def pullback(self, inputs, outputs, gout):
        (gz,) = gout
        retval = []

        # The following 3 lines verify that gz is complex when the
        # output is complex. The rest of this function make this supposition.
        output_type = get_scalar_type(
            self.resolve_dtypes(tuple(i.dtype for i in inputs))[1][0]
        )
        if output_type in complex_types:
            if gz.type not in complex_types:
                raise TypeError(
                    "Mul with output_type "
                    + str(output_type)
                    + " expected gz type to be complex, got gz with type "
                    + str(gz.type)
                )

        if output_type in discrete_types:
            return [ipt.zeros_like(dtype=config.floatX) for ipt in inputs]

        for input in inputs:
            if gz.type in complex_types:
                # zr+zi = (xr + xi)(yr + yi)
                # zr+zi = (xr*yr - xi*yi) + (xr yi + xi yr )
                otherprod = mul(*(difference(inputs, [input])))
                yr = real(otherprod)
                yi = imag(otherprod)
                if input.type in complex_types:
                    retval += [
                        complex(
                            yr * real(gz) + yi * imag(gz), yr * imag(gz) - yi * real(gz)
                        )
                    ]
                else:
                    retval += [yr * real(gz) + yi * imag(gz)]
            else:
                retval += [mul(*([gz, *difference(inputs, [input])]))]
        return retval


mul = Mul()


class Sub(BinaryScalarOp):
    name = "sub"

    def resolve_dtypes(self, input_dtypes):
        inputs, outputs = _resolve_numeric_dtypes(input_dtypes)
        if outputs == ("bool",):
            raise TypeError("bool output not supported")
        return inputs, outputs

    nfunc_spec = ("subtract", 2, 1)

    def impl(self, x, y):
        return x - y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        return f"{z} = {x} - {y};"

    def pullback(self, inputs, outputs, gout):
        (x, y) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            return [
                x.zeros_like(dtype=config.floatX),
                y.zeros_like(dtype=config.floatX),
            ]

        first_part = gz
        second_part = -gz

        return first_part, second_part


sub = Sub()


class TrueDiv(BinaryScalarOp):
    name = "true_div"

    nfunc_spec = ("true_divide", 2, 1)

    def resolve_dtypes(self, input_dtypes):
        inputs = resolve_weak_dtypes(input_dtypes)
        if all(dtype in discrete_dtypes for dtype in inputs):
            inputs = tuple(
                dtype if isinstance(given, str) else config.floatX
                for given, dtype in zip(input_dtypes, inputs, strict=True)
            )
            return inputs, (config.floatX,)
        return super().resolve_dtypes(input_dtypes)

    def impl(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if all(a.dtype in discrete_dtypes for a in (x, y)):
            return np.sctypeDict[config.floatX](float(x) / y)
        else:
            return x / y

    def c_code(self, node, name, inputs, outputs, sub):
        # we generate good c code only when both are complex!
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types or node.inputs[1].type in complex_types:
            raise NotImplementedError("type not supported", type)
        if (
            node.inputs[0].type in discrete_types
            and node.inputs[1].type in discrete_types
        ):
            return f"{z} = ((double){x}) / {y};"
        return f"{z} = {x} / {y};"

    def pullback(self, inputs, outputs, gout):
        (x, y) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()

        # If the output of this op is discrete, then it
        # it is locally flat everywhere, so the gradient
        # through it is 0.
        # This is different from it not being connected
        # to the output; x/y is still a function of x
        # and y; it's just a step function.
        if all(a.dtype in discrete_dtypes for a in (x, y)):
            return [
                x.zeros_like(dtype=config.floatX),
                y.zeros_like(dtype=config.floatX),
            ]

        first_part = gz / y

        if y.type in complex_types:
            raise NotImplementedError()

        second_part = -(gz * x) / (y * y)

        return first_part, second_part


true_div = TrueDiv()


class IntDiv(BinaryScalarOp):
    name = "int_div"

    nfunc_spec = ("floor_divide", 2, 1)
    complex_error = ComplexError(
        "PyTensor does not support integer division (//) on "
        "complex numbers, since numpy deprecated it."
    )

    def impl(self, x, y):
        return x // y

    def c_support_code(self, **kwargs):
        # We use a macro as python use % as a special string character,
        # and the output of c_code may be run through another level
        # of string formatting.
        return "#define PYTENSOR_MACRO_MOD(x,y) (x % y)"

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        fail = sub["fail"]

        t = node.inputs[0].type.upcast(*[i.type for i in node.inputs[1:]])
        if t in map(str, discrete_types):
            x_div_y_pp = f"({x} / {y})"
            x_div_y_mp = f"((-{x}) / {y})"
            x_mod_y_mp = f"PYTENSOR_MACRO_MOD((-{x}), {y})"
            x_div_y_pm = f"({x} / (-{y}))"
            x_mod_y_pm = f"PYTENSOR_MACRO_MOD({x}, (-{y}))"
            x_div_y_mm = f"((-{x}) / (-{y}))"
            # If we are in a gpuarray kernel, %(fail)s exits the kernel,
            # and we do not have any error report, and we cannot set
            # Python error messages either, so for now we just call the
            # cuda function, which return a binary pattern of all 1s.
            div_zero = dedent(
                f"""
                #ifdef KERNEL
                    {z} = {x_div_y_pp};
                #else
                    PyErr_SetString(PyExc_ZeroDivisionError, "integer division by zero");
                    {fail}
                #endif
                """
            )
        elif t in map(str, float_types):
            # We need to call different functions of math.h
            # depending on the type
            if t == "float32":
                floor = "floorf"
                fmod = "fmodf"
            elif t == "float64":
                floor = "floor"
                fmod = "fmod"
            else:
                raise NotImplementedError("type not supported", t)

            x_div_y_pp = f"{floor}({x} / {y})"
            x_div_y_mp = f"{floor}((-{x}) / {y})"
            x_mod_y_mp = f"{fmod}((-{x}), {y})"
            x_div_y_pm = f"{floor}({x} / (-{y}))"
            x_mod_y_pm = f"{fmod}({x}, (-{y}))"
            x_div_y_mm = f"{floor}((-{x}) / (-{y}))"
            div_zero = f"{z} = {x_div_y_pp};"
        elif t in complex_types:
            raise self.complex_error
        else:
            raise NotImplementedError("type not supported", t)

        return dedent(
            f"""
            if ({y} == 0) {{
                {div_zero};
            }} else if ({y} < 0) {{
                if ({x} < 0) {{
                    {z} = {x_div_y_mm};
                }} else {{
                    {z} = - {x_div_y_pm} - (({x_mod_y_pm} == 0) ? 0 : 1);
                }}
            }} else {{
                if ({x} < 0) {{
                    {z} = - {x_div_y_mp} - (({x_mod_y_mp} == 0) ? 0 : 1);
                }} else {{
                    {z} = {x_div_y_pp};
                }}
            }}
            """
        )

    def c_code_cache_version(self):
        return (6,)

    def pullback(self, inputs, outputs, g_output):
        return [inp.zeros_like(dtype=config.floatX) for inp in inputs]


int_div = IntDiv()


floor_div = int_div


def mod_check(x, y):
    if as_scalar(x).type in complex_types or as_scalar(y).type in complex_types:
        # Currently forbidden.
        raise Mod.complex_error
    else:
        return mod(x, y)


class Mod(BinaryScalarOp):
    name = "mod"

    nfunc_spec = ("mod", 2, 1)
    complex_error = ComplexError(
        "PyTensor does not support the mod operator (%) on "
        "complex numbers, since numpy deprecated it."
    )

    def impl(self, x, y):
        if isinstance(x, builtins.complex) or isinstance(y, builtins.complex):
            raise self.complex_error
        return x % y

    def c_code_cache_version(self):
        return (9,)

    def c_support_code(self, **kwargs):
        # We use a macro as python use % as a special string character,
        # and the output of c_code may be run through another level
        # of string formatting.
        return "#define PYTENSOR_MACRO_MOD(x, y) (x % y)"

    def c_code(self, node, name, inputs, outputs, sub):
        """
        We want the result to have the same sign as Python, not the other
        implementation of mod.

        """
        (x, y) = inputs
        (z,) = outputs
        fail = sub["fail"]
        t = node.inputs[0].type.upcast(*[i.type for i in node.inputs[1:]])
        if (
            str(t) in map(str, discrete_types)
            or t in ("uint8", "int8", "uint16", "int16")
            or t in ("uint32", "int32", "uint64", "int64")
            or t in discrete_types
        ):
            # The above or's should not be needed anymore. However, for now we
            # keep them out of safety, and verify they are useless with an
            # assert.
            assert str(t) in map(str, discrete_types)
            x_mod_y = f"PYTENSOR_MACRO_MOD({x}, {y})"
            x_mod_ymm = f"PYTENSOR_MACRO_MOD(-{x}, -{y})"
            x_mod_ypm = f"PYTENSOR_MACRO_MOD({x}, -{y})"
            x_mod_ymp = f"PYTENSOR_MACRO_MOD(-{x}, {y})"
            # If we are in a gpuarray kernel, %(fail)s exits the kernel,
            # and we do not have any error report, and we cannot set
            # Python error messages either, so for now we just call the
            # cuda function, returning a binary pattern depending on dtype
            mod_zero = dedent(
                f"""
                #ifdef KERNEL
                    {z} = {x_mod_y};
                #else
                    PyErr_SetString(PyExc_ZeroDivisionError, "integer modulo by zero");
                    {fail}
                #endif
                """
            )
        elif (
            str(t) in map(str, float_types)
            or t in ("float32", "float64")
            or t in float_types
        ):
            # The above or's should not be needed anymore. However, for now we
            # keep them out of safety, and verify they are useless with an
            # assert.
            assert str(t) in map(str, float_types)
            x_mod_y = f"fmod({x}, {y})"
            x_mod_ymm = f"fmod(-{x}, -{y})"
            x_mod_ypm = f"fmod({x}, -{y})"
            x_mod_ymp = f"fmod(-{x}, {y})"
            mod_zero = f"{z} = {x_mod_y};"
        elif str(t) in map(str, complex_types):
            raise self.complex_error
        else:
            raise NotImplementedError("type not supported", t)

        return dedent(
            f"""
            if ({y} == 0) {{
                {mod_zero};
            }} else if ({y} < 0){{
                if ({x} < 0){{
                    {z} = -({x_mod_ymm});
                }} else {{
                    {z} = ({x_mod_ypm}) + ({x_mod_ypm} != 0 ? {y} : 0);
                }}
            }} else {{
                if ({x} < 0){{
                    {z} = - {x_mod_ymp} + ({x_mod_ymp} != 0 ? {y} : 0);
                }} else {{
                    {z} = {x_mod_y};
                }}
            }}
            """
        )

    def pullback(self, inputs, outputs, gout):
        (x, y) = inputs
        (gz,) = gout
        if outputs[0].type in discrete_types:
            # The gradient does not flow in if the output is discrete
            return [
                x.zeros_like(dtype=config.floatX),
                y.zeros_like(dtype=config.floatX),
            ]
        return [gz, -(x // y) * gz]


mod = Mod()


class Pow(BinaryScalarOp):
    name = "pow"

    resolve_dtypes = _resolve_numeric_bool_to_int8_dtypes

    nfunc_spec = ("power", 2, 1)

    def impl(self, x, y):
        return np.power(x, y)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types or node.inputs[1].type in complex_types:
            raise NotImplementedError("type not supported", type)
        return f"{z} = pow({x}, {y});"

    def pullback(self, inputs, outputs, gout):
        (x, y) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()

        if outputs[0].type in discrete_types:
            return [
                x.zeros_like(dtype=config.floatX),
                y.zeros_like(dtype=config.floatX),
            ]

        first_part = gz * y * x ** (y - 1)

        second_part = gz * log(x) * x**y
        second_part = switch(eq(x, 0), 0, second_part)

        return (first_part, second_part)

    def c_code_contiguous(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if not config.lib__amdlibm:
            raise MethodNotDefined()

        # We compare the dtype AND the broadcast flag
        # as this function do not broadcast
        if (
            node.inputs[0].type == node.outputs[0].type
            and node.inputs[1].type == node.outputs[0].type
            and None not in node.inputs[0].type.shape
            and None not in node.inputs[1].type.shape
            and
            # amdlibm 3.0 do not have a float64 version of this SIMD function
            node.inputs[0].dtype == "float32"
            and node.inputs[1].dtype == "float32"
        ):
            dtype = "float"
            fct = "amd_vrsa_powf"
            return f"""
        npy_intp n = PyArray_SIZE({z});
        {dtype} * x = ({dtype}*) PyArray_DATA({x});
        {dtype} * y = ({dtype}*) PyArray_DATA({y});
        {dtype} * z = ({dtype}*) PyArray_DATA({z});
        {fct}(n, x, y, z);
        """
        # We compare the dtype and check we broadcast a scalar
        elif (
            node.inputs[0].type == node.outputs[0].type
            and node.inputs[1].dtype == node.outputs[0].dtype
            and all(node.inputs[1].broadcastable)
            and
            # amdlibm 3.0 do not have a float64 version of this SIMD function
            node.inputs[0].dtype == "float32"
            and node.inputs[1].dtype == "float32"
        ):
            dtype = "float"
            fct = "amd_vrsa_powxf"
            return f"""
        npy_intp n = PyArray_SIZE({z});
        {dtype} * x = ({dtype}*) PyArray_DATA({x});
        {dtype} * y = ({dtype}*) PyArray_DATA({y});
        {dtype} * z = ({dtype}*) PyArray_DATA({z});
        {fct}(n, x, *y, z);
        """

        raise MethodNotDefined()


pow = Pow()


class Clip(ScalarOp):
    name = "clip"

    def resolve_dtypes(self, input_dtypes):
        inputs, outputs = _resolve_numeric_dtypes(input_dtypes)
        if any(dtype.startswith("complex") for dtype in inputs):
            raise TypeError("complex types are not supported")
        return inputs, outputs

    nin = 3
    # The numpy.clip don't work correctly when the min is bigger then the max,
    # So we do not use nfunc_spec = ('clip', 3, 1)

    def impl(self, x, min, max):
        if x < min:
            return min
        elif x > max:
            return max
        else:
            return x

    def c_code(self, node, name, inputs, outputs, sub):
        (x, min, max) = inputs
        (z,) = outputs
        return f"{z} = {x} < {min} ? {min} : {x} > {max} ? {max} : {x};"

    def pullback(self, inputs, outputs, gout):
        (x, mn, mx) = inputs
        (gz,) = gout
        assert gz.type not in complex_types
        gx = ((x >= mn) & (x <= mx)) * gz
        gmn = (x < mn) * gz
        gmx = (x > mx) * gz

        def handle_int(v):
            if outputs[0].type in int_types:
                return v.zeros_like(dtype=config.floatX)
            return v

        return list(map(handle_int, [gx, gmn, gmx]))


# Don't allow complex even if numpy do
# As there is no mathematical reason for this function on complex
clip = Clip()


class Second(BinaryScalarOp):
    name = "second"

    def resolve_dtypes(self, input_dtypes):
        inputs = tuple(resolve_weak_dtypes((dtype,))[0] for dtype in input_dtypes)
        return inputs, (inputs[1],)

    def impl(self, x, y):
        return y

    def c_code(self, node, name, inputs, outputs, sub):
        (_x, y) = inputs
        (z,) = outputs
        return f"{z} = {y};"

    def connection_pattern(self, node):
        # x is never connected because its elements are never used
        # y is connected because its elements are copied over

        return [[False], [True]]

    def pullback(self, inputs, outputs, gout):
        (_x, y) = inputs
        (gz,) = gout
        if y.type in continuous_types:
            # x is disconnected because the elements of x are not used
            return disconnected_type(), gz
        else:
            # when y is discrete, we assume the function can be extended
            # to deal with real-valued inputs by rounding them to the
            # nearest integer. f(x+eps) thus equals f(x) so the gradient
            # is zero, not disconnected or undefined
            return disconnected_type(), y.zeros_like(dtype=config.floatX)


second = Second()


class Identity(UnaryScalarOp):
    name = "identity"

    preserves_zero = True
    monotonic_increasing = True

    def impl(self, input):
        return input

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return f"{z} = {x};"

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in continuous_types:
            return (gz,)
        else:
            return (x.zeros_like(dtype=config.floatX),)


identity = Identity()


# CASTING OPERATIONS
class Cast(UnaryScalarOp):
    __props__ = ("o_type",)

    def __init__(self, o_type):
        if not isinstance(o_type, ScalarType):
            raise TypeError(o_type)
        super().__init__()
        self.o_type = o_type
        self.ctor = np.dtype(o_type.dtype).type

    def resolve_dtypes(self, input_dtypes):
        return resolve_weak_dtypes(input_dtypes), (self.o_type.dtype,)

    def __str__(self):
        return f"{self.__class__.__name__}{{{self.o_type.dtype}}}"

    def impl(self, input):
        return self.ctor(input)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.outputs[0].type == bool:
            return f"{z} = ({x}) ? 1 : 0;"
        return f"{z} = ({node.outputs[0].type.dtype_specs()[1]}){x};"

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if self.o_type in continuous_types:
            return [gz]
        else:
            return [x.zeros_like(dtype=config.floatX)]

    def c_code_cache_version(self):
        s = super().c_code_cache_version()
        if s:
            return (4, *s)
        else:
            return s


convert_to_bool: Cast = Cast(bool)
convert_to_int8: Cast = Cast(int8)
convert_to_int16: Cast = Cast(int16)
convert_to_int32: Cast = Cast(int32)
convert_to_int64: Cast = Cast(int64)
convert_to_uint8: Cast = Cast(uint8)
convert_to_uint16: Cast = Cast(uint16)
convert_to_uint32: Cast = Cast(uint32)
convert_to_uint64: Cast = Cast(uint64)
convert_to_float16: Cast = Cast(float16)
convert_to_float32: Cast = Cast(float32)
convert_to_float64: Cast = Cast(float64)
convert_to_complex64: Cast = Cast(complex64)
convert_to_complex128: Cast = Cast(complex128)

_cast_mapping = {
    "bool": convert_to_bool,
    "int8": convert_to_int8,
    "int16": convert_to_int16,
    "int32": convert_to_int32,
    "int64": convert_to_int64,
    "uint8": convert_to_uint8,
    "uint16": convert_to_uint16,
    "uint32": convert_to_uint32,
    "uint64": convert_to_uint64,
    "float16": convert_to_float16,
    "float32": convert_to_float32,
    "float64": convert_to_float64,
    "complex64": convert_to_complex64,
    "complex128": convert_to_complex128,
}


def cast(x, dtype):
    """
    Symbolically cast `x` to a ScalarType of given `dtype`.

    """
    if dtype == "floatX":
        dtype = config.floatX

    _x = as_scalar(x)
    if _x.type.dtype == dtype:
        return _x
    if _x.type.dtype.startswith("complex") and not dtype.startswith("complex"):
        raise TypeError(
            "Casting from complex to real is ambiguous: consider"
            " real(), imag(), angle() or abs()"
        )
    return _cast_mapping[dtype](_x)


class Abs(UnaryScalarOp):
    resolve_dtypes = _resolve_real_dtypes

    preserves_zero = True
    non_negative = True
    nfunc_spec = ("abs", 1, 1)

    def impl(self, x):
        return np.abs(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        if x.type in float_types:
            return (gz * sign(x),)
        return (gz * x / _abs(x),)  # formula works for complex and real

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        type = node.inputs[0].type
        if type in int_types:
            return f"{z} = abs({x});"
        if type in float_types:
            return f"{z} = fabs({x});"
        if type in complex_types:
            return f"{z} = sqrt(get_real({x}) * get_real({x}) + get_imag({x}) * get_imag({x}));"
        if node.outputs[0].type == bool:
            return f"{z} = ({x}) ? 1 : 0;"
        if type in uint_types:
            # uint are always already absolute value.
            return f"{z} = {x};"
        raise NotImplementedError("type not supported", type)


abs = Abs()


class Sign(UnaryScalarOp):
    name = "sign"

    resolve_dtypes = _resolve_nonbool_dtypes

    preserves_zero = True
    monotonic_increasing = True

    nfunc_spec = ("sign", 1, 1)

    def impl(self, x):
        # casting to output type is handled by filter
        return np.sign(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (_gz,) = gout
        rval = x.zeros_like()

        if rval.type in discrete_types:
            rval = rval.astype(config.floatX)

        return [rval]

    def c_code(self, node, name, inputs, outputs, sub):
        # casting is done by compiler
        # TODO: use copysign
        (x,) = inputs
        (z,) = outputs
        type = node.inputs[0].type
        if type in float_types:
            return (
                f"{z} = ({x} > 0) ? 1. : (({x} < 0) ? -1. : (isnan({x}) ? NAN : 0.));"
            )
        if type in int_types:
            return f"{z} = ({x} >= 0) ? ({x} == 0) ? 0 : 1 : -1;"
        if type in complex_types:
            real_type = get_scalar_type(_real_dtype(type.dtype)).dtype_specs()[1]
            hypot = "hypotf" if type == complex64 else "hypot"
            # Scoped so that fused Composites with several Signs don't redeclare locals
            return f"""
            {{
                {real_type} real = get_real({x});
                {real_type} imag = get_imag({x});
                {real_type} magnitude = {hypot}(real, imag);
                if (isnan(magnitude)) {{
                    set_real(&{z}, NAN);
                    set_imag(&{z}, NAN);
                }} else if (isinf(magnitude)) {{
                    if (isinf(real) && isinf(imag)) {{
                        set_real(&{z}, NAN);
                        set_imag(&{z}, NAN);
                    }} else if (isinf(real)) {{
                        set_real(&{z}, real > 0 ? 1. : -1.);
                        set_imag(&{z}, 0.);
                    }} else {{
                        set_real(&{z}, 0.);
                        set_imag(&{z}, imag > 0 ? 1. : -1.);
                    }}
                }} else if (magnitude == 0) {{
                    set_real(&{z}, 0.);
                    set_imag(&{z}, 0.);
                }} else {{
                    set_real(&{z}, real / magnitude);
                    set_imag(&{z}, imag / magnitude);
                }}
            }}
            """
        raise NotImplementedError("type not supported", type)

    def c_code_cache_version(self):
        s = super().c_code_cache_version()
        if s:
            return (5, *s)
        else:  # if parent is unversioned, we are too
            return s


sign = Sign()


class Ceil(UnaryScalarOp):
    name = "ceil"

    resolve_dtypes = _resolve_float_no_complex_dtypes

    preserves_zero = True
    monotonic_increasing = True

    nfunc_spec = ("ceil", 1, 1)

    def impl(self, x):
        return np.ceil(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (_gz,) = gout
        rval = x.zeros_like()

        if rval.type in discrete_types:
            rval = rval.astype(config.floatX)

        return [rval]

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = ceil(({cast}){x});"


ceil = Ceil()


class Floor(UnaryScalarOp):
    name = "floor"

    resolve_dtypes = _resolve_float_no_complex_dtypes

    preserves_zero = True
    monotonic_increasing = True

    nfunc_spec = ("floor", 1, 1)

    def impl(self, x):
        return np.floor(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (_gz,) = gout
        rval = x.zeros_like()

        if rval.type in discrete_types:
            rval = rval.astype(config.floatX)

        return [rval]

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = floor(({cast}){x});"


floor = Floor()


class Trunc(UnaryScalarOp):
    name = "trunc"

    resolve_dtypes = _resolve_float_no_complex_dtypes

    preserves_zero = True
    monotonic_increasing = True

    nfunc_spec = ("trunc", 1, 1)

    def impl(self, x):
        return np.trunc(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (_gz,) = gout
        return [x.zeros_like(dtype=config.floatX)]

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return f"{z} = {x} >= 0? floor({x}): -floor(-{x});"


trunc = Trunc()


class RoundHalfToEven(UnaryScalarOp):
    resolve_dtypes = _resolve_float_only_dtypes

    preserves_zero = True
    monotonic_increasing = True

    """
    This function implement the same rounding than numpy: Round half to even.

    c/c++ round fct IS DIFFERENT!
    See http://en.wikipedia.org/wiki/Rounding for more details.

    """

    nfunc_spec = ("around", 1, 1)

    def impl(self, x):
        return np.round(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (_gz,) = gout
        rval = x.zeros_like()

        if rval.type in discrete_types:
            rval = rval.astype(config.floatX)

        return [rval]

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        typ = node.outputs[0].type.dtype
        if typ not in ("float32", "float64"):
            raise NotImplementedError("The output should be float32 or float64")
        if typ == "float32":
            ctype = "float"
            floor_function = "floorf"
        else:
            ctype = "double"
            floor_function = "floor"
        return f"""
        /* Code inspired from NumPy npy_rint implementation. */
        {{
            {ctype} y, r;
            y = {floor_function}({x});
            r = {x} - y;
            if(r > 0.5) {{
                y += 1;
            }} else if(r == 0.5) {{
                r = y - 2.0*{floor_function}(0.5*y);
                /*
                If y is even, then r == 0
                If y is odd,  then r == 1
                So we can just add r to y, so that
                y will be incremented only if he's odd.
                */
                y += (int)r;
            }}
            {z} = y;
        }}
        """


round_half_to_even = RoundHalfToEven()


def round_half_away_from_zero_(a):
    if a > 0:
        return np.floor(a + 0.5)
    else:
        return np.ceil(a - 0.5)


round_half_away_from_zero_vec64 = np.vectorize(
    round_half_away_from_zero_, doc="round_half_away_from_zero_vec64"
)
round_half_away_from_zero_vec32 = np.vectorize(
    round_half_away_from_zero_,
    doc="round_half_away_from_zero_vec32",
    otypes=["float32"],
)


def round_half_away_from_zero_vec(a):
    if getattr(a, "dtype", None) == np.float32:
        return round_half_away_from_zero_vec32(a)
    return round_half_away_from_zero_vec64(a)


class RoundHalfAwayFromZero(UnaryScalarOp):
    resolve_dtypes = _resolve_float_only_dtypes

    preserves_zero = True
    monotonic_increasing = True

    """
    Implement the same rounding algo as c round() fct.

    numpy.round fct IS DIFFERENT!
    See http://en.wikipedia.org/wiki/Rounding for more details.

    """

    def impl(self, x):
        return round_half_away_from_zero_vec(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (_gz,) = gout
        rval = x.zeros_like()

        if rval.type in discrete_types:
            rval = rval.astype(config.floatX)

        return [rval]

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.outputs[0].type.dtype in ("float32", "float64"):
            return f"{z} = round({x});"
        else:
            raise NotImplementedError("The output should be float32 or float64")


round_half_away_from_zero = RoundHalfAwayFromZero()


class Neg(UnaryScalarOp):
    name = "neg"

    resolve_dtypes = _resolve_nonbool_dtypes

    preserves_zero = True
    monotonic_decreasing = True
    # We can use numpy.negative here, because even if it gives unexpected
    # results on Boolean arrays, it will be passed other dtypes as PyTensor
    # does not have a Boolean type for tensors.
    nfunc_spec = ("negative", 1, 1)

    def impl(self, x):
        return -x

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (-gz,)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return f"{z} = -{x};"


neg = Neg()

pprint.assign(add, printing.OperatorPrinter("+", -2, "either"))
pprint.assign(mul, printing.OperatorPrinter("*", -1, "either"))
pprint.assign(sub, printing.OperatorPrinter("-", -2, "left"))
pprint.assign(neg, printing.OperatorPrinter("-", 0, "either"))
pprint.assign(true_div, printing.OperatorPrinter("/", -1, "left"))
pprint.assign(int_div, printing.OperatorPrinter("//", -1, "left"))
pprint.assign(pow, printing.OperatorPrinter("**", 1, "right"))
pprint.assign(mod, printing.OperatorPrinter("%", -1, "left"))


class Reciprocal(UnaryScalarOp):
    """Multiplicative inverse."""

    name = "reciprocal"

    resolve_dtypes = _resolve_float_dtypes

    nfunc_spec = ("reciprocal", 1, 1)

    def impl(self, x):
        return np.float32(1.0) / x

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (-gz / (x * x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return f"{z} = 1.0 / {x};"


reciprocal = Reciprocal()


class Log(UnaryScalarOp):
    """
    log base e.

    """

    name = "log"

    resolve_dtypes = _resolve_float_dtypes

    monotonic_increasing = True
    nfunc_spec = ("log", 1, 1)
    amd_float32 = "amd_vrsa_logf"
    amd_float64 = "amd_vrda_log"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.log will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.log(x, dtype=np.float32)
        return np.log(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / x,)

    def c_code(self, node, name, inputs, outputs, sub):
        # todo: the version using log2 seems to be very slightly faster
        # on some machines for some reason, check if it's worth switching
        # return f"{z} = log2({x}) * 0.69314718055994529;"
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = log(({cast}){x});"


log = Log()


class Log2(UnaryScalarOp):
    """
    log base 2.

    """

    name = "log2"

    resolve_dtypes = _resolve_float_dtypes

    monotonic_increasing = True
    nfunc_spec = ("log2", 1, 1)
    amd_float32 = "amd_vrsa_log2f"
    amd_float64 = "amd_vrda_log2"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.log2 will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.log2(x, dtype=np.float32)
        return np.log2(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / (x * np.array(math.log(2.0), dtype=x.dtype)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = log2(({cast}){x});"


log2 = Log2()


class Log10(UnaryScalarOp):
    """
    log base 10.

    """

    name = "log10"

    resolve_dtypes = _resolve_float_dtypes

    monotonic_increasing = True
    nfunc_spec = ("log10", 1, 1)
    amd_float32 = "amd_vrsa_log10f"
    amd_float64 = "amd_vrda_log10"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.log10 will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.log10(x, dtype=np.float32)
        return np.log10(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / (x * np.array(math.log(10.0), dtype=x.dtype)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = log10(({cast}){x});"


log10 = Log10()


class Log1p(UnaryScalarOp):
    name = "log1p"

    resolve_dtypes = _resolve_float_dtypes

    preserves_zero = True
    monotonic_increasing = True
    """
    log(1+x).

    """

    nfunc_spec = ("log1p", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.log1p will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.log1p(x, dtype=np.float32)
        return np.log1p(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return [gz / (1 + x)]

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = log1p(({cast}){x});"


log1p = Log1p()


class Exp(UnaryScalarOp):
    name = "exp"

    resolve_dtypes = _resolve_float_dtypes

    monotonic_increasing = True
    non_negative = True
    nfunc_spec = ("exp", 1, 1)
    amd_float32 = "amd_vrsa_expf"
    amd_float64 = "amd_vrda_exp"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.exp will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.exp(x, dtype=np.float32)
        return np.exp(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * exp(x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = exp(({cast}){x});"


exp = Exp()


class Exp2(UnaryScalarOp):
    name = "exp2"

    resolve_dtypes = _resolve_float_dtypes

    monotonic_increasing = True
    non_negative = True
    nfunc_spec = ("exp2", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.exp2 will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.exp2(x, dtype=np.float32)
        return np.exp2(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * exp2(x) * log(np.array(2, dtype=x.dtype)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = exp2(({cast}){x});"


exp2 = Exp2()


class Expm1(UnaryScalarOp):
    name = "expm1"

    resolve_dtypes = _resolve_float_dtypes

    preserves_zero = True
    monotonic_increasing = True
    nfunc_spec = ("expm1", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.expm1 will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.expm1(x, dtype=np.float32)
        return np.expm1(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * exp(x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = expm1(({cast}){x});"

    def c_code_cache_version(self):
        return (5,)


expm1 = Expm1()


class Sqr(UnaryScalarOp):
    name = "sqr"

    preserves_zero = True
    non_negative = True
    nfunc_spec = ("square", 1, 1)

    def impl(self, x):
        return x * x

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * x * 2,)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return f"{z} = {x} * {x};"


sqr = Sqr()


class Sqrt(UnaryScalarOp):
    name = "sqrt"

    resolve_dtypes = _resolve_float_dtypes

    preserves_zero = True
    monotonic_increasing = True

    nfunc_spec = ("sqrt", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.sqrt will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.sqrt(x, dtype=np.float32)
        return np.sqrt(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return ((gz * 0.5) / sqrt(x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = sqrt(({cast}){x});"


sqrt = Sqrt()


class Cos(UnaryScalarOp):
    name = "cos"

    resolve_dtypes = _resolve_float_dtypes

    nfunc_spec = ("cos", 1, 1)
    amd_float32 = "amd_vrsa_cosf"
    amd_float64 = "amd_vrda_cos"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.cos will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.cos(x, dtype=np.float32)
        return np.cos(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (-gz * sin(x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = cos(({cast}){x});"


cos = Cos()


class ArcCos(UnaryScalarOp):
    name = "arccos"

    resolve_dtypes = _resolve_float_dtypes

    monotonic_decreasing = True
    nfunc_spec = ("arccos", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arccos will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.arccos(x, dtype=np.float32)
        return np.arccos(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (-gz / sqrt(np.array(1, dtype=x.dtype) - sqr(x)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = acos(({cast}){x});"


arccos = ArcCos()


class Sin(UnaryScalarOp):
    name = "sin"

    resolve_dtypes = _resolve_float_dtypes

    preserves_zero = True
    nfunc_spec = ("sin", 1, 1)
    amd_float32 = "amd_vrsa_sinf"
    amd_float64 = "amd_vrda_sin"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.sin will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.sin(x, dtype=np.float32)
        return np.sin(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * cos(x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = sin(({cast}){x});"


sin = Sin()


class ArcSin(UnaryScalarOp):
    name = "arcsin"

    resolve_dtypes = _resolve_float_dtypes

    preserves_zero = True
    monotonic_increasing = True
    nfunc_spec = ("arcsin", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arcsin will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.arcsin(x, dtype=np.float32)
        return np.arcsin(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / sqrt(np.array(1, dtype=x.dtype) - sqr(x)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = asin(({cast}){x});"


arcsin = ArcSin()


class Tan(UnaryScalarOp):
    name = "tan"

    resolve_dtypes = _resolve_float_dtypes

    preserves_zero = True
    nfunc_spec = ("tan", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.tan will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.tan(x, dtype=np.float32)
        return np.tan(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / sqr(cos(x)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = tan(({cast}){x});"


tan = Tan()


class ArcTan(UnaryScalarOp):
    name = "arctan"

    resolve_dtypes = _resolve_float_dtypes

    preserves_zero = True
    monotonic_increasing = True
    nfunc_spec = ("arctan", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arctan will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.arctan(x, dtype=np.float32)
        return np.arctan(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / (np.array(1, dtype=x.dtype) + sqr(x)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = atan(({cast}){x});"


arctan = ArcTan()


class ArcTan2(BinaryScalarOp):
    name = "arctan2"

    resolve_dtypes = _resolve_float_dtypes

    nfunc_spec = ("arctan2", 2, 1)

    def impl(self, y, x):
        # If x and y are int8 or uint8, numpy.arctan2 will compute the result
        # in half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            y_dtype = str(getattr(x, "dtype", ""))
            if y_dtype in ("int8", "uint8"):
                return np.arctan2(y, x, dtype=np.float32)
        return np.arctan2(y, x)

    def pullback(self, inputs, outputs, gout):
        (y, x) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        else:
            if outputs[0].type in discrete_types:
                if x.type in discrete_types:
                    gx = x.zeros_like(dtype=config.floatX)
                else:
                    gx = x.zeros_like()
                if y.type in discrete_types:
                    gy = y.zeros_like(dtype=config.floatX)
                else:
                    gy = y.zeros_like()
                return [gx, gy]

            # If the output is float, the gradient should flow,
            # even if the inputs are ints
            return [gz * x / (sqr(x) + sqr(y)), gz * neg(y) / (sqr(x) + sqr(y))]

    def c_code(self, node, name, inputs, outputs, sub):
        (y, x) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types or node.inputs[1].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = atan2(({cast}){y}, ({cast}){x});"


arctan2 = ArcTan2()


class Cosh(UnaryScalarOp):
    """
    cosh(x) = (exp(x) + exp(-x)) / 2.

    """

    name = "cosh"

    resolve_dtypes = _resolve_float_dtypes

    non_negative = True

    nfunc_spec = ("cosh", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.cosh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.cosh(x, dtype=np.float32)
        return np.cosh(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * sinh(x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = cosh(({cast}){x});"


cosh = Cosh()


class ArcCosh(UnaryScalarOp):
    name = "arccosh"

    resolve_dtypes = _resolve_float_dtypes

    monotonic_increasing = True
    nfunc_spec = ("arccosh", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arccosh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.arccosh(x, dtype=np.float32)
        return np.arccosh(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / sqrt(sqr(x) - np.array(1, dtype=x.dtype)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = acosh(({cast}){x});"


arccosh = ArcCosh()


class Sinh(UnaryScalarOp):
    name = "sinh"

    resolve_dtypes = _resolve_float_dtypes

    preserves_zero = True
    monotonic_increasing = True
    """
    sinh(x) = (exp(x) - exp(-x)) / 2.

    """

    nfunc_spec = ("sinh", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.sinh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.sinh(x, dtype=np.float32)
        return np.sinh(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * cosh(x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = sinh(({cast}){x});"


sinh = Sinh()


class ArcSinh(UnaryScalarOp):
    name = "arcsinh"

    resolve_dtypes = _resolve_float_dtypes

    preserves_zero = True
    monotonic_increasing = True
    nfunc_spec = ("arcsinh", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arcsinh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.arcsinh(x, dtype=np.float32)
        return np.arcsinh(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / sqrt(sqr(x) + np.array(1, dtype=x.dtype)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = asinh(({cast}){x});"


arcsinh = ArcSinh()


class Tanh(UnaryScalarOp):
    name = "tanh"

    resolve_dtypes = _resolve_float_dtypes

    preserves_zero = True
    monotonic_increasing = True
    """
    tanh(x) = sinh(x) / cosh(x)
            = (exp(2*x) - 1) / (exp(2*x) + 1).

    """

    nfunc_spec = ("tanh", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.tanh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.tanh(x, dtype=np.float32)
        return np.tanh(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * (1 - sqr(tanh(x))),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = tanh(({cast}){x});"


tanh = Tanh()


class ArcTanh(UnaryScalarOp):
    name = "arctanh"

    resolve_dtypes = _resolve_float_dtypes

    preserves_zero = True
    monotonic_increasing = True
    nfunc_spec = ("arctanh", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arctanh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.arctanh(x, dtype=np.float32)
        return np.arctanh(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / (np.array(1, dtype=x.dtype) - sqr(x)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = atanh(({cast}){x});"


arctanh = ArcTanh()


class Real(UnaryScalarOp):
    name = "real"

    resolve_dtypes = _resolve_real_dtypes

    preserves_zero = True
    """
    Extract the real coordinate of a complex number.

    """

    nfunc_spec = ("real", 1, 1)

    def impl(self, x):
        return np.real(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            return [complex(gz, 0)]
        elif x.type in float_types:
            return [gz]
        else:
            return [x.zeros_like(dtype=config.floatX)]

    def c_code(self, *args, **kwargs):
        raise NotImplementedError()


real = Real()


class Imag(UnaryScalarOp):
    name = "imag"

    resolve_dtypes = _resolve_real_dtypes

    preserves_zero = True
    nfunc_spec = ("imag", 1, 1)

    def impl(self, x):
        return np.imag(x)

    def pullback(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            return [complex(0, gz)]
        elif x.type in float_types:
            return [second(x, 0)]
        else:
            return [x.zeros_like(dtype=config.floatX)]

    def c_code(self, *args, **kwargs):
        raise NotImplementedError()


imag = Imag()


class Complex(BinaryScalarOp):
    name = "complex"

    def resolve_dtypes(self, input_dtypes):
        inputs = resolve_weak_dtypes(input_dtypes)
        if any(dtype.startswith("complex") for dtype in inputs):
            raise TypeError("Complex components must be real")
        dtype = upcast(*inputs)
        output = (
            "complex128"
            if dtype in ("float64", "int64", "uint64", "int32", "uint32")
            else "complex64"
        )
        component = "float64" if output == "complex128" else "float32"
        inputs = tuple(
            dtype if isinstance(given, str) else component
            for given, dtype in zip(input_dtypes, inputs, strict=True)
        )
        return inputs, (output,)

    def impl(self, x, y):
        return builtins.complex(x, y)

    def pullback(self, inputs, outputs, gout):
        (x, y) = inputs
        (gz,) = gout
        return [cast(real(gz), x.type.dtype), cast(imag(gz), y.type.dtype)]

    def c_code(self, *args, **kwargs):
        raise NotImplementedError()


complex = Complex()


class Conj(UnaryScalarOp):
    name = "conj"

    resolve_dtypes = _resolve_numeric_bool_to_int8_dtypes

    preserves_zero = True
    nfunc_spec = ("conj", 1, 1)

    def impl(self, x):
        return np.conj(x)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            # For non complex, th
            raise NotImplementedError("type have no c code", node.inputs[0].type)
        return f"{z} = {x};"


conj = Conj()


class ScalarInnerGraphOp(ScalarOp, HasInnerGraph):
    """Includes boilerplate code for Python and C-implementation of Scalar Ops with inner graph."""

    __props__ = ("fgraph",)

    def __init__(self, *args, **kwargs):
        self.prepare_node_called = set()
        super().__init__(*args, **kwargs)

    def _validate_inner_graph(self, fgraph):
        """Validate that all ops in the inner graph are ScalarOps."""
        for node in fgraph.apply_nodes:
            if not isinstance(node.op, ScalarOp):
                raise TypeError(
                    f"The fgraph of {self.__class__.__name__} must be exclusively "
                    "composed of scalar operations."
                )

    @property
    def py_perform_fn(self):
        if hasattr(self, "_py_perform_fn"):
            return self._py_perform_fn

        from pytensor.link.utils import fgraph_to_python

        def python_convert(op, node=None, **kwargs):
            assert node is not None

            n_outs = len(node.outputs)

            if n_outs > 1:

                def _perform(*inputs, outputs=[[None] for _ in range(n_outs)]):
                    op.perform(node, inputs, outputs)
                    return tuple(o[0] for o in outputs)

            else:

                def _perform(*inputs, outputs=[[None]]):
                    op.perform(node, inputs, outputs)
                    return outputs[0][0]

            return _perform

        self._py_perform_fn = fgraph_to_python(self.fgraph, python_convert)
        return self._py_perform_fn

    def impl(self, *inputs):
        output_storage = [[None] for i in range(self.nout)]
        self.perform(None, inputs, output_storage)
        ret = to_return_values([storage[0] for storage in output_storage])
        if self.nout > 1:
            ret = tuple(ret)
        return ret

    def c_code_cache_version(self):
        rval = list(self.c_code_cache_version_outer())
        for x in self.fgraph.toposort():
            xv = x.op.c_code_cache_version()
            if xv:
                rval.append(xv)
            else:
                return ()
        return tuple(rval)

    def c_header_dirs(self, **kwargs):
        rval = list(
            chain.from_iterable(
                subnode.op.c_header_dirs(**kwargs) for subnode in self.fgraph.toposort()
            )
        )
        return rval

    def c_support_code(self, **kwargs):
        # Remove duplicate code blocks by using a `set`
        rval = {
            subnode.op.c_support_code(**kwargs).strip()
            for subnode in self.fgraph.toposort()
        }
        return "\n".join(sorted(rval))

    def c_support_code_apply(self, node, name):
        # Ensure nodenames is populated (side effect of c_code_template)
        _ = self.c_code_template
        rval = []
        for subnode, subnodename in zip(
            self.fgraph.toposort(), self.nodenames, strict=True
        ):
            subnode_support_code = subnode.op.c_support_code_apply(
                subnode, subnodename % dict(nodename=name)
            )
            if subnode_support_code:
                rval.append(subnode_support_code)
        # there should be no need to remove duplicate code blocks because
        # each block should have been specialized for the given nodename.
        # Any block that isn't specialized should be returned via
        # c_support_code instead of c_support_code_apply.
        return "\n".join(rval)

    def prepare_node(self, node, storage_map, compute_map, impl):
        if impl not in self.prepare_node_called:
            for n in applys_between(self.inputs, self.outputs):
                n.op.prepare_node(n, None, None, impl)
            self.prepare_node_called.add(impl)

    def __getstate__(self):
        rval = dict(self.__dict__)
        rval.pop("_c_code", None)
        rval.pop("_py_perform_fn", None)
        rval.pop("_name", None)
        rval.pop("prepare_node_called", None)
        return rval

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.prepare_node_called = set()


class Composite(ScalarInnerGraphOp):
    """
    Composite is an Op that takes a graph of scalar operations and
    produces c code for the whole graph. Its purpose is to implement loop
    fusion.

    Composite depends on all the Ops in its graph having C code.

    """

    _name = None

    def __init__(
        self,
        inputs,
        outputs,
        name="Composite",
    ):
        self.name = name

        for i in inputs:
            assert i not in outputs  # This isn't supported, use identity

        # Composite inner graphs have no inplace ops, so structurally-identical
        # nodes can be safely deduplicated.
        self.fgraph = FrozenFunctionGraph.from_io(inputs, outputs, dedup_nodes=True)
        self._validate_inner_graph(self.fgraph)

        self.inputs = self.fgraph.inputs
        self.outputs = self.fgraph.outputs
        self.inputs_type = tuple(inp.type for inp in self.inputs)
        self.outputs_type = tuple(out.type for out in self.outputs)
        self.nin = len(self.inputs)
        self.nout = len(self.outputs)
        super().__init__()

    def __str__(self):
        if self._name is not None:
            return self._name

        if len(self.fgraph.outputs) > 1 or len(self.fgraph.apply_nodes) > 10:
            self._name = "Composite{...}"
        else:
            outputs_str = ", ".join(pprint(output) for output in self.fgraph.outputs)
            self._name = f"Composite{{{outputs_str}}}"

        return self._name

    def clone(self):
        return self  # Op is immutable

    def resolve_dtypes(self, input_dtypes):
        expected = tuple(t.dtype for t in self.inputs_type)
        inputs = tuple(
            dtype if isinstance(dtype, str) else target
            for dtype, target in zip(input_dtypes, expected, strict=True)
        )
        if inputs != expected:
            node = self.make_node(*(get_scalar_type(dtype)() for dtype in inputs))
            return inputs, tuple(out.dtype for out in node.outputs)
        return inputs, tuple(t.dtype for t in self.outputs_type)

    def make_node(self, *inputs):
        inputs = [
            inp if isinstance(inp, ScalarVariable) else as_scalar(inp) for inp in inputs
        ]

        if self.inputs_type == tuple(inp.type for inp in inputs):
            return super().make_node(*inputs)

        if len(inputs) != self.nin:
            raise ValueError("Number of inputs does not match expected")

        # First try to coerce each input to its inner-graph input type.
        try:
            filtered_inputs = [
                inner_inp.type.filter_variable(inp)
                for inp, inner_inp in zip(inputs, self.inputs)
            ]
        except TypeError:
            pass
        else:
            return super().make_node(*filtered_inputs)

        # The new input types are incompatible with the inner graph.
        # Try to make a new inner graph with rebuild_strict=False.
        unfrozen_fgraph = self.fgraph.unfreeze()
        new_inner_inputs = [i.type() for i in inputs]
        new_outputs = clone_replace(
            unfrozen_fgraph.outputs,
            replace=dict(zip(unfrozen_fgraph.inputs, new_inner_inputs)),
            rebuild_strict=False,
        )
        return Composite(new_inner_inputs, new_outputs).make_node(*inputs)

    def perform(self, node, inputs, output_storage):
        outputs = self.py_perform_fn(*inputs)
        # zip strict not specified because we are in a hot loop
        for storage, out_val in zip(output_storage, outputs):
            storage[0] = out_val

    def pullback(self, inputs, outputs, output_grads):
        raise NotImplementedError("grad is not implemented for Composite")

    @property
    def c_code_template(self):
        from pytensor.link.c.interface import CLinkerType

        if hasattr(self, "_c_code"):
            return self._c_code

        fg = self.fgraph
        subd = {e: f"%(i{i})s" for i, e in enumerate(fg.inputs)}

        for var in fg.variables:
            if var.owner is None:
                if var not in fg.inputs:
                    # This is an orphan
                    if isinstance(var, Constant) and isinstance(var.type, CLinkerType):
                        subd[var] = f"({var.type.c_literal(var.data)})"
                    else:
                        raise ValueError(
                            "All orphans in the fgraph to Composite must"
                            " be Constant, CLinkerType instances."
                        )
            elif any(i.dtype == "float16" for i in var.owner.inputs) or any(
                o.dtype == "float16" for o in var.owner.outputs
            ):
                # flag for elemwise ops to check.
                self.inner_float16 = True

        self.nodenames = nodenames = []  # Used by self.c_support_code_apply

        _c_code = "{\n"
        i = 0
        for j, node in enumerate(fg.toposort()):
            for output in node.outputs:
                if output not in subd:
                    i += 1
                    name = f"V%(id)s_tmp{i}"
                    subd[output] = name
                    _c_code += f"{output.type.dtype_specs()[1]} {name};\n"

            nodename = f"%(nodename)s_subnode{j}"
            nodenames.append(nodename)

            s = node.op.c_code(
                node,
                nodename,
                [subd[input] for input in node.inputs],
                [subd[output] for output in node.outputs],
                dict(fail="%(fail)s", id=f"%(id)s_{j}"),
            )
            _c_code += s
            _c_code += "\n"

        # Copy the temporary outputs to the real outputs
        for i, output in enumerate(fg.outputs):
            _c_code += f"%(o{i})s = {subd[output]};\n"

        _c_code += "}\n"

        self._c_code = _c_code

        return self._c_code

    def c_code(self, node, nodename, inames, onames, sub):
        d = dict(
            chain(
                zip((f"i{i}" for i in range(len(inames))), inames, strict=True),
                zip((f"o{i}" for i in range(len(onames))), onames, strict=True),
            ),
            **sub,
        )
        d["nodename"] = nodename
        if "id" not in sub:
            # The use of a dummy id is safe as the code is in a separate block.
            # It won't generate conflicting variable name.
            d["id"] = "_DUMMY_ID_"

        return self.c_code_template % d

    def c_code_cache_version_outer(self) -> tuple[int, ...]:
        return (7,)
