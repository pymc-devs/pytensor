import errno
import sys
from pathlib import Path

from pytensor.compile.compilelock import lock_ctx
from pytensor.configdefaults import config
from pytensor.link.c import cmodule


# TODO These two lines may be removed in the future, when we are 100% sure
# no one has an old cutils_ext.so lying around anymore.
(config.compiledir / "cutils_ext.so").unlink(missing_ok=True)


def compile_cutils():
    """
    Do just the compilation of cutils_ext.

    """
    code = """
        #include <Python.h>
        #include "pytensor_mod_helper.h"

        extern "C"{
        static PyObject *
        run_cthunk(PyObject *self, PyObject *args)
        {
          PyObject *py_cthunk = NULL;
          if(!PyArg_ParseTuple(args,"O",&py_cthunk))
            return NULL;

          if (!PyCObject_Check(py_cthunk)) {
            PyErr_SetString(PyExc_ValueError,
                           "Argument to run_cthunk must be a PyCObject.");
            return NULL;
          }
          void * ptr_addr = PyCObject_AsVoidPtr(py_cthunk);
          int (*fn)(void*) = (int (*)(void*))(ptr_addr);
          void* it = PyCObject_GetDesc(py_cthunk);
          int failure = fn(it);

          return Py_BuildValue("i", failure);
         }
         static PyMethodDef CutilsExtMethods[] = {
            {"run_cthunk",  run_cthunk, METH_VARARGS|METH_KEYWORDS,
             "Run an pytensor cthunk."},
            {NULL, NULL, 0, NULL}        /* Sentinel */
        };"""

    # This is not the most efficient code, but it is written this way to
    # highlight the changes needed to make 2.x code compile under python 3.
    code = code.replace("<Python.h>", '"numpy/npy_3kcompat.h"', 1)
    code = code.replace("PyCObject", "NpyCapsule")
    code += """
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "cutils_ext",
        NULL,
        -1,
        CutilsExtMethods,
    };

    PyMODINIT_FUNC
    PyInit_cutils_ext(void) {
        return PyModule_Create(&moduledef);
    }
    }
    """

    loc = config.compiledir / "cutils_ext"
    if not loc.exists():
        try:
            loc.mkdir()
        except OSError as e:
            assert e.errno == errno.EEXIST
            assert loc.exists(), loc

    args = cmodule.GCC_compiler.compile_args(march_flags=False)
    cmodule.GCC_compiler.compile_str("cutils_ext", code, location=loc, preargs=args)


try:
    # See gh issue #728 for why these lines are here. Summary: compiledir
    # must be at the beginning of the path to avoid conflicts with any other
    # cutils_ext modules that might exist. An __init__.py file must be created
    # for the same reason. Note that these 5 lines may seem redundant (they are
    # repeated in compile_str()) but if another cutils_ext does exist then it
    # will be imported and compile_str won't get called at all.
    sys.path.insert(0, str(config.compiledir))
    location = config.compiledir / "cutils_ext"
    if not location.exists():
        try:
            location.mkdir()
        except OSError as e:
            assert e.errno == errno.EEXIST
            assert location.exists(), location
    (location / "__init__.py").touch(exist_ok=True)

    try:
        from cutils_ext.cutils_ext import *  # noqa
    except ImportError:
        with lock_ctx():
            # Ensure no-one else is currently modifying the content of the compilation
            # directory. This is important to prevent multiple processes from trying to
            # compile the cutils_ext module simultaneously.
            try:
                # We must retry to import it as some other process could
                # have been compiling it between the first failed import
                # and when we receive the lock
                from cutils_ext.cutils_ext import *  # noqa
            except ImportError:
                compile_cutils()
                from cutils_ext.cutils_ext import *  # noqa
finally:
    if config.compiledir.resolve() == Path(sys.path[0]).resolve():
        del sys.path[0]
