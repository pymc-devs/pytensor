#ifndef PYTENSOR_MOD_HELPER
#define PYTENSOR_MOD_HELPER

#include <Python.h>

#ifndef _WIN32
#define MOD_PUBLIC __attribute__((visibility ("default")))
#else
/* MOD_PUBLIC is only used in PyMODINIT_FUNC, which is declared
 * and implemented in mod.cu/cpp, not in headers, so dllexport
 * is always correct. */
#define MOD_PUBLIC __declspec( dllexport )
#endif

#ifdef __cplusplus
#define PYTENSOR_EXTERN extern "C"
#else
#define PYTENSOR_EXTERN
#endif

/* We need to redefine PyMODINIT_FUNC to add MOD_PUBLIC in the middle */
#undef PyMODINIT_FUNC
#define PyMODINIT_FUNC PYTENSOR_EXTERN MOD_PUBLIC PyObject *

#endif
