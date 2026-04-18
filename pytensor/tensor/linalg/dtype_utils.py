import numpy as np


_float32 = np.dtype("float32")
_float64 = np.dtype("float64")
_complex64 = np.dtype("complex64")
_complex128 = np.dtype("complex128")


def _to_lapack_dtype(dt: np.dtype) -> np.dtype:
    """Map a single numpy dtype to the nearest LAPACK-supported dtype."""
    if dt.kind == "c":
        return _complex64 if dt.itemsize <= 8 else _complex128
    if (dt.kind == "f" and dt.itemsize > 4) or (dt.kind in "ibu" and dt.itemsize > 2):
        return _float64
    return _float32


_COMPLEX_TO_REAL = {_complex64: _float32, _complex128: _float64}


def linalg_output_dtype(*input_dtypes: np.dtype | str) -> str:
    """Map one or more input dtypes to the working dtype LAPACK would use.

    Matches ``scipy.linalg.lapack.find_best_lapack_type``: each input is mapped to its nearest LAPACK type
    independently, then the results are combined via ``numpy.result_type``.
    """
    if len(input_dtypes) == 1:
        return _to_lapack_dtype(np.dtype(input_dtypes[0])).name
    lapack_dtypes = [_to_lapack_dtype(np.dtype(dt)) for dt in input_dtypes]
    return np.result_type(*lapack_dtypes).name


def linalg_real_output_dtype(*input_dtypes: np.dtype | str) -> str:
    """Companion to ``linalg_output_dtype`` for functions that map C -> R.

    Use for outputs that are mathematically real regardless of input type: singular values (SVD), eigenvalues of
    hermitian matrices (eigh), log-determinant (slogdet), norms, etc.
    """
    dt = np.dtype(linalg_output_dtype(*input_dtypes))
    return _COMPLEX_TO_REAL.get(dt, dt).name
