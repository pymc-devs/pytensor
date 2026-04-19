from collections.abc import Sequence
from functools import reduce

import numpy as np

from pytensor.tensor.variable import TensorVariable


def linalg_output_dtype(*input_dtypes: np.dtype | str) -> str:
    """Map one or more input dtypes to the working dtype LAPACK would use.

    Matches ``scipy.linalg.lapack.find_best_lapack_type``: each input is mapped to its nearest LAPACK type
    independently, then the results are combined via ``numpy.result_type``.
    """

    def _to_lapack_dtype(dt: np.dtype) -> np.dtype:
        """Map a single numpy dtype to the nearest LAPACK-supported dtype."""
        if dt.kind == "c":
            return np.dtype("complex64") if dt.itemsize <= 8 else np.dtype("complex128")
        if (dt.kind == "f" and dt.itemsize > 4) or (
            dt.kind in "ibu" and dt.itemsize > 2
        ):
            return np.dtype("float64")
        return np.dtype("float32")

    return np.result_type(*(_to_lapack_dtype(np.dtype(dt)) for dt in input_dtypes)).name


def linalg_real_output_dtype(*input_dtypes: np.dtype | str) -> str:
    """Companion to ``linalg_output_dtype`` for functions that map C -> R.

    Use for outputs that are mathematically real regardless of input type: singular values (SVD), eigenvalues of
    hermitian matrices (eigh), log-determinant (slogdet), norms, etc.
    """
    dt = np.dtype(linalg_output_dtype(*input_dtypes))
    if dt.kind == "c":
        return np.finfo(dt).dtype.name
    return dt.name


def _largest_common_dtype(tensors: Sequence[TensorVariable]) -> np.dtype:
    return reduce(lambda l, r: np.promote_types(l, r), [x.dtype for x in tensors])
