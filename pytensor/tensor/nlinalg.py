"""Deprecated: use ``pytensor.tensor.linalg`` instead."""

import warnings


_MOVED_NAMES: dict[str, str] = {
    "Eig": "pytensor.tensor._linalg.decomposition.eigen",
    "Eigh": "pytensor.tensor._linalg.decomposition.eigen",
    "EighGrad": "pytensor.tensor._linalg.decomposition.eigen",
    "_zero_disconnected": "pytensor.tensor._linalg.decomposition.eigen",
    "eig": "pytensor.tensor._linalg.decomposition.eigen",
    "eigh": "pytensor.tensor._linalg.decomposition.eigen",
    "SVD": "pytensor.tensor._linalg.decomposition.svd",
    "svd": "pytensor.tensor._linalg.decomposition.svd",
    "MatrixInverse": "pytensor.tensor._linalg.inverse",
    "MatrixPinv": "pytensor.tensor._linalg.inverse",
    "TensorInv": "pytensor.tensor._linalg.inverse",
    "inv": "pytensor.tensor._linalg.inverse",
    "matrix_inverse": "pytensor.tensor._linalg.inverse",
    "pinv": "pytensor.tensor._linalg.inverse",
    "tensorinv": "pytensor.tensor._linalg.inverse",
    "KroneckerProduct": "pytensor.tensor._linalg.products",
    "kron": "pytensor.tensor._linalg.products",
    "matrix_dot": "pytensor.tensor._linalg.products",
    "matrix_power": "pytensor.tensor._linalg.products",
    "Lstsq": "pytensor.tensor._linalg.solve.lstsq",
    "TensorSolve": "pytensor.tensor._linalg.solve.lstsq",
    "lstsq": "pytensor.tensor._linalg.solve.lstsq",
    "tensorsolve": "pytensor.tensor._linalg.solve.lstsq",
    "Det": "pytensor.tensor._linalg.summary",
    "SLogDet": "pytensor.tensor._linalg.summary",
    "_multi_svd_norm": "pytensor.tensor._linalg.summary",
    "det": "pytensor.tensor._linalg.summary",
    "norm": "pytensor.tensor._linalg.summary",
    "slogdet": "pytensor.tensor._linalg.summary",
    "trace": "pytensor.tensor._linalg.summary",
}


def __getattr__(name: str):
    if name in _MOVED_NAMES:
        mod_path = _MOVED_NAMES[name]
        warnings.warn(
            f"Importing {name!r} from 'pytensor.tensor.nlinalg' is deprecated. "
            f"Use 'from pytensor.tensor.linalg import {name}' instead. "
            "Imports from nlinalg will be removed in PyTensor 3.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        import importlib

        mod = importlib.import_module(mod_path)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_MOVED_NAMES.keys())
