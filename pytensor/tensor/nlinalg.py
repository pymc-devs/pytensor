"""Deprecated: use ``pytensor.tensor.linalg`` instead."""

import warnings


_MOVED_NAMES: dict[str, str] = {
    "Eig": "pytensor.tensor.linalg.decomposition.eigen",
    "Eigh": "pytensor.tensor.linalg.decomposition.eigen",
    "EighGrad": "pytensor.tensor.linalg.decomposition.eigen",
    "_zero_disconnected": "pytensor.tensor.linalg.decomposition.eigen",
    "eig": "pytensor.tensor.linalg.decomposition.eigen",
    "eigh": "pytensor.tensor.linalg.decomposition.eigen",
    "SVD": "pytensor.tensor.linalg.decomposition.svd",
    "svd": "pytensor.tensor.linalg.decomposition.svd",
    "MatrixInverse": "pytensor.tensor.linalg.inverse",
    "MatrixPinv": "pytensor.tensor.linalg.inverse",
    "TensorInv": "pytensor.tensor.linalg.inverse",
    "inv": "pytensor.tensor.linalg.inverse",
    "matrix_inverse": "pytensor.tensor.linalg.inverse",
    "pinv": "pytensor.tensor.linalg.inverse",
    "tensorinv": "pytensor.tensor.linalg.inverse",
    "KroneckerProduct": "pytensor.tensor.linalg.products",
    "kron": "pytensor.tensor.linalg.products",
    "matrix_dot": "pytensor.tensor.linalg.products",
    "matrix_power": "pytensor.tensor.linalg.products",
    "Lstsq": "pytensor.tensor.linalg.solvers.lstsq",
    "TensorSolve": "pytensor.tensor.linalg.solvers.lstsq",
    "lstsq": "pytensor.tensor.linalg.solvers.lstsq",
    "tensorsolve": "pytensor.tensor.linalg.solvers.lstsq",
    "Det": "pytensor.tensor.linalg.summary",
    "SLogDet": "pytensor.tensor.linalg.summary",
    "_multi_svd_norm": "pytensor.tensor.linalg.summary",
    "det": "pytensor.tensor.linalg.summary",
    "norm": "pytensor.tensor.linalg.summary",
    "slogdet": "pytensor.tensor.linalg.summary",
    "trace": "pytensor.tensor.linalg.summary",
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
