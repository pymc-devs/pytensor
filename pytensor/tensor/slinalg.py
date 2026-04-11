"""Deprecated: use ``pytensor.tensor.linalg`` instead."""

import warnings


_MOVED_NAMES: dict[str, str] = {
    "BaseBlockDiagonal": "pytensor.tensor.linalg.constructors",
    "BlockDiagonal": "pytensor.tensor.linalg.constructors",
    "_largest_common_dtype": "pytensor.tensor.linalg.constructors",
    "block_diag": "pytensor.tensor.linalg.constructors",
    "Cholesky": "pytensor.tensor.linalg.decomposition.cholesky",
    "cholesky": "pytensor.tensor.linalg.decomposition.cholesky",
    "Eigvalsh": "pytensor.tensor.linalg.decomposition.eigen",
    "EigvalshGrad": "pytensor.tensor.linalg.decomposition.eigen",
    "eigvalsh": "pytensor.tensor.linalg.decomposition.eigen",
    "LU": "pytensor.tensor.linalg.decomposition.lu",
    "LUFactor": "pytensor.tensor.linalg.decomposition.lu",
    "PivotToPermutations": "pytensor.tensor.linalg.decomposition.lu",
    "lu": "pytensor.tensor.linalg.decomposition.lu",
    "lu_factor": "pytensor.tensor.linalg.decomposition.lu",
    "pivot_to_permutation": "pytensor.tensor.linalg.decomposition.lu",
    "QR": "pytensor.tensor.linalg.decomposition.qr",
    "qr": "pytensor.tensor.linalg.decomposition.qr",
    "QZ": "pytensor.tensor.linalg.decomposition.schur",
    "Schur": "pytensor.tensor.linalg.decomposition.schur",
    "ordqz": "pytensor.tensor.linalg.decomposition.schur",
    "qz": "pytensor.tensor.linalg.decomposition.schur",
    "schur": "pytensor.tensor.linalg.decomposition.schur",
    "Expm": "pytensor.tensor.linalg.products",
    "expm": "pytensor.tensor.linalg.products",
    "SolveBase": "pytensor.tensor.linalg.solvers.core",
    "_default_b_ndim": "pytensor.tensor.linalg.solvers.core",
    "Solve": "pytensor.tensor.linalg.solvers.general",
    "lu_solve": "pytensor.tensor.linalg.solvers.general",
    "solve": "pytensor.tensor.linalg.solvers.general",
    "CholeskySolve": "pytensor.tensor.linalg.solvers.psd",
    "cho_solve": "pytensor.tensor.linalg.solvers.psd",
    "SolveTriangular": "pytensor.tensor.linalg.solvers.triangular",
    "solve_triangular": "pytensor.tensor.linalg.solvers.triangular",
    "solve_continuous_lyapunov": "pytensor.tensor.linalg.solvers.linear_control",
    "solve_discrete_are": "pytensor.tensor.linalg.solvers.linear_control",
    "solve_discrete_lyapunov": "pytensor.tensor.linalg.solvers.linear_control",
    "solve_sylvester": "pytensor.tensor.linalg.solvers.linear_control",
}


def __getattr__(name: str):
    if name in _MOVED_NAMES:
        mod_path = _MOVED_NAMES[name]
        warnings.warn(
            f"Importing {name!r} from 'pytensor.tensor.slinalg' is deprecated. "
            f"Use 'from pytensor.tensor.linalg import {name}' instead. "
            "Imports from slinalg will be removed in PyTensor 3.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        import importlib

        mod = importlib.import_module(mod_path)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_MOVED_NAMES.keys())
