"""Deprecated: use ``pytensor.tensor.linalg`` instead."""

import warnings


_MOVED_NAMES: dict[str, str] = {
    "BaseBlockDiagonal": "pytensor.tensor._linalg.constructors",
    "BlockDiagonal": "pytensor.tensor._linalg.constructors",
    "_largest_common_dtype": "pytensor.tensor._linalg.constructors",
    "block_diag": "pytensor.tensor._linalg.constructors",
    "Cholesky": "pytensor.tensor._linalg.decomposition.cholesky",
    "cholesky": "pytensor.tensor._linalg.decomposition.cholesky",
    "Eigvalsh": "pytensor.tensor._linalg.decomposition.eigen",
    "EigvalshGrad": "pytensor.tensor._linalg.decomposition.eigen",
    "eigvalsh": "pytensor.tensor._linalg.decomposition.eigen",
    "LU": "pytensor.tensor._linalg.decomposition.lu",
    "LUFactor": "pytensor.tensor._linalg.decomposition.lu",
    "PivotToPermutations": "pytensor.tensor._linalg.decomposition.lu",
    "lu": "pytensor.tensor._linalg.decomposition.lu",
    "lu_factor": "pytensor.tensor._linalg.decomposition.lu",
    "pivot_to_permutation": "pytensor.tensor._linalg.decomposition.lu",
    "QR": "pytensor.tensor._linalg.decomposition.qr",
    "qr": "pytensor.tensor._linalg.decomposition.qr",
    "QZ": "pytensor.tensor._linalg.decomposition.schur",
    "Schur": "pytensor.tensor._linalg.decomposition.schur",
    "ordqz": "pytensor.tensor._linalg.decomposition.schur",
    "qz": "pytensor.tensor._linalg.decomposition.schur",
    "schur": "pytensor.tensor._linalg.decomposition.schur",
    "Expm": "pytensor.tensor._linalg.products",
    "expm": "pytensor.tensor._linalg.products",
    "SolveBase": "pytensor.tensor._linalg.solve.core",
    "_default_b_ndim": "pytensor.tensor._linalg.solve.core",
    "Solve": "pytensor.tensor._linalg.solve.general",
    "lu_solve": "pytensor.tensor._linalg.solve.general",
    "solve": "pytensor.tensor._linalg.solve.general",
    "CholeskySolve": "pytensor.tensor._linalg.solve.psd",
    "cho_solve": "pytensor.tensor._linalg.solve.psd",
    "SolveTriangular": "pytensor.tensor._linalg.solve.triangular",
    "solve_triangular": "pytensor.tensor._linalg.solve.triangular",
    "solve_continuous_lyapunov": "pytensor.tensor._linalg.solve.linear_control",
    "solve_discrete_are": "pytensor.tensor._linalg.solve.linear_control",
    "solve_discrete_lyapunov": "pytensor.tensor._linalg.solve.linear_control",
    "solve_sylvester": "pytensor.tensor._linalg.solve.linear_control",
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
