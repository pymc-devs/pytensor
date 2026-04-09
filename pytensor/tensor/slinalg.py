import warnings


_deprecated_names = {
    "solve_continuous_lyapunov",
    "solve_discrete_are",
    "solve_discrete_lyapunov",
}


def __getattr__(name):
    if name in _deprecated_names:
        warnings.warn(
            f"{name} has been moved from tensor/slinalg.py as part of a reorganization "
            "of linear algebra routines in Pytensor. Imports from slinalg.py will fail in Pytensor 3.0.\n"
            f"Please use the stable user-facing linalg API: from pytensor.tensor.linalg import {name}",
            DeprecationWarning,
            stacklevel=2,
        )
        from pytensor.tensor._linalg.solve import linear_control

        return getattr(linear_control, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


# Re-exports: decomposition ops that were moved to _linalg/decomposition/
# These are kept here for backwards compatibility.
# Re-exports: constructor ops that were moved to _linalg/constructors.py
from pytensor.tensor._linalg.constructors import (  # noqa: E402, F401
    BaseBlockDiagonal,
    BlockDiagonal,
    _largest_common_dtype,
    block_diag,
)
from pytensor.tensor._linalg.decomposition.cholesky import (  # noqa: E402, F401
    Cholesky,
    cholesky,
)
from pytensor.tensor._linalg.decomposition.eigen import (  # noqa: E402, F401
    Eigvalsh,
    EigvalshGrad,
    eigvalsh,
)
from pytensor.tensor._linalg.decomposition.lu import (  # noqa: E402, F401
    LU,
    LUFactor,
    PivotToPermutations,
    lu,
    lu_factor,
    pivot_to_permutation,
)
from pytensor.tensor._linalg.decomposition.qr import QR, qr  # noqa: E402, F401
from pytensor.tensor._linalg.decomposition.schur import (  # noqa: E402, F401
    QZ,
    Schur,
    ordqz,
    qz,
    schur,
)

# Re-exports: product ops that were moved to _linalg/products.py
from pytensor.tensor._linalg.products import Expm, expm  # noqa: E402, F401

# Re-exports: solve ops that were moved to _linalg/solve/
from pytensor.tensor._linalg.solve.core import (  # noqa: E402, F401
    SolveBase,
    _default_b_ndim,
)
from pytensor.tensor._linalg.solve.general import (  # noqa: E402, F401
    Solve,
    lu_solve,
    solve,
)
from pytensor.tensor._linalg.solve.psd import (  # noqa: E402, F401
    CholeskySolve,
    cho_solve,
)
from pytensor.tensor._linalg.solve.triangular import (  # noqa: E402, F401
    SolveTriangular,
    solve_triangular,
)


__all__ = [
    "block_diag",
    "cho_solve",
    "cholesky",
    "eigvalsh",
    "expm",
    "lu",
    "lu_factor",
    "lu_solve",
    "ordqz",
    "pivot_to_permutation",
    "qr",
    "qz",
    "schur",
    "solve",
    "solve_triangular",
]
