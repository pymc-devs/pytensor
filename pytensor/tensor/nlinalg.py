# Re-exports: decomposition ops that were moved to _linalg/decomposition/
# These are kept here for backwards compatibility.
from pytensor.tensor._linalg.decomposition.eigen import (  # noqa: F401
    Eig,
    Eigh,
    EighGrad,
    _zero_disconnected,
    eig,
    eigh,
)
from pytensor.tensor._linalg.decomposition.svd import SVD, svd  # noqa: F401

# Re-exports: inverse ops that were moved to _linalg/inverse.py
from pytensor.tensor._linalg.inverse import (  # noqa: F401
    MatrixInverse,
    MatrixPinv,
    TensorInv,
    inv,
    matrix_inverse,
    pinv,
    tensorinv,
)

# Re-exports: product ops that were moved to _linalg/products.py
from pytensor.tensor._linalg.products import (  # noqa: F401
    KroneckerProduct,
    kron,
    matrix_dot,
    matrix_power,
)

# Re-exports: solve ops that were moved to _linalg/solve/
from pytensor.tensor._linalg.solve.lstsq import (  # noqa: F401
    Lstsq,
    TensorSolve,
    lstsq,
    tensorsolve,
)

# Re-exports: summary ops that were moved to _linalg/summary.py
from pytensor.tensor._linalg.summary import (  # noqa: F401
    Det,
    SLogDet,
    _multi_svd_norm,
    det,
    norm,
    slogdet,
    trace,
)


__all__ = [
    "det",
    "eig",
    "eigh",
    "inv",
    "kron",
    "lstsq",
    "matrix_dot",
    "matrix_power",
    "norm",
    "pinv",
    "slogdet",
    "svd",
    "tensorinv",
    "tensorsolve",
    "trace",
]
