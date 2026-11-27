import numpy as np
from scipy.linalg import get_lapack_funcs

from pytensor.link.python.dispatch.basic import python_funcify
from pytensor.tensor.linalg.decomposition.qr import QR


@python_funcify.register(QR)
def python_funcify_QR(op, node=None, **kwargs):
    mode = op.mode
    pivoting = op.pivoting
    overwrite_a = op.overwrite_a
    call_and_get_lwork = op._call_and_get_lwork

    def qr(x):
        M, N = x.shape

        if pivoting:
            (geqp3,) = get_lapack_funcs(("geqp3",), (x,))
            factor, jpvt, tau, *_ = call_and_get_lwork(
                geqp3, x, lwork=-1, overwrite_a=overwrite_a
            )
            jpvt -= 1  # geqp3 returns 1-based indices
        else:
            (geqrf,) = get_lapack_funcs(("geqrf",), (x,))
            factor, tau, *_ = call_and_get_lwork(
                geqrf, x, lwork=-1, overwrite_a=overwrite_a
            )

        if mode not in ("economic", "raw") or M < N:
            R = np.triu(factor)
        else:
            R = np.triu(factor[:N, :])

        if mode == "r":
            return (R, jpvt) if pivoting else R
        if mode == "raw":
            return (factor, tau, R, jpvt) if pivoting else (factor, tau, R)

        (orgqr,) = get_lapack_funcs(("orgqr",), (factor,))
        if M < N:
            Q, *_ = call_and_get_lwork(
                orgqr, factor[:, :M], tau, lwork=-1, overwrite_a=1
            )
        elif mode == "economic":
            Q, *_ = call_and_get_lwork(orgqr, factor, tau, lwork=-1, overwrite_a=1)
        else:
            square = np.empty((M, M), dtype=factor.dtype.char)
            square[:, :N] = factor
            Q, *_ = call_and_get_lwork(orgqr, square, tau, lwork=-1, overwrite_a=1)

        return (Q, R, jpvt) if pivoting else (Q, R)

    return qr
