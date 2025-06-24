from pytensor.link.numba.dispatch import numba_funcify
from pytensor.link.numba.dispatch.basic import numba_njit
from pytensor.link.numba.dispatch.linalg.dot.banded import _gbmv
from pytensor.link.numba.dispatch.slinalg import _COMPLEX_DTYPE_NOT_SUPPORTED_MSG
from pytensor.tensor.blas import BandedGEMV
from pytensor.tensor.type import complex_dtypes


@numba_funcify.register(BandedGEMV)
def numba_funcify_BandedGEMV(op, node, **kwargs):
    kl = op.lower_diags
    ku = op.upper_diags
    overwrite_y = op.overwrite_y
    trans = int(op.transpose)
    dtype = node.inputs[0].dtype

    if dtype in complex_dtypes:
        raise NotImplementedError(_COMPLEX_DTYPE_NOT_SUPPORTED_MSG.format(op=op))

    @numba_njit(cache=False)
    def banded_gemv(A, x, y, alpha, beta):
        return _gbmv(
            A=A,
            x=x,
            kl=kl,
            ku=ku,
            y=y,
            alpha=alpha,
            beta=beta,
            overwrite_y=overwrite_y,
            trans=trans,
        )

    return banded_gemv
