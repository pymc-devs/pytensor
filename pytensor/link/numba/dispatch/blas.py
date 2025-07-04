from pytensor.link.numba.dispatch import numba_funcify
from pytensor.link.numba.dispatch.basic import numba_njit
from pytensor.link.numba.dispatch.linalg.dot.banded import _gbmv
from pytensor.link.numba.dispatch.linalg.dot.general import _matrix_vector_product
from pytensor.link.numba.dispatch.slinalg import _COMPLEX_DTYPE_NOT_SUPPORTED_MSG
from pytensor.tensor.blas import BandedGEMV, Gemv
from pytensor.tensor.type import complex_dtypes


@numba_funcify.register(Gemv)
def numba_funcify_Gemv(op, node, **kwargs):
    """
    Function to handle the Gemv operation in Numba.
    """
    overwrite_y = op.inplace

    @numba_njit()
    def numba_gemv(y, alpha, A, x, beta):
        """
        Numba implementation of the Gemv operation.
        """
        return _matrix_vector_product(
            alpha=alpha,
            A=A,
            x=x,
            beta=beta,
            y=y,
            overwrite_y=overwrite_y,
        )

    return numba_gemv


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
