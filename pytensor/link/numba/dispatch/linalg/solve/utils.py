from scipy import linalg

from pytensor.link.numba.dispatch import basic as numba_basic


@numba_basic.numba_njit(inline="always")
def _solve_check_input_shapes(A, B):
    if A.shape[0] != B.shape[0]:
        raise linalg.LinAlgError("Dimensions of A and B do not conform")
    if A.shape[-2] != A.shape[-1]:
        raise linalg.LinAlgError("Last 2 dimensions of A must be square")
