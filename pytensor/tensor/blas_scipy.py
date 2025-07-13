"""
Implementations of BLAS Ops based on scipy's BLAS bindings.
"""

from scipy.linalg.blas import get_blas_funcs

from pytensor.tensor.blas import Ger


class ScipyGer(Ger):
    def perform(self, node, inputs, output_storage):
        cA, calpha, cx, cy = inputs
        (cZ,) = output_storage
        A = cA
        ger_func = get_blas_funcs("ger", dtype=cA.dtype)
        if A.flags["C_CONTIGUOUS"]:
            # Work on transposed system to avoid copying
            A = ger_func(calpha, cy, cx, a=A.T, overwrite_a=self.destructive).T
        else:
            A = ger_func(calpha, cx, cy, a=A, overwrite_a=self.destructive)
        cZ[0] = A


scipy_ger_no_inplace = ScipyGer(False)
scipy_ger_inplace = ScipyGer(True)
