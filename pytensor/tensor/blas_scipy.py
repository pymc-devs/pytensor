"""
Implementations of BLAS Ops based on scipy's BLAS bindings.
"""

from pytensor.tensor.blas import Ger


class ScipyGer(Ger):
    def perform(self, node, inputs, output_storage):
        from scipy.linalg.blas import get_blas_funcs

        cA, calpha, cx, cy = inputs
        (cZ,) = output_storage
        # N.B. some versions of scipy (e.g. mine) don't actually work
        # in-place on a, even when I tell it to.
        A = cA
        local_ger = get_blas_funcs("ger", dtype=cA.dtype)
        if A.size == 0:
            # We don't have to compute anything, A is empty.
            # We need this special case because Numpy considers it
            # C-contiguous, which is confusing.
            if not self.destructive:
                # Sometimes numpy thinks empty matrices can share memory,
                # so here to stop DebugMode from complaining.
                A = A.copy()
        elif A.flags["C_CONTIGUOUS"]:
            A = local_ger(calpha, cy, cx, a=A.T, overwrite_a=int(self.destructive)).T
        else:
            A = local_ger(calpha, cx, cy, a=A, overwrite_a=int(self.destructive))
        cZ[0] = A


scipy_ger_no_inplace = ScipyGer(False)
scipy_ger_inplace = ScipyGer(True)
