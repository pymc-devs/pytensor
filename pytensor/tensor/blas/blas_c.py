from pytensor.link.c.op import COp
from pytensor.link.c.params_type import ParamsType
from pytensor.scalar import bool as bool_t
from pytensor.tensor.blas._core import ldflags
from pytensor.tensor.blas.c_code.blas_headers import (
    blas_header_text,
    blas_header_version,
)
from pytensor.tensor.blas.c_code.codegen import gemv_c_code, ger_c_code
from pytensor.tensor.blas.gemv import Gemv
from pytensor.tensor.blas.ger import Ger


class BaseBLAS(COp):
    def c_libraries(self, **kwargs):
        return ldflags()

    def c_compile_args(self, **kwargs):
        return ldflags(libs=False, flags=True)

    def c_lib_dirs(self, **kwargs):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self, **kwargs):
        return ldflags(libs=False, include_dir=True)

    def c_support_code(self, **kwargs):
        return blas_header_text()


# ##### ####### #######
# GER
# ##### ####### #######


class CGer(BaseBLAS, Ger):
    """C implementation of GER (rank-1 update): Z = A + alpha * outer(x, y)."""

    params_type = ParamsType(
        destructive=bool_t,
    )

    def c_code(self, node, name, inp, out, sub):
        return ger_c_code(node, name, inp, out, sub)

    def c_code_cache_version(self):
        return (11, blas_header_version())


cger_inplace = CGer(True)
cger_no_inplace = CGer(False)


# ##### ####### #######
# GEMV
# ##### ####### #######


class CGemv(BaseBLAS, Gemv):
    params_type = ParamsType(
        inplace=bool_t,
    )

    def __init__(self, inplace):
        super().__init__(inplace)

    def c_code(self, node, name, inp, out, sub):
        return gemv_c_code(node, name, inp, out, sub)

    def c_code_cache_version(self):
        return (18, blas_header_version(), must_initialize_y_gemv())


cgemv_inplace = CGemv(inplace=True)
cgemv_no_inplace = CGemv(inplace=False)


def must_initialize_y_gemv():
    if must_initialize_y_gemv._force_init_beta is None:
        from pytensor.link.c.cmodule import GCC_compiler

        """
        Test issue 1569.
        Namely when evaluating

            beta*y + alpha*dot(A, x)

        where we set y * beta = zeros of the correct dimensions we
        do not actually set y = zeros and instead let the BLAS
        perform beta*y with uninitialized memory for
        speed. Occasionally the memory contains values that are
        equivalent to NaN in which case the product beta*y contains
        NaN's for correctly implemented BLAS libraries. In this
        situation, since we are introducing the NaN's, we need to test
        whether the BLAS performs correctly. If it *does*, i.e. it
        actually performs the multiplication beta*y which will result
        in NaN's in the result, then we need initialize the memory to
        zeros.
        """
        test_code = """
#include <math.h>
extern "C" void dgemv_(char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int *);
int main() {
  double A[2][2] = {{1., 1.}, {1., 1.}};
  double x[2] = {1., 1.};
  double y[2] = {NAN, NAN};
  const int s = 2;
  const int inc = 1;
  const double alpha = 1.0;
  const double beta = 0.0;

  dgemv_("T", &s, &s, &alpha, A, &s, x, &inc, &beta, &y, &inc);

  return (isnan(y[0]) || isnan(y[1]) ? 1 : 0;
}
"""
        res = GCC_compiler.try_compile_tmp(
            test_code,
            tmp_prefix="check_beta_",
            flags=ldflags(libs=True, flags=True, libs_dir=True),
            try_run=True,
        )
        if res:
            if res[0]:
                must_initialize_y_gemv._force_init_beta = res[1]
            else:
                must_initialize_y_gemv._force_init_beta = False
        else:
            must_initialize_y_gemv._force_init_beta = False

    return must_initialize_y_gemv._force_init_beta


must_initialize_y_gemv._force_init_beta = None  # type: ignore[attr-defined]
