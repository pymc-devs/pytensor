from collections.abc import Hashable

from pytensor.configdefaults import config
from pytensor.graph.basic import Apply
from pytensor.graph.utils import MethodNotDefined
from pytensor.link.c.dispatch.basic import CImpl, c_funcify
from pytensor.link.c.params_type import Params, ParamsType
from pytensor.scalar import bool as bool_t
from pytensor.tensor.blas._core import ldflags, must_initialize_y_gemv
from pytensor.tensor.blas.c_code.blas_headers import (
    blas_header_text,
    blas_header_version,
)
from pytensor.tensor.blas.c_code.codegen import gemm_c_code, gemv_c_code, ger_c_code
from pytensor.tensor.blas.gemm import Gemm
from pytensor.tensor.blas.gemv import Gemv
from pytensor.tensor.blas.ger import Ger


class BlasImpl(CImpl):
    """Base for the BLAS C implementations: link flags, headers, and the ``inplace`` param.

    The generated code reads ``inplace`` out of a `ParamsType` rather than baking it in, so one
    compiled module serves both forms of an op.
    """

    params_type = ParamsType(inplace=bool_t)

    def get_params(self, node: Apply) -> Params:
        return self.params_type.get_params(self.op)

    def c_support_code(self, **kwargs) -> str:
        return blas_header_text()

    def c_headers(self, **kwargs) -> list[str]:
        return []

    def c_libraries(self, **kwargs) -> list[str]:
        return ldflags()

    def c_compile_args(self, **kwargs) -> list[str]:
        return ldflags(libs=False, flags=True)

    def c_lib_dirs(self, **kwargs) -> list[str]:
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self, **kwargs) -> list[str]:
        return ldflags(libs=False, include_dir=True)


class GemmImpl(BlasImpl):
    """C implementation of `Gemm`."""

    op: Gemm

    def c_support_code(self, **kwargs) -> str:
        # BLAS declarations plus the MOD macro and compute_strides helper the GEMM templates
        # in codegen.py expect.
        mod_str = """
        #ifndef MOD
        #define MOD %
        #endif
        void compute_strides(npy_intp *shape, int N_shape, int type_size, npy_intp *res) {
            int s;
            res[N_shape - 1] = type_size;
            for (int i = N_shape - 1; i > 0; i--) {
                s = shape[i];
                res[i - 1] = res[i] * (s > 0 ? s : 1);
            }
        }
        """
        return blas_header_text() + mod_str

    def c_code_cache_version(self) -> tuple[Hashable, ...]:
        return (8, 14, blas_header_version())

    def c_code(self, node, name, inp, out, sub) -> str:
        if node.inputs[0].type.dtype.startswith("complex"):
            raise MethodNotDefined("GemmImpl.c_code")
        return gemm_c_code(node, name, inp, out, sub)


class GemvImpl(BlasImpl):
    """C implementation of `Gemv`."""

    op: Gemv

    def c_code_cache_version(self) -> tuple[Hashable, ...]:
        return (18, blas_header_version(), must_initialize_y_gemv())

    def c_code(self, node, name, inp, out, sub) -> str:
        # No `blas__ldflags` guard: the fallback header supplies its own [sd]gemv_ and [sd]dot_.
        if node.outputs[0].dtype not in ("float32", "float64"):
            raise MethodNotDefined("GemvImpl.c_code")
        return gemv_c_code(node, name, inp, out, sub)


class GerImpl(BlasImpl):
    """C implementation of `Ger`."""

    op: Ger

    def c_code_cache_version(self) -> tuple[Hashable, ...]:
        return (11, blas_header_version())

    def c_code(self, node, name, inp, out, sub) -> str:
        # Unlike gemv, the fallback header defines no [sd]ger_, so without link flags this code
        # would not link.
        if not config.blas__ldflags or node.outputs[0].dtype not in (
            "float32",
            "float64",
        ):
            raise MethodNotDefined("GerImpl.c_code")
        return ger_c_code(node, name, inp, out, sub)


@c_funcify.register(Gemm)
def c_funcify_Gemm(op, node=None, **kwargs) -> GemmImpl:
    return GemmImpl(op)


@c_funcify.register(Gemv)
def c_funcify_Gemv(op, node=None, **kwargs) -> GemvImpl:
    return GemvImpl(op)


@c_funcify.register(Ger)
def c_funcify_Ger(op, node=None, **kwargs) -> GerImpl:
    return GerImpl(op)
