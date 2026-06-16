import numpy as np

import pytensor.scalar
from pytensor.graph.basic import Apply
from pytensor.graph.utils import InconsistencyError, MethodNotDefined
from pytensor.link.c.op import COp
from pytensor.link.c.params_type import ParamsType
from pytensor.printing import FunctionPrinter, pprint
from pytensor.scalar import bool as bool_t
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.blas._codegen import (
    dot22_c_code,
    dot22scalar_c_code,
    gemm_c_code,
)
from pytensor.tensor.blas._core import (
    ldflags,
    view_roots,
)
from pytensor.tensor.blas.blas_headers import (
    blas_header_text,
    blas_header_version,
)
from pytensor.tensor.type import DenseTensorType, tensor


class GemmRelated(COp):
    """Base class for Gemm and Dot22.

    This class provides a kind of templated gemm Op. The C code itself is
    emitted by the generators in :mod:`pytensor.tensor.blas._codegen`.
    """

    __props__: tuple[str, ...] = ()

    def c_support_code(self, **kwargs):
        # BLAS declarations plus the MOD macro and compute_strides helper used by
        # the GEMM templates in _codegen.py.
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

    def c_headers(self, **kwargs):
        return []

    def c_libraries(self, **kwargs):
        return ldflags()

    # code_cache_version is built by subclasses from
    # build_gemm_version

    def c_compile_args(self, **kwargs):
        return ldflags(libs=False, flags=True)

    def c_lib_dirs(self, **kwargs):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self, **kwargs):
        return ldflags(libs=False, include_dir=True)

    def build_gemm_version(self):
        return (14, blas_header_version())


class Gemm(GemmRelated):
    """In-place version of matrix-matrix multiplication (with accumulation).

    When a and b are scalars and x, y, and z are matrices, then

        gemm(z,a,x,y,b)

    is similar to

        b*z + a*dot(x,y)

    The difference between the two is that the top form is destructive
    on z, whereas the bottom form is not.  Gemm works in-place on the
    storage associated with z, and the L{Variable} returned by Gemm
    has a storage that will be aliased to the storage of the z
    argument. Because of this in-place computation, an L{Apply} of
    this op will destroy the L{Variable} z on which it operates.  (See
    L{DestructiveOps} for an explanation of what destroying means in
    the context of pytensor graphs. See L{BlasLapackSupport} for more
    optimized linear algebra operations.)

    """

    E_rank = "gemm only works for rank 2"
    E_scalar = "gemm requires scalar argument"
    E_z_uniq = "argument z aliased to x or y"  # TODO: justify / delete this
    E_mixed = "gemm requires matching dtypes"
    E_float = "gemm requires floating-point dtypes"

    __props__ = ("inplace",)
    params_type = ParamsType(
        inplace=bool_t,
    )
    check_input = False

    def __init__(self, inplace):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def __str__(self):
        if self.inplace:
            inplace_str = "inplace"
        else:
            inplace_str = "no_inplace"
        return f"{self.__class__.__name__}{{{inplace_str}}}"

    def __setstate__(self, dct):
        self.__dict__.update(dct)

        # Correctly reload older pickles where destroy_map were not
        # saved
        if "destroy_map" not in self.__dict__ and self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, *inputs):
        inputs = list(map(as_tensor_variable, inputs))

        if any(not isinstance(i.type, DenseTensorType) for i in inputs):
            raise NotImplementedError("Only dense tensor types are supported")

        if len(inputs) != 5:
            raise TypeError(
                f"Wrong number of inputs for {self} (expected 5, got {len(inputs)})"
            )
        z, a, x, y, b = inputs

        zr, xr, yr = (set(view_roots(i)) for i in (z, x, y))

        # We want the gemm to be inplace. When this op is inplace, it
        # declare to be inplace only on z. So to make it safe, we
        # raise an error if z can be a view on x or y.

        # I don't know if PyTensor currently can support that case. As
        # this case don't happen in our code, I won't spent time
        # investigating this. So the assert is for safety.  I also
        # think there is another mechanism that would prevent this,
        # but I don't what to modify old code and have chance to break
        # something.
        if self.inplace:
            if zr.intersection(xr):
                raise InconsistencyError(Gemm.E_z_uniq, (z, x))
            if zr.intersection(yr):
                raise InconsistencyError(Gemm.E_z_uniq, (z, y))

        if z.ndim != 2:
            raise TypeError(Gemm.E_rank, z)
        if a.ndim != 0:
            raise TypeError(Gemm.E_scalar, a)
        if x.ndim != 2:
            raise TypeError(Gemm.E_rank, x)
        if y.ndim != 2:
            raise TypeError(Gemm.E_rank, y)
        if b.ndim != 0:
            raise TypeError(Gemm.E_scalar, b)

        if not (z.dtype == a.dtype == x.dtype == y.dtype == b.dtype):
            raise TypeError(Gemm.E_mixed, (z.dtype, a.dtype, x.dtype, y.dtype, b.dtype))

        if not z.dtype.startswith("float") and not z.dtype.startswith("complex"):
            raise TypeError(Gemm.E_float, (z.dtype))

        output = z.type()
        return Apply(self, inputs, [output])

    def perform(self, node, inp, out):
        z, a, x, y, b = inp
        (zout,) = out
        assert a.shape == ()
        assert b.shape == ()
        if not self.inplace:
            z = z.copy()  # the original z will not be changed
        if z.shape == ():
            z.itemset(z * a + b * np.dot(x, y))
            zout[0] = z
        else:
            # Broadcast Z if needed
            if (x.shape[0] > z.shape[0]) or (y.shape[1] > z.shape[1]):
                z = np.broadcast_to(
                    z, (max(x.shape[0], z.shape[0]), max(y.shape[1], z.shape[1]))
                ).copy()
            if b == 0.0:
                if a == 1.0:
                    z[:] = np.dot(x, y)
                elif a == -1.0:
                    z[:] = -np.dot(x, y)
                else:
                    z[:] = a * np.dot(x, y)
            elif b == 1.0:
                if a == 1.0:
                    z += np.dot(x, y)
                elif a == -1.0:
                    z -= np.dot(x, y)
                else:
                    z += a * np.dot(x, y)
            else:
                z *= b
                z += a * np.dot(x, y)
            zout[0] = z

    def infer_shape(self, node, input_shapes):
        z_shape, _, x_shape, y_shape, _ = input_shapes
        return [
            (
                pytensor.scalar.maximum(z_shape[0], x_shape[0]),
                pytensor.scalar.maximum(z_shape[1], y_shape[1]),
            )
        ]

    def c_code(self, node, name, inp, out, sub):
        if node.inputs[0].type.dtype.startswith("complex"):
            raise MethodNotDefined(f"{self.__class__.__name__}.c_code")
        return gemm_c_code(node, name, inp, out, sub)

    def c_code_cache_version(self):
        gv = self.build_gemm_version()
        if gv:
            return (8, *gv)
        else:
            return gv


gemm_inplace = Gemm(inplace=True)
gemm_no_inplace = Gemm(inplace=False)
# For the user interface. PyTensor optimization will make them inplace
gemm = gemm_no_inplace
pprint.assign(gemm_inplace, FunctionPrinter(["gemm_inplace"]))
pprint.assign(gemm_no_inplace, FunctionPrinter(["gemm_no_inplace"]))


class Dot22(GemmRelated):
    """Compute a matrix-matrix product.

    This is a specialization of the more general Dot().

    """

    check_input = False

    def make_node(self, x, y):
        x = as_tensor_variable(x)
        y = as_tensor_variable(y)

        if any(not isinstance(i.type, DenseTensorType) for i in (x, y)):
            raise NotImplementedError("Only dense tensor types are supported")

        dtypes = ("float16", "float32", "float64", "complex64", "complex128")
        if x.type.ndim != 2 or x.type.dtype not in dtypes:
            raise TypeError(x)
        if y.type.ndim != 2 or y.type.dtype not in dtypes:
            raise TypeError(y)
        if y.type.dtype != x.type.dtype:
            raise TypeError("dtype mismatch to Dot22")
        outputs = [tensor(dtype=x.type.dtype, shape=(x.type.shape[0], y.type.shape[1]))]
        return Apply(self, [x, y], outputs)

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = np.dot(*inputs)

    def infer_shape(self, node, input_shapes):
        return [[input_shapes[0][0], input_shapes[1][1]]]

    def c_code(self, node, name, inp, out, sub):
        if node.inputs[0].type.dtype.startswith("complex"):
            raise MethodNotDefined(f"{self.__class__.__name__}.c_code")
        if len(self.c_libraries()) <= 0:
            raise NotImplementedError()
        return dot22_c_code(node, name, inp, out, sub)

    def c_code_cache_version(self):
        gv = self.build_gemm_version()
        if gv:
            return (2, *gv)
        else:
            return gv


_dot22 = Dot22()


class Dot22Scalar(GemmRelated):
    """Compute a matrix-matrix product.

    This is a specialization of the more general Dot()
    Used to call optimized gemm implementation.
    Also used to generate a gemm later.
    compute scalar*dot(x,y).

    """

    check_input = False

    def make_node(self, x, y, a):
        if any(not isinstance(i.type, DenseTensorType) for i in (x, y, a)):
            raise NotImplementedError("Only dense tensor types are supported")

        if a.ndim != 0:
            raise TypeError(Gemm.E_scalar, a)
        if x.ndim != 2:
            raise TypeError(Gemm.E_rank, x)
        if y.ndim != 2:
            raise TypeError(Gemm.E_rank, y)

        if not (a.dtype == x.dtype == y.dtype):
            raise TypeError(
                "Dot22Scalar requires matching dtypes", (a.dtype, x.dtype, y.dtype)
            )

        if not a.dtype.startswith("float") and not a.dtype.startswith("complex"):
            raise TypeError("Dot22Scalar requires float or complex args", a.dtype)

        sz = (x.type.shape[0], y.type.shape[1])
        outputs = [tensor(dtype=x.type.dtype, shape=sz)]
        return Apply(self, [x, y, a], outputs)

    def perform(self, node, inp, out):
        x, y, scalar = inp
        (z,) = out
        try:
            z[0] = np.asarray(scalar * np.dot(x, y))
        except ValueError as e:
            # The error raised by numpy has no shape information, we
            # mean to add that
            e.args = (*e.args, x.shape, y.shape)
            raise

    def infer_shape(self, node, input_shapes):
        return [[input_shapes[0][0], input_shapes[1][1]]]

    def c_code(self, node, name, inp, out, sub):
        if node.inputs[0].type.dtype.startswith("complex"):
            raise MethodNotDefined(f"{self.__class__.__name__}.c_code")
        if len(self.c_libraries()) <= 0:
            raise NotImplementedError()
        return dot22scalar_c_code(node, name, inp, out, sub)

    def c_code_cache_version(self):
        gv = self.build_gemm_version()
        if gv:
            return (2, *gv)
        else:
            return gv


_dot22scalar = Dot22Scalar()
