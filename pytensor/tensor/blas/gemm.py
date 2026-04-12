import numpy as np

import pytensor.scalar
from pytensor.graph.basic import Apply
from pytensor.graph.utils import InconsistencyError, MethodNotDefined
from pytensor.link.c.op import COp
from pytensor.link.c.params_type import ParamsType
from pytensor.printing import FunctionPrinter, pprint
from pytensor.scalar import bool as bool_t
from pytensor.tensor.basic import as_tensor_variable
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

    This class provides a kind of templated gemm Op.
    """

    __props__: tuple[str, ...] = ()

    def c_support_code(self, **kwargs):
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

    declare_NS = """
        int unit = 0;

        int type_num = PyArray_DESCR(%(_x)s)->type_num;
        int type_size = PyArray_ITEMSIZE(%(_x)s); // in bytes

        npy_intp* Nx = PyArray_DIMS(%(_x)s);
        npy_intp* Ny = PyArray_DIMS(%(_y)s);
        npy_intp* Nz = 0; //PyArray_DIMS(%(_zout)s);

        npy_intp* Sx = PyArray_STRIDES(%(_x)s);
        npy_intp* Sy = PyArray_STRIDES(%(_y)s);
        npy_intp* Sz = 0; //PyArray_STRIDES(%(_zout)s);

        //strides for x, y, z in dimensions 0, 1
        int sx_0, sx_1, sy_0, sy_1, sz_0, sz_1;
        """

    # implement if you don't have an inplace props
    # setup_z_Nz_Sz = None
    # otherwise implement
    # setup_z_Nz_Sz_inplace = None
    # setup_z_Nz_Sz_outplace = None

    check_xyz_rank2 = """
        if (PyArray_NDIM(%(_x)s) != 2) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(x) != 2. rank(x) is %%d.",
                         PyArray_NDIM(%(_x)s));
            %(fail)s;
        }
        if (PyArray_NDIM(%(_y)s) != 2) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(y) != 2. rank(y) is %%d.", PyArray_NDIM(%(_y)s));
            %(fail)s;
        }
        if (%(_zout)s && PyArray_NDIM(%(_zout)s) != 2) {
            PyErr_Format(PyExc_NotImplementedError,
                         "rank(z) != 2. rank(z) is %%d.", PyArray_NDIM(%(_zout)s));
            %(fail)s;
        }
        """
    check_xyz_double_or_float = """
        if ((PyArray_DESCR(%(_x)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_x)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(x) is not double or float"); %(fail)s;}

        if ((PyArray_DESCR(%(_y)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_y)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(y) is not double or float"); %(fail)s;}

        if ((PyArray_DESCR(%(_zout)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_zout)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(z) is not double or float"); %(fail)s;}

        if ((PyArray_DESCR(%(_x)s)->type_num != PyArray_DESCR(%(_y)s)->type_num)
            ||(PyArray_DESCR(%(_x)s)->type_num != PyArray_DESCR(%(_zout)s)->type_num))
        { PyErr_SetString(PyExc_NotImplementedError, "type(x), type(y), type(z) are not all the same"); %(fail)s; }
        """

    # it is not necessary that a or b have the same type as x,y,z
    check_ab_double_or_float = """
        if ((PyArray_DESCR(%(_a)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_a)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(a) is not double or float"); %(fail)s;}

        if ((PyArray_DESCR(%(_b)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_b)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(b) is not double or float"); %(fail)s;}
        """

    # broadcast_xy = None

    check_dims = """
        if (Nx[0] !=1 && Nz[0] != 1 && Nx[0] != Nz[0])
        {
            PyErr_Format(PyExc_ValueError,
                "Shape mismatch: x has %%ld rows but z has %%ld rows",
                (long int)Nx[0], (long int)Nz[0]);
            %(fail)s;
        }
        if (Nx[1] != Ny[0])
        {
            PyErr_Format(PyExc_ValueError,
                "Shape mismatch: x has %%ld cols (and %%ld rows) but y has %%ld rows (and %%ld cols)",
                (long int)Nx[1], (long int)Nx[0], (long int)Ny[0], (long int)Ny[1]);
            %(fail)s;
        }
        if (Ny[1] != 1 && Nz[1]!= 1 && Ny[1] != Nz[1])
        {
            PyErr_Format(PyExc_ValueError,
                "Shape mismatch: y has %%ld cols but z has %%ld cols",
                (long int)Ny[1], (long int)Nz[1]);
            %(fail)s;
        }

        // We must not raise an error when Nx[1] == 0. This would disable cases
        // that numpy.dot accept.
        """

    check_strides = """
        /*
        If some matrices are not contiguous on either dimensions,
        or have invalid strides, copy their content into a contiguous one
        */
        if ((Sx[0] < 1) || (Sx[1] < 1) || (Sx[0] MOD type_size) || (Sx[1] MOD type_size)
            || ((Sx[0] != type_size) && (Sx[1] != type_size)))
        {
            PyArrayObject * _x_copy = (PyArrayObject *) PyArray_Copy(%(_x)s);
            if (!_x_copy)
                %(fail)s
            Py_XDECREF(%(_x)s);
            %(_x)s = _x_copy;
            Sx = PyArray_STRIDES(%(_x)s);
            if ((Sx[0] < 1) || (Sx[1] < 1)) {
                compute_strides(Nx, 2, type_size, Sx);
            }
        }

        if ((Sy[0] < 1) || (Sy[1] < 1) || (Sy[0] MOD type_size) || (Sy[1] MOD type_size)
            || ((Sy[0] != type_size) && (Sy[1] != type_size)))
        {
            PyArrayObject * _y_copy = (PyArrayObject *) PyArray_Copy(%(_y)s);
            if (!_y_copy)
                %(fail)s
            Py_XDECREF(%(_y)s);
            %(_y)s = _y_copy;
            Sy = PyArray_STRIDES(%(_y)s);
            if ((Sy[0] < 1) || (Sy[1] < 1)) {
                compute_strides(Ny, 2, type_size, Sy);
            }
        }

        if ((Sz[0] < 1) || (Sz[1] < 1) || (Sz[0] MOD type_size) || (Sz[1] MOD type_size)
            || ((Sz[0] != type_size) && (Sz[1] != type_size)))
        {
            PyArrayObject * _z_copy = (PyArrayObject *) PyArray_Copy(%(_zout)s);
            if (!_z_copy)
                %(fail)s
            Py_XDECREF(%(_zout)s);
            %(_zout)s = _z_copy;
            Sz = PyArray_STRIDES(%(_zout)s);
            if ((Sz[0] < 1) || (Sz[1] < 1)) {
                compute_strides(Nz, 2, type_size, Sz);
            }
        }
        """

    encode_strides_in_unit = """
        /*
        encode the stride structure of _x,_y,_zout into a single integer
        */
        unit |= ((Sx[1] == type_size || Nx[1]==1) ? 0x0 : (Sx[0] == type_size || Nx[0]==1) ? 0x1 : 0x2) << 8;
        unit |= ((Sy[1] == type_size || Ny[1]==1) ? 0x0 : (Sy[0] == type_size || Ny[0]==1) ? 0x1 : 0x2) << 4;
        unit |= ((Sz[1] == type_size || Nz[1]==1) ? 0x0 : (Sz[0] == type_size || Nz[0]==1) ? 0x1 : 0x2) << 0;
        """

    compute_strides = """
        /* create appropriate strides for malformed matrices that are row or column
         * vectors, or empty matrices.
         * In that case, the value of the stride does not really matter, but
         * some versions of BLAS insist that:
         *  - they are not smaller than the number of elements in the array,
         *  - they are not 0.
         */
        sx_0 = (Nx[0] > 1) ? Sx[0]/type_size : (Nx[1] + 1);
        sx_1 = (Nx[1] > 1) ? Sx[1]/type_size : (Nx[0] + 1);
        sy_0 = (Ny[0] > 1) ? Sy[0]/type_size : (Ny[1] + 1);
        sy_1 = (Ny[1] > 1) ? Sy[1]/type_size : (Ny[0] + 1);
        sz_0 = (Nz[0] > 1) ? Sz[0]/type_size : (Nz[1] + 1);
        sz_1 = (Nz[1] > 1) ? Sz[1]/type_size : (Nz[0] + 1);
        """

    begin_switch_typenum = """
        switch (type_num)
        {
        """

    case_float = """
            case NPY_FLOAT:
            {
        """

    # case_float_ab_constants = None

    case_float_gemm = """
                float* x = (float*)PyArray_DATA(%(_x)s);
                float* y = (float*)PyArray_DATA(%(_y)s);
                float* z = (float*)PyArray_DATA(%(_zout)s);
                char N = 'N';
                char T = 'T';
                int Nz0 = Nz[0], Nz1 = Nz[1], Nx1 = Nx[1];
                switch(unit)
                {
                    case 0x000: sgemm_(&N, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_0, &b, z, &sz_0); break;
                    case 0x100: sgemm_(&N, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_1, &b, z, &sz_0); break;
                    case 0x010: sgemm_(&T, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_0, &b, z, &sz_0); break;
                    case 0x110: sgemm_(&T, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_1, &b, z, &sz_0); break;
                    case 0x001: sgemm_(&T, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_0, &b, z, &sz_1); break;
                    case 0x101: sgemm_(&N, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_0, &b, z, &sz_1); break;
                    case 0x011: sgemm_(&T, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_1, &b, z, &sz_1); break;
                    case 0x111: sgemm_(&N, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_1, &b, z, &sz_1); break;
                    default: PyErr_SetString(PyExc_ValueError, "some matrix has no unit stride"); %(fail)s;
                };
        """

    case_double = """
            }
            break;
            case NPY_DOUBLE:
            {
        """

    # case_double_ab_constants = None

    case_double_gemm = """
                double* x = (double*)PyArray_DATA(%(_x)s);
                double* y = (double*)PyArray_DATA(%(_y)s);
                double* z = (double*)PyArray_DATA(%(_zout)s);
                char N = 'N';
                char T = 'T';
                int Nz0 = Nz[0], Nz1 = Nz[1], Nx1 = Nx[1];
                switch(unit)
                {
                    case 0x000: dgemm_(&N, &N, &Nz1, &Nz0, &Nx1, &a, y,
                                       &sy_0, x, &sx_0, &b, z, &sz_0); break;
                    case 0x100: dgemm_(&N, &T, &Nz1, &Nz0, &Nx1, &a, y,
                                       &sy_0, x, &sx_1, &b, z, &sz_0); break;
                    case 0x010: dgemm_(&T, &N, &Nz1, &Nz0, &Nx1, &a, y,
                                       &sy_1, x, &sx_0, &b, z, &sz_0); break;
                    case 0x110: dgemm_(&T, &T, &Nz1, &Nz0, &Nx1, &a, y,
                                       &sy_1, x, &sx_1, &b, z, &sz_0); break;
                    case 0x001: dgemm_(&T, &T, &Nz0, &Nz1, &Nx1, &a, x,
                                       &sx_0, y, &sy_0, &b, z, &sz_1); break;
                    case 0x101: dgemm_(&N, &T, &Nz0, &Nz1, &Nx1, &a, x,
                                       &sx_1, y, &sy_0, &b, z, &sz_1); break;
                    case 0x011: dgemm_(&T, &N, &Nz0, &Nz1, &Nx1, &a, x,
                                       &sx_0, y, &sy_1, &b, z, &sz_1); break;
                    case 0x111: dgemm_(&N, &N, &Nz0, &Nz1, &Nx1, &a, x,
                                       &sx_1, y, &sy_1, &b, z, &sz_1); break;
                    default: PyErr_SetString(PyExc_ValueError,
                                             "some matrix has no unit stride");
                             %(fail)s;
                };
        """

    end_switch_typenum = """
            }
            break;
        }
        """

    def build_gemm_call(self):
        if hasattr(self, "inplace"):
            setup_z_Nz_Sz = f"if(%(params)s->inplace){{{self.setup_z_Nz_Sz_inplace}}}else{{{self.setup_z_Nz_Sz_outplace}}}"
        else:
            setup_z_Nz_Sz = self.setup_z_Nz_Sz

        return "".join(
            (
                self.declare_NS,
                self.check_xyz_rank2,
                setup_z_Nz_Sz,
                self.check_xyz_double_or_float,
                self.check_ab_double_or_float,
                self.broadcast_xy,
                self.check_dims,
                self.check_strides,
                self.encode_strides_in_unit,
                self.compute_strides,
                self.begin_switch_typenum,
                self.case_float,
                self.case_float_ab_constants,
                self.case_float_gemm,
                self.case_double,
                self.case_double_ab_constants,
                self.case_double_gemm,
                self.end_switch_typenum,
            )
        )

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

    def __getstate__(self):
        rval = self.__dict__.copy()
        # Do not serialize the setup code, it will be restored in __setstate__
        # depending on the value of 'inplace'
        rval.pop("setup_z_Nz_Sz", None)
        return rval

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

    def infer_shape(self, fgraph, node, input_shapes):
        z_shape, _, x_shape, y_shape, _ = input_shapes
        return [
            (
                pytensor.scalar.maximum(z_shape[0], x_shape[0]),
                pytensor.scalar.maximum(z_shape[1], y_shape[1]),
            )
        ]

    setup_z_Nz_Sz_inplace = """
        // Needs broadcasting
        if (PyArray_DIMS(%(_z)s)[0] < Nx[0] || PyArray_DIMS(%(_z)s)[1] < Ny[1]){

            npy_intp dims[2];
            dims[0] = (PyArray_DIMS(%(_z)s)[0] >= Nx[0]) ? PyArray_DIMS(%(_z)s)[0] : Nx[0];
            dims[1] = (PyArray_DIMS(%(_z)s)[1] >= Ny[1]) ? PyArray_DIMS(%(_z)s)[1] : Ny[1];

            // Check if we need to allocate new array
            if((NULL == %(_zout)s)
                || (PyArray_DIMS(%(_zout)s)[0] != dims[0])
                || (PyArray_DIMS(%(_zout)s)[1] != dims[1]))
            {
                // fprintf(stderr, "Gemm Allocating z output array with shape (%%i %%i)\\n", dims[0], dims[1]);
                Py_XDECREF(%(_zout)s);
                %(_zout)s = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(_z)s));
            }

            // fprintf(stderr, "Gemm Broadcasting Z into shape (%%i %%i)\\n", dims[0], dims[1]);
            if(PyArray_CopyInto(%(_zout)s, %(_z)s) == -1)
            {
                %(fail)s;
            }

        } else {
            if (%(_zout)s != %(_z)s)
            {
                Py_XDECREF(%(_zout)s);
                %(_zout)s = %(_z)s;
                Py_INCREF(%(_zout)s);
            }
        }

        Nz = PyArray_DIMS(%(_zout)s);
        Sz = PyArray_STRIDES(%(_zout)s);
        """

    setup_z_Nz_Sz_outplace = """
        npy_intp dims[2];
        dims[0] = (PyArray_DIMS(%(_z)s)[0] >= Nx[0]) ? PyArray_DIMS(%(_z)s)[0] : Nx[0];
        dims[1] = (PyArray_DIMS(%(_z)s)[1] >= Ny[1]) ? PyArray_DIMS(%(_z)s)[1] : Ny[1];

        // Check if we need to allocate new array
        if ((NULL == %(_zout)s)
            || (PyArray_DIMS(%(_zout)s)[0] != dims[0])
            || (PyArray_DIMS(%(_zout)s)[1] != dims[1]))
        {
            Py_XDECREF(%(_zout)s);
            %(_zout)s = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(_z)s));
            // fprintf(stderr, "Gemm Allocating z output array with shape (%%i %%i)\\n", dims[0], dims[1]);
            if(!%(_zout)s) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemm_no_inplace output");
                %(fail)s
            }
        }

        // fprintf(stderr, "Gemm Broadcasting Z into shape (%%i %%i)\\n", dims[0], dims[1]);
        if(PyArray_CopyInto(%(_zout)s, %(_z)s) == -1)
        {
            %(fail)s
        }

        Nz = PyArray_DIMS(%(_zout)s);
        Sz = PyArray_STRIDES(%(_zout)s);
        """

    broadcast_xy = """
        // Broadcast X if needed
        if (Nz[0] > Nx[0])
        {
            npy_intp dims[2];
            dims[0] = Nz[0];
            dims[1] = Nx[1];
            // fprintf(stderr, "Gemm Broadcasting X into shape (%%i %%i)\\n", dims[0], dims[1]);
            PyArrayObject *x_new = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(_x)s));
            if(!x_new) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemm_inplace input");
                %(fail)s
            }

            if(PyArray_CopyInto(x_new, %(_x)s) == -1)
            {
                %(fail)s
            }

            Py_DECREF(%(_x)s);
            %(_x)s = x_new;

            Nx = PyArray_DIMS(%(_x)s);
            Sx = PyArray_STRIDES(%(_x)s);
        }

        // Broadcast Y if needed
        if (Nz[1] > Ny[1])
        {
            npy_intp dims[2];
            dims[0] = Ny[0];
            dims[1] = Nz[1];
            // fprintf(stderr, "Gemm Broadcasting Y into shape (%%i %%i)\\n", dims[0], dims[1]);
            PyArrayObject *y_new = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_TYPE(%(_x)s));
            if(!y_new) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc gemm_inplace input");
                %(fail)s
            }

            if(PyArray_CopyInto(y_new, %(_y)s) == -1)
            {
                %(fail)s
            }

            Py_DECREF(%(_y)s);
            %(_y)s = y_new;

            Ny = PyArray_DIMS(%(_y)s);
            Sy = PyArray_STRIDES(%(_y)s);
        }

    """

    case_float_ab_constants = """
        #define REAL float
        float a = (PyArray_DESCR(%(_a)s)->type_num == NPY_FLOAT)
        ? (REAL)(((float*)PyArray_DATA(%(_a)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_a)s))[0]);
        float b = (PyArray_DESCR(%(_b)s)->type_num == NPY_FLOAT) ?
        (REAL)(((float*)PyArray_DATA(%(_b)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_b)s))[0]);
        #undef REAL
        """
    case_double_ab_constants = """
        #define REAL double
        double a = (PyArray_DESCR(%(_a)s)->type_num == NPY_FLOAT)
        ? (REAL)(((float*)PyArray_DATA(%(_a)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_a)s))[0]);
        double b = (PyArray_DESCR(%(_b)s)->type_num == NPY_FLOAT) ?
        (REAL)(((float*)PyArray_DATA(%(_b)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_b)s))[0]);
        #undef REAL
        """

    def c_code(self, node, name, inp, out, sub):
        _z, _a, _x, _y, _b = inp
        (_zout,) = out
        if node.inputs[0].type.dtype.startswith("complex"):
            raise MethodNotDefined(f"{self.__class__.__name__}.c_code")
        full_code = self.build_gemm_call() % dict(locals(), **sub)
        return full_code

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

    def infer_shape(self, fgraph, node, input_shapes):
        return [[input_shapes[0][0], input_shapes[1][1]]]

    setup_z_Nz_Sz = """
        if ((NULL == %(_zout)s)
            || (PyArray_DIMS(%(_zout)s)[0] != PyArray_DIMS(%(_x)s)[0])
            || (PyArray_DIMS(%(_zout)s)[1] != PyArray_DIMS(%(_y)s)[1]))
        {
            if (NULL != %(_zout)s) Py_XDECREF(%(_zout)s);
            npy_intp dims[2];
            dims[0] = PyArray_DIMS(%(_x)s)[0];
            dims[1] = PyArray_DIMS(%(_y)s)[1];
            %(_zout)s = (PyArrayObject*)PyArray_SimpleNew(2, dims,
                            PyArray_TYPE(%(_x)s));
            //fprintf(stderr, "Dot Allocating %%i %%i\\n", dims[0], dims[1]);
            if(!%(_zout)s) {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc dot22 output");
                %(fail)s
            }
        }
        Nz = PyArray_DIMS(%(_zout)s);
        Sz = PyArray_STRIDES(%(_zout)s);

        """
    broadcast_xy = ""
    check_ab_double_or_float = ""
    case_float_ab_constants = """
                float a = 1.0;
                float b = 0.0;
        """
    case_double_ab_constants = """
                double a = 1.0;
                double b = 0.0;
        """

    def c_code(self, node, name, inp, out, sub):  # DEBUG
        _x, _y = inp
        (_zout,) = out
        if node.inputs[0].type.dtype.startswith("complex"):
            raise MethodNotDefined(f"{self.__class__.__name__}.c_code")
        if len(self.c_libraries()) <= 0:
            raise NotImplementedError()
        full_code = self.build_gemm_call() % dict(locals(), **sub)
        return full_code

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

    def infer_shape(self, fgraph, node, input_shapes):
        return [[input_shapes[0][0], input_shapes[1][1]]]

    setup_z_Nz_Sz = Dot22.setup_z_Nz_Sz
    broadcast_xy = ""

    check_ab_double_or_float = """
        if ((PyArray_DESCR(%(_a)s)->type_num != NPY_DOUBLE)
            && (PyArray_DESCR(%(_a)s)->type_num != NPY_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError,
                         "type(a) is not double or float"); %(fail)s;}

        """
    case_float_ab_constants = """
        #define REAL float
        float a = (PyArray_DESCR(%(_a)s)->type_num == NPY_FLOAT)
        ? (REAL)(((float*)PyArray_DATA(%(_a)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_a)s))[0]);
        #undef REAL
        float b = 0.0;
        """

    case_double_ab_constants = """
        #define REAL double
        double a = (PyArray_DESCR(%(_a)s)->type_num == NPY_FLOAT)
        ? (REAL)(((float*)PyArray_DATA(%(_a)s))[0])
        : (REAL)(((double*)PyArray_DATA(%(_a)s))[0]);
        #undef REAL
        double b = 0.0;
        """

    def c_code(self, node, name, inp, out, sub):
        _x, _y, _a = inp
        (_zout,) = out
        if node.inputs[0].type.dtype.startswith("complex"):
            raise MethodNotDefined(f"{self.__class__.__name__}.c_code")
        if len(self.c_libraries()) <= 0:
            raise NotImplementedError()
        full_code = self.build_gemm_call() % dict(locals(), **sub)
        return full_code

    def c_code_cache_version(self):
        gv = self.build_gemm_version()
        if gv:
            return (2, *gv)
        else:
            return gv


_dot22scalar = Dot22Scalar()
