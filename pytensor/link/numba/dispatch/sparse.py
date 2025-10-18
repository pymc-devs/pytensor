import numpy as np
import scipy as sp
import scipy.sparse
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)

from pytensor.link.numba.dispatch.basic import numba_funcify, numba_njit
from pytensor.sparse import (
    CSM,
    CSMProperties,
    SparseDenseMultiply,
    SparseDenseVectorMultiply,
)


class CSMatrixType(types.Type):
    """A Numba `Type` modeled after the base class `scipy.sparse.compressed._cs_matrix`."""

    name: str

    @staticmethod
    def instance_class(data, indices, indptr, shape):
        raise NotImplementedError()

    def __init__(self):
        # TODO: Accept dtype again
        #  Actually accept data type, so that in can have a layout other than "A"
        self.dtype = types.float64
        # TODO: Most times data/indices/indptr are C-contiguous, allow setting those
        self.data = types.Array(self.dtype, 1, "A")
        self.indices = types.Array(types.int32, 1, "A")
        self.indptr = types.Array(types.int32, 1, "A")
        self.shape = types.UniTuple(types.int64, 2)
        super().__init__(self.name)

    @property
    def key(self):
        return (self.name, self.dtype)


make_attribute_wrapper(CSMatrixType, "data", "data")
make_attribute_wrapper(CSMatrixType, "indices", "indices")
make_attribute_wrapper(CSMatrixType, "indptr", "indptr")
make_attribute_wrapper(CSMatrixType, "shape", "shape")


class CSRMatrixType(CSMatrixType):
    name = "csr_matrix"

    @staticmethod
    def instance_class(data, indices, indptr, shape):
        return sp.sparse.csr_matrix((data, indices, indptr), shape, copy=False)


class CSCMatrixType(CSMatrixType):
    name = "csc_matrix"

    @staticmethod
    def instance_class(data, indices, indptr, shape):
        return sp.sparse.csc_matrix((data, indices, indptr), shape, copy=False)


@typeof_impl.register(sp.sparse.csc_matrix)
def typeof_csc_matrix(val, c):
    # data = typeof_impl(val.data, c)
    return CSCMatrixType()


@typeof_impl.register(sp.sparse.csr_matrix)
def typeof_csr_matrix(val, c):
    # data = typeof_impl(val.data, c)
    return CSRMatrixType()


@register_model(CSRMatrixType)
class CSRMatrixModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", fe_type.data),
            ("indices", fe_type.indices),
            ("indptr", fe_type.indptr),
            ("shape", fe_type.shape),
        ]
        super().__init__(dmm, fe_type, members)


@register_model(CSCMatrixType)
class CSCMatrixModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", fe_type.data),
            ("indices", fe_type.indices),
            ("indptr", fe_type.indptr),
            ("shape", fe_type.shape),
        ]
        super().__init__(dmm, fe_type, members)


@unbox(CSCMatrixType)
@unbox(CSRMatrixType)
def unbox_matrix(typ, obj, c):
    struct_ptr = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    data = c.pyapi.object_getattr_string(obj, "data")
    indices = c.pyapi.object_getattr_string(obj, "indices")
    indptr = c.pyapi.object_getattr_string(obj, "indptr")
    shape = c.pyapi.object_getattr_string(obj, "shape")

    struct_ptr.data = c.unbox(typ.data, data).value
    struct_ptr.indices = c.unbox(typ.indices, indices).value
    struct_ptr.indptr = c.unbox(typ.indptr, indptr).value
    struct_ptr.shape = c.unbox(typ.shape, shape).value

    c.pyapi.decref(data)
    c.pyapi.decref(indices)
    c.pyapi.decref(indptr)
    c.pyapi.decref(shape)

    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    is_error = c.builder.load(is_error_ptr)

    res = NativeValue(struct_ptr._getvalue(), is_error=is_error)

    return res


@box(CSCMatrixType)
@box(CSRMatrixType)
def box_matrix(typ, val, c):
    struct_ptr = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)

    data_obj = c.box(typ.data, struct_ptr.data)
    indices_obj = c.box(typ.indices, struct_ptr.indices)
    indptr_obj = c.box(typ.indptr, struct_ptr.indptr)
    shape_obj = c.box(typ.shape, struct_ptr.shape)

    # Why incref here, just to decref later?
    c.pyapi.incref(data_obj)
    c.pyapi.incref(indices_obj)
    c.pyapi.incref(indptr_obj)
    c.pyapi.incref(shape_obj)

    cls_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.instance_class))
    obj = c.pyapi.call_function_objargs(
        cls_obj, (data_obj, indices_obj, indptr_obj, shape_obj)
    )

    c.pyapi.decref(data_obj)
    c.pyapi.decref(indices_obj)
    c.pyapi.decref(indptr_obj)
    c.pyapi.decref(shape_obj)

    return obj


def _intrinsic_cs_codegen(context, builder, sig, args):
    matrix_type = sig.return_type
    struct = cgutils.create_struct_proxy(matrix_type)(context, builder)
    data, indices, indptr, shape = args
    struct.data = data
    struct.indices = indices
    struct.indptr = indptr
    struct.shape = shape
    # TODO: Check why do we use use impl_ret_borrowed, whereas numba numpy array uses impl_ret_new_ref
    #  Is it because we create a struct_proxy. What is that even?
    return impl_ret_borrowed(
        context,
        builder,
        matrix_type,
        struct._getvalue(),
    )


@intrinsic
def csr_matrix_from_components(typingctx, data, indices, indptr, shape):
    # TODO: put dtype back in
    sig = CSRMatrixType()(data, indices, indptr, shape)
    return sig, _intrinsic_cs_codegen


@intrinsic
def csc_matrix_from_components(typingctx, data, indices, indptr, shape):
    sig = CSCMatrixType()(data, indices, indptr, shape)
    return sig, _intrinsic_cs_codegen


@overload(sp.sparse.csr_matrix)
def overload_csr_matrix(arg1, shape, dtype=None):
    if not isinstance(arg1, types.Tuple) or len(arg1) != 3:
        return None
    if isinstance(shape, types.NoneType):
        return None

    def impl(arg1, shape, dtype=None):
        data, indices, indptr = arg1
        return csr_matrix_from_components(data, indices, indptr, shape)

    return impl


@overload(sp.sparse.csc_matrix)
def overload_csc_matrix(arg1, shape, dtype=None):
    if not isinstance(arg1, types.Tuple) or len(arg1) != 3:
        return None
    if isinstance(shape, types.NoneType):
        return None

    def impl(arg1, shape, dtype=None):
        data, indices, indptr = arg1
        return csc_matrix_from_components(data, indices, indptr, shape)

    return impl


@overload(np.shape)
def overload_sparse_shape(x):
    if isinstance(x, CSMatrixType):
        return lambda x: x.shape


@overload_attribute(CSMatrixType, "ndim")
def overload_sparse_ndim(matrix):
    if not isinstance(matrix, CSMatrixType):
        return

    def ndim(matrix):
        return 2

    return ndim


@overload_method(CSMatrixType, "copy")
def overload_sparse_copy(matrix):
    match matrix:
        case CSRMatrixType():
            builder = csr_matrix_from_components
        case CSCMatrixType():
            builder = csc_matrix_from_components
        case _:
            return

    def copy(matrix):
        return builder(
            matrix.data.copy(),
            matrix.indices.copy(),
            matrix.indptr.copy(),
            matrix.shape,
        )

    return copy


@overload_method(CSMatrixType, "astype")
def overload_sparse_astype(matrix, dtype):
    match matrix:
        case CSRMatrixType():
            builder = csr_matrix_from_components
        case CSCMatrixType():
            builder = csc_matrix_from_components
        case _:
            return

    def astype(matrix, dtype):
        return builder(
            matrix.data.astype(dtype),
            matrix.indices.copy(),
            matrix.indptr.copy(),
            matrix.shape,
        )

    return astype


@numba_funcify.register(CSMProperties)
def numba_funcify_CSMProperties(op, **kwargs):
    @numba_njit
    def csm_properties(x):
        # Reconsider this int32/int64. Scipy/base PyTensor use int32 for indices/indptr.
        # But this seems to be legacy mistake and devs would choose int64 nowadays, and may move there.
        return x.data, x.indices, x.indptr, np.asarray(x.shape, dtype="int64")

    return csm_properties


@numba_funcify.register(CSM)
def numba_funcify_CSM(op, **kwargs):
    format = op.format

    @numba_njit
    def csm_constructor(data, indices, indptr, shape):
        constructor_arg = (data, indices, indptr)
        shape_arg = (shape[0], shape[1])
        if format == "csr":
            return sp.sparse.csr_matrix(constructor_arg, shape=shape_arg)
        else:
            return sp.sparse.csc_matrix(constructor_arg, shape=shape_arg)

    return csm_constructor


@numba_funcify.register(SparseDenseMultiply)
@numba_funcify.register(SparseDenseVectorMultiply)
def numba_funcify_SparseDenseMultiply(op, node, **kwargs):
    x, y = node.inputs
    [z] = node.outputs
    out_dtype = z.type.dtype
    format = z.type.format
    same_dtype = x.type.dtype == out_dtype

    if y.ndim == 0:

        @numba_njit
        def sparse_multiply_scalar(x, y):
            if same_dtype:
                z = x.copy()
            else:
                z = x.astype(out_dtype)
            # Numba doesn't know how to handle in-place mutation / assignment of fields
            # z.data *= y
            z_data = z.data
            z_data *= y
            return z

        return sparse_multiply_scalar

    elif y.ndim == 1:

        @numba_njit
        def sparse_dense_multiply(x, y):
            assert x.shape[1] == y.shape[0]
            if same_dtype:
                z = x.copy()
            else:
                z = x.astype(out_dtype)

            M, N = x.shape
            indices = x.indices
            indptr = x.indptr
            z_data = z.data
            if format == "csc":
                for j in range(0, N):
                    for i_idx in range(indptr[j], indptr[j + 1]):
                        z_data[i_idx] *= y[j]
                return z

            else:
                for i in range(0, M):
                    for j_idx in range(indptr[i], indptr[i + 1]):
                        j = indices[j_idx]
                        z_data[j_idx] *= y[j]

            return z

        return sparse_dense_multiply

    else:  # y.ndim == 2

        @numba_njit
        def sparse_dense_multiply(x, y):
            assert x.shape == y.shape
            if same_dtype:
                z = x.copy()
            else:
                z = x.astype(out_dtype)

            M, N = x.shape
            indices = x.indices
            indptr = x.indptr
            z_data = z.data
            if format == "csc":
                for j in range(0, N):
                    for i_idx in range(indptr[j], indptr[j + 1]):
                        i = indices[i_idx]
                        z_data[i_idx] *= y[i, j]
                return z

            else:
                for i in range(0, M):
                    for j_idx in range(indptr[i], indptr[i + 1]):
                        j = indices[j_idx]
                        z_data[j_idx] *= y[i, j]

            return z

        return sparse_dense_multiply
