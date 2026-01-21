import numpy as np
import scipy as sp
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed, lower_constant
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


class CSMatrixType(types.Type):
    """A Numba `Type` modeled after the base class `scipy.sparse.compressed._cs_matrix`."""

    name: str

    @staticmethod
    def instance_class(data, indices, indptr, shape):
        raise NotImplementedError()

    def __init__(self, data_type, indices_type, indptr_type):
        self._key = (data_type, indices_type, indptr_type)
        self.data = data_type
        self.indices = indices_type
        self.indptr = indptr_type
        self.shape = types.UniTuple(types.int32, 2)
        super().__init__(self.name)

    @property
    def key(self):
        return self._key


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
@typeof_impl.register(sp.sparse.csr_matrix)
def typeof_cs_matrix(val, ctx):
    match val:
        case sp.sparse.csc_matrix():
            numba_type = CSCMatrixType
        case sp.sparse.csr_matrix():
            numba_type = CSRMatrixType
        case _:
            raise ValueError(f"val of type {type(val)} not recognized")
    return numba_type(
        typeof_impl(val.data, ctx),
        typeof_impl(val.indices, ctx),
        typeof_impl(val.indptr, ctx),
    )


@register_model(CSCMatrixType)
@register_model(CSRMatrixType)
class CSMatrixModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", fe_type.data),
            ("indices", fe_type.indices),
            ("indptr", fe_type.indptr),
            ("shape", fe_type.shape),
        ]
        super().__init__(dmm, fe_type, members)


@unbox(CSMatrixType)
def unbox_cs_matrix(typ, obj, c):
    struct_ptr = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    # Get attributes from python object
    data = c.pyapi.object_getattr_string(obj, "data")
    indices = c.pyapi.object_getattr_string(obj, "indices")
    indptr = c.pyapi.object_getattr_string(obj, "indptr")
    shape = c.pyapi.object_getattr_string(obj, "shape")

    # Unbox them into llvm struct
    struct_ptr.data = c.unbox(typ.data, data).value
    struct_ptr.indices = c.unbox(typ.indices, indices).value
    struct_ptr.indptr = c.unbox(typ.indptr, indptr).value
    struct_ptr.shape = c.unbox(typ.shape, shape).value

    # Decref created attributes
    c.pyapi.decref(data)
    c.pyapi.decref(indices)
    c.pyapi.decref(indptr)
    c.pyapi.decref(shape)

    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    is_error = c.builder.load(is_error_ptr)
    res = NativeValue(struct_ptr._getvalue(), is_error=is_error)

    return res


@box(CSMatrixType)
def box_cs_matrix(typ, val, c):
    struct_ptr = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)

    data_obj = c.box(typ.data, struct_ptr.data)
    indices_obj = c.box(typ.indices, struct_ptr.indices)
    indptr_obj = c.box(typ.indptr, struct_ptr.indptr)
    shape_obj = c.box(typ.shape, struct_ptr.shape)

    # Call scipy.sparse.cs[c|r]_matrix
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
    return impl_ret_borrowed(
        context,
        builder,
        matrix_type,
        struct._getvalue(),
    )


@intrinsic
def csr_matrix_from_components(typingctx, data, indices, indptr, shape):
    sig = CSRMatrixType(data, indices, indptr)(data, indices, indptr, shape)
    return sig, _intrinsic_cs_codegen


@intrinsic
def csc_matrix_from_components(typingctx, data, indices, indptr, shape):
    sig = CSCMatrixType(data, indices, indptr)(data, indices, indptr, shape)
    return sig, _intrinsic_cs_codegen


@lower_constant(CSRMatrixType)
@lower_constant(CSCMatrixType)
def cs_matrix_constant(context, builder, ty, pyval):
    data_const = context.make_constant_array(builder, ty.data, pyval.data)
    indices_const = context.make_constant_array(builder, ty.indices, pyval.indices)
    indptr_const = context.make_constant_array(builder, ty.indptr, pyval.indptr)
    shape = context.get_constant_generic(builder, ty.shape, pyval.shape)
    args = (data_const, indices_const, indptr_const, shape)

    sig = ty(*args)
    return _intrinsic_cs_codegen(context, builder, sig, args)


@overload(sp.sparse.csr_matrix)
def overload_csr_matrix(arg1, shape, dtype=None):
    if not isinstance(arg1, types.BaseAnonymousTuple) or len(arg1) != 3:
        return None
    if isinstance(shape, types.NoneType):
        return None

    def impl(arg1, shape, dtype=None):
        data, indices, indptr = arg1
        int32_shape = (types.int32(shape[0]), types.int32(shape[1]))
        return csr_matrix_from_components(data, indices, indptr, int32_shape)

    return impl


@overload(sp.sparse.csc_matrix)
def overload_csc_matrix(arg1, shape, dtype=None):
    if not isinstance(arg1, types.BaseAnonymousTuple) or len(arg1) != 3:
        return None
    if isinstance(shape, types.NoneType):
        return None

    def impl(arg1, shape, dtype=None):
        data, indices, indptr = arg1
        int32_shape = (types.int32(shape[0]), types.int32(shape[1]))
        return csc_matrix_from_components(data, indices, indptr, int32_shape)

    return impl


@overload(np.shape)
def overload_sparse_shape(matrix):
    if isinstance(matrix, CSMatrixType):
        return lambda matrix: matrix.shape


@overload_attribute(CSMatrixType, "ndim")
def overload_sparse_ndim(matrix):
    return lambda matrix: 2


@overload_attribute(CSMatrixType, "T")
def overload_sparse_T(matrix):
    match matrix:
        case CSRMatrixType():
            builder = csc_matrix_from_components
        case CSCMatrixType():
            builder = csr_matrix_from_components
        case _:
            return

    def transpose(matrix):
        n_row, n_col = matrix.shape
        return builder(
            matrix.data.copy(),
            matrix.indices.copy(),
            matrix.indptr.copy(),
            (n_col, n_row),
        )

    return transpose


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


@overload_method(CSCMatrixType, "tocsr")
def overload_tocsr(matrix):
    def to_csr(matrix):
        n_row, n_col = matrix.shape
        csc_ptr = matrix.indptr.view(np.uint32)
        csc_ind = matrix.indices.view(np.uint32)
        csc_data = matrix.data
        nnz = csc_ptr[n_col]

        csr_ptr = np.empty(n_row + 1, dtype=np.uint32)
        csr_ind = np.empty(nnz, dtype=np.uint32)
        csr_data = np.empty(nnz, dtype=matrix.data.dtype)

        csr_ptr[:n_row] = 0

        for n in range(nnz):
            csr_ptr[csc_ind[n]] += 1

        cumsum = 0
        for row in range(n_row):
            temp = csr_ptr[row]
            csr_ptr[row] = cumsum
            cumsum += temp
        csr_ptr[n_row] = nnz

        for col_idx in range(n_col):
            for jj in range(csc_ptr[col_idx], csc_ptr[col_idx + 1]):
                row_idx = csc_ind[jj]
                dest = csr_ptr[row_idx]

                csr_ind[dest] = col_idx
                csr_data[dest] = csc_data[jj]

                csr_ptr[row_idx] += 1

        last = 0
        for row_idx in range(n_row + 1):
            temp = csr_ptr[row_idx]
            csr_ptr[row_idx] = last
            last = temp

        return csr_matrix_from_components(
            csr_data, csr_ind.view(np.int32), csr_ptr.view(np.int32), matrix.shape
        )

    return to_csr


@overload_method(CSRMatrixType, "tocsc")
def overload_tocsc(matrix):
    def to_csc(matrix):
        n_row, n_col = matrix.shape
        csr_ptr = matrix.indptr.view(np.uint32)
        csr_ind = matrix.indices.view(np.uint32)
        csr_data = matrix.data
        nnz = csr_ptr[n_row]

        csc_ptr = np.empty(n_col + 1, dtype=np.uint32)
        csc_ind = np.empty(nnz, dtype=np.uint32)
        csc_data = np.empty(nnz, dtype=matrix.data.dtype)

        csc_ptr[:n_col] = 0

        for n in range(nnz):
            csc_ptr[csr_ind[n]] += 1

        cumsum = 0
        for col in range(n_col):
            temp = csc_ptr[col]
            csc_ptr[col] = cumsum
            cumsum += temp
        csc_ptr[n_col] = nnz

        for row in range(n_row):
            for jj in range(csr_ptr[row], csr_ptr[row + 1]):
                col = csr_ind[jj]
                dest = csc_ptr[col]

                csc_ind[dest] = row
                csc_data[dest] = csr_data[jj]

                csc_ptr[col] += 1

        last = 0
        for col in range(n_col + 1):
            temp = csc_ptr[col]
            csc_ptr[col] = last
            last = temp

        return csc_matrix_from_components(
            csc_data, csc_ind.view(np.int32), csc_ptr.view(np.int32), matrix.shape
        )

    return to_csc


@overload_method(CSMatrixType, "toarray")
def overload_toarray(matrix):
    match matrix:
        case CSRMatrixType():

            def to_array(matrix):
                indptr = matrix.indptr.view(np.uint32)
                indices = matrix.indices.view(np.uint32)
                n_row = matrix.shape[0]
                dense = np.zeros(matrix.shape, dtype=matrix.data.dtype)
                for row_idx in range(n_row):
                    for k in range(indptr[row_idx], indptr[row_idx + 1]):
                        col_idx = indices[k]
                        dense[row_idx, col_idx] = matrix.data[k]
                return dense

            return to_array
        case CSCMatrixType():

            def to_array(matrix):
                indptr = matrix.indptr.view(np.uint32)
                indices = matrix.indices.view(np.uint32)
                n_col = matrix.shape[1]
                dense = np.zeros(matrix.shape, dtype=matrix.data.dtype)
                for col_idx in range(n_col):
                    for k in range(indptr[col_idx], indptr[col_idx + 1]):
                        row_idx = indices[k]
                        dense[row_idx, col_idx] = matrix.data[k]
                return dense

            return to_array
        case _:
            return
