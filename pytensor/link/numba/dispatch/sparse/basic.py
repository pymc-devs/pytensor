import numpy as np
import scipy as sp
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

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    register_funcify_default_op_cache_key,
)
from pytensor.sparse import (
    CSM,
    CSMProperties,
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
        self.shape = types.UniTuple(types.int64, 2)
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


@box(CSMatrixType)
def box_cs_matrix(typ, val, c):
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
    sig = CSRMatrixType(data, indices, indptr)(data, indices, indptr, shape)
    return sig, _intrinsic_cs_codegen


@intrinsic
def csc_matrix_from_components(typingctx, data, indices, indptr, shape):
    sig = CSCMatrixType(data, indices, indptr)(data, indices, indptr, shape)
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
def overload_sparse_shape(matrix):
    if isinstance(matrix, CSMatrixType):
        return lambda matrix: matrix.shape


@overload_attribute(CSMatrixType, "ndim")
def overload_sparse_ndim(matrix):
    return lambda matrix: 2


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


@register_funcify_default_op_cache_key(CSMProperties)
def numba_funcify_CSMProperties(op, **kwargs):
    @numba_basic.numba_njit
    def csm_properties(x):
        # Reconsider this int32/int64. Scipy/base PyTensor use int32 for indices/indptr.
        # But this seems to be legacy mistake and devs would choose int64 nowadays, and may move there.
        return x.data, x.indices, x.indptr, np.asarray(x.shape, dtype="int64")

    return csm_properties


@register_funcify_default_op_cache_key(CSM)
def numba_funcify_CSM(op, **kwargs):
    format = op.format

    @numba_basic.numba_njit
    def csm_constructor(data, indices, indptr, shape):
        constructor_arg = (data, indices, indptr)
        shape_arg = (shape[0], shape[1])
        if format == "csr":
            return sp.sparse.csr_matrix(constructor_arg, shape=shape_arg)
        else:
            return sp.sparse.csc_matrix(constructor_arg, shape=shape_arg)

    return csm_constructor
