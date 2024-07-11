import scipy

import pytensor
import pytensor.scalar as ps
from pytensor.configdefaults import config
from pytensor.graph.basic import Apply
from pytensor.graph.rewriting.basic import (
    PatternNodeRewriter,
    WalkingGraphRewriter,
    node_rewriter,
)
from pytensor.link.c.op import COp, _NoPythonCOp
from pytensor.misc.safe_asarray import _asarray
from pytensor.sparse import basic as sparse
from pytensor.sparse.basic import (
    CSC,
    CSR,
    csm_data,
    csm_grad,
    csm_indices,
    csm_indptr,
    csm_properties,
    usmm,
)
from pytensor.tensor import blas
from pytensor.tensor.basic import as_tensor_variable, cast
from pytensor.tensor.math import mul, neg, sub
from pytensor.tensor.rewriting.basic import register_canonicalize, register_specialize
from pytensor.tensor.shape import shape, specify_shape
from pytensor.tensor.type import TensorType, tensor


_is_sparse_variable = sparse._is_sparse_variable
_is_dense = sparse._is_dense


@node_rewriter([csm_properties])
def local_csm_properties_csm(fgraph, node):
    """
    If we find csm_properties(CSM(*args)), then we can replace that with the
    *args directly.

    """
    if node.op == csm_properties:
        (csm,) = node.inputs
        if csm.owner and (csm.owner.op == CSC or csm.owner.op == CSR):
            return csm.owner.inputs

    return False


register_specialize(local_csm_properties_csm)


# This is tested in tests/test_basic.py:test_remove0
@node_rewriter([sparse.Remove0])
def local_inplace_remove0(fgraph, node):
    """Rewrite to insert inplace versions of `Remove0`."""
    # If inplace is not enabled, enable it and replace that op with a
    # new op which has inplace enabled
    if isinstance(node.op, sparse.Remove0) and not node.op.inplace:
        new_op = node.op.__class__(inplace=True)
        new_node = new_op(*node.inputs)
        return [new_node]
    return False


pytensor.compile.optdb.register(
    "local_inplace_remove0",
    WalkingGraphRewriter(
        local_inplace_remove0, failure_callback=WalkingGraphRewriter.warn_inplace
    ),
    "fast_run",
    "inplace",
    position=60,
)


class AddSD_ccode(_NoPythonCOp):
    """
    Add a sparse and a dense matrix.

    Parameters
    ----------
    x
        A sparse matrix.
    y
        A dense matrix

    Returns
    -------
    matrix
        `x`+`y`

    Notes
    -----
    The grad implemented is structured on `x`.

    """

    __props__ = ("format", "inplace")

    def __init__(self, format, inplace=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Should we do inplace addition or not ?
        self.inplace = inplace
        self.format = format
        if self.inplace:
            self.destroy_map = {0: [3]}

    def __str__(self):
        inp = ""
        if self.inplace:
            inp = ",inplace"
        return f"{self.__class__.__name__}{{{self.format}{inp}}}"

    def make_node(self, x, y):
        x, y = sparse.as_sparse_variable(x), as_tensor_variable(y)
        out_dtype = ps.upcast(x.type.dtype, y.type.dtype)
        if self.inplace:
            assert out_dtype == y.dtype

        indices, indptr, data = csm_indices(x), csm_indptr(x), csm_data(x)
        # We either use CSC or CSR depending on the format of input
        assert self.format == x.type.format
        # The magic number two here arises because L{scipy.sparse}
        # objects must be matrices (have dimension 2)
        assert y.type.ndim == 2
        out = TensorType(
            dtype=out_dtype, shape=tuple(1 if s == 1 else None for s in y.type.shape)
        )()
        return Apply(self, [data, indices, indptr, y], [out])

    def c_code(self, node, name, inputs, outputs, sub):
        (_data, _indices, _indptr, y) = inputs
        (z,) = outputs
        inplace = int(self.inplace)
        format = {"csc": 0, "csr": 1}[self.format]
        out_typenum = node.outputs[0].type.dtype_specs()[2]
        code = f"""
                Py_XDECREF({z});
                if (!{inplace}){{
                    if(PyArray_TYPE({y}) != {out_typenum}){{
                        {z} = (PyArrayObject *) PyArray_FromArray({y},  PyArray_DescrFromType({out_typenum}), 0);
                    }}else{{
                        {z} = (PyArrayObject *) PyArray_NewCopy({y}, NPY_CORDER);
                    }}
                }}else{{
                  {z} = {y};
                  Py_XINCREF({z});
                }}

                npy_intp N =  PyArray_DIMS({_indptr})[0]-1;

                const dtype_{_indptr}* __restrict__ indptr = (dtype_{_indptr}*)PyArray_DATA({_indptr});
                const dtype_{_indices}* __restrict__ indices = (dtype_{_indices}*)PyArray_DATA({_indices});
                const dtype_{_data}* __restrict__ data = (dtype_{_data}*)PyArray_DATA({_data});

                dtype_{y}* ydata = (dtype_{y}*)PyArray_DATA({y});
                dtype_{z}* zdata = (dtype_{z}*)PyArray_DATA({z});
                npy_intp Yi = PyArray_STRIDES({y})[0]/PyArray_DESCR({y})->elsize;
                npy_intp Yj = PyArray_STRIDES({y})[1]/PyArray_DESCR({y})->elsize;

                npy_intp pos;
                if ({format} == 0){{
                for (npy_intp col = 0; col < N; ++col){{
                  for (dtype_{_indptr} ind = indptr[col]; ind < indptr[col+1]; ++ind){{
                    npy_intp row = indices[ind];
                    pos = row * Yi + col * Yj;
                    zdata[pos] = ydata[pos] + data[ind];
                  }}
                }}
                }}else{{
                for (npy_intp row = 0; row < N; ++row){{
                  for (dtype_{_indptr} ind = indptr[row]; ind < indptr[row+1]; ++ind){{
                    npy_intp col = indices[ind];
                    pos = row * Yi + col * Yj;
                    zdata[pos] = ydata[pos] + data[ind];
                  }}
                 }}
                }}
             """
        return code

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[3]]

    def c_code_cache_version(self):
        return (2,)


@node_rewriter([sparse.AddSD])
def local_inplace_addsd_ccode(fgraph, node):
    """Rewrite to insert inplace versions of `AddSD`."""
    if isinstance(node.op, sparse.AddSD) and config.cxx:
        out_dtype = ps.upcast(*node.inputs)
        if out_dtype != node.inputs[1].dtype:
            return
        new_node = AddSD_ccode(format=node.inputs[0].type.format, inplace=True)(
            *node.inputs
        )
        return [new_node]
    return False


pytensor.compile.optdb.register(
    "local_inplace_addsd_ccode",
    WalkingGraphRewriter(
        local_inplace_addsd_ccode, failure_callback=WalkingGraphRewriter.warn_inplace
    ),
    "fast_run",
    "inplace",
    position=60,
)


@register_canonicalize("fast_compile")
@register_specialize
@node_rewriter([sparse.DenseFromSparse])
def local_dense_from_sparse_sparse_from_dense(fgraph, node):
    if isinstance(node.op, sparse.DenseFromSparse):
        inp = node.inputs[0]
        if inp.owner and isinstance(inp.owner.op, sparse.SparseFromDense):
            return inp.owner.inputs


@node_rewriter([sparse.AddSD])
def local_addsd_ccode(fgraph, node):
    """
    Convert AddSD to faster AddSD_ccode.

    """
    if isinstance(node.op, sparse.AddSD) and config.cxx:
        new_node = AddSD_ccode(format=node.inputs[0].type.format)(*node.inputs)
        return [new_node]
    return False


pytensor.compile.optdb.register(
    "local_addsd_ccode",
    WalkingGraphRewriter(local_addsd_ccode),
    # Must be after local_inplace_addsd_ccode at 60
    "fast_run",
    position=61,
)


class StructuredDotCSC(COp):
    """
    Structured Dot CSC is like `dot`, except that only the gradient wrt non-zero
    elements of a sparse matrix are calculated and propagated.

    The output is presumed to be a dense matrix, and is represented by a
    `TensorType` instance.

    Notes
    -----
    The gradient. implemented is structured.

    This `Op` is used as a rewritten form of `StructuredDot`.

    """

    __props__ = ()

    def make_node(self, a_val, a_ind, a_ptr, a_nrows, b):
        dtype_out = ps.upcast(a_val.type.dtype, b.type.dtype)
        r = Apply(
            self,
            [a_val, a_ind, a_ptr, a_nrows, b],
            [
                tensor(
                    dtype=dtype_out, shape=(None, 1 if b.type.shape[1] == 1 else None)
                )
            ],
        )
        return r

    def perform(self, node, inputs, outputs):
        (a_val, a_ind, a_ptr, a_nrows, b) = inputs
        (out,) = outputs
        a = scipy.sparse.csc_matrix(
            (a_val, a_ind, a_ptr), (a_nrows, b.shape[0]), copy=False
        )
        # out[0] = a.dot(b)
        out[0] = _asarray(a * b, dtype=node.outputs[0].type.dtype)
        assert _is_dense(out[0])  # scipy 0.7 automatically converts to dense

    def c_code(self, node, name, inputs, outputs, sub):
        # C-implementation of the dot product of the sparse matrix A and matrix
        # B.
        # @param a_val: non-zero values of the sparse matrix
        # @param a_ind: column indices of the non-null values (.indices of a
        # scipy.csc_matrix)
        # @param a_ptr: a_ptr indicates col indices for col. i are in the range
        # a_ptr[i]:a_ptr[i+1]
        # @param n_rows: number of rows of sparse matrix
        # @param b: dense matrix to perform dot product with, as in dot(a, b)
        # @param z: return value
        # @param sub: TODO, not too sure, something to do with weave probably

        (a_val, a_ind, a_ptr, a_nrows, b) = inputs
        (z,) = outputs
        if node.inputs[0].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for a_val")
        if node.inputs[4].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for b")

        typenum_z = node.outputs[0].type.dtype_specs()[2]  # retrieve dtype number
        typenum_a_val = node.inputs[0].type.dtype_specs()[2]  # retrieve dtype number
        typenum_b = node.inputs[4].type.dtype_specs()[2]  # retrieve dtype number

        fail = sub["fail"]
        rval = f"""

        if (PyArray_NDIM({a_val}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(a_val) != 1"); {fail};}}
        if (PyArray_NDIM({a_ind}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(a_ind) != 1"); {fail};}}
        if (PyArray_NDIM({a_ptr}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(a_ptr) != 1"); {fail};}}
        if (PyArray_NDIM({a_nrows}) != 0) {{PyErr_SetString(PyExc_NotImplementedError, "rank(nrows) != 0"); {fail};}}
        if (PyArray_NDIM({b}) != 2) {{PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 2"); {fail};}}

        if (PyArray_TYPE({a_val}) != {typenum_a_val}) {{
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for a_val"); {fail};}}

        if (PyArray_TYPE({b}) != {typenum_b}) {{
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for b"); {fail};}}

        if (PyArray_TYPE({a_ind}) != NPY_INT32) {{
        PyErr_SetString(PyExc_NotImplementedError, "a_ind dtype not INT32"); {fail};}}

        if (PyArray_TYPE({a_ptr}) != NPY_INT32)
        {{PyErr_SetString(PyExc_NotImplementedError, "a_ptr dtype not INT32"); {fail};}}

        if (PyArray_TYPE({a_nrows}) != NPY_INT32)
        {{PyErr_SetString(PyExc_NotImplementedError, "a_nrows dtype not INT32"); {fail};}}

        if (PyArray_DIMS({a_val})[0] != PyArray_DIMS({a_ind})[0])
        {{PyErr_SetString(PyExc_NotImplementedError, "a_val and a_ind have different lengths"); {fail};}}

        if (PyArray_DIMS({a_ptr})[0] != PyArray_DIMS({b})[0]+1)
        {{PyErr_SetString(PyExc_NotImplementedError, "a's number of columns doesn't match b's rows"); {fail};}}

        if ((!{z})
            || (PyArray_DIMS({z})[0] != ((npy_int32 *)PyArray_DATA({a_nrows}))[0])
            || (PyArray_DIMS({z})[1] != PyArray_DIMS({b})[1])
            )
        {{
            {{Py_XDECREF({z});}}
            npy_intp dims[] = {{0, 0}};
            dims[0] = ((npy_int32 *)PyArray_DATA({a_nrows}))[0];
            dims[1] = PyArray_DIMS({b})[1];
            {z} = (PyArrayObject*) PyArray_SimpleNew(2, dims, {typenum_z});
        }}

        {{
            // sparse array has size MxK, dense KxN, output MxN
            npy_intp M = PyArray_DIMS({z})[0];
            npy_intp N = PyArray_DIMS({z})[1];
            npy_intp K = PyArray_DIMS({b})[0];
            if (N > 0x7fffffffL)
            {{PyErr_SetString(PyExc_NotImplementedError, "array too big (overflows int32 index)"); {fail};}}

            // strides tell you how many bytes to skip to go to next column/row entry
            npy_intp Szm = PyArray_STRIDES({z})[0] / PyArray_DESCR({z})->elsize;
            npy_intp Szn = PyArray_STRIDES({z})[1] / PyArray_DESCR({z})->elsize;
            //npy_intp Sbm = PyArray_STRIDES({b})[0] / PyArray_DESCR({b})->elsize;
            npy_intp Sbn = PyArray_STRIDES({b})[1] / PyArray_DESCR({b})->elsize;
            npy_intp Sval = PyArray_STRIDES({a_val})[0] / PyArray_DESCR({a_val})->elsize;
            npy_intp Sind = PyArray_STRIDES({a_ind})[0] / PyArray_DESCR({a_ind})->elsize;
            npy_intp Sptr = PyArray_STRIDES({a_ptr})[0] / PyArray_DESCR({a_ptr})->elsize;

            // pointers to access actual data in the arrays passed as params.
            dtype_{z}*     __restrict__ Dz   = (dtype_{z}*)PyArray_DATA({z});
            const dtype_{a_val}* __restrict__ Dval = (dtype_{a_val}*)PyArray_DATA({a_val});
            const npy_int32 * __restrict__ Dind = (npy_int32*)PyArray_DATA({a_ind});
            const npy_int32 * __restrict__ Dptr = (npy_int32*)PyArray_DATA({a_ptr});

            //npy_intp nnz = PyArray_DIMS({a_ind})[0];

            //clear the output array
            memset(Dz, 0, M*N*sizeof(dtype_{z}));

            //iterate over the sparse array, making the most of an entry wherever we find it.
            //
            // Normal matrix matrix multiply: A MxK, B KxN =>  Z = AB
            // for m
            //   for n
            //     for k
            //        z[m, n] += a[m, k] * b[k, n]
            // Here instead: Z =
            // for k
            //   for m (sparse)
            //     for n
            //        z[m, n] += a[m, k] * b[k, n]

            // loop over inner dimension
            for (npy_int32 k = 0; k < K; ++k)
            {{
                // get pointer to k-th row of dense matrix
                const dtype_{b}* __restrict__ bk = (dtype_{b}*)(PyArray_BYTES({b}) + PyArray_STRIDES({b})[0] * k);

                // loop over sparse column indices through index pointer array
                // (amounts to looping over rows M of sparse matrix)

                for (npy_int32 m_idx = Dptr[k * Sptr]; m_idx < Dptr[(k+1) * Sptr]; ++m_idx)
                {{
                    npy_int32 m = Dind[m_idx * Sind]; // row index of non-null value for column K
                    const dtype_{a_val} Amk = Dval[m_idx * Sval]; // actual value at that location

                    // pointer to m-th row of the output matrix Z
                    dtype_{z}* __restrict__ zm = (dtype_{z}*)(PyArray_BYTES({z}) + PyArray_STRIDES({z})[0] * m);

                    //RESOLVE: a.shape[0] equals z.shape[0], why is this not an equality constraint?
                    if (m >= PyArray_DIMS({z})[0])
                    {{PyErr_SetString(PyExc_NotImplementedError, "illegal row index in a"); {fail};}}

                    // loop over final dimension (cols of dense matrix) and perform dot product
                    if ((Szn == 1) && (Sbn == 1)) {{
                        for(npy_int32 n = 0; n < N; ++n)
                        {{
                            zm[n] += Amk * bk[n];
                        }}
                    }}
                    else
                    {{
                        for(npy_int32 n = 0; n < N; ++n)
                        {{
                            zm[n*Szn] += Amk * bk[n*Sbn];
                        }}
                    }}
                }}
            }}
        }}
        """

        return rval

    def c_code_cache_version(self):
        return (3,)


sd_csc = StructuredDotCSC()


class StructuredDotCSR(COp):
    """
    Structured Dot CSR is like dot, except that only the
    gradient wrt non-zero elements of a sparse matrix
    are calculated and propagated.

    The output is presumed to be a dense matrix, and is represented by a
    `TensorType` instance.

    Notes
    -----
    The gradient implemented is structured.

    This `Op` is used as a rewritten form of `StructuredDot`.

    """

    __props__ = ()

    def make_node(self, a_val, a_ind, a_ptr, b):
        self.dtype_out = ps.upcast(a_val.type.dtype, b.type.dtype)
        r = Apply(
            self,
            [a_val, a_ind, a_ptr, b],
            [
                tensor(
                    dtype=self.dtype_out,
                    shape=(None, 1 if b.type.shape[1] == 1 else None),
                )
            ],
        )
        return r

    def perform(self, node, inputs, outputs):
        (a_val, a_ind, a_ptr, b) = inputs
        (out,) = outputs
        a = scipy.sparse.csr_matrix(
            (a_val, a_ind, a_ptr), (len(a_ptr) - 1, b.shape[0]), copy=True
        )  # use view_map before setting this to False
        # out[0] = a.dot(b)
        out[0] = a * b
        # scipy 0.7 automatically converts to dense, but not .6 sometimes
        assert _is_dense(out[0])

    def c_code(self, node, name, inputs, outputs, sub):
        """
        C-implementation of the dot product of the sparse matrix A and matrix B.

        Parameters
        ----------
        a_val
            Non-zero values of the sparse matrix.
        a_ind
            Column indices of the non-null values (.indices of a
            scipy.csc_matrix).
        a_ptr
            Indicates col indices for col. i are in the range
            a_ptr[i]:a_ptr[i+1].
        n_cols
            Number of columns of sparse matrix.
        b
            Dense matrix to perform dot product with, as in dot(a, b).
        z
            Return value.
        sub
            TODO, not too sure, something to do with weave probably.

        """
        (a_val, a_ind, a_ptr, b) = inputs
        (z,) = outputs
        typenum_z = TensorType(self.dtype_out, []).dtype_specs()[2]
        if node.inputs[0].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for a_val")
        if node.inputs[3].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for b")

        fail = sub["fail"]
        return f"""
        if (PyArray_NDIM({a_val}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(a_val) != 1"); {fail};}}
        if (PyArray_NDIM({a_ind}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(a_ind) != 1"); {fail};}}
        if (PyArray_NDIM({a_ptr}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(a_ptr) != 1"); {fail};}}
        if (PyArray_NDIM({b}) != 2) {{PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 2"); {fail};}}

        if (PyArray_TYPE({a_ind}) != NPY_INT32) {{
        PyErr_SetString(PyExc_NotImplementedError, "a_ind dtype not INT32"); {fail};}}

        if (PyArray_TYPE({a_ptr}) != NPY_INT32)
        {{PyErr_SetString(PyExc_NotImplementedError, "a_ptr dtype not INT32"); {fail};}}

        if (PyArray_DIMS({a_val})[0] != PyArray_DIMS({a_ind})[0])
        {{PyErr_SetString(PyExc_NotImplementedError, "a_val and a_ind have different lengths"); {fail};}}

        if ((!{z})
            || (PyArray_DIMS({z})[0] != PyArray_DIMS({a_ptr})[0]-1) //a's rows
            || (PyArray_DIMS({z})[1] != PyArray_DIMS({b})[1])       //b's columns
            )
        {{
            {{Py_XDECREF({z});}}
            npy_intp dims[] = {{0, 0}};
            dims[0] = PyArray_DIMS({a_ptr})[0]-1;
            dims[1] = PyArray_DIMS({b})[1];
            {z} = (PyArrayObject*) PyArray_SimpleNew(2, dims, {typenum_z});
        }}

        {{
            // sparse array has size MxK, dense KxN, output MxN
            npy_intp M = PyArray_DIMS({z})[0];
            npy_intp N = PyArray_DIMS({z})[1];
            npy_intp K = PyArray_DIMS({b})[0];
            if (N > 0x7fffffffL)
            {{PyErr_SetString(PyExc_NotImplementedError, "array too big (overflows int32 index)"); {fail};}}

            // strides tell you how many bytes to skip to go to next column/row entry
            npy_intp Szm = PyArray_STRIDES({z})[0] / PyArray_DESCR({z})->elsize;
            npy_intp Szn = PyArray_STRIDES({z})[1] / PyArray_DESCR({z})->elsize;
            npy_intp Sbm = PyArray_STRIDES({b})[0] / PyArray_DESCR({b})->elsize;
            npy_intp Sbn = PyArray_STRIDES({b})[1] / PyArray_DESCR({b})->elsize;
            npy_intp Sval = PyArray_STRIDES({a_val})[0] / PyArray_DESCR({a_val})->elsize;
            npy_intp Sind = PyArray_STRIDES({a_ind})[0] / PyArray_DESCR({a_ind})->elsize;
            npy_intp Sptr = PyArray_STRIDES({a_ptr})[0] / PyArray_DESCR({a_ptr})->elsize;

            // pointers to access actual data in the arrays passed as params.
            dtype_{z}* __restrict__ Dz = (dtype_{z}*)PyArray_DATA({z});
            const dtype_{a_val}* __restrict__ Dval = (dtype_{a_val}*)PyArray_DATA({a_val});
            const npy_int32 * __restrict__ Dind = (npy_int32*)PyArray_DATA({a_ind});
            const npy_int32 * __restrict__ Dptr = (npy_int32*)PyArray_DATA({a_ptr});

            //npy_intp nnz = PyArray_DIMS({a_ind})[0];

            //clear the output array
            memset(Dz, 0, M*N*sizeof(dtype_{z}));

            //iterate over the sparse array, making the most of an entry wherever we find it.
            // Normal matrix matrix multiply:
            // for m
            //   for n
            //     for k
            //        z[m, n] += a[m, k] * b[k, n]
            // Here instead:
            // for m
            //   for k (sparse)
            //     for n
            //        z[m, n] += a[m, k] * b[k, n]

            // loop over inner dimension
            for (npy_int64 m = 0; m < M; ++m)
            {{
                // pointer to m-th row of the output matrix Z
                dtype_{z}* __restrict__ zm = (dtype_{z}*)(PyArray_BYTES({z}) + PyArray_STRIDES({z})[0] * m);

                // loop over sparse rows indices through index pointer array
                // (amounts to looping over cols k of sparse matrix)
                for (npy_int32 k_idx = Dptr[m * Sptr]; k_idx < Dptr[(m+1) * Sptr]; ++k_idx)
                {{
                    npy_int32 k = Dind[k_idx * Sind]; // col index of non-null value for row m
                    const dtype_{a_val} Amk = Dval[k_idx * Sval]; // actual value at that location

                    // get pointer to k-th row of dense matrix
                    const dtype_{b}* __restrict__ bk = (dtype_{b}*)(PyArray_BYTES({b}) + PyArray_STRIDES({b})[0] * k);

                    // loop over final dimension (cols of dense matrix) and perform dot product
                    for(npy_int32 n = 0; n < N; ++n)
                    {{
                        zm[n*Szn] += Amk * bk[n*Sbn];
                    }}
                }}
            }}
        }}

        """

    def c_code_cache_version(self):
        return (2,)


sd_csr = StructuredDotCSR()


# register a specialization to replace StructuredDot -> StructuredDotCSx
# This is tested in tests/test_basic.py:792
@node_rewriter([sparse._structured_dot])
def local_structured_dot(fgraph, node):
    if node.op == sparse._structured_dot:
        a, b = node.inputs
        if a.type.format == "csc":
            a_val, a_ind, a_ptr, a_shape = csm_properties(a)
            a_nsparse = a_shape[0]
            return [sd_csc(a_val, a_ind, a_ptr, a_nsparse, b)]
        if a.type.format == "csr":
            a_val, a_ind, a_ptr, a_shape = csm_properties(a)
            return [sd_csr(a_val, a_ind, a_ptr, b)]
    return False


# Commented out because
# a) it is only slightly faster than scipy these days, and sometimes a little
# slower, and
# b) the resulting graphs make it very difficult for an op to do size checking
# on the matrices involved.  dimension mismatches are hard to detect sensibly.
# register_specialize(local_structured_dot)


class UsmmCscDense(_NoPythonCOp):
    """Performs ``alpha * x @ y + z``.

    ``x`` and ``y`` are a matrices, ``z`` is a dense matrix, and ``alpha`` is a
    scalar.  The result is a dense matrix.

    Notes
    -----
    The gradient is not implemented for this `Op`.

    This is an optimized version of `Usmm` when ``x`` is in CSC format and ``y`` is dense.

    """

    __props__ = ("inplace",)

    def __init__(self, inplace):
        self.inplace = inplace
        if inplace:
            self.destroy_map = {0: [6]}

    def __str__(self):
        if self.inplace:
            return "UsmmCscDense{inplace}"
        else:
            return "UsmmCscDense{no_inplace}"

    def make_node(self, alpha, x_val, x_ind, x_ptr, x_nrows, y, z):
        alpha = as_tensor_variable(alpha)
        x_val = as_tensor_variable(x_val)
        x_ind = as_tensor_variable(x_ind)
        x_ptr = as_tensor_variable(x_ptr)
        x_nrows = as_tensor_variable(x_nrows)
        y = as_tensor_variable(y)
        z = as_tensor_variable(z)
        assert x_ind.dtype == "int32"
        assert x_ptr.dtype == "int32"
        assert x_nrows.dtype == "int32"
        assert alpha.ndim == 2 and alpha.type.shape == (1, 1)
        assert x_val.ndim == 1
        assert y.ndim == 2
        assert z.ndim == 2

        dtype_out = ps.upcast(
            alpha.type.dtype, x_val.type.dtype, y.type.dtype, z.type.dtype
        )

        if dtype_out not in ("float32", "float64"):
            raise NotImplementedError("only float types are supported in operands")

        if self.inplace:
            assert z.type.dtype == dtype_out

        # axpy work only with the same dtype, so we should upcast the input
        if dtype_out != alpha.type.dtype:
            alpha = cast(alpha, dtype_out)
        if dtype_out != x_val.type.dtype:
            x_val = cast(x_val, dtype_out)
        if dtype_out != y.type.dtype:
            y = cast(y, dtype_out)
        if dtype_out != z.type.dtype:
            z = cast(z, dtype_out)

        r = Apply(
            self,
            [alpha, x_val, x_ind, x_ptr, x_nrows, y, z],
            [
                tensor(
                    dtype=dtype_out, shape=(None, 1 if y.type.shape[1] == 1 else None)
                )
            ],
        )
        return r

    def c_support_code(self, **kwargs):
        return blas.blas_header_text()

    def c_libraries(self, **kwargs):
        return blas.ldflags()

    def c_compile_args(self, **kwargs):
        return blas.ldflags(libs=False, flags=True)

    def c_lib_dirs(self, **kwargs):
        return blas.ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self, **kwargs):
        return blas.ldflags(libs=False, include_dir=True)

    def c_code(self, node, name, inputs, outputs, sub):
        alpha, x_val, x_ind, x_ptr, x_nrows, y, z = inputs
        zn = outputs[0]
        if node.inputs[1].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for x_val")
        if node.inputs[5].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for y")
        if node.inputs[6].type.dtype != node.outputs[0].type.dtype:
            raise NotImplementedError("z and output must have same type")

        if node.inputs[1].type.dtype == "float32":
            conv_type = "float"
            axpy = "saxpy_"
        else:
            conv_type = "double"
            axpy = "daxpy_"
        # retrieve dtype numbers
        typenum_alpha = node.inputs[0].type.dtype_specs()[2]
        typenum_x_val = node.inputs[1].type.dtype_specs()[2]
        typenum_y = node.inputs[5].type.dtype_specs()[2]
        typenum_z = node.inputs[6].type.dtype_specs()[2]
        typenum_zn = node.outputs[0].type.dtype_specs()[2]

        inplace = int(self.inplace)

        fail = sub["fail"]
        rval = f"""

        if (PyArray_NDIM({x_val}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(x_val) != 1"); {fail};}}
        if (PyArray_NDIM({x_ind}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(x_ind) != 1"); {fail};}}
        if (PyArray_NDIM({x_ptr}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(x_ptr) != 1"); {fail};}}
        if (PyArray_NDIM({x_nrows}) != 0) {{PyErr_SetString(PyExc_NotImplementedError, "rank(nrows) != 0"); {fail};}}
        if (PyArray_NDIM({y}) != 2) {{PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 2"); {fail};}}

        if (PyArray_TYPE({x_val}) != {typenum_x_val}) {{
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for x_val"); {fail};}}

        if (PyArray_TYPE({y}) != {typenum_y}) {{
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for y"); {fail};}}

        if (PyArray_TYPE({z}) != {typenum_z}) {{
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for z"); {fail};}}

        if (PyArray_TYPE({alpha}) != {typenum_alpha}) {{
        PyErr_SetString(PyExc_NotImplementedError, "Invalid type for alpha"); {fail};}}

        if (PyArray_TYPE({x_ind}) != NPY_INT32) {{
        PyErr_SetString(PyExc_NotImplementedError, "x_ind dtype not INT32"); {fail};}}

        if (PyArray_TYPE({x_ptr}) != NPY_INT32)
        {{PyErr_SetString(PyExc_NotImplementedError, "x_ptr dtype not INT32"); {fail};}}

        if (PyArray_TYPE({x_nrows}) != NPY_INT32)
        {{PyErr_SetString(PyExc_NotImplementedError, "x_nrows dtype not INT32"); {fail};}}

        if (PyArray_DIMS({x_val})[0] != PyArray_DIMS({x_ind})[0])
        {{PyErr_SetString(PyExc_NotImplementedError, "x_val and x_ind have different lengths"); {fail};}}

        if (PyArray_DIMS({x_ptr})[0] != PyArray_DIMS({y})[0]+1)
        {{PyErr_SetString(PyExc_NotImplementedError, "x's number of columns doesn't match y's rows"); {fail};}}

        if (PyArray_DIMS({z})[0] != ((npy_int32 *)PyArray_DATA({x_nrows}))[0] || PyArray_DIMS({z})[1] != PyArray_DIMS({y})[1])
        {{PyErr_SetString(PyExc_NotImplementedError, "The dimension of the allocated output doesn't match the correct output size."); {fail};}}

        if (PyArray_SIZE({alpha}) != 1)
        {{PyErr_SetString(PyExc_NotImplementedError, "The number of element in alpha must be 1"); {fail};}}

        if (PyArray_NDIM({alpha}) != 2)
        {{PyErr_SetString(PyExc_NotImplementedError, "The number dimension of alpha must be 2"); {fail};}}

        if (PyArray_NDIM({x_val}) != 1)
        {{PyErr_SetString(PyExc_NotImplementedError, "The number dimension of x_val must be 1"); {fail};}}

        if (PyArray_NDIM({y}) != 2)
        {{PyErr_SetString(PyExc_NotImplementedError, "The number dimension of y must be 2"); {fail};}}

        if (PyArray_NDIM({z}) != 2)
        {{PyErr_SetString(PyExc_NotImplementedError, "The number dimension of z must be 2"); {fail};}}

        if ({inplace})
        {{
            if ({typenum_zn} != {typenum_z}) {{
            PyErr_SetString(PyExc_NotImplementedError, "When inplace the output dtype must be the same as the input"); {fail};}}

            Py_XDECREF({zn});
            {zn} = {z};
            Py_INCREF({zn});
        }}
        else if (!{zn}
            || (PyArray_DIMS({zn})[0] != ((npy_int32 *)PyArray_DATA({x_nrows}))[0])
            || (PyArray_DIMS({zn})[1] != PyArray_DIMS({y})[1])
            )
        {{
            {{Py_XDECREF({zn});}}
            npy_intp dims[] = {{0, 0}};
            dims[0] = ((npy_int32 *)PyArray_DATA({x_nrows}))[0];
            dims[1] = PyArray_DIMS({y})[1];
            {zn} = (PyArrayObject*) PyArray_SimpleNew(2, dims, {typenum_zn});
        }}

        {{
            // sparse array has size MxK, dense KxN, output MxN
            npy_intp M = PyArray_DIMS({zn})[0];
            npy_intp N = PyArray_DIMS({zn})[1];
            npy_intp K = PyArray_DIMS({y})[0];

            // pointers to access actual data in the arrays passed as params.
            const dtype_{x_val}* __restrict__ Dval = (dtype_{x_val}*)PyArray_DATA({x_val});
            const npy_int32 * __restrict__ Dind = (npy_int32*)PyArray_DATA({x_ind});
            const npy_int32 * __restrict__ Dptr = (npy_int32*)PyArray_DATA({x_ptr});
            const dtype_{alpha} alpha = ((dtype_{alpha}*)PyArray_DATA({alpha}))[0];

            npy_intp Sz = PyArray_STRIDES({z})[1] / PyArray_DESCR({z})->elsize;
            npy_intp Szn = PyArray_STRIDES({zn})[1] / PyArray_DESCR({zn})->elsize;
            npy_intp Sval = PyArray_STRIDES({x_val})[0] / PyArray_DESCR({x_val})->elsize;
            npy_intp Sind = PyArray_STRIDES({x_ind})[0] / PyArray_DESCR({x_ind})->elsize;
            npy_intp Sptr = PyArray_STRIDES({x_ptr})[0] / PyArray_DESCR({x_ptr})->elsize;
            npy_intp Sy = PyArray_STRIDES({y})[1] / PyArray_DESCR({y})->elsize;

            // blas expects ints; convert here (rather than just making N etc ints) to avoid potential overflow in the negative-stride correction
            if ((N > 0x7fffffffL)||(Sy > 0x7fffffffL)||(Szn > 0x7fffffffL)||(Sy < -0x7fffffffL)||(Szn < -0x7fffffffL))
            {{PyErr_SetString(PyExc_NotImplementedError, "array too big for BLAS (overflows int32 index)"); {fail};}}
            int N32 = N;
            int Sy32 = Sy;
            int Szn32 = Szn;

            if (!({inplace}))
            {{
                if (PyArray_CopyInto({zn}, {z}))
                {{
                    Py_XDECREF({zn});
                    {fail};
                }}
            }}

            for (npy_intp k = 0; k < K; ++k)
            {{
                for (npy_int32 m_idx = Dptr[k * Sptr]; m_idx < Dptr[(k+1)*Sptr]; ++m_idx)
                {{
                    const npy_int32 m = Dind[m_idx * Sind]; // row index of non-null value for column K

                    const dtype_{x_val} Amk = alpha * Dval[m_idx * Sval]; // actual value at that location

                    dtype_{y}* y_row = (dtype_{y}*)(PyArray_BYTES({y}) + PyArray_STRIDES({y})[0] * k);
                    // axpy expects pointer to the beginning of memory arrays,
                    // so when the stride is negative, we need to get the
                    // last element
                    if (Sy < 0)
                        y_row += (K - 1) * Sy;

                    dtype_{zn}* z_row = (dtype_{zn}*)(PyArray_BYTES({zn}) + PyArray_STRIDES({zn})[0] * m);
                    if (Szn < 0)
                        z_row += (N - 1) * Szn;

                    {axpy}(&N32, ({conv_type}*)&Amk, ({conv_type}*)y_row, &Sy32, ({conv_type}*)z_row, &Szn32);
                }}
            }}
        }}
        """

        return rval

    def c_code_cache_version(self):
        return (3, blas.blas_header_version())


usmm_csc_dense = UsmmCscDense(inplace=False)
usmm_csc_dense_inplace = UsmmCscDense(inplace=True)


# This is tested in tests/test_basic.py:UsmmTests
local_usmm = PatternNodeRewriter(
    (
        sub,
        "z",
        (
            mul,
            {
                "pattern": "alpha",
                "constraint": lambda expr: (
                    all(s == 1 for s in expr.type.shape) and config.blas__ldflags
                ),
            },
            (sparse._dot, "x", "y"),
        ),
    ),
    (usmm, (neg, "alpha"), "x", "y", "z"),
)
register_specialize(local_usmm, name="local_usmm")


# register a specialization to replace usmm_csc_dense -> usmm_csc_dense_inplace
# This is tested in tests/test_basic.py:UsmmTests
@node_rewriter([usmm_csc_dense])
def local_usmm_csc_dense_inplace(fgraph, node):
    if node.op == usmm_csc_dense:
        return [usmm_csc_dense_inplace(*node.inputs)]


register_specialize(local_usmm_csc_dense_inplace, "cxx_only", "inplace")


# This is tested in tests/test_basic.py:UsmmTests
@node_rewriter([usmm])
def local_usmm_csx(fgraph, node):
    """
    usmm -> usmm_csc_dense

    """
    if node.op == usmm:
        alpha, x, y, z = node.inputs

        x_is_sparse_variable = _is_sparse_variable(x)
        y_is_sparse_variable = _is_sparse_variable(y)

        if x_is_sparse_variable and not y_is_sparse_variable:
            if x.type.format == "csc":
                x_val, x_ind, x_ptr, x_shape = csm_properties(x)
                x_nsparse = x_shape[0]
                dtype_out = ps.upcast(
                    alpha.type.dtype, x.type.dtype, y.type.dtype, z.type.dtype
                )
                if dtype_out not in ("float32", "float64"):
                    return False
                # Sparse cast is not implemented.
                if y.type.dtype != dtype_out:
                    return False

                return [usmm_csc_dense(alpha, x_val, x_ind, x_ptr, x_nsparse, y, z)]
    return False


register_specialize(local_usmm_csx, "cxx_only")


class CSMGradC(_NoPythonCOp):
    __props__ = ()

    def make_node(self, a_val, a_ind, a_ptr, a_dim, b_val, b_ind, b_ptr, b_dim):
        return Apply(
            self,
            [a_val, a_ind, a_ptr, a_dim, b_val, b_ind, b_ptr, b_dim],
            [b_val.type()],
        )

    def c_code(self, node, name, inputs, outputs, sub):
        # retrieve dtype number
        (a_val, a_ind, a_ptr, a_dim, b_val, b_ind, b_ptr, b_dim) = inputs
        (z,) = outputs
        typenum_z = node.outputs[0].type.dtype_specs()[2]
        if node.inputs[0].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for a_val")
        if node.inputs[3].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for b_val")

        fail = sub["fail"]
        return f"""
        if (PyArray_NDIM({a_val}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(a_val) != 1"); {fail};}}
        if (PyArray_NDIM({a_ind}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(a_ind) != 1"); {fail};}}
        if (PyArray_NDIM({a_ptr}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(a_ptr) != 1"); {fail};}}
        if (PyArray_NDIM({b_val}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(b_val) != 1"); {fail};}}
        if (PyArray_NDIM({b_ind}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(b_ind) != 1"); {fail};}}
        if (PyArray_NDIM({b_ptr}) != 1) {{PyErr_SetString(PyExc_NotImplementedError, "rank(b_ptr) != 1"); {fail};}}

        if (PyArray_TYPE({a_ind}) != NPY_INT32) {{
        PyErr_SetString(PyExc_NotImplementedError, "a_ind dtype not INT32"); {fail};}}

        if (PyArray_TYPE({a_ptr}) != NPY_INT32)
        {{PyErr_SetString(PyExc_NotImplementedError, "a_ptr dtype not INT32"); {fail};}}

        if (PyArray_TYPE({b_ind}) != NPY_INT32) {{
        PyErr_SetString(PyExc_NotImplementedError, "b_ind dtype not INT32"); {fail};}}

        if (PyArray_TYPE({b_ptr}) != NPY_INT32)
        {{PyErr_SetString(PyExc_NotImplementedError, "b_ptr dtype not INT32"); {fail};}}

        if (PyArray_DIMS({a_val})[0] != PyArray_DIMS({a_ind})[0])
        {{PyErr_SetString(PyExc_NotImplementedError, "a_val and a_ind have different lengths"); {fail};}}

        if (PyArray_DIMS({b_val})[0] != PyArray_DIMS({b_ind})[0])
        {{PyErr_SetString(PyExc_NotImplementedError, "b_val and b_ind have different lengths"); {fail};}}

        if (PyArray_DIMS({a_ptr})[0] != PyArray_DIMS({b_ptr})[0])
        {{PyErr_SetString(PyExc_NotImplementedError, "a_ptr and b_ptr have different lengths"); {fail};}}

        if ((!{z}) || (PyArray_DIMS({z})[0] != PyArray_DIMS({a_val})[0]))
        {{
            {{Py_XDECREF({z});}}
            npy_intp dims[] = {{0}};
            dims[0] = PyArray_DIMS({a_val})[0];
            {z} = (PyArrayObject*) PyArray_SimpleNew(1, dims, {typenum_z});
        }}

        {{
            // sparse array has size MxK, dense KxN, output MxN
            npy_intp M = PyArray_DIMS({a_ptr})[0] - 1;
            npy_intp a_dim_0 = ((npy_int32 *)PyArray_DATA({a_dim}))[0];
            npy_intp a_dim_1 = ((npy_int32 *)PyArray_DATA({a_dim}))[1];

            npy_intp sp_dim = (M == a_dim_0)?a_dim_1:a_dim_0;

            // strides tell you how many bytes to skip to go to next column/row entry
            npy_intp Sz = PyArray_STRIDES({z})[0] / PyArray_DESCR({z})->elsize;
            npy_intp Sa_val = PyArray_STRIDES({a_val})[0] / PyArray_DESCR({a_val})->elsize;
            npy_intp Sa_ind = PyArray_STRIDES({a_ind})[0] / PyArray_DESCR({a_ind})->elsize;
            npy_intp Sa_ptr = PyArray_STRIDES({a_ptr})[0] / PyArray_DESCR({a_ptr})->elsize;
            npy_intp Sb_val = PyArray_STRIDES({b_val})[0] / PyArray_DESCR({b_val})->elsize;
            npy_intp Sb_ind = PyArray_STRIDES({b_ind})[0] / PyArray_DESCR({b_ind})->elsize;
            npy_intp Sb_ptr = PyArray_STRIDES({b_ptr})[0] / PyArray_DESCR({b_ptr})->elsize;

            // pointers to access actual data in the arrays passed as params.
            dtype_{z}* __restrict__ Dz = (dtype_{z}*)PyArray_DATA({z});
            const dtype_{a_val}* __restrict__ Da_val = (dtype_{a_val}*)PyArray_DATA({a_val});
            const npy_int32 * __restrict__ Da_ind = (npy_int32*)PyArray_DATA({a_ind});
            const npy_int32 * __restrict__ Da_ptr = (npy_int32*)PyArray_DATA({a_ptr});
            const dtype_{b_val}* __restrict__ Db_val = (dtype_{b_val}*)PyArray_DATA({b_val});
            const npy_int32 * __restrict__ Db_ind = (npy_int32*)PyArray_DATA({b_ind});
            const npy_int32 * __restrict__ Db_ptr = (npy_int32*)PyArray_DATA({b_ptr});

            npy_intp nnz = PyArray_DIMS({a_ind})[0];

            dtype_{b_val} b_row[sp_dim];

            //clear the output array
            for (npy_int64 i = 0; i < nnz; ++i)
            {{
                Dz[i*Sz] = 0;
            }}
            memset(b_row, 0, sp_dim*sizeof(dtype_{b_val}));

            // loop over inner dimension
            for (npy_int64 m = 0; m < M; ++m)
            {{
                for (npy_int32 j_ptr = Db_ptr[m * Sb_ptr];
                    j_ptr < Db_ptr[(m + 1) * Sb_ptr]; j_ptr++) {{
                    b_row[Db_ind[j_ptr * Sb_ind]] += Db_val[j_ptr*Sb_val];
                }}

                for (npy_int32 j_ptr = Da_ptr[m * Sa_ptr];
                    j_ptr < Da_ptr[(m + 1) * Sa_ptr]; j_ptr++) {{
                    Dz[j_ptr*Sz] = b_row[Da_ind[j_ptr * Sa_ind]];
                }}

                for (npy_int32 j_ptr = Db_ptr[m * Sb_ptr];
                    j_ptr < Db_ptr[(m + 1) * Sb_ptr]; j_ptr++) {{
                    b_row[Db_ind[j_ptr * Sb_ind]] = 0;
                }}
            }}
        }}

        """

    def c_code_cache_version(self):
        return (3,)


csm_grad_c = CSMGradC()


@node_rewriter([csm_grad(None)])
def local_csm_grad_c(fgraph, node):
    """
    csm_grad(None) -> csm_grad_c

    """
    if node.op == csm_grad(None):
        return [csm_grad_c(*node.inputs)]
    return False


# DISABLED AS IT IS BROKEN FOR UNSORTED INDICES!
# register_specialize(local_csm_grad_c, 'cxx_only')


class MulSDCSC(_NoPythonCOp):
    """Multiplication of sparse matrix by a broadcasted dense vector element-wise.

    Notes
    -----

    This `Op` is used as a rewritten form of `mul_s_d`.

    """

    __props__ = ()

    def make_node(self, a_data, a_indices, a_indptr, b):
        """

        Parameters
        ----------
        a_data
            Sparse matrix data.
        a_indices
            Sparse matrix indices.
        a_indptr
            Sparse matrix indptr.
        b
            Tensor type matrix.

        Returns
        -------
        The multiplication of the two matrices element-wise.

        Notes
        -----
        `a_data`, `a_indices` and `a_indptr` must be the properties of a sparse
        matrix in csc format.

        The dtype of `a_data`, i.e. the dtype of the sparse matrix, cannot be a
        complex type.

        """
        assert b.type.ndim == 2
        return Apply(
            self,
            [a_data, a_indices, a_indptr, b],
            [tensor(dtype=b.dtype, shape=(None,))],
        )

    def c_code_cache_version(self):
        return (3,)

    def c_code(self, node, name, inputs, outputs, sub):
        (
            _data,
            _indices,
            _indptr,
            _b,
        ) = inputs
        (_zout,) = outputs
        if node.inputs[0].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for a")
        if node.inputs[3].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for b")

        fail = sub["fail"]
        return f"""
        if (PyArray_NDIM({_b}) != 2) {{
            PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 2");
            {fail};}}
        if (PyArray_NDIM({_data}) != 1) {{
            PyErr_SetString(PyExc_NotImplementedError, "rank(data) != 1");
            {fail};}}
        if (PyArray_NDIM({_indices}) != 1) {{
            PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1");
            {fail};}}
        if (PyArray_NDIM({_indptr}) != 1) {{
            PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1");
            {fail};}}

        if( PyArray_TYPE({_indices}) != NPY_INT32) {{
        PyErr_SetString(PyExc_NotImplementedError, "C"); {fail};}}

        if( PyArray_TYPE({_indptr}) != NPY_INT32)
        {{PyErr_SetString(PyExc_NotImplementedError, "D"); {fail};}}

        if (!{_zout} ||
            (PyArray_DIMS({_zout})[0] != PyArray_DIMS({_indices})[0]) ||
            !(PyArray_ISCONTIGUOUS({_zout})))
        {{
            Py_XDECREF({_zout});
            {_zout} = (PyArrayObject*) PyArray_SimpleNew(1,
                  PyArray_DIMS({_indices}), PyArray_TYPE({_b}));
            if (!{_zout})
            {{
                PyErr_SetString(PyExc_MemoryError,
                    "Could not allocate output memory.");
                {fail};
            }}
        }}

        {{ //makes it compile even though labels jump over variable definitions.
            const npy_intp nnz = PyArray_DIMS({_indices})[0];
            //TODO: error checking with this
            const npy_intp N =  PyArray_DIMS({_indptr})[0]-1;

            const dtype_{_data} * const __restrict__ data = (dtype_{_data}*)PyArray_DATA({_data});
            const npy_int32 * const __restrict__ indptr = (npy_int32 *)PyArray_DATA({_indptr});
            const npy_int32 * const __restrict__ indices = (npy_int32 *)PyArray_DATA({_indices});

            dtype_{_zout} * const __restrict__ zout = (dtype_{_zout}*)PyArray_DATA({_zout});

            const npy_intp Sb = PyArray_STRIDES({_b})[0];

            // loop over columns
            for (npy_intp j = 0; j < N; ++j)
            {{
                // for each non-null value in the sparse column
                for (npy_int32 i_idx = indptr[j]; i_idx < indptr[j+1]; ++i_idx)
                {{
                    // extract row index of non-null value
                    npy_int32 i = indices[i_idx];

                    // extract i-th row of dense matrix
                    const dtype_{_b}* __restrict__ b_row = (dtype_{_b}*)(PyArray_BYTES({_b}) + Sb * i);

                    // write resulting gradient to sparse output
                    zout[i_idx] = data[i_idx] * b_row[j];
                }}
            }}
        }}

        """

    def __str__(self):
        return self.__class__.__name__


mul_s_d_csc = MulSDCSC()


class MulSDCSR(_NoPythonCOp):
    """Multiplication of sparse matrix by a broadcasted dense vector element-wise.

    Notes
    -----

    This `Op` is used as a rewritten form of `mul_s_d`.

    """

    __props__ = ()

    def make_node(self, a_data, a_indices, a_indptr, b):
        """

        Parameters
        ----------
        a_data
            Sparse matrix data.
        a_indices
            Sparse matrix indices.
        a_indptr
            Sparse matrix indptr.
        b
            Tensor type matrix.

        Returns
        -------
        The multiplication of the two matrix element wise.

        Notes
        -----
        `a_data`, `a_indices` and `a_indptr` must be the properties
        of a sparse matrix in csr format.

        The dtype of `a_data`, i.e. the dtype of the sparse matrix,
        cannot be a complex type.

        """
        assert b.type.ndim == 2
        return Apply(
            self,
            [a_data, a_indices, a_indptr, b],
            [tensor(dtype=b.dtype, shape=(None,))],
        )

    def c_code_cache_version(self):
        return (3,)

    def c_code(self, node, name, inputs, outputs, sub):
        (
            _data,
            _indices,
            _indptr,
            _b,
        ) = inputs
        (_zout,) = outputs
        if node.inputs[0].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for a")
        if node.inputs[3].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for b")

        fail = sub["fail"]
        return f"""
        if (PyArray_NDIM({_b}) != 2) {{
            PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 2");
            {fail};}}
        if (PyArray_NDIM({_data}) != 1) {{
            PyErr_SetString(PyExc_NotImplementedError, "rank(data) != 1");
            {fail};}}
        if (PyArray_NDIM({_indices}) != 1) {{
            PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1");
            {fail};}}
        if (PyArray_NDIM({_indptr}) != 1) {{
            PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1");
            {fail};}}

        if( PyArray_TYPE({_indices}) != NPY_INT32) {{
        PyErr_SetString(PyExc_NotImplementedError, "C"); {fail};}}

        if( PyArray_TYPE({_indptr}) != NPY_INT32)
        {{PyErr_SetString(PyExc_NotImplementedError, "D"); {fail};}}

        if (!{_zout} ||
            (PyArray_DIMS({_zout})[0] != PyArray_DIMS({_indices})[0]) ||
            !(PyArray_ISCONTIGUOUS({_zout})))
        {{
            Py_XDECREF({_zout});
            {_zout} = (PyArrayObject*) PyArray_SimpleNew(1,
                    PyArray_DIMS({_indices}), PyArray_TYPE({_b}));
            if (!{_zout})
            {{
                PyErr_SetString(PyExc_MemoryError,
                    "Could not allocate output memory.");
                {fail};
            }}
        }}

        {{ //makes it compile even though labels jump over variable definitions.
            const npy_intp nnz = PyArray_DIMS({_indices})[0];
            //TODO: error checking with this
            const npy_intp N =  PyArray_DIMS({_indptr})[0]-1;

            const dtype_{_data} * const __restrict__ data = (dtype_{_data}*)PyArray_DATA({_data});
            const npy_int32 * const __restrict__ indptr = (npy_int32 *)PyArray_DATA({_indptr});
            const npy_int32 * const __restrict__ indices = (npy_int32 *)PyArray_DATA({_indices});

            dtype_{_zout} * const __restrict__ zout = (dtype_{_zout}*)PyArray_DATA({_zout});

            const npy_intp Sb = PyArray_STRIDES({_b})[0];

            // loop over columns
            for (npy_intp j = 0; j < N; ++j)
            {{
                // extract i-th row of dense matrix
                const dtype_{_b}* __restrict__ b_row = (dtype_{_b}*)(PyArray_BYTES({_b}) + Sb * j);

                // for each non-null value in the sparse column
                for (npy_int32 i_idx = indptr[j]; i_idx < indptr[j+1]; ++i_idx)
                {{
                    // extract row index of non-null value
                    npy_int32 i = indices[i_idx];

                    // write resulting gradient to sparse output
                    zout[i_idx] = data[i_idx] * b_row[i];
                }}
            }}
        }}

        """

    def __str__(self):
        return self.__class__.__name__


mul_s_d_csr = MulSDCSR()


# register a specialization to replace MulSD -> MulSDCSX
@node_rewriter([sparse.mul_s_d])
def local_mul_s_d(fgraph, node):
    if node.op == sparse.mul_s_d:
        x, y = node.inputs

        x_is_sparse_variable = _is_sparse_variable(x)

        if x_is_sparse_variable:
            svar = x
            dvar = y
        else:
            svar = y
            dvar = x

        if dvar.type.ndim != 2:
            return False
        if svar.type.format == "csc":
            CSx = sparse.CSC
            mul_s_d_csx = mul_s_d_csc
        elif svar.type.format == "csr":
            CSx = sparse.CSR
            mul_s_d_csx = mul_s_d_csr
        else:
            raise NotImplementedError
        if x.dtype != y.dtype:
            # mul_s_d_csx don't support that case
            return

        c_data = mul_s_d_csx(
            sparse.csm_data(svar),
            sparse.csm_indices(svar),
            sparse.csm_indptr(svar),
            dvar,
        )

        return [
            CSx(
                c_data,
                sparse.csm_indices(svar),
                sparse.csm_indptr(svar),
                sparse.csm_shape(svar),
            )
        ]

    return False


register_specialize(local_mul_s_d, "cxx_only")


class MulSVCSR(_NoPythonCOp):
    """Multiplication of sparse matrix by a broadcasted dense vector element-wise.


    Notes
    -----

    This `Op` is used as a rewritten form of `MulSV`.

    """

    __props__ = ()

    def make_node(self, a_data, a_indices, a_indptr, b):
        """

        Parameters
        ----------
        a_data
            Sparse matrix data.
        a_indices
            Sparse matrix indices.
        a_indptr
            Sparse matrix indptr.
        b
            Tensor type matrix.

        Returns
        -------
        The multiplication of the two matrix element wise.

        Notes
        -----
        `a_data`, `a_indices` and `a_indptr` must be the properties
        of a sparse matrix in csr format.

        The dtype of `a_data`, i.e. the dtype of the sparse matrix,
        cannot be a complex type.

        """
        assert b.type.ndim == 1
        return Apply(
            self,
            [a_data, a_indices, a_indptr, b],
            [tensor(dtype=b.dtype, shape=(None,))],
        )

    def c_code_cache_version(self):
        return (2,)

    def c_code(self, node, name, inputs, outputs, sub):
        (
            _data,
            _indices,
            _indptr,
            _b,
        ) = inputs
        (_zout,) = outputs
        if node.inputs[0].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for a")
        if node.inputs[3].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for b")

        fail = sub["fail"]
        return f"""
        if (PyArray_NDIM({_b}) != 1) {{
            PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 1");
            {fail};
        }}
        if (PyArray_NDIM({_data}) != 1) {{
            PyErr_SetString(PyExc_NotImplementedError, "rank(data) != 1");
            {fail};
        }}
        if (PyArray_NDIM({_indices}) != 1) {{
            PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1");
            {fail};
        }}
        if (PyArray_NDIM({_indptr}) != 1) {{
            PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1");
            {fail};
        }}

        if( PyArray_TYPE({_indices}) != NPY_INT32) {{
        PyErr_SetString(PyExc_NotImplementedError, "C"); {fail};}}

        if( PyArray_TYPE({_indptr}) != NPY_INT32)
        {{PyErr_SetString(PyExc_NotImplementedError, "D"); {fail};}}

        if (!{_zout}
            || PyArray_DIMS({_zout})[0] != PyArray_DIMS({_indices})[0]
            || !PyArray_ISCONTIGUOUS({_zout}))
        {{
            Py_XDECREF({_zout});
            {_zout} = (PyArrayObject*) PyArray_SimpleNew(1,
                    PyArray_DIMS({_indices}), PyArray_TYPE({_b}));
        }}

        {{ //makes it compile even though labels jump over variable definitions.
            const npy_intp nnz = PyArray_DIMS({_indices})[0];
            //TODO: error checking with this
            const npy_intp N =  PyArray_DIMS({_indptr})[0]-1;

            const dtype_{_data} * const __restrict__ data = (dtype_{_data}*)PyArray_DATA({_data});
            const npy_int32 * const __restrict__ indptr = (npy_int32 *)PyArray_DATA({_indptr});
            const npy_int32 * const __restrict__ indices = (npy_int32 *)PyArray_DATA({_indices});

            const dtype_{_b}* __restrict__ Db = (dtype_{_b}*)PyArray_DATA({_b});

            dtype_{_zout} * const __restrict__ zout = (dtype_{_zout}*)PyArray_DATA({_zout});

            const npy_intp Sb = PyArray_STRIDES({_b})[0] / PyArray_DESCR({_b})->elsize;

            // loop over rows
            for (npy_intp j = 0; j < N; ++j)
            {{
                // for each non-null value in the sparse column
                for (npy_int32 i_idx = indptr[j]; i_idx < indptr[j+1]; ++i_idx)
                {{
                    // extract row index of non-null value
                    npy_int32 i = indices[i_idx];

                    zout[i_idx] = data[i_idx] * Db[i * Sb];
                }}
            }}
        }}

        """

    def __str__(self):
        return self.__class__.__name__


mul_s_v_csr = MulSVCSR()


# register a specialization to replace MulSV -> MulSVCSR
@node_rewriter([sparse.mul_s_v])
def local_mul_s_v(fgraph, node):
    if node.op == sparse.mul_s_v:
        x, y = node.inputs

        x_is_sparse_variable = _is_sparse_variable(x)

        if x_is_sparse_variable:
            svar = x
            dvar = y
        else:
            svar = y
            dvar = x

        if dvar.type.ndim != 1:
            return False
        elif svar.type.format == "csr":
            CSx = sparse.CSR
            mul_s_v_csx = mul_s_v_csr
        else:
            return False

        s_val, s_ind, s_ptr, s_shape = sparse.csm_properties(svar)

        c_data = mul_s_v_csx(s_val, s_ind, s_ptr, dvar)

        return [CSx(c_data, s_ind, s_ptr, s_shape)]

    return False


register_specialize(local_mul_s_v, "cxx_only")


class StructuredAddSVCSR(_NoPythonCOp):
    """Structured addition of a sparse matrix and a dense vector.

    The elements of the vector are are only added to the corresponding
    non-zero elements. Therefore, this operation outputs another sparse
    matrix.

    Notes
    -----

    This `Op` is used as a rewritten form of `StructuredAddSV`.

    """

    __props__ = ()

    def make_node(self, a_data, a_indices, a_indptr, b):
        """

        Parameters
        ----------
        a_data
            Sparse matrix data.
        a_indices
            Sparse matrix indices.
        a_indptr
            Sparse matrix indptr.
        b
            Tensor type vector.

        Returns
        -------
        A sparse matrix containing the addition of the vector to the data of the
        sparse matrix.

        """
        b = as_tensor_variable(b)
        a_data = as_tensor_variable(a_data)
        a_indices = as_tensor_variable(a_indices)
        a_indptr = as_tensor_variable(a_indptr)
        assert a_data.type.ndim == 1
        assert a_indices.type.ndim == 1
        assert a_indptr.type.ndim == 1
        assert b.type.ndim == 1
        return Apply(
            self,
            [a_data, a_indices, a_indptr, b],
            [tensor(dtype=b.dtype, shape=(None,))],
        )

    def c_code_cache_version(self):
        return (3,)

    def c_code(self, node, name, inputs, outputs, sub):
        (
            _data,
            _indices,
            _indptr,
            _b,
        ) = inputs
        (_zout,) = outputs
        if node.inputs[0].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for a")
        if node.inputs[3].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for b")

        fail = sub["fail"]
        return f"""
        if (PyArray_NDIM({_b}) != 1) {{
            PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 1");
            {fail};
        }}
        if (PyArray_NDIM({_data}) != 1) {{
            PyErr_SetString(PyExc_NotImplementedError, "rank(data) != 1");
            {fail};
        }}
        if (PyArray_NDIM({_indices}) != 1) {{
            PyErr_SetString(PyExc_NotImplementedError, "rank(indices) != 1");
            {fail};
        }}
        if (PyArray_NDIM({_indptr}) != 1) {{
            PyErr_SetString(PyExc_NotImplementedError, "rank(indptr) != 1");
            {fail};
        }}

        if( PyArray_TYPE({_indices}) != NPY_INT32) {{
        PyErr_SetString(PyExc_NotImplementedError, "C"); {fail};}}

        if( PyArray_TYPE({_indptr}) != NPY_INT32)
        {{PyErr_SetString(PyExc_NotImplementedError, "D"); {fail};}}

        if (!{_zout}
            || (PyArray_DIMS({_zout})[0] != PyArray_DIMS({_indices})[0])
            || !(PyArray_ISCONTIGUOUS({_zout})))
        {{
            Py_XDECREF({_zout});
            {_zout} = (PyArrayObject*) PyArray_SimpleNew(1,
                    PyArray_DIMS({_indices}), PyArray_TYPE({_b}));
            if (!{_zout})
            {{
                PyErr_SetString(PyExc_MemoryError,
                    "Could not allocate output memory.");
                {fail};
            }}
        }}

        {{ //makes it compile even though labels jump over variable definitions.
            const npy_intp nnz = PyArray_DIMS({_indices})[0];
            //TODO: error checking with this
            const npy_intp N =  PyArray_DIMS({_indptr})[0]-1;

            const dtype_{_data} * const __restrict__ data = (dtype_{_data}*)PyArray_DATA({_data});
            const npy_int32 * const __restrict__ indptr = (npy_int32 *)PyArray_DATA({_indptr});
            const npy_int32 * const __restrict__ indices = (npy_int32 *)PyArray_DATA({_indices});

            const dtype_{_b}* __restrict__ Db = (dtype_{_b}*)PyArray_DATA({_b});

            dtype_{_zout} * const __restrict__ zout = (dtype_{_zout}*)PyArray_DATA({_zout});

            const npy_intp Sb = PyArray_STRIDES({_b})[0] / PyArray_DESCR({_b})->elsize;

            // loop over columns
            for (npy_intp j = 0; j < N; ++j)
            {{
                // for each non-null value in the sparse column
                for (npy_int32 i_idx = indptr[j]; i_idx < indptr[j+1]; ++i_idx)
                {{
                    // extract row index of non-null value
                    npy_int32 i = indices[i_idx];

                    // write resulting gradient to sparse output
                    zout[i_idx] = data[i_idx] + Db[i * Sb];
                }}
            }}
        }}

        """

    def __str__(self):
        return self.__class__.__name__


structured_add_s_v_csr = StructuredAddSVCSR()


# register a specialization to replace
# structured_add_s_v -> structured_add_s_v_csr
@node_rewriter([sparse.structured_add_s_v])
def local_structured_add_s_v(fgraph, node):
    if node.op == sparse.structured_add_s_v:
        x, y = node.inputs

        x_is_sparse_variable = _is_sparse_variable(x)
        # y_is_sparse_variable = _is_sparse_variable(y)

        if x_is_sparse_variable:
            svar = x
            dvar = y
        else:
            svar = y
            dvar = x

        if dvar.type.ndim != 1:
            return False
        elif svar.type.format == "csr":
            CSx = sparse.CSR
            structured_add_s_v_csx = structured_add_s_v_csr
        else:
            return False

        s_val, s_ind, s_ptr, s_shape = sparse.csm_properties(svar)

        c_data = structured_add_s_v_csx(s_val, s_ind, s_ptr, dvar)

        return [CSx(c_data, s_ind, s_ptr, s_shape)]

    return False


register_specialize(local_structured_add_s_v, "cxx_only")


class SamplingDotCSR(_NoPythonCOp):
    r"""
    An operator optimized for calculating the dot product :math:`x y^\top = z`
    when one only wants to calculate a subset of :math:`z`.

    This is equivalent to :math:`p \circ (x \cdot y^\top)` where :math:`\circ` is
    the element-wise product, :math:`x` and :math:`y` operands of the dot
    product, and :math:`p` is a matrix that contains 1 when the corresponding
    element of :math:`z` should be calculated and 0 when it shouldn't. Note
    that `SamplingDot` has a different interface than ``dot`` because
    `SamplingDot` requires :math:`x` to be a :math:`m \times k` matrix while
    :math:`y` is a :math:`n \times k` matrix instead of the usual :math:``k
    \times n` matrix.

    Notes
    -----
    It will work if the pattern is not binary value, but if the
    pattern doesn't have a high sparsity proportion it will be slower
    then a more optimized dot followed by a normal element-wise
    multiplication.

    If we have the input of mixed dtype, we insert cast element-wise
    in the graph to be able to call BLAS function as they don't
    allow mixed dtype.

    This `Op` is used as a rewritten form of `SamplingDot`.

    """

    __props__ = ()

    def make_node(self, x, y, p_data, p_ind, p_ptr, p_ncols):
        """

        Parameters
        ----------
        x
            Tensor matrix.
        y
            Tensor matrix.
        p_data
            Sparse matrix data.
        p_ind
            Sparse matrix indices.
        p_ptr
            Sparse matric indptr.
        p_ncols
            Sparse matrix number of columns.

        Returns
        -------
        A dense matrix containing the dot product of :math:`x` by :math:`y^\top` only
        where :math:`p` is 1.

        """
        x = as_tensor_variable(x)
        y = as_tensor_variable(y)
        p_data = as_tensor_variable(p_data)
        p_ind = as_tensor_variable(p_ind)
        p_ptr = as_tensor_variable(p_ptr)
        p_ncols = as_tensor_variable(p_ncols)

        assert p_ncols.dtype == "int32"

        dtype_out = ps.upcast(x.type.dtype, y.type.dtype, p_data.type.dtype)
        dot_out = ps.upcast(x.type.dtype, y.type.dtype)

        # We call blas ?dot function that take only param of the same type
        x = cast(x, dot_out)
        y = cast(y, dot_out)

        return Apply(
            self,
            [x, y, p_data, p_ind, p_ptr, p_ncols],
            [
                tensor(dtype=dtype_out, shape=(None,)),
                tensor(dtype=p_ind.type.dtype, shape=(None,)),
                tensor(dtype=p_ptr.type.dtype, shape=(None,)),
            ],
        )

    def c_code_cache_version(self):
        return (4, blas.blas_header_version())

    def c_support_code(self, **kwargs):
        return blas.blas_header_text()

    def c_libraries(self, **kwargs):
        return blas.ldflags()

    def c_compile_args(self, **kwargs):
        return blas.ldflags(libs=False, flags=True)

    def c_lib_dirs(self, **kwargs):
        return blas.ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self, **kwargs):
        return blas.ldflags(libs=False, include_dir=True)

    def c_code(self, node, name, inputs, outputs, sub):
        x, y, p_data, p_ind, p_ptr, p_ncols = inputs
        z_data, z_ind, z_ptr = outputs
        if node.inputs[0].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for x")
        if node.inputs[1].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for y")
        if node.inputs[2].type.dtype in ("complex64", "complex128"):
            raise NotImplementedError("Complex types are not supported for pattern")

        dot_out = ps.upcast(node.inputs[0].type.dtype, node.inputs[1].type.dtype)

        if dot_out == "float32":
            conv_type = "float"
            cdot = "sdot_"
        else:
            conv_type = "double"
            cdot = "ddot_"

        # retrieve dtype number
        typenum_x = node.inputs[0].type.dtype_specs()[2]
        typenum_y = node.inputs[1].type.dtype_specs()[2]
        typenum_p = node.inputs[2].type.dtype_specs()[2]
        typenum_zd = TensorType(node.outputs[0].dtype, []).dtype_specs()[2]
        typenum_zi = TensorType(node.outputs[1].dtype, []).dtype_specs()[2]
        typenum_zp = TensorType(node.outputs[2].dtype, []).dtype_specs()[2]

        fail = sub["fail"]
        rval = f"""
        if (PyArray_NDIM({x}) != 2) {{
PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 2"); {fail};}}
        if (PyArray_NDIM({y}) != 2) {{
PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 2"); {fail};}}

        if (PyArray_TYPE({x}) != {typenum_x}) {{
            PyErr_SetString(PyExc_NotImplementedError,
                            "Invalid type for x");
            {fail};}}

        if (PyArray_TYPE({y}) != {typenum_y}) {{
            PyErr_SetString(PyExc_NotImplementedError,
                            "Invalid type for y");
            {fail};}}

        if (PyArray_TYPE({p_data}) != {typenum_p}) {{
            PyErr_SetString(PyExc_NotImplementedError,
                            "Invalid type for pattern");
            {fail};}}

        if (PyArray_DIMS({x})[1] != PyArray_DIMS({y})[1]) {{
            PyErr_SetString(PyExc_NotImplementedError,
              "x's number of columns doesn't match y's rows! Note: sampling_dot is different from dot because y is assumed to be transposed.");
            {fail};}}

        if (PyArray_DIMS({y})[0] != ((npy_int32 *)PyArray_DATA({p_ncols}))[0] ||
            PyArray_DIMS({x})[0] != (PyArray_DIMS({p_ptr})[0] - 1))
        {{PyErr_SetString(PyExc_NotImplementedError,
        "The dimension of the pattern and the output must match"); {fail};}}

        // Allocate output
        if (!{z_data}
            || (PyArray_DIMS({z_data})[0] != PyArray_DIMS({p_data})[0])
            || (PyArray_TYPE({z_data}) != {typenum_zd})
            || !(PyArray_ISCONTIGUOUS({z_data})))
         {{
            {{Py_XDECREF({z_data});}}
            npy_intp dims[] = {{0}};
            dims[0] = PyArray_DIMS({p_data})[0];
            {z_data} = (PyArrayObject*) PyArray_SimpleNew(1, dims,
                                                            {typenum_zd});
        }}
        if (!{z_ind}
            || (PyArray_DIMS({z_ind})[0] != PyArray_DIMS({p_ind})[0])
            || (PyArray_TYPE({z_ind}) != {typenum_zi})
            || !(PyArray_ISCONTIGUOUS({z_ind})))
        {{
            {{Py_XDECREF({z_ind});}}
            npy_intp dims[] = {{0}};
            dims[0] = PyArray_DIMS({p_ind})[0];
            {z_ind} = (PyArrayObject*) PyArray_SimpleNew(1, dims,
                                                           {typenum_zi});
        }}
        if (!{z_ptr}
            || (PyArray_DIMS({z_ptr})[0] != PyArray_DIMS({p_ptr})[0])
            || (PyArray_TYPE({z_ptr}) != {typenum_zp})
            || !(PyArray_ISCONTIGUOUS({z_ptr})))
        {{
            {{Py_XDECREF({z_ptr});}}
            npy_intp dims[] = {{0}};
            dims[0] = PyArray_DIMS({p_ptr})[0];
            {z_ptr} = (PyArrayObject*) PyArray_SimpleNew(1, dims,
                                                           {typenum_zp});
        }}

        {{
            // Product of MxK and NxK, output MxN
            npy_intp M = PyArray_DIMS({x})[0];
            npy_intp N = PyArray_DIMS({y})[0];
            npy_intp K = PyArray_DIMS({y})[1];

            // pointers to access actual data in the arrays passed as params.
            const dtype_{x}* __restrict__ Dx = (dtype_{x}*)PyArray_DATA({x});
            const dtype_{y}* __restrict__ Dy = (dtype_{y}*)PyArray_DATA({y});
            const dtype_{p_data}* __restrict__ Dpd = (dtype_{p_data}*)PyArray_DATA({p_data});
            const dtype_{p_ind}* __restrict__ Dpi = (dtype_{p_ind}*)PyArray_DATA({p_ind});
            const dtype_{p_ptr}* __restrict__ Dpp = (dtype_{p_ptr}*)PyArray_DATA({p_ptr});
            dtype_{z_data}* __restrict__ Dzd = (dtype_{z_data}*)PyArray_DATA({z_data});
            dtype_{z_ind}* __restrict__ Dzi = (dtype_{z_ind}*)PyArray_DATA({z_ind});
            dtype_{z_ptr}* __restrict__ Dzp = (dtype_{z_ptr}*)PyArray_DATA({z_ptr});

            const npy_intp Sdx = PyArray_STRIDES({x})[1]/PyArray_DESCR({x})->elsize;
            const npy_intp Sdy = PyArray_STRIDES({y})[1]/PyArray_DESCR({y})->elsize;
            const npy_intp Sdpd = PyArray_STRIDES({p_data})[0] / PyArray_DESCR({p_data})->elsize;
            const npy_intp Sdpi = PyArray_STRIDES({p_ind})[0] / PyArray_DESCR({p_ind})->elsize;
            const npy_intp Sdpp = PyArray_STRIDES({p_ptr})[0] / PyArray_DESCR({p_ptr})->elsize;
            const npy_intp Sdzd = PyArray_STRIDES({z_data})[0] / PyArray_DESCR({z_data})->elsize;
            const npy_intp Sdzi = PyArray_STRIDES({z_ind})[0] / PyArray_DESCR({z_ind})->elsize;
            const npy_intp Sdzp = PyArray_STRIDES({z_ptr})[0] / PyArray_DESCR({z_ptr})->elsize;

            memcpy(Dzi, Dpi, PyArray_DIMS({p_ind})[0]*sizeof(dtype_{p_ind}));
            memcpy(Dzp, Dpp, PyArray_DIMS({p_ptr})[0]*sizeof(dtype_{p_ptr}));

            // blas expects ints; convert here (rather than just making K etc ints) to avoid potential overflow in the negative-stride correction
            if ((K > 0x7fffffffL)||(Sdx > 0x7fffffffL)||(Sdy > 0x7fffffffL)||(Sdx < -0x7fffffffL)||(Sdy < -0x7fffffffL))
            {{PyErr_SetString(PyExc_NotImplementedError, "array too big for BLAS (overflows int32 index)"); {fail};}}
            int K32 = K;
            int Sdx32 = Sdx;
            int Sdy32 = Sdy;

            for (npy_intp m = 0; m < M; ++m) {{
                for (npy_int32 n_idx = Dpp[m * Sdpp]; n_idx < Dpp[(m+1)*Sdpp]; ++n_idx) {{
                    const npy_int32 n = Dpi[n_idx * Sdpi]; // row index of non-null value for column K

                    const dtype_{x}* x_row = (dtype_{x}*)(PyArray_BYTES({x}) + PyArray_STRIDES({x})[0] * m);

                    const dtype_{y}* y_col = (dtype_{y}*)(PyArray_BYTES({y}) + PyArray_STRIDES({y})[0] * n);
                    // dot expects pointer to the beginning of memory arrays,
                    // so when the stride is negative, we need to get the
                    // last element
                    if (Sdx < 0)
                        x_row += (K - 1) * Sdx;
                    if (Sdy < 0)
                        y_col += (K - 1) * Sdy;

                    Dzd[n_idx * Sdzd] = Dpd[n_idx * Sdpd] * {cdot}(&K32, (const {conv_type}*)x_row, &Sdx32, (const {conv_type}*)y_col, &Sdy32);
                }}
            }}
        }}
        """

        return rval


sampling_dot_csr = SamplingDotCSR()


# register a specialization to replace SamplingDot -> SamplingDotCsr
@node_rewriter([sparse.sampling_dot])
def local_sampling_dot_csr(fgraph, node):
    if not config.blas__ldflags:
        # The C implementation of SamplingDotCsr relies on BLAS routines
        return
    if node.op == sparse.sampling_dot:
        x, y, p = node.inputs
        if p.type.format == "csr":
            p_data, p_ind, p_ptr, p_shape = sparse.csm_properties(p)

            z_data, z_ind, z_ptr = sampling_dot_csr(
                x, y, p_data, p_ind, p_ptr, p_shape[1]
            )
            # This is a hack that works around some missing `Type`-related
            # static shape narrowing.  More specifically,
            # `TensorType.convert_variable` currently won't combine the static
            # shape information from `old_out.type` and `new_out.type`, only
            # the broadcast patterns, and, since `CSR.make_node` doesn't do
            # that either, we use `specify_shape` to produce an output `Type`
            # with the same level of static shape information as the original
            # `old_out`.
            old_out = node.outputs[0]
            new_out = specify_shape(
                sparse.CSR(z_data, z_ind, z_ptr, p_shape), shape(old_out)
            )
            return [new_out]
    return False


register_specialize(local_sampling_dot_csr, "cxx_only", name="local_sampling_dot_csr")
