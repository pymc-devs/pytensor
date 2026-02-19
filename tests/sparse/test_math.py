import time
from itertools import product

import numpy as np
import pytest
import scipy.sparse as scipy_sparse

import pytensor
import pytensor.sparse.math as psm
import pytensor.tensor as pt
from pytensor.compile import get_default_mode
from pytensor.configdefaults import config
from pytensor.link.numba import NumbaLinker
from pytensor.scalar import upcast
from pytensor.sparse.basic import (
    CSR,
    CSMProperties,
    SparseTensorType,
    _is_dense_variable,
    _is_sparse,
    _is_sparse_variable,
    _mtypes,
    all_dtypes,
    as_sparse_variable,
    complex_dtypes,
    csc_matrix,
    csr_matrix,
    float_dtypes,
    sparse_formats,
)
from pytensor.sparse.math import (
    Dot,
    SamplingDot,
    StructuredDot,
    TrueDot,
    Usmm,
    add,
    ge,
    gt,
    le,
    lt,
    mul_s_v,
    multiply,
    sampling_dot,
    structured_add_s_v,
    structured_dot,
    true_dot,
)
from pytensor.sparse.rewriting import UsmmCscDense
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.subtensor import Subtensor
from pytensor.tensor.type import (
    TensorType,
    matrix,
    scalar,
    vector,
)
from tests import unittest_tools as utt
from tests.sparse.test_basic import (
    as_sparse_format,
    random_lil,
    sparse_random_inputs,
    verify_grad_sparse,
)


def eval_outputs(outputs):
    return pytensor.function([], outputs)()[0]


class TestAddMul:
    def test_AddSS(self):
        self._testSS(add)

    def test_AddSD(self):
        self._testSD(add)

    def test_AddDS(self):
        self._testDS(add)

    def test_MulSS(self):
        self._testSS(
            multiply,
            np.array([[1.0, 0], [3, 0], [0, 6]]),
            np.array([[1.0, 2], [3, 0], [0, 6]]),
        )

    def test_MulSD(self):
        self._testSD(
            multiply,
            np.array([[1.0, 0], [3, 0], [0, 6]]),
            np.array([[1.0, 2], [3, 0], [0, 6]]),
        )

    def test_MulDS(self):
        self._testDS(
            multiply,
            np.array([[1.0, 0], [3, 0], [0, 6]]),
            np.array([[1.0, 2], [3, 0], [0, 6]]),
        )

    def _testSS(self, op, array1=None, array2=None):
        if array1 is None:
            array1 = np.array([[1.0, 0], [3, 0], [0, 6]])
        if array2 is None:
            array2 = np.asarray([[0, 2.0], [0, 4], [5, 0]])

        for mtype1, mtype2 in product(_mtypes, _mtypes):
            for dtype1, dtype2 in [
                ("float64", "int8"),
                ("int8", "float64"),
                ("float64", "float64"),
            ]:
                a = mtype1(array1).astype(dtype1)
                aR = as_sparse_variable(a)
                assert aR.data is not a
                assert _is_sparse(a)
                assert _is_sparse_variable(aR)

                b = mtype2(array2).astype(dtype2)
                bR = as_sparse_variable(b)
                assert bR.data is not b
                assert _is_sparse(b)
                assert _is_sparse_variable(bR)

                apb = op(aR, bR)
                assert _is_sparse_variable(apb)

                assert apb.type.format == aR.type.format, apb.type.format

                val = eval_outputs([apb])
                assert val.shape == (3, 2)
                if op is add:
                    assert np.all(val.todense() == array1 + array2)
                    if dtype1.startswith("float") and dtype2.startswith("float"):
                        verify_grad_sparse(op, [a, b], structured=False)
                elif op is multiply:
                    assert np.all(val.todense() == array1 * array2)
                    if dtype1.startswith("float") and dtype2.startswith("float"):
                        verify_grad_sparse(op, [a, b], structured=False)

    def _testSD(self, op, array1=None, array2=None):
        if array1 is None:
            array1 = np.array([[1.0, 0], [3, 0], [0, 6]])
        if array2 is None:
            array2 = np.asarray([[0, 2.0], [0, 4], [5, 0]])

        for mtype in _mtypes:
            for a in [
                np.array(array1),
                pt.as_tensor_variable(array1),
                pytensor.shared(array1),
            ]:
                for dtype1, dtype2 in [
                    ("float64", "int8"),
                    ("int8", "float64"),
                    # Needed to test the grad
                    ("float32", "float64"),
                ]:
                    a = a.astype(dtype1)
                    b = mtype(array2).astype(dtype2)
                    bR = as_sparse_variable(b)
                    assert bR.data is not b  # constants are copied
                    assert _is_sparse(b)
                    assert _is_sparse_variable(bR)

                    apb = op(a, bR)

                    val = eval_outputs([apb])
                    assert val.shape == (3, 2)
                    if op is add:
                        assert _is_dense_variable(apb)
                        assert np.all(val == array1 + b)
                        ans = np.array([[1.0, 2], [3, 4], [5, 6]])
                        assert np.all(val == ans)
                        if hasattr(a, "owner") and a.owner is not None:
                            continue
                        if dtype1.startswith("float") and dtype2.startswith("float"):
                            verify_grad_sparse(op, [a, b], structured=True)
                    elif op is multiply:
                        assert _is_sparse_variable(apb)
                        assert np.all(val.todense() == b.multiply(array1))
                        assert np.all(
                            val.todense() == np.array([[1, 0], [9, 0], [0, 36]])
                        )
                        if hasattr(a, "owner") and a.owner is not None:
                            continue
                        if dtype1.startswith("float") and dtype2.startswith("float"):
                            verify_grad_sparse(op, [a, b], structured=False)

    def _testDS(self, op, array1=None, array2=None):
        if array1 is None:
            array1 = np.array([[1.0, 0], [3, 0], [0, 6]])
        if array2 is None:
            array2 = np.asarray([[0, 2.0], [0, 4], [5, 0]])

        for mtype in _mtypes:
            for b in [
                np.asarray(array2),
                pt.as_tensor_variable(array2),
                pytensor.shared(array2),
            ]:
                for dtype1, dtype2 in [
                    ("float64", "int8"),
                    ("int8", "float64"),
                ]:
                    a = mtype(array1).astype(dtype1)
                    aR = as_sparse_variable(a)
                    assert aR.data is not a
                    assert _is_sparse(a)
                    assert _is_sparse_variable(aR)
                    b = b.astype(dtype2)

                    apb = op(aR, b)

                    val = eval_outputs([apb])
                    assert val.shape == (3, 2)
                    if op is add:
                        assert _is_dense_variable(apb)
                        assert np.all(val == a + array2)
                        ans = np.array([[1.0, 2], [3, 4], [5, 6]])
                        assert np.all(val == ans)
                        if dtype1.startswith("float") and dtype2.startswith("float"):
                            verify_grad_sparse(op, [a, b], structured=True)
                    elif op is multiply:
                        assert _is_sparse_variable(apb)
                        ans = np.array([[1, 0], [9, 0], [0, 36]])
                        assert np.all(val.todense() == (a.multiply(array2)))
                        assert np.all(val.todense() == ans)
                        if dtype1.startswith("float") and dtype2.startswith("float"):
                            verify_grad_sparse(op, [a, b], structured=False)


class TestComparison:
    # took from tensor basic_test.py
    def _rand_ranged(self, min, max, shape):
        return np.asarray(
            np.random.random(shape) * (max - min) + min, dtype=config.floatX
        )

    tests = [
        lambda x, y: x > y,
        lambda x, y: x < y,
        lambda x, y: x >= y,
        lambda x, y: x <= y,
    ]

    testsDic = {
        gt: lambda x, y: x > y,
        lt: lambda x, y: x < y,
        ge: lambda x, y: x >= y,
        le: lambda x, y: x <= y,
    }

    def __generalized_ss_test(self, pytensorp, symbolicType, testOp, scipyType):
        x = symbolicType()
        y = symbolicType()

        op = pytensorp(x, y)
        f = pytensor.function([x, y], op)

        m1 = scipyType(random_lil((10, 40), config.floatX, 3))
        m2 = scipyType(random_lil((10, 40), config.floatX, 3))

        assert np.array_equal(f(m1, m2).data, testOp(m1, m2).data)

    def __generalized_sd_test(self, pytensorp, symbolicType, testOp, scipyType):
        x = symbolicType()
        y = matrix()

        op = pytensorp(x, y)
        f = pytensor.function([x, y], op)

        m1 = scipyType(random_lil((10, 40), config.floatX, 3))
        m2 = self._rand_ranged(1000, -1000, [10, 40])

        assert np.array_equal(f(m1, m2).data, testOp(m1, m2).data)

    def __generalized_ds_test(self, pytensorp, symbolicType, testOp, scipyType):
        x = symbolicType()
        y = matrix()

        op = pytensorp(y, x)
        f = pytensor.function([y, x], op)

        m1 = scipyType(random_lil((10, 40), config.floatX, 3))
        m2 = self._rand_ranged(1000, -1000, [10, 40])

        assert np.array_equal(f(m2, m1).data, testOp(m2, m1).data)

    def test_ss_csr_comparison(self):
        for op in self.tests:
            self.__generalized_ss_test(op, csr_matrix, op, scipy_sparse.csr_matrix)

    def test_ss_csc_comparison(self):
        for op in self.tests:
            self.__generalized_ss_test(op, csc_matrix, op, scipy_sparse.csc_matrix)

    def test_sd_csr_comparison(self):
        for op in self.tests:
            self.__generalized_sd_test(op, csr_matrix, op, scipy_sparse.csr_matrix)

    def test_sd_csc_comparison(self):
        for op in self.tests:
            self.__generalized_sd_test(op, csc_matrix, op, scipy_sparse.csc_matrix)

    def test_ds_csc_comparison(self):
        for op in self.testsDic:
            self.__generalized_ds_test(
                op,
                csc_matrix,
                self.testsDic[op],
                scipy_sparse.csc_matrix,
            )

    def test_ds_csr_comparison(self):
        for op in self.testsDic:
            self.__generalized_ds_test(
                op,
                csr_matrix,
                self.testsDic[op],
                scipy_sparse.csr_matrix,
            )

    def test_equality_case(self):
        # Test assuring normal behaviour when values in the matrices are equal
        x = csc_matrix()
        y = matrix()

        m1 = scipy_sparse.csc_matrix((2, 2), dtype=pytensor.config.floatX)
        m2 = np.asarray([[0, 0], [0, 0]], dtype=pytensor.config.floatX)

        for func in self.testsDic:
            op = func(y, x)
            f = pytensor.function([y, x], op)
            assert np.array_equal(f(m2, m1), self.testsDic[func](m2, m1))


class TestStructuredDot:
    def test_structureddot_csc_grad(self):
        spmat = scipy_sparse.csc_matrix(random_lil((4, 3), "float32", 3))
        mat = np.asarray(np.random.standard_normal((3, 2)), "float32")
        verify_grad_sparse(structured_dot, [spmat, mat], structured=True)

        def buildgraph_T(spmat, mat):
            return structured_dot(mat.T, spmat.T)

        verify_grad_sparse(buildgraph_T, [spmat, mat], structured=True)

    def test_structureddot_csr_grad(self):
        spmat = scipy_sparse.csr_matrix(random_lil((4, 3), "float64", 3))
        mat = np.asarray(np.random.standard_normal((3, 2)), "float64")
        verify_grad_sparse(structured_dot, [spmat, mat], structured=True)

        def buildgraph_T(spmat, mat):
            return structured_dot(mat.T, spmat.T)

        verify_grad_sparse(buildgraph_T, [spmat, mat], structured=True)

    def test_upcast(self):
        typenames = (
            "float32",
            "int64",
            "int8",
            "int32",
            "int16",
            "float64",
            "complex64",
            "complex128",
        )
        for dense_dtype in typenames:
            for sparse_dtype in typenames:
                correct_dtype = upcast(sparse_dtype, dense_dtype)
                a = SparseTensorType("csc", dtype=sparse_dtype)()
                b = matrix(dtype=dense_dtype)
                d = structured_dot(a, b)
                assert d.type.dtype == correct_dtype

                f = pytensor.function([a, b], d)

                M, N, K, nnz = (4, 3, 5, 3)
                spmat = scipy_sparse.csc_matrix(random_lil((M, N), sparse_dtype, nnz))
                spmat.dtype = np.dtype(sparse_dtype)
                mat = np.asarray(
                    np.random.standard_normal((N, K)) * 9, dtype=dense_dtype
                )
                pytensor_result = f(spmat, mat)
                scipy_result = spmat * mat
                assert pytensor_result.shape == scipy_result.shape
                assert pytensor_result.dtype == scipy_result.dtype
                utt.assert_allclose(scipy_result, pytensor_result)

    @pytest.mark.xfail(
        reason="The optimization from structured_dot -> structured_dot_csc is currently disabled",
    )
    def test_opt_unpack(self):
        """
        Test that a graph involving structured_dot(assembled_csc_matrix) is optimized to be just a structured_dot_csc
        Op and no assembly of a csc_matrix.
        """

        kerns = TensorType(dtype="int64", shape=(None,))("kerns")
        spmat = scipy_sparse.random_array(shape=(4, 6), density=0.2, format="csc")

        images = TensorType(dtype="float32", shape=(None, None))("images")

        cscmat = pytensor.sparse.CSC(
            kerns, spmat.indices[: spmat.size], spmat.indptr, spmat.shape
        )
        f = pytensor.function([kerns, images], structured_dot(cscmat, images.T))

        sdcscpresent = False
        for node in f.maker.fgraph.toposort():
            # print node.op
            assert not isinstance(node.op, pytensor.sparse.CSM)
            assert not isinstance(node.op, CSMProperties)
            if isinstance(
                f.maker.fgraph.toposort()[1].op,
                pytensor.sparse.rewriting.StructuredDotCSC,
            ):
                sdcscpresent = True
        assert sdcscpresent

        kernvals = np.array(spmat.data[: spmat.size])
        bsize = 3
        imvals = 1.0 * np.array(
            np.arange(bsize * spmat.shape[1]).reshape(bsize, spmat.shape[1]),
            dtype="float32",
        )
        f(kernvals, imvals)

    @pytest.mark.parametrize(
        "sparse_format_a",
        (
            "csc",
            "csr",
            pytest.param(
                "bsr",
                marks=pytest.mark.xfail(
                    isinstance(get_default_mode().linker, NumbaLinker),
                    reason="Numba does not support bsr",
                ),
            ),
        ),
    )
    @pytest.mark.parametrize(
        "sparse_format_b",
        (
            "csc",
            "csr",
            pytest.param(
                "bsr",
                marks=pytest.mark.xfail(
                    isinstance(get_default_mode().linker, NumbaLinker),
                    reason="Numba does not support bsr",
                ),
            ),
        ),
    )
    def test_dot_sparse_sparse(self, sparse_format_a, sparse_format_b):
        sparse_dtype = "float64"
        sp_mat = {
            "csc": scipy_sparse.csc_matrix,
            "csr": scipy_sparse.csr_matrix,
            "bsr": scipy_sparse.csr_matrix,
        }
        a = SparseTensorType(sparse_format_a, dtype=sparse_dtype)()
        b = SparseTensorType(sparse_format_b, dtype=sparse_dtype)()
        d = pt.dot(a, b)
        f = pytensor.function([a, b], d)
        for M, N, K, nnz in [
            (4, 3, 2, 3),
            (40, 30, 20, 3),
            (40, 30, 20, 30),
            (400, 3000, 200, 6000),
        ]:
            a_val = sp_mat[sparse_format_a](random_lil((M, N), sparse_dtype, nnz))
            b_val = sp_mat[sparse_format_b](random_lil((N, K), sparse_dtype, nnz))
            f(a_val, b_val)  # TODO: Test something

    def test_tensor_dot_types(self):
        x = csc_matrix("x")
        x_d = pt.matrix("x_d")
        y = csc_matrix("y")

        res = pt.dot(x, y)
        op_types = {
            type(n.op) for n in pytensor.graph.traversal.applys_between([x, y], [res])
        }
        assert StructuredDot in op_types
        assert pt.math.Dot not in op_types

        res = pt.dot(x_d, y)
        op_types = {
            type(n.op) for n in pytensor.graph.traversal.applys_between([x, y], [res])
        }
        assert StructuredDot in op_types
        assert pt.math.Dot not in op_types

        res = pt.dot(x, x_d)
        op_types = {
            type(n.op) for n in pytensor.graph.traversal.applys_between([x, y], [res])
        }
        assert StructuredDot in op_types
        assert pt.math.Dot not in op_types

        res = pt.dot(pt.second(1, x), y)
        op_types = {
            type(n.op) for n in pytensor.graph.traversal.applys_between([x, y], [res])
        }
        assert StructuredDot in op_types
        assert pt.math.Dot not in op_types

    def test_csc_correct_output_faster_than_scipy(self):
        sparse_dtype = "float64"
        dense_dtype = "float64"

        a = SparseTensorType("csc", dtype=sparse_dtype)()
        b = matrix(dtype=dense_dtype)
        d = pt.dot(a, b)
        f = pytensor.function([a, b], d)

        for M, N, K, nnz in [
            (4, 3, 2, 3),
            (40, 30, 20, 3),
            (40, 30, 20, 30),
            (400, 3000, 200, 6000),
        ]:
            spmat = scipy_sparse.csc_matrix(random_lil((M, N), sparse_dtype, nnz))
            mat = np.asarray(np.random.standard_normal((N, K)), dense_dtype)
            pytensor_times = []
            scipy_times = []
            for i in range(5):
                t0 = time.perf_counter()
                pytensor_result = f(spmat, mat)
                t1 = time.perf_counter()
                scipy_result = spmat * mat
                t2 = time.perf_counter()

                pytensor_times.append(t1 - t0)
                scipy_times.append(t2 - t1)

            pytensor_time = np.min(pytensor_times)
            scipy_time = np.min(scipy_times)

            # speedup = scipy_time / pytensor_time
            # print scipy_times
            # print pytensor_times
            # print ('M=%(M)s N=%(N)s K=%(K)s nnz=%(nnz)s pytensor_time'
            #       '=%(pytensor_time)s speedup=%(speedup)s') % locals()

            # fail if PyTensor is slower than scipy by more than a certain amount
            overhead_tol = 0.003  # seconds overall
            overhead_rtol = 1.2  # times as long
            utt.assert_allclose(scipy_result, pytensor_result)
            if pytensor.config.mode == "FAST_RUN" and pytensor.config.cxx:
                assert pytensor_time <= overhead_rtol * scipy_time + overhead_tol

    def test_csr_correct_output_faster_than_scipy(self):
        sparse_dtype = "float32"
        dense_dtype = "float32"

        a = SparseTensorType("csr", dtype=sparse_dtype)()
        b = matrix(dtype=dense_dtype)
        d = pt.dot(a, b)
        f = pytensor.function([a, b], d)

        for M, N, K, nnz in [
            (4, 3, 2, 3),
            (40, 30, 20, 3),
            (40, 30, 20, 30),
            (400, 3000, 200, 6000),
        ]:
            spmat = scipy_sparse.csr_matrix(random_lil((M, N), sparse_dtype, nnz))
            mat = np.asarray(np.random.standard_normal((N, K)), dense_dtype)
            t0 = time.perf_counter()
            pytensor_result = f(spmat, mat)
            t1 = time.perf_counter()
            scipy_result = spmat * mat
            t2 = time.perf_counter()

            pytensor_time = t1 - t0
            scipy_time = t2 - t1
            # print 'pytensor took', pytensor_time,
            # print 'scipy took', scipy_time
            overhead_tol = 0.002  # seconds
            overhead_rtol = 1.1  # times as long
            utt.assert_allclose(scipy_result, pytensor_result)
            if pytensor.config.mode == "FAST_RUN" and pytensor.config.cxx:
                assert pytensor_time <= overhead_rtol * scipy_time + overhead_tol, (
                    pytensor_time,
                    overhead_rtol * scipy_time + overhead_tol,
                    scipy_time,
                    overhead_rtol,
                    overhead_tol,
                )


class TestDots(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        x_size = (10, 100)
        y_size = (100, 1000)

        self.x_csr = scipy_sparse.csr_matrix(
            np.random.binomial(1, 0.5, x_size), dtype=pytensor.config.floatX
        )
        self.x_csc = scipy_sparse.csc_matrix(
            np.random.binomial(1, 0.5, x_size), dtype=pytensor.config.floatX
        )
        self.y = np.asarray(
            np.random.uniform(-1, 1, y_size), dtype=pytensor.config.floatX
        )
        self.y_csr = scipy_sparse.csr_matrix(
            np.random.binomial(1, 0.5, y_size), dtype=pytensor.config.floatX
        )
        self.y_csc = scipy_sparse.csc_matrix(
            np.random.binomial(1, 0.5, y_size), dtype=pytensor.config.floatX
        )
        self.v_10 = np.asarray(
            np.random.uniform(-1, 1, 10), dtype=pytensor.config.floatX
        )
        self.v_100 = np.asarray(
            np.random.uniform(-1, 1, 100), dtype=pytensor.config.floatX
        )

    def test_csr_dense(self):
        x = csr_matrix("x")
        y = matrix("y")
        v = vector("v")

        for x, y, x_v, y_v in [
            (x, y, self.x_csr, self.y),
            (x, v, self.x_csr, self.v_100),
            (v, x, self.v_10, self.x_csr),
        ]:
            f_a = pytensor.function([x, y], psm.dot(x, y))

            def f_b(x, y):
                return x * y

            utt.assert_allclose(f_a(x_v, y_v), f_b(x_v, y_v))

            # Test infer_shape
            self._compile_and_check(
                [x, y], [psm.dot(x, y)], [x_v, y_v], (Dot, Usmm, UsmmCscDense)
            )

    def test_csc_dense(self):
        x = csc_matrix("x")
        y = matrix("y")
        v = vector("v")

        for x, y, x_v, y_v in [
            (x, y, self.x_csc, self.y),
            (x, v, self.x_csc, self.v_100),
            (v, x, self.v_10, self.x_csc),
        ]:
            f_a = pytensor.function([x, y], psm.dot(x, y))

            def f_b(x, y):
                return x * y

            utt.assert_allclose(f_a(x_v, y_v), f_b(x_v, y_v))

            self._compile_and_check(
                [x, y], [psm.dot(x, y)], [x_v, y_v], (Dot, Usmm, UsmmCscDense)
            )

    def test_sparse_sparse(self):
        for d1, d2 in [
            ("float32", "float32"),
            ("float32", "float64"),
            ("float64", "float32"),
            ("float64", "float64"),
            ("float32", "int16"),
            ("float32", "complex64"),
        ]:
            for x_f, y_f in [
                ("csc", "csc"),
                ("csc", "csr"),
                ("csr", "csc"),
                ("csr", "csr"),
            ]:
                x = SparseTensorType(format=x_f, dtype=d1)("x")
                y = SparseTensorType(format=x_f, dtype=d2)("x")

                def f_a(x, y):
                    return x * y

                f_b = pytensor.function([x, y], psm.dot(x, y))

                vx = getattr(self, "x_" + x_f).astype(d1)
                vy = getattr(self, "y_" + y_f).astype(d2)
                utt.assert_allclose(f_a(vx, vy).toarray(), f_b(vx, vy))

                f_a = pytensor.function([x, y], psm.dot(x, y).shape)

                def f_b2(x, y):
                    return (x * y).shape

                assert np.all(f_a(vx, vy) == f_b2(vx, vy))
                topo = f_a.maker.fgraph.toposort()
                assert not any(
                    isinstance(node.op, Dot | Usmm | UsmmCscDense) for node in topo
                )

    def test_int32_dtype(self):
        size = 9
        intX = "int32"

        C = matrix("C", dtype=intX)
        I = matrix("I", dtype=intX)

        fI = I.flatten()
        data = pt.ones_like(fI)
        indptr = pt.arange(data.shape[0] + 1, dtype="int32")

        m1 = CSR(data, fI, indptr, (8, size))
        m2 = psm.dot(m1, C)
        y = m2.reshape(shape=(2, 4, 9), ndim=3)

        f = pytensor.function(inputs=[I, C], outputs=y)
        i = np.asarray([[4, 3, 7, 7], [2, 8, 4, 5]], dtype=intX)
        a = np.asarray(
            np.random.default_rng().integers(0, 100, (size, size)), dtype=intX
        )
        f(i, a)

    def test_csr_dense_grad(self):
        # shortcut: testing csc in float32, testing csr in float64

        # allocate a random sparse matrix
        spmat = scipy_sparse.csr_matrix(random_lil((4, 3), "float64", 3))
        mat = np.asarray(np.random.standard_normal((2, 4)), "float64")

        def buildgraph_T(mat):
            return Dot()(mat, spmat)

        utt.verify_grad(buildgraph_T, [mat])


class TestUsmm:
    def setup_method(self):
        x_size = (10, 100)
        y_size = (100, 200)
        z_size = (x_size[0], y_size[1])

        self.rng = np.random.default_rng(seed=utt.fetch_seed())
        self.x = np.asarray(
            self.rng.binomial(1, 0.5, x_size), dtype=pytensor.config.floatX
        )
        self.y = np.asarray(
            self.rng.uniform(-1, 1, y_size), dtype=pytensor.config.floatX
        )
        self.z = np.asarray(
            self.rng.uniform(-1, 1, z_size), dtype=pytensor.config.floatX
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("dtype1", ["float32", "float64", "int16", "complex64"])
    @pytest.mark.parametrize("dtype2", ["float32", "float64", "int16", "complex64"])
    @pytest.mark.parametrize("can_inplace", [False, True])
    @pytest.mark.parametrize("format1", ["dense", "csc", "csr"])
    @pytest.mark.parametrize("format2", ["dense", "csc", "csr"])
    def test_basic(self, dtype1, dtype2, can_inplace, format1, format2):
        def mat(format, name, dtype):
            if format == "dense":
                return matrix(name, dtype=dtype)
            else:
                return pytensor.sparse.matrix(format, name, dtype=dtype)

        if format1 == "dense" and format2 == "dense":
            pytest.skip("Skipping dense-dense case")

        dtype3 = upcast(dtype1, dtype2)
        dtype4 = dtype3 if can_inplace else "int32"
        inplace = can_inplace

        x = mat(format1, "x", dtype1)
        y = mat(format2, "y", dtype2)

        a = scalar("a", dtype=dtype3)
        z = pytensor.shared(np.asarray(self.z, dtype=dtype4).copy())

        def f_b(z, a, x, y):
            return z - a * (x * y)

        x_data = np.asarray(self.x, dtype=dtype1)
        if format1 != "dense":
            x_data = as_sparse_format(x_data, format1)
        y_data = np.asarray(self.y, dtype=dtype2)
        if format2 != "dense":
            y_data = as_sparse_format(y_data, format2)
        a_data = np.asarray(1.5, dtype=dtype3)
        z_data = np.asarray(self.z, dtype=dtype4)

        f_b_out = f_b(z_data, a_data, x_data, y_data)

        # To make it easier to check the toposort
        mode = get_default_mode().excluding("fusion")

        if inplace:
            updates = [(z, z - a * psm.dot(x, y))]
            f_a = pytensor.function([a, x, y], [], updates=updates, mode=mode)
            f_a(a_data, x_data, y_data)
            f_a_out = z.get_value(borrow=True)
        else:
            f_a = pytensor.function([a, x, y], z - a * psm.dot(x, y), mode=mode)
            f_a_out = f_a(a_data, x_data, y_data)

        # As we do a dot product of 2 vector of 100 element,
        # This mean we can have 2*100*eps abs error.
        if f_a_out.dtype in ["float64", "complex128"]:
            atol = 3e-8
            rtol = 1e-5
        else:
            atol = None
            rtol = None
        utt.assert_allclose(f_a_out, f_b_out, rtol=rtol, atol=atol)
        topo = f_a.maker.fgraph.toposort()
        up = upcast(dtype1, dtype2, dtype3, dtype4)

        fast_compile = pytensor.config.mode == "FAST_COMPILE"

        if not pytensor.config.blas__ldflags:
            # Usmm should not be inserted, because it relies on BLAS
            assert len(topo) == 4, topo
            assert isinstance(topo[0].op, psm.Dot)
            assert isinstance(topo[1].op, DimShuffle)
            assert isinstance(topo[2].op, Elemwise) and isinstance(
                topo[2].op.scalar_op, pytensor.scalar.Mul
            )
            assert isinstance(topo[3].op, Elemwise) and isinstance(
                topo[3].op.scalar_op, pytensor.scalar.Sub
            )
        elif (
            y.type.dtype == up
            and format1 == "csc"
            and format2 == "dense"
            and "cxx_only" not in f_a.maker.linker.incompatible_rewrites
            and up in ("float32", "float64")
        ):
            # The op UsmmCscDense should be inserted
            assert (
                sum(
                    isinstance(node.op, Elemwise)
                    and isinstance(node.op.scalar_op, pytensor.scalar.basic.Cast)
                    for node in topo
                )
                == len(topo) - 5
            )
            new_topo = [
                node
                for node in topo
                if not (
                    isinstance(node.op, Elemwise)
                    and isinstance(node.op.scalar_op, pytensor.scalar.basic.Cast)
                )
            ]
            topo = new_topo
            assert len(topo) == 5, topo

            # Usmm is tested at the same time in debugmode
            # Check if the optimization local_usmm and local_usmm_csx is
            # applied

            def check_once(x):
                assert sum(isinstance(n.op, x) for n in topo) == 1

            check_once(CSMProperties)
            check_once(DimShuffle)
            check_once(Subtensor)
            check_once(UsmmCscDense)
            check_once(Elemwise)
            if inplace:
                assert topo[4].op.inplace
        elif not fast_compile:
            # The op Usmm should be inserted
            assert len(topo) == 3, topo
            assert isinstance(topo[0].op, DimShuffle)
            assert topo[1].op == pytensor.tensor.neg
            assert isinstance(topo[2].op, psm.Usmm)

    @pytest.mark.parametrize(
        "params",
        [
            ("float32", "float64", "int16", "complex64", "csc", "dense"),
            ("float32", "float64", "int16", "complex64", "csr", "dense"),
        ],
    )
    def test_infer_shape(self, params):
        def mat(format, name, dtype):
            if format == "dense":
                return matrix(name, dtype=dtype)
            else:
                return pytensor.sparse.matrix(format, name, dtype=dtype)

        dtype1, dtype2, dtype3, dtype4, format1, format2 = params

        if format1 == "dense" and format2 == "dense":
            pytest.skip("Skipping dense-dense case; Usmm won't be used.")

        x = mat(format1, "x", dtype1)
        y = mat(format2, "y", dtype2)
        a = scalar("a", dtype=dtype3)
        z = pytensor.shared(np.asarray(self.z, dtype=dtype4).copy())

        def f_b(z, a, x, y):
            return z - a * (x * y)

        x_data = np.asarray(self.x, dtype=dtype1)
        if format1 != "dense":
            x_data = as_sparse_format(x_data, format1)
        y_data = np.asarray(self.y, dtype=dtype2)
        if format2 != "dense":
            y_data = as_sparse_format(y_data, format2)
        a_data = np.asarray(1.5, dtype=dtype3)
        z_data = np.asarray(self.z, dtype=dtype4)

        f_b_out = f_b(z_data, a_data, x_data, y_data)

        # To make it easier to check the toposort
        mode = pytensor.compile.mode.get_default_mode().excluding("fusion")

        # test infer_shape of Dot got applied
        f_shape = pytensor.function([a, x, y], (z - a * psm.dot(x, y)).shape, mode=mode)
        assert all(f_shape(a_data, x_data, y_data) == f_b_out.shape)
        topo = f_shape.maker.fgraph.toposort()
        assert not any(isinstance(node.op, Dot | Usmm | UsmmCscDense) for node in topo)


class TestTrueDot(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op = true_dot
        self.op_class = TrueDot

    def test_op_ss(self):
        for format in ["csc", "csr"]:
            for dtype in all_dtypes:
                variable, data = sparse_random_inputs(
                    format, shape=(10, 10), out_dtype=dtype, n=2, p=0.1
                )

                f = pytensor.function(variable, self.op(*variable))

                tested = f(*data)

                x, y = (m.toarray() for m in data)
                expected = np.dot(x, y)

                assert tested.format == format
                assert tested.dtype == expected.dtype
                tested = tested.toarray()
                utt.assert_allclose(tested, expected)

    def test_op_sd(self):
        for format in ["csc", "csr"]:
            for dtype in all_dtypes:
                variable, data = sparse_random_inputs(
                    format, shape=(10, 10), out_dtype=dtype, n=2, p=0.1
                )
                variable[1] = TensorType(dtype=dtype, shape=(None, None))()
                data[1] = data[1].toarray()

                f = pytensor.function(variable, self.op(*variable))

                tested = f(*data)
                expected = np.dot(data[0].toarray(), data[1])

                assert tested.format == format
                assert tested.dtype == expected.dtype
                tested = tested.toarray()
                utt.assert_allclose(tested, expected)

    def test_infer_shape(self):
        for format in ["csc", "csr"]:
            for dtype in all_dtypes:
                (x,), (x_value,) = sparse_random_inputs(
                    format, shape=(9, 10), out_dtype=dtype, p=0.1
                )
                (y,), (y_value,) = sparse_random_inputs(
                    format, shape=(10, 24), out_dtype=dtype, p=0.1
                )
                variable = [x, y]
                data = [x_value, y_value]
                self._compile_and_check(
                    variable, [self.op(*variable)], data, self.op_class
                )

    def test_grad(self):
        for format in ["csc", "csr"]:
            for dtype in float_dtypes:
                (_x,), (x_value,) = sparse_random_inputs(
                    format, shape=(9, 10), out_dtype=dtype, p=0.1
                )
                (_y,), (y_value,) = sparse_random_inputs(
                    format, shape=(10, 24), out_dtype=dtype, p=0.1
                )
                data = [x_value, y_value]
                verify_grad_sparse(self.op, data, structured=False)


class TestSamplingDot(utt.InferShapeTester):
    x = [matrix() for t in range(2)]
    x.append(csr_matrix())
    a = [
        np.array(
            np.random.default_rng().integers(1, 6, size=(4, 3)) - 1,
            dtype=pytensor.config.floatX,
        ),
        np.array(
            np.random.default_rng().integers(1, 6, size=(5, 3)) - 1,
            dtype=pytensor.config.floatX,
        ),
        np.array(
            np.random.default_rng().integers(1, 3, size=(4, 5)) - 1,
            dtype=pytensor.config.floatX,
        ),
    ]
    a[2] = scipy_sparse.csr_matrix(a[2])

    def setup_method(self):
        super().setup_method()
        self.op_class = SamplingDot

    def test_op(self):
        f = pytensor.function(self.x, sampling_dot(*self.x))
        tested = f(*self.a)
        x, y, p = self.a
        expected = p.multiply(np.dot(x, y.T))
        utt.assert_allclose(expected.toarray(), tested.toarray())
        assert tested.format == "csr"
        assert tested.dtype == expected.dtype

    def test_negative_stride(self):
        f = pytensor.function(self.x, sampling_dot(*self.x))
        a2 = [self.a[0][::-1, :], self.a[1][:, ::-1], self.a[2]]
        tested = f(*a2)
        x, y, p = a2
        expected = p.multiply(np.dot(x, y.T))
        utt.assert_allclose(expected.toarray(), tested.toarray())
        assert tested.format == "csr"
        assert tested.dtype == expected.dtype

    def test_infer_shape(self):
        self._compile_and_check(
            self.x,
            [sampling_dot(*self.x)],
            self.a,
            self.op_class,
            excluding=["local_sampling_dot_csr"],
        )

    def test_grad(self):
        def _helper(x, y):
            return sampling_dot(x, y, self.a[2])

        verify_grad_sparse(_helper, self.a[:2])


class TestMulSV:
    def test_mul_s_v_grad(self):
        sp_types = {"csc": scipy_sparse.csc_matrix, "csr": scipy_sparse.csr_matrix}
        for format in ("csr", "csc"):
            for dtype in ("float32", "float64"):
                spmat = sp_types[format](random_lil((4, 3), dtype, 3))
                mat = np.asarray(np.random.random(3), dtype=dtype)
                verify_grad_sparse(mul_s_v, [spmat, mat], structured=True)

    def test_mul_s_v(self):
        sp_types = {"csc": scipy_sparse.csc_matrix, "csr": scipy_sparse.csr_matrix}
        for format in ("csr", "csc"):
            for dtype in ("float32", "float64"):
                x = SparseTensorType(format, dtype=dtype)()
                y = vector(dtype=dtype)
                f = pytensor.function([x, y], mul_s_v(x, y))

                spmat = sp_types[format](random_lil((4, 3), dtype, 3))
                mat = np.asarray(np.random.random(3), dtype=dtype)

                out = f(spmat, mat)
                utt.assert_allclose(spmat.toarray() * mat, out.toarray())


class TestStructuredAddSV:
    def test_structured_add_s_v_grad(self):
        sp_types = {"csc": scipy_sparse.csc_matrix, "csr": scipy_sparse.csr_matrix}
        for format in ("csr", "csc"):
            for dtype in ("float32", "float64"):
                spmat = sp_types[format](random_lil((4, 3), dtype, 3))
                mat = np.asarray(np.random.random(3), dtype=dtype)
                verify_grad_sparse(structured_add_s_v, [spmat, mat], structured=True)

    def test_structured_add_s_v(self):
        sp_types = {"csc": scipy_sparse.csc_matrix, "csr": scipy_sparse.csr_matrix}
        for format in ("csr", "csc"):
            for dtype in ("float32", "float64"):
                x = SparseTensorType(format, dtype=dtype)()
                y = vector(dtype=dtype)
                f = pytensor.function([x, y], structured_add_s_v(x, y))

                spmat = sp_types[format](random_lil((4, 3), dtype, 3))
                spones = spmat.copy()
                spones.data = np.ones_like(spones.data)
                mat = np.asarray(np.random.random(3), dtype=dtype)

                out = f(spmat, mat)
                utt.assert_allclose(
                    spones.multiply(spmat + mat).toarray(), out.toarray()
                )


class TestAddSSData(utt.InferShapeTester):
    x = {}
    a = {}

    def setup_method(self):
        super().setup_method()
        self.op_class = psm.AddSSData

        for format in sparse_formats:
            variable = getattr(pytensor.sparse, format + "_matrix")

            a_val = np.array(
                np.random.default_rng(utt.fetch_seed()).integers(1, 4, size=(3, 4)) - 1,
                dtype=pytensor.config.floatX,
            )
            constant = as_sparse_format(a_val, format)

            self.x[format] = [variable() for t in range(2)]
            self.a[format] = [constant for t in range(2)]

    def test_op(self):
        for format in sparse_formats:
            f = pytensor.function(self.x[format], psm.add_s_s_data(*self.x[format]))

            tested = f(*self.a[format])
            expected = 2 * self.a[format][0]

            utt.assert_allclose(expected.toarray(), tested.toarray())
            assert tested.format == expected.format
            assert tested.dtype == expected.dtype

    def test_infer_shape(self):
        for format in sparse_formats:
            self._compile_and_check(
                self.x[format],
                [psm.add_s_s_data(*self.x[format])],
                self.a[format],
                self.op_class,
            )

    def test_grad(self):
        for format in sparse_formats:
            verify_grad_sparse(self.op_class(), self.a[format], structured=True)


def test_useless_conj():
    x = SparseTensorType("csr", dtype="complex128")()
    assert x.conj() is not x
    x = SparseTensorType("csr", dtype="float64")()
    assert x.conj() is x


class TestSpSum(utt.InferShapeTester):
    possible_axis = [None, 0, 1]

    def setup_method(self):
        super().setup_method()
        self.op_class = psm.SpSum
        self.op = psm.sp_sum

    @pytest.mark.parametrize("op_type", ["func", "method"])
    def test_op(self, op_type):
        for format in sparse_formats:
            for axis in self.possible_axis:
                variable, data = sparse_random_inputs(format, shape=(10, 10))

                if op_type == "func":
                    z = psm.sp_sum(variable[0], axis=axis)
                if op_type == "method":
                    z = variable[0].sum(axis=axis)

                if axis is None:
                    assert z.type.broadcastable == ()
                else:
                    assert z.type.broadcastable == (False,)

                f = pytensor.function(variable, self.op(variable[0], axis=axis))
                tested = f(*data)
                expected = data[0].todense().sum(axis).ravel()
                utt.assert_allclose(expected, tested)

    def test_infer_shape(self):
        for format in sparse_formats:
            for axis in self.possible_axis:
                variable, data = sparse_random_inputs(format, shape=(9, 10))
                self._compile_and_check(
                    variable, [self.op(variable[0], axis=axis)], data, self.op_class
                )

    def test_grad(self):
        for format in sparse_formats:
            for axis in self.possible_axis:
                for struct in [True, False]:
                    _variable, data = sparse_random_inputs(format, shape=(9, 10))
                    verify_grad_sparse(
                        self.op_class(axis=axis, sparse_grad=struct),
                        data,
                        structured=struct,
                    )


def structure_function(f, index=0):
    def structured_function(*args):
        pattern = args[index]
        evaluated = f(*args)
        evaluated[pattern == 0] = 0
        return evaluated

    return structured_function


def elemwise_checker(
    op, expected_f, gap=None, test_dtypes=None, grad_test=True, name=None, gap_grad=None
):
    if test_dtypes is None:
        test_dtypes = all_dtypes

    class TestElemwise:
        def setup_method(self):
            super().setup_method()
            self.op = op
            self.expected_f = expected_f
            self.gap = gap
            if gap_grad is not None:
                self.gap_grad = gap_grad
            else:
                self.gap_grad = gap
            assert eval(self.__class__.__name__) is self.__class__

        def test_op(self):
            for format in sparse_formats:
                for dtype in test_dtypes:
                    if dtype == "int8" or dtype == "uint8":
                        continue

                    if dtype.startswith("uint"):
                        if self.gap and len(self.gap) == 2 and self.gap[0] < 0:
                            if self.gap[1] >= 1:
                                self.gap = (0, self.gap[1])
                            else:
                                raise TypeError(
                                    "Gap not suitable for", dtype, self.__name__
                                )

                    variable, data = sparse_random_inputs(
                        format, shape=(4, 7), out_dtype=dtype, gap=self.gap
                    )

                    f = pytensor.function(variable, self.op(*variable))

                    tested = f(*data)
                    data = [m.toarray() for m in data]
                    expected = self.expected_f(*data)

                    assert tested.format == format
                    tested = tested.toarray()

                    try:
                        utt.assert_allclose(expected, tested)
                    except AssertionError:
                        raise AssertionError(self.__name__)

                for dtype in ["int8", "uint8"]:
                    if dtype in test_dtypes:
                        if self.gap:
                            domain = self.gap
                            if dtype == "uint8":
                                if len(domain) == 2 and domain[0] < 0:
                                    if domain[1] >= 1:
                                        domain = (0, domain[1])
                                    else:
                                        raise TypeError(
                                            "Gap not suitable for", dtype, self.__name__
                                        )

                        else:
                            domain = (0, 5)

                        variable, data = sparse_random_inputs(
                            format, shape=(4, 7), out_dtype=dtype, gap=domain
                        )

                        f = pytensor.function(variable, self.op(*variable))

                        old_value = (
                            pt.math.float32_atol,
                            pt.math.float32_rtol,
                            pt.math.float64_atol,
                            pt.math.float64_rtol,
                        )
                        pt.math.float32_atol = 1e-4
                        pt.math.float32_rtol = 1e-3
                        pt.math.float64_atol = 1e-3
                        pt.math.float64_rtol = 1e-4
                        try:
                            tested = f(*data)
                        finally:
                            (
                                pt.math.float32_atol,
                                pt.math.float32_rtol,
                                pt.math.float64_atol,
                                pt.math.float64_rtol,
                            ) = old_value

                        data = [m.toarray().astype("float32") for m in data]
                        expected = self.expected_f(*data)

                        assert tested.format == format
                        tested = tested.toarray()

                        try:
                            utt.assert_allclose(tested, expected, rtol=1e-2)
                        except AssertionError:
                            raise AssertionError(self.__name__)

        if grad_test:

            def test_grad(self):
                for format in sparse_formats:
                    for dtype in float_dtypes:
                        _variable, data = sparse_random_inputs(
                            format, shape=(4, 7), out_dtype=dtype, gap=self.gap_grad
                        )

                        verify_grad_sparse(self.op, data, structured=True)

    if name is None:
        name = op.__name__.capitalize() + "Tester"
    TestElemwise.__name__ = name
    if hasattr(TestElemwise, "__qualname__"):
        TestElemwise.__qualname__ = name
    assert "Roundhalftoeven" not in TestElemwise.__name__

    return TestElemwise


StructuredSigmoidTester = elemwise_checker(
    psm.structured_sigmoid,
    structure_function(lambda x: 1.0 / (1.0 + np.exp(-x))),
    test_dtypes=[
        m for m in all_dtypes if (m not in complex_dtypes and not m.startswith("uint"))
    ],
    gap=(-5, 5),
    name="StructuredSigmoidTester",
)

StructuredExpTester = elemwise_checker(
    psm.structured_exp, structure_function(np.exp), name="StructuredExpTester"
)

StructuredLogTester = elemwise_checker(
    psm.structured_log,
    structure_function(np.log),
    gap=(0.5, 10),
    name="StructuredLogTester",
)

StructuredPowTester = elemwise_checker(
    lambda x: psm.structured_pow(x, 2),
    structure_function(lambda x: np.power(x, 2)),
    name="StructuredPowTester",
)

StructuredMinimumTester = elemwise_checker(
    lambda x: psm.structured_minimum(x, 2),
    structure_function(lambda x: np.minimum(x, 2)),
    name="StructuredMinimumTester",
)

StructuredMaximumTester = elemwise_checker(
    lambda x: psm.structured_maximum(x, 2),
    structure_function(lambda x: np.maximum(x, 2)),
    name="StructuredMaximumTester",
)

StructuredAddTester = elemwise_checker(
    lambda x: psm.structured_add(x, 2),
    structure_function(lambda x: np.add(x, 2)),
    name="StructuredAddTester",
)

SinTester = elemwise_checker(psm.sin, np.sin)

TanTester = elemwise_checker(psm.tan, np.tan, gap=(-1, 1))

ArcsinTester = elemwise_checker(
    psm.arcsinh, np.arcsin, gap=(-1, 1), gap_grad=(-0.99, 0.99)
)

ArctanTester = elemwise_checker(psm.arctan, np.arctan)

SinhTester = elemwise_checker(psm.sinh, np.sinh)

ArcsinhTester = elemwise_checker(psm.arcsinh, np.arcsinh, gap=(-1, 1))

TanhTester = elemwise_checker(psm.tanh, np.tanh, gap=(-1, 1))

ArctanhTester = elemwise_checker(
    psm.arctanh, np.arctanh, gap=(-0.9, 1), gap_grad=(-0.9, 0.95)
)

RintTester = elemwise_checker(
    psm.rint, np.rint, grad_test=False, test_dtypes=float_dtypes
)

SgnTester = elemwise_checker(
    psm.sign,
    np.sign,
    grad_test=False,
    test_dtypes=[
        m for m in all_dtypes if (m not in complex_dtypes and not m.startswith("uint"))
    ],
)

CeilTester = elemwise_checker(
    psm.ceil,
    np.ceil,
    grad_test=False,
    test_dtypes=[m for m in all_dtypes if m not in complex_dtypes],
)

FloorTester = elemwise_checker(
    psm.floor,
    np.floor,
    grad_test=False,
    test_dtypes=[m for m in all_dtypes if m not in complex_dtypes],
)

Log1pTester = elemwise_checker(psm.log1p, np.log1p, gap=(0.5, 10))

Expm1Tester = elemwise_checker(psm.expm1, np.expm1)

Deg2radTester = elemwise_checker(
    psm.deg2rad,
    np.deg2rad,
    test_dtypes=[m for m in all_dtypes if m not in complex_dtypes],
)

Rad2degTester = elemwise_checker(
    psm.rad2deg,
    np.rad2deg,
    test_dtypes=[m for m in all_dtypes if m not in complex_dtypes],
)


TruncTester = elemwise_checker(
    psm.trunc,
    np.trunc,
    test_dtypes=[m for m in all_dtypes if m not in complex_dtypes],
    grad_test=False,
)


SqrTester = elemwise_checker(psm.sqr, lambda x: x * x)

SqrtTester = elemwise_checker(psm.sqrt, np.sqrt, gap=(0, 10))

ConjTester = elemwise_checker(psm.conjugate, np.conj, grad_test=False)

NegTester = elemwise_checker(psm.neg, np.negative, name="TestNeg")
