from functools import partial

import numpy as np
import numpy.linalg
import pytest
from numpy.testing import assert_array_almost_equal

import pytensor
from pytensor import function
from pytensor.configdefaults import config
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.math import _allclose
from pytensor.tensor.nlinalg import (
    SVD,
    Eig,
    MatrixInverse,
    TensorInv,
    det,
    eig,
    eigh,
    kron,
    lstsq,
    matrix_dot,
    matrix_inverse,
    matrix_power,
    norm,
    pinv,
    qr,
    slogdet,
    svd,
    tensorinv,
    tensorsolve,
    trace,
)
from pytensor.tensor.type import (
    lmatrix,
    lscalar,
    matrix,
    scalar,
    tensor,
    tensor3,
    tensor4,
    vector,
)
from tests import unittest_tools as utt


def test_pseudoinverse_correctness():
    rng = np.random.default_rng(utt.fetch_seed())
    d1 = rng.integers(4) + 2
    d2 = rng.integers(4) + 2
    r = rng.standard_normal((d1, d2)).astype(config.floatX)

    x = matrix()
    xi = pinv(x)

    ri = function([x], xi)(r)
    assert ri.shape[0] == r.shape[1]
    assert ri.shape[1] == r.shape[0]
    assert ri.dtype == r.dtype
    # Note that pseudoinverse can be quite imprecise so I prefer to compare
    # the result with what np.linalg returns
    assert _allclose(ri, np.linalg.pinv(r))


def test_pseudoinverse_grad():
    rng = np.random.default_rng(utt.fetch_seed())
    d1 = rng.integers(4) + 2
    d2 = rng.integers(4) + 2
    r = rng.standard_normal((d1, d2)).astype(config.floatX)

    utt.verify_grad(pinv, [r])


class TestMatrixInverse(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = MatrixInverse
        self.op = matrix_inverse
        self.rng = np.random.default_rng(utt.fetch_seed())

    def test_inverse_correctness(self):
        r = self.rng.standard_normal((4, 4)).astype(config.floatX)

        x = matrix()
        xi = self.op(x)

        ri = function([x], xi)(r)
        assert ri.shape == r.shape
        assert ri.dtype == r.dtype

        rir = np.dot(ri, r)
        rri = np.dot(r, ri)

        assert _allclose(np.identity(4), rir), rir
        assert _allclose(np.identity(4), rri), rri

    def test_infer_shape(self):
        r = self.rng.standard_normal((4, 4)).astype(config.floatX)

        x = matrix()
        xi = self.op(x)

        self._compile_and_check([x], [xi], [r], self.op_class, warn=False)


def test_matrix_dot():
    rng = np.random.default_rng(utt.fetch_seed())
    n = rng.integers(4) + 2
    rs = []
    xs = []
    for k in range(n):
        rs += [rng.standard_normal((4, 4)).astype(config.floatX)]
        xs += [matrix()]
    sol = matrix_dot(*xs)

    pytensor_sol = function(xs, sol)(*rs)
    numpy_sol = rs[0]
    for r in rs[1:]:
        numpy_sol = np.dot(numpy_sol, r)

    assert _allclose(numpy_sol, pytensor_sol)


def test_qr_modes():
    rng = np.random.default_rng(utt.fetch_seed())

    A = matrix("A", dtype=config.floatX)
    a = rng.random((4, 4)).astype(config.floatX)

    f = function([A], qr(A))
    t_qr = f(a)
    n_qr = np.linalg.qr(a)
    assert _allclose(n_qr, t_qr)

    for mode in ["reduced", "r", "raw"]:
        f = function([A], qr(A, mode))
        t_qr = f(a)
        n_qr = np.linalg.qr(a, mode)
        if isinstance(n_qr, list | tuple):
            assert _allclose(n_qr[0], t_qr[0])
            assert _allclose(n_qr[1], t_qr[1])
        else:
            assert _allclose(n_qr, t_qr)

    try:
        n_qr = np.linalg.qr(a, "complete")
        f = function([A], qr(A, "complete"))
        t_qr = f(a)
        assert _allclose(n_qr, t_qr)
    except TypeError as e:
        assert "name 'complete' is not defined" in str(e)


class TestSvd(utt.InferShapeTester):
    op_class = SVD

    def setup_method(self):
        super().setup_method()
        self.rng = np.random.default_rng(utt.fetch_seed())
        self.A = matrix(dtype=config.floatX)
        self.op = svd

    @pytest.mark.parametrize(
        "core_shape", [(3, 3), (4, 3), (3, 4)], ids=["square", "tall", "wide"]
    )
    @pytest.mark.parametrize(
        "full_matrix", [True, False], ids=["full=True", "full=False"]
    )
    @pytest.mark.parametrize(
        "compute_uv", [True, False], ids=["compute_uv=True", "compute_uv=False"]
    )
    @pytest.mark.parametrize(
        "batched", [True, False], ids=["batched=True", "batched=False"]
    )
    @pytest.mark.parametrize(
        "test_imag", [True, False], ids=["test_imag=True", "test_imag=False"]
    )
    def test_svd(self, core_shape, full_matrix, compute_uv, batched, test_imag):
        dtype = config.floatX
        if test_imag:
            dtype = "complex128" if dtype.endswith("64") else "complex64"
        shape = core_shape if not batched else (10, *core_shape)
        A = tensor("A", shape=shape, dtype=dtype)
        a = self.rng.random(shape).astype(dtype)

        outputs = svd(A, compute_uv=compute_uv, full_matrices=full_matrix)
        outputs = outputs if isinstance(outputs, list) else [outputs]
        fn = function(inputs=[A], outputs=outputs)

        np_fn = np.vectorize(
            partial(np.linalg.svd, compute_uv=compute_uv, full_matrices=full_matrix),
            signature=outputs[0].owner.op.core_op.gufunc_signature,
        )

        np_outputs = np_fn(a)
        pt_outputs = fn(a)

        np_outputs = np_outputs if isinstance(np_outputs, tuple) else [np_outputs]

        for np_val, pt_val in zip(np_outputs, pt_outputs, strict=True):
            assert _allclose(np_val, pt_val)

    def test_svd_infer_shape(self):
        self.validate_shape((4, 4), full_matrices=True, compute_uv=True)
        self.validate_shape((4, 4), full_matrices=False, compute_uv=True)
        self.validate_shape((2, 4), full_matrices=False, compute_uv=True)
        self.validate_shape((4, 2), full_matrices=False, compute_uv=True)
        self.validate_shape((4, 4), compute_uv=False)

    def validate_shape(self, shape, compute_uv=True, full_matrices=True):
        A = self.A
        A_v = self.rng.random(shape).astype(config.floatX)
        outputs = self.op(A, full_matrices=full_matrices, compute_uv=compute_uv)
        if not compute_uv:
            outputs = [outputs]
        self._compile_and_check([A], outputs, [A_v], self.op_class, warn=False)

    @pytest.mark.parametrize(
        "compute_uv, full_matrices, gradient_test_case",
        [(False, False, 0)]
        + [(True, False, i) for i in range(8)]
        + [(True, True, i) for i in range(8)],
        ids=(
            ["compute_uv=False, full_matrices=False"]
            + [
                f"compute_uv=True, full_matrices=False, gradient={grad}"
                for grad in ["U", "s", "V", "U+s", "s+V", "U+V", "U+s+V", "None"]
            ]
            + [
                f"compute_uv=True, full_matrices=True, gradient={grad}"
                for grad in ["U", "s", "V", "U+s", "s+V", "U+V", "U+s+V", "None"]
            ]
        ),
    )
    @pytest.mark.parametrize(
        "shape", [(3, 3), (4, 3), (3, 4)], ids=["(3,3)", "(4,3)", "(3,4)"]
    )
    @pytest.mark.parametrize(
        "batched", [True, False], ids=["batched=True", "batched=False"]
    )
    def test_grad(self, compute_uv, full_matrices, gradient_test_case, shape, batched):
        rng = np.random.default_rng(utt.fetch_seed())
        if batched:
            shape = (4, *shape)

        A_v = self.rng.normal(size=shape).astype(config.floatX)
        if full_matrices:
            with pytest.raises(
                NotImplementedError,
                match="Gradient of svd not implemented for full_matrices=True",
            ):
                U, s, V = svd(
                    self.A, compute_uv=compute_uv, full_matrices=full_matrices
                )
                pytensor.grad(s.sum(), self.A)

        elif compute_uv:

            def svd_fn(A, case=0):
                U, s, V = svd(A, compute_uv=compute_uv, full_matrices=full_matrices)
                if case == 0:
                    return U.sum()
                elif case == 1:
                    return s.sum()
                elif case == 2:
                    return V.sum()
                elif case == 3:
                    return U.sum() + s.sum()
                elif case == 4:
                    return s.sum() + V.sum()
                elif case == 5:
                    return U.sum() + V.sum()
                elif case == 6:
                    return U.sum() + s.sum() + V.sum()
                elif case == 7:
                    # All inputs disconnected
                    return as_tensor_variable(3.0)

            utt.verify_grad(
                partial(svd_fn, case=gradient_test_case),
                [A_v],
                rng=rng,
            )

        else:
            utt.verify_grad(
                partial(svd, compute_uv=compute_uv, full_matrices=full_matrices),
                [A_v],
                rng=rng,
            )


def test_tensorsolve():
    rng = np.random.default_rng(utt.fetch_seed())

    A = tensor4("A", dtype=config.floatX)
    B = matrix("B", dtype=config.floatX)
    X = tensorsolve(A, B)
    fn = function([A, B], [X])

    # slightly modified example from np.linalg.tensorsolve docstring
    a = np.eye(2 * 3 * 4).astype(config.floatX)
    a.shape = (2 * 3, 4, 2, 3 * 4)
    b = rng.random((2 * 3, 4)).astype(config.floatX)

    n_x = np.linalg.tensorsolve(a, b)
    t_x = fn(a, b)
    assert _allclose(n_x, t_x)

    # check the type upcast now
    C = tensor4("C", dtype="float32")
    D = matrix("D", dtype="float64")
    Y = tensorsolve(C, D)
    fn = function([C, D], [Y])

    c = np.eye(2 * 3 * 4, dtype="float32")
    c.shape = (2 * 3, 4, 2, 3 * 4)
    d = rng.random((2 * 3, 4)).astype("float64")
    n_y = np.linalg.tensorsolve(c, d)
    t_y = fn(c, d)
    assert _allclose(n_y, t_y)
    assert n_y.dtype == Y.dtype

    # check the type upcast now
    E = tensor4("E", dtype="int32")
    F = matrix("F", dtype="float64")
    Z = tensorsolve(E, F)
    fn = function([E, F], [Z])

    e = np.eye(2 * 3 * 4, dtype="int32")
    e.shape = (2 * 3, 4, 2, 3 * 4)
    f = rng.random((2 * 3, 4)).astype("float64")
    n_z = np.linalg.tensorsolve(e, f)
    t_z = fn(e, f)
    assert _allclose(n_z, t_z)
    assert n_z.dtype == Z.dtype


def test_inverse_singular():
    singular = np.array([[1, 0, 0]] + [[0, 1, 0]] * 2, dtype=config.floatX)
    a = matrix()
    f = function([a], matrix_inverse(a))
    with pytest.raises(np.linalg.LinAlgError):
        f(singular)


def test_inverse_grad():
    rng = np.random.default_rng(utt.fetch_seed())
    r = rng.standard_normal((4, 4))
    utt.verify_grad(matrix_inverse, [r], rng=np.random)

    rng = np.random.default_rng(utt.fetch_seed())

    r = rng.standard_normal((4, 4))
    utt.verify_grad(matrix_inverse, [r], rng=np.random)


def test_det():
    rng = np.random.default_rng(utt.fetch_seed())

    r = rng.standard_normal((5, 5)).astype(config.floatX)
    x = matrix()
    f = pytensor.function([x], det(x))
    assert np.allclose(np.linalg.det(r), f(r))


def test_det_non_square_raises():
    with pytest.raises(ValueError, match="Determinant not defined"):
        det(tensor("x", shape=(5, 7)))


def test_det_grad():
    rng = np.random.default_rng(utt.fetch_seed())

    r = rng.standard_normal((5, 5)).astype(config.floatX)
    utt.verify_grad(det, [r], rng=np.random)


def test_det_shape():
    x = matrix()
    assert det(x).type.shape == ()


def test_slogdet():
    rng = np.random.default_rng(utt.fetch_seed())

    r = rng.standard_normal((5, 5)).astype(config.floatX)
    x = matrix()
    f = pytensor.function([x], slogdet(x))
    f_sign, f_det = f(r)
    sign, det = np.linalg.slogdet(r)
    assert np.equal(sign, f_sign)
    assert np.allclose(det, f_det)
    # check numpy array types is returned
    # see https://github.com/pymc-devs/pytensor/issues/799
    sign, logdet = slogdet(x)
    det = sign * pytensor.tensor.exp(logdet)
    assert_array_almost_equal(det.eval({x: [[1]]}), np.array(1.0))


def test_trace():
    rng = np.random.default_rng(utt.fetch_seed())
    x = matrix()
    with pytest.warns(FutureWarning):
        g = trace(x)
    f = pytensor.function([x], g)

    for shp in [(2, 3), (3, 2), (3, 3)]:
        m = rng.random(shp).astype(config.floatX)
        v = np.trace(m)
        assert v == f(m)

    xx = vector()
    ok = False
    try:
        with pytest.warns(FutureWarning):
            trace(xx)
    except TypeError:
        ok = True
    except ValueError:
        ok = True

    assert ok


class TestEig(utt.InferShapeTester):
    op_class = Eig
    op = eig
    dtype = "float64"

    def setup_method(self):
        super().setup_method()
        self.rng = np.random.default_rng(utt.fetch_seed())
        self.A = matrix(dtype=self.dtype)
        self.X = np.asarray(self.rng.random((5, 5)), dtype=self.dtype)
        self.S = self.X.dot(self.X.T)

    def test_infer_shape(self):
        A = self.A
        S = self.S
        self._compile_and_check(
            [A],  # pytensor.function inputs
            self.op(A),  # pytensor.function outputs
            # S must be square
            [S],
            self.op_class,
            warn=False,
        )

    def test_eval(self):
        A = matrix(dtype=self.dtype)
        assert [e.eval({A: [[1]]}) for e in self.op(A)] == [[1.0], [[1.0]]]
        x = [[0, 1], [1, 0]]
        w, v = (e.eval({A: x}) for e in self.op(A))
        assert_array_almost_equal(np.dot(x, v), w * v)


class TestEigh(TestEig):
    op = staticmethod(eigh)

    def test_uplo(self):
        S = self.S
        a = matrix(dtype=self.dtype)
        wu, vu = (out.eval({a: S}) for out in self.op(a, "U"))
        wl, vl = (out.eval({a: S}) for out in self.op(a, "L"))
        assert_array_almost_equal(wu, wl)
        assert_array_almost_equal(vu * np.sign(vu[0, :]), vl * np.sign(vl[0, :]))

    def test_grad(self):
        X = self.X
        # We need to do the dot inside the graph because Eigh needs a
        # matrix that is hermitian
        utt.verify_grad(lambda x: self.op(x.dot(x.T))[0], [X], rng=self.rng)
        utt.verify_grad(lambda x: self.op(x.dot(x.T))[1], [X], rng=self.rng)
        utt.verify_grad(lambda x: self.op(x.dot(x.T), "U")[0], [X], rng=self.rng)
        utt.verify_grad(lambda x: self.op(x.dot(x.T), "U")[1], [X], rng=self.rng)


class TestEighFloat32(TestEigh):
    dtype = "float32"

    def test_uplo(self):
        super().test_uplo()

    def test_grad(self):
        super().test_grad()


class TestLstsq:
    def test_correct_solution(self):
        x = lmatrix()
        y = lmatrix()
        z = lscalar()
        b = lstsq(x, y, z)
        f = function([x, y, z], b)
        TestMatrix1 = np.asarray([[2, 1], [3, 4]])
        TestMatrix2 = np.asarray([[17, 20], [43, 50]])
        TestScalar = np.asarray(1)
        f = function([x, y, z], b)
        m = f(TestMatrix1, TestMatrix2, TestScalar)
        assert np.allclose(TestMatrix2, np.dot(TestMatrix1, m[0]))

    def test_wrong_coefficient_matrix(self):
        x = vector()
        y = vector()
        z = scalar()
        b = lstsq(x, y, z)
        f = function([x, y, z], b)
        with pytest.raises(np.linalg.linalg.LinAlgError):
            f([2, 1], [2, 1], 1)

    def test_wrong_rcond_dimension(self):
        x = vector()
        y = vector()
        z = vector()
        b = lstsq(x, y, z)
        f = function([x, y, z], b)
        with pytest.raises(np.linalg.LinAlgError):
            f([2, 1], [2, 1], [2, 1])


class TestMatrixPower:
    @config.change_flags(compute_test_value="raise")
    @pytest.mark.parametrize("n", [-1, 0, 1, 2, 3, 4, 5, 11])
    def test_numpy_compare(self, n):
        a = np.array([[0.1231101, 0.72381381], [0.28748201, 0.43036511]]).astype(
            config.floatX
        )
        A = matrix("A", dtype=config.floatX)
        A.tag.test_value = a
        Q = matrix_power(A, n)
        n_p = np.linalg.matrix_power(a, n)
        assert np.allclose(n_p, Q.get_test_value())

    def test_non_square_matrix(self):
        A = matrix("A", dtype=config.floatX)
        Q = matrix_power(A, 3)
        f = function([A], [Q])
        a = np.array(
            [
                [0.47497769, 0.81869379],
                [0.74387558, 0.31780172],
                [0.54381007, 0.28153101],
            ]
        ).astype(config.floatX)
        with pytest.raises(ValueError):
            f(a)


class TestNorm:
    def test_wrong_type_of_ord_for_vector(self):
        with pytest.raises(ValueError, match="Invalid norm order 'fro' for vectors"):
            norm([2, 1], "fro")

    def test_wrong_type_of_ord_for_matrix(self):
        ord = 0
        with pytest.raises(ValueError, match=f"Invalid norm order for matrices: {ord}"):
            norm([[2, 1], [3, 4]], ord)

    def test_non_tensorial_input(self):
        with pytest.raises(
            ValueError,
            match="Cannot compute norm when core_dims < 1 or core_dims > 3, found: core_dims = 0",
        ):
            norm(3, ord=2)

    def test_invalid_axis_input(self):
        axis = scalar("i", dtype="int")
        with pytest.raises(
            TypeError, match="'axis' must be None, an integer, or a tuple of integers"
        ):
            norm([[1, 2], [3, 4]], axis=axis)

    @pytest.mark.parametrize(
        "ord",
        [None, np.inf, -np.inf, 1, -1, 2, -2],
        ids=["None", "inf", "-inf", "1", "-1", "2", "-2"],
    )
    @pytest.mark.parametrize("core_dims", [(4,), (4, 3)], ids=["vector", "matrix"])
    @pytest.mark.parametrize("batch_dims", [(), (2,)], ids=["no_batch", "batch"])
    @pytest.mark.parametrize("test_imag", [True, False], ids=["complex", "real"])
    @pytest.mark.parametrize(
        "keepdims", [True, False], ids=["keep_dims=True", "keep_dims=False"]
    )
    def test_numpy_compare(
        self,
        ord: float,
        core_dims: tuple[int, ...],
        batch_dims: tuple[int, ...],
        test_imag: bool,
        keepdims: bool,
        axis=None,
    ):
        is_matrix = len(core_dims) == 2
        has_batch = len(batch_dims) > 0
        if ord in [np.inf, -np.inf] and not is_matrix:
            pytest.skip("Infinity norm not defined for vectors")
        if test_imag and is_matrix and ord == -2:
            pytest.skip("Complex matrices not supported")
        if has_batch and not is_matrix:
            # Handle batched vectors by row-normalizing a matrix
            axis = (-1,)

        rng = np.random.default_rng(utt.fetch_seed())

        if test_imag:
            x_real, x_imag = rng.standard_normal((2, *batch_dims, *core_dims)).astype(
                config.floatX
            )
            dtype = "complex128" if config.floatX.endswith("64") else "complex64"
            X = (x_real + 1j * x_imag).astype(dtype)
        else:
            X = rng.standard_normal(batch_dims + core_dims).astype(config.floatX)

        if batch_dims == ():
            np_norm = np.linalg.norm(X, ord=ord, axis=axis, keepdims=keepdims)
        else:
            np_norm = np.stack(
                [np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims) for x in X]
            )

        pt_norm = norm(X, ord=ord, axis=axis, keepdims=keepdims)
        f = function([], pt_norm, mode="FAST_COMPILE")

        utt.assert_allclose(np_norm, f())


class TestTensorInv(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.A = tensor4("A", dtype=config.floatX)
        self.B = tensor3("B", dtype=config.floatX)
        self.a = np.random.random((4, 6, 8, 3)).astype(config.floatX)
        self.b = np.random.random((2, 15, 30)).astype(config.floatX)
        self.b1 = np.random.random((30, 2, 15)).astype(
            config.floatX
        )  # for ind=1 since we need prod(b1.shape[:ind]) == prod(b1.shape[ind:])

    def test_infer_shape(self):
        A = self.A
        Ai = tensorinv(A)
        self._compile_and_check(
            [A],  # pytensor.function inputs
            [Ai],  # pytensor.function outputs
            [self.a],  # value to substitute
            TensorInv,
        )

    def test_eval(self):
        A = self.A
        Ai = tensorinv(A)
        n_ainv = np.linalg.tensorinv(self.a)
        tf_a = function([A], [Ai])
        t_ainv = tf_a(self.a)
        assert _allclose(n_ainv, t_ainv)

        B = self.B
        Bi = tensorinv(B)
        Bi1 = tensorinv(B, ind=1)
        n_binv = np.linalg.tensorinv(self.b)
        n_binv1 = np.linalg.tensorinv(self.b1, ind=1)
        tf_b = function([B], [Bi])
        tf_b1 = function([B], [Bi1])
        t_binv = tf_b(self.b)
        t_binv1 = tf_b1(self.b1)
        assert _allclose(t_binv, n_binv)
        assert _allclose(t_binv1, n_binv1)


class TestKron(utt.InferShapeTester):
    rng = np.random.default_rng(43)

    def setup_method(self):
        self.op = kron
        super().setup_method()

    def test_vec_vec_kron_raises(self):
        x = vector()
        y = vector()
        with pytest.raises(
            TypeError, match="kron: inputs dimensions must sum to 3 or more"
        ):
            kron(x, y)

    @pytest.mark.parametrize("shp0", [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
    @pytest.mark.parametrize("shp1", [(6,), (6, 7), (6, 7, 8), (6, 7, 8, 9)])
    def test_perform(self, shp0, shp1):
        if len(shp0) + len(shp1) == 2:
            pytest.skip("Sum of shp0 and shp1 must be more than 2")
        x = tensor(dtype="floatX", shape=(None,) * len(shp0))
        a = np.asarray(self.rng.random(shp0)).astype(config.floatX)
        y = tensor(dtype="floatX", shape=(None,) * len(shp1))
        f = function([x, y], kron(x, y))
        b = self.rng.random(shp1).astype(config.floatX)
        out = f(a, b)
        # Using the np.kron to compare outputs
        np_val = np.kron(a, b)
        np.testing.assert_allclose(out, np_val)

    @pytest.mark.parametrize(
        "i, shp0, shp1",
        [(0, (2, 3), (6, 7)), (1, (2, 3), (4, 3, 5)), (2, (2, 4, 3), (4, 3, 5))],
    )
    def test_kron_commutes_with_inv(self, i, shp0, shp1):
        if (pytensor.config.floatX == "float32") & (i == 2):
            pytest.skip("Half precision insufficient for test 3 to pass")
        x = tensor(dtype="floatX", shape=(None,) * len(shp0))
        a = np.asarray(self.rng.random(shp0)).astype(config.floatX)
        y = tensor(dtype="floatX", shape=(None,) * len(shp1))
        b = self.rng.random(shp1).astype(config.floatX)
        lhs_f = function([x, y], pinv(kron(x, y)))
        rhs_f = function([x, y], kron(pinv(x), pinv(y)))
        atol = 1e-4 if config.floatX == "float32" else 1e-12
        np.testing.assert_allclose(lhs_f(a, b), rhs_f(a, b), atol=atol)
