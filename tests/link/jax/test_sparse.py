import numpy as np
import pytest
import scipy.sparse


jax = pytest.importorskip("jax")
from jax.experimental.sparse import BCOO

import pytensor.sparse as ps
import pytensor.tensor as pt
from pytensor import function
from pytensor.graph import Constant, FunctionGraph
from pytensor.tensor.type import DenseTensorType
from tests.link.jax.test_basic import compare_jax_and_py


def assert_bcoo_arrays_allclose(a1, a2):
    assert isinstance(a1, BCOO)
    assert isinstance(a1, BCOO)
    np.testing.assert_allclose(a1.todense(), a2.todense())


@pytest.mark.parametrize("sparse_type", ("csc", "csr"))
def test_sparse_io(sparse_type):
    """Test explicit (non-shared) input and output sparse types in JAX."""
    sparse_mat = ps.matrix(format=sparse_type, name="csc", dtype="float32")
    sparse_mat_out = sparse_mat.T

    with pytest.warns(
        UserWarning,
        match="SparseTypes are implicitly converted to sparse BCOO arrays",
    ):
        fn = function([sparse_mat], sparse_mat_out, mode="JAX")

    sp_sparse_mat = scipy.sparse.random(
        5, 40, density=0.25, format=sparse_type, dtype="float32"
    )
    jx_sparse_mat = BCOO.from_scipy_sparse(sp_sparse_mat)

    sp_res = fn(sp_sparse_mat)
    jx_res = fn(jx_sparse_mat)
    assert_bcoo_arrays_allclose(sp_res, jx_sparse_mat.T)
    assert_bcoo_arrays_allclose(jx_res, jx_sparse_mat.T)

    # Chained applications
    assert_bcoo_arrays_allclose(fn(fn(sp_sparse_mat)), jx_sparse_mat)
    assert_bcoo_arrays_allclose(fn(fn(jx_sparse_mat)), jx_sparse_mat)


@pytest.mark.parametrize(
    "op, x_type, y_type",
    [
        (ps.dot, pt.vector, ps.matrix),
        (ps.dot, pt.matrix, ps.matrix),
        (ps.dot, ps.matrix, pt.vector),
        (ps.dot, ps.matrix, pt.matrix),
        # structured_dot only allows matrix @ matrix
        (ps.structured_dot, pt.matrix, ps.matrix),
        (ps.structured_dot, ps.matrix, pt.matrix),
        (ps.structured_dot, ps.matrix, ps.matrix),
    ],
)
@pytest.mark.parametrize("x_constant", (False, True))
@pytest.mark.parametrize("y_constant", (False, True))
def test_sparse_dot(x_type, y_type, op, x_constant, y_constant):
    inputs = []
    test_values = []

    if x_type is ps.matrix:
        x_test = scipy.sparse.random(5, 40, density=0.25, format="csr", dtype="float32")
        x_pt = ps.as_sparse_variable(x_test, name="x")
    else:
        if x_type is pt.vector:
            x_test = np.arange(40, dtype="float32")
        else:
            x_test = np.arange(5 * 40, dtype="float32").reshape(5, 40)
        x_pt = pt.as_tensor_variable(x_test, name="x")
    assert isinstance(x_pt, Constant)

    if not x_constant:
        x_pt = x_pt.type(name="x")
        inputs.append(x_pt)
        test_values.append(x_test)

    if y_type is ps.matrix:
        y_test = scipy.sparse.random(40, 3, density=0.25, format="csc", dtype="float32")
        y_pt = ps.as_sparse_variable(y_test, name="y")
    else:
        if y_type is pt.vector:
            y_test = np.arange(40, dtype="float32")
        else:
            y_test = np.arange(40 * 3, dtype="float32").reshape(40, 3)
        y_pt = pt.as_tensor_variable(y_test, name="y")
    assert isinstance(y_pt, Constant)

    if not y_constant:
        y_pt = y_pt.type(name="y")
        inputs.append(y_pt)
        test_values.append(y_test)

    dot_pt = op(x_pt, y_pt)
    fgraph = FunctionGraph(inputs, [dot_pt])

    def assert_fn(x, y):
        [x] = x
        [y] = y
        if hasattr(x, "todense"):
            x = x.todense()
        if hasattr(y, "todense"):
            y = y.todense()
        np.testing.assert_allclose(x, y)

    compare_jax_and_py(
        fgraph,
        test_values,
        must_be_device_array=isinstance(dot_pt.type, DenseTensorType),
        assert_fn=assert_fn,
    )
