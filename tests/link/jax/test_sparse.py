import numpy as np
import pytest
import scipy.sparse

import pytensor.sparse as ps
import pytensor.tensor as pt
from pytensor import function
from pytensor.graph import FunctionGraph
from tests.link.jax.test_basic import compare_jax_and_py


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
    ],
)
def test_sparse_dot_constant_sparse(x_type, y_type, op):
    inputs = []
    test_values = []

    if x_type is ps.matrix:
        x_sp = scipy.sparse.random(5, 40, density=0.25, format="csr", dtype="float32")
        x_pt = ps.as_sparse_variable(x_sp, name="x")
    else:
        x_pt = x_type("x", dtype="float32")
        if x_pt.ndim == 1:
            x_test = np.arange(40, dtype="float32")
        else:
            x_test = np.arange(5 * 40, dtype="float32").reshape(5, 40)
        inputs.append(x_pt)
        test_values.append(x_test)

    if y_type is ps.matrix:
        y_sp = scipy.sparse.random(40, 3, density=0.25, format="csc", dtype="float32")
        y_pt = ps.as_sparse_variable(y_sp, name="y")
    else:
        y_pt = y_type("y", dtype="float32")
        if y_pt.ndim == 1:
            y_test = np.arange(40, dtype="float32")
        else:
            y_test = np.arange(40 * 3, dtype="float32").reshape(40, 3)
        inputs.append(y_pt)
        test_values.append(y_test)

    dot_pt = op(x_pt, y_pt)
    fgraph = FunctionGraph(inputs, [dot_pt])
    compare_jax_and_py(fgraph, test_values, jax_mode="JAX")


def test_sparse_dot_non_const_raises():
    x_pt = pt.vector("x")

    y_sp = scipy.sparse.random(40, 3, density=0.25, format="csc", dtype="float32")
    y_pt = ps.as_sparse_variable(y_sp, name="y").type()

    out = ps.dot(x_pt, y_pt)

    msg = "JAX sparse dot only implemented for constant sparse inputs"

    with pytest.raises(NotImplementedError, match=msg):
        function([x_pt, y_pt], out, mode="JAX")

    y_pt_shared = ps.shared(y_sp, name="y")

    out = ps.dot(x_pt, y_pt_shared)

    with pytest.raises(NotImplementedError, match=msg):
        function([x_pt], out, mode="JAX")
