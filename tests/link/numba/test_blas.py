import itertools

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import In, Mode, config, function
from pytensor.compile import get_mode
from pytensor.graph import RewriteDatabaseQuery
from pytensor.link.numba import NumbaLinker
from pytensor.tensor.blas import Gemv, banded_gemv
from tests.link.numba.test_basic import compare_numba_and_py, numba_inplace_mode
from tests.tensor.test_slinalg import _make_banded_A


numba_blas_mode = Mode(
    NumbaLinker(),
    RewriteDatabaseQuery(
        include=["fast_run", "numba", "BlasOpt"],
        exclude=[
            "cxx_only",
            "c_blas",
            "local_careduce_fusion",
            "scan_save_mem_prealloc",
        ],
    ),
)


def test_banded_dot():
    rng = np.random.default_rng()

    A = pt.tensor("A", shape=(10, 10), dtype=config.floatX)
    A_val = _make_banded_A(rng.normal(size=(10, 10)), kl=1, ku=1).astype(config.floatX)

    x = pt.tensor("x", shape=(10,), dtype=config.floatX)
    x_val = rng.normal(size=(10,)).astype(config.floatX)

    output = banded_gemv(A, x, upper_diags=1, lower_diags=1)

    fn, _ = compare_numba_and_py(
        [A, x],
        output,
        test_inputs=[A_val, x_val],
        numba_mode=numba_inplace_mode,
        eval_obj_mode=False,
    )

    for stride in [2, -1, -2]:
        x_shape = (10 * abs(stride),)
        x_val = rng.normal(size=x_shape).astype(config.floatX)
        x_val = x_val[::stride]

        nb_output = fn(A_val, x_val)
        expected = A_val @ x_val

        np.testing.assert_allclose(
            nb_output,
            expected,
            strict=True,
            err_msg=f"Test failed for stride = {stride}",
        )


def test_numba_gemv():
    rng = np.random.default_rng()
    A = pt.tensor("A", shape=(10, 10))
    x = pt.tensor("x", shape=(10,))
    y = pt.tensor("y", shape=(10,))
    alpha, beta = pt.dscalars("alpha", "beta")

    output = alpha * A @ x + beta * y

    A_val = rng.normal(size=(10, 10)).astype(config.floatX)
    x_val = rng.normal(size=(10,)).astype(config.floatX)
    y_val = rng.normal(size=(10,)).astype(config.floatX)
    alpha_val, beta_val = rng.normal(size=(2,)).astype(config.floatX)

    fn, _ = compare_numba_and_py(
        [A, x, y, alpha, beta],
        output,
        test_inputs=[A_val, x_val, y_val, alpha_val, beta_val],
        numba_mode=numba_blas_mode,
        eval_obj_mode=False,
    )
    assert any(isinstance(node.op, Gemv) for node in fn.maker.fgraph.toposort())

    for stride, matrix in itertools.product([2, -1, -2], ["x", "y"]):
        shape = (10 * abs(stride),)
        val = rng.normal(size=shape).astype(config.floatX)

        if matrix == "x":
            x_val = val[::stride]
        else:
            y_val = val[::stride]

        nb_output = fn(A_val, x_val, y_val, alpha_val, beta_val)
        expected = alpha_val * A_val @ x_val + beta_val * y_val

        np.testing.assert_allclose(
            nb_output,
            expected,
            strict=True,
            err_msg=f"Test failed for stride = {stride}",
        )


@pytest.mark.parametrize("size", [10, 100, 1000], ids=str)
@pytest.mark.parametrize("use_blas_gemv", [True, False], ids=["numba+blas", "numba"])
def test_numba_gemv_benchmark(size, use_blas_gemv, benchmark):
    rng = np.random.default_rng()
    mode = numba_blas_mode if use_blas_gemv else get_mode("NUMBA")

    A = pt.tensor("A", shape=(None, None))
    x = pt.tensor("x", shape=(None,))
    y = pt.tensor("y", shape=(None,))
    alpha, beta = pt.dscalars("alpha", "beta")

    out = alpha * (A @ x) + beta * y
    fn = function([A, x, In(y, mutable=True), alpha, beta], out, mode=mode)

    if use_blas_gemv:
        assert any(isinstance(node.op, Gemv) for node in fn.maker.fgraph.toposort())
    else:
        assert not any(isinstance(node.op, Gemv) for node in fn.maker.fgraph.toposort())

    A_val = rng.normal(size=(size, size)).astype(config.floatX)
    x_val = rng.normal(size=(size,)).astype(config.floatX)
    y_val = rng.normal(size=(size,)).astype(config.floatX)
    alpha_val, beta_val = rng.normal(size=(2,)).astype(config.floatX)

    res = fn(A=A_val, x=x_val, y=y_val, alpha=alpha_val, beta=beta_val)
    np.testing.assert_allclose(res, y_val)

    benchmark(fn, A=A_val, x=x_val, y=y_val, alpha=alpha_val, beta=beta_val)
