import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config, function, grad
from pytensor.tensor.special import xlog1py, xlogy
from pytensor.tensor.symbolic import TensorSymbolicOp


class SymbolicAdd(TensorSymbolicOp):
    def build_inner_graph(self, x, y):
        return [x + y]


@pytest.mark.parametrize("op", [SymbolicAdd(), pt.logaddexp, xlogy, xlog1py])
@pytest.mark.parametrize("floatX", ["float32", "float64"])
@config.change_flags(cast_policy="numpy+floatX")
def test_weak_scalar_promotion(op, floatX):
    with config.change_flags(floatX=floatX):
        x = pt.fvector("x")
        for inputs in ((x, 2), (2.5, x)):
            out = op(*inputs)
            assert out.dtype == "float32"
            assert all(inp.dtype == "float32" for inp in out.owner.inputs)
        assert op(x, np.float64(2)).dtype == "float64"


@config.change_flags(cast_policy="numpy+floatX", floatX="float64")
def test_logsumexp_preserves_float32():
    assert pt.logsumexp(pt.fvector()).dtype == "float32"


@pytest.mark.parametrize("op", [pt.logaddexp, xlogy, xlog1py])
@config.change_flags(cast_policy="numpy+floatX", floatX="float64")
def test_integer_literal_in_float_operation(op):
    assert op(pt.bvector(), 128).dtype == "float32"


@pytest.mark.parametrize("mode", ["FAST_COMPILE", "CVM", "NUMBA"])
@config.change_flags(cast_policy="numpy+floatX", floatX="float64")
def test_weak_scalar_evaluation(mode):
    if mode == "NUMBA":
        pytest.importorskip("numba")
    x = pt.fvector("x")
    out = [pt.cos(pt.sin(x)) + 2 * x, pt.logaddexp(x, 2), xlogy(2, x)]
    out.append(grad(out[1].sum(), x))
    f = function([x], out, mode=mode)
    value = np.array([0.25, 0.5, 1], dtype="float32")
    expected = [
        np.cos(np.sin(value)) + 2 * value,
        np.logaddexp(value, 2),
        2 * np.log(value),
        1 / (1 + np.exp(2 - value)),
    ]
    for result, reference in zip(f(value), expected, strict=True):
        assert result.dtype == "float32"
        np.testing.assert_allclose(result, reference, rtol=1e-6)
