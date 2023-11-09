import numpy as np
import pytest

from pytensor import config
from pytensor.graph import FunctionGraph
from pytensor.tensor import tensor
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.math import Dot, matmul
from tests.link.jax.test_basic import compare_jax_and_py
from tests.tensor.test_blockwise import check_blockwise_runtime_broadcasting


jax = pytest.importorskip("jax")


def test_runtime_broadcasting():
    check_blockwise_runtime_broadcasting("JAX")


# Equivalent blockwise to matmul but with dumb signature
odd_matmul = Blockwise(Dot(), signature="(i00,i01),(i10,i11)->(o00,o01)")


@pytest.mark.parametrize("matmul_op", (matmul, odd_matmul))
def test_matmul(matmul_op):
    rng = np.random.default_rng(14)
    a = tensor("a", shape=(2, 3, 5))
    b = tensor("b", shape=(2, 5, 3))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (a, b)
    ]

    out = matmul_op(a, b)
    assert isinstance(out.owner.op, Blockwise)
    fg = FunctionGraph([a, b], [out])
    fn, _ = compare_jax_and_py(fg, test_values)

    # Check we are not adding any unnecessary stuff
    jaxpr = str(jax.make_jaxpr(fn.vm.jit_fn)(*test_values))
    jaxpr = jaxpr.replace("name=jax_funcified_fgraph", "name=matmul")
    expected_jaxpr = str(jax.make_jaxpr(jax.jit(jax.numpy.matmul))(*test_values))
    assert jaxpr == expected_jaxpr
