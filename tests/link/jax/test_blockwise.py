import numpy as np
import pytest

from pytensor import config
from pytensor.tensor import add, as_tensor, constant, iscalar, lmatrix, tensor
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.math import Dot, matmul
from tests.link.jax.test_basic import compare_jax_and_py
from tests.tensor.test_blockwise import check_blockwise_runtime_broadcasting
from tests.unittest_tools import RewriteTester


jax = pytest.importorskip("jax")

from pytensor.link.jax.dispatch.blockwise import JAXUnbatchedScalar
from pytensor.tensor.rewriting.jax import blockwise_unbatched_scalar_inputs


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
    fn, _ = compare_jax_and_py([a, b], [out], test_values)

    # Check we are not adding any unnecessary stuff
    jaxpr = str(jax.make_jaxpr(fn.vm.jit_fn)(*test_values))
    jaxpr = jaxpr.replace("name=jax_funcified_fgraph", "name=matmul")
    expected_jaxpr = str(jax.make_jaxpr(jax.jit(jax.numpy.matmul))(*test_values))
    assert jaxpr == expected_jaxpr


def test_unbatched_scalar_inputs_rewrite():
    x = lmatrix("x")
    n, m = iscalar("n"), iscalar("m")
    two = constant(2, dtype="int32")
    scalar_add = Blockwise(add, signature="(),(),(),()->()")
    # n is padded by a DimShuffle, m by a DimShuffle of a MakeVector
    out = scalar_add(x, n, as_tensor([m]), two)

    result = RewriteTester(
        [x, n, m],
        [out],
        include=None,
        custom_rewrite=blockwise_unbatched_scalar_inputs,
    )
    result.assert_graph(
        scalar_add(
            x,
            JAXUnbatchedScalar(2)(n),
            JAXUnbatchedScalar(2)(m),
            two[None, None],
        )
    )
