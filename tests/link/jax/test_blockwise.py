from functools import partial

import numpy as np
import pytest

from pytensor import config
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.mode import get_mode
from pytensor.tensor import add, as_tensor, constant, iscalar, lmatrix, lscalar, tensor
from pytensor.tensor.blockwise import Blockwise, JAXBlockwise
from pytensor.tensor.math import Dot, matmul, mul
from pytensor.tensor.rewriting.jax import blockwise_unbatched_scalar_inputs
from tests.link.jax.test_basic import compare_jax_and_py
from tests.tensor.test_blockwise import check_blockwise_runtime_broadcasting
from tests.unittest_tools import RewriteTester


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
    fn, _ = compare_jax_and_py([a, b], [out], test_values)

    # Check we are not adding any unnecessary stuff
    jaxpr = str(jax.make_jaxpr(fn.vm.jit_fn)(*test_values))
    jaxpr = jaxpr.replace("name=jax_funcified_fgraph", "name=matmul")
    expected_jaxpr = str(jax.make_jaxpr(jax.jit(jax.numpy.matmul))(*test_values))
    assert jaxpr == expected_jaxpr


def test_unbatched_scalar_inputs_rewrite():
    x = lmatrix("x")
    n, m = iscalar("n"), iscalar("m")
    two_int32 = constant(2, dtype="int32")
    scalar_add = Blockwise(add, signature="(),(),(),(),()->()")
    # n is padded by a DimShuffle, m by a DimShuffle of a MakeVector
    out = scalar_add(x, n, as_tensor([m]), two_int32.astype("int64"), n)

    result = RewriteTester(
        [x, n, m],
        [out],
        include=None,
        custom_rewrite=blockwise_unbatched_scalar_inputs,
    )
    two_int64 = constant(2, dtype="int64")
    inner_inputs = [inp.type() for inp in (x, n, m, two_int64, n)]
    lowered_op = JAXBlockwise(inner_inputs, [scalar_add(*inner_inputs)])
    result.assert_graph(lowered_op(x, n, m, two_int64, n))

    # A broadcastable input with core dims is not a scalar and stays vectorized
    core = tensor("core", shape=(1, 1))
    batched = tensor("batched", shape=(None,))
    out = Blockwise(mul, signature="(n),()->(n)")(core, batched)
    result = RewriteTester(
        [core, batched],
        [out],
        include=None,
        custom_rewrite=blockwise_unbatched_scalar_inputs,
    )
    result.assert_graph(out)


def test_unbatched_scalar_inputs():
    x = tensor("x", dtype="int32", shape=(2, 3))
    n = iscalar("n")
    v = tensor("v", shape=(None,))
    scalar_add = Blockwise(add, signature="(),(),(),()->()")
    out = scalar_add(x, n, v.shape[0], constant(2, dtype="int32"))

    fn, _ = compare_jax_and_py(
        [x, n, v],
        [out],
        [np.ones((2, 3), dtype="int32"), np.int32(3), np.zeros(4)],
        assert_fn=partial(np.testing.assert_array_equal, strict=True),
        jax_mode="JAX",
    )
    np.testing.assert_array_equal(
        fn(np.ones((2, 3), dtype="int32"), np.int32(7), np.zeros(6)),
        [np.full((2, 3), 16, dtype=out.dtype)],
        strict=True,
    )

    # With every input unbatched there is nothing to vectorize over
    a, b = iscalar("a"), lscalar("b")
    core_op = OpFromGraph([a, b], [a + b, a - b])
    blockwise_op = Blockwise(core_op, signature="(),()->(),()")
    compare_jax_and_py(
        [n, v],
        blockwise_op(n[None, None], v.shape[0], return_list=True),
        [np.int32(3), np.zeros(4)],
        assert_fn=partial(np.testing.assert_array_equal, strict=True),
        jax_mode=get_mode("JAX").excluding("local_useless_unbatched_blockwise"),
    )
