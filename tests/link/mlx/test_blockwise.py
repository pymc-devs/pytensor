import numpy as np

import pytensor.tensor as pt
from pytensor.tensor import tensor
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.math import Dot
from tests.link.mlx.test_basic import compare_mlx_and_py


# Equivalent blockwise to matmul but with dumb signature
odd_matmul = Blockwise(Dot(), signature="(i00,i01),(i10,i11)->(o00,o01)")


# @pytest.mark.parametrize("matmul_op", (matmul, odd_matmul))
# def test_matmul(matmul_op):
# rng = np.random.default_rng(14)
# a = tensor("a", shape=(2, 3, 5))
# b = tensor("b", shape=(2, 5, 3))
# test_values = [
# rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (a, b)
# ]
#
# out = matmul_op(a, b)
# assert isinstance(out.owner.op, Blockwise)
# fn, _ = compare_mlx_and_py([a, b], [out], test_values)
#
## Check we are not adding any unnecessary stuff
# jaxpr = str(jax.make_jaxpr(fn.vm.jit_fn)(*test_values))
# jaxpr = jaxpr.replace("name=jax_funcified_fgraph", "name=matmul")
# expected_jaxpr = str(jax.make_jaxpr(jax.jit(jax.numpy.matmul))(*test_values))
# assert jaxpr == expected_jaxpr


# conv1d
# (2, 100)
# (8, 100)
# mode = valid


def test_blockwise_conv1d():
    rng = np.random.default_rng(14)
    a = tensor("a", shape=(2, 100))
    b = tensor("b", shape=(2, 8))

    # a_test = np.broadcast_to(np.arange(100), (2, 100))
    a_test = rng.normal(size=(2, 100))
    b_test = rng.normal(size=(2, 8))
    # b_test = np.concatenate(
    # [
    #  np.ones((1, 8)),
    #  np.zeros((1, 8)),
    #  np.zeros((1, 8)),
    # np.array([1, 0, 0, 0, 0, 0, 0, 0]).reshape(1, 8),
    # np.array([1, 0, 0, 0, 0, 0, 0, 0]).reshape(1, 8),
    # ],
    # axis=0,
    # )

    test_values = [a_test, b_test]

    out = pt.signal.convolve1d(a, b, mode="valid")

    # assert isinstance(out.owner.op, Blockwise)
    compare_mlx_and_py([a, b], [out], test_values, must_be_device_array=True)
