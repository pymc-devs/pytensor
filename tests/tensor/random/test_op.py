import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config, function
from pytensor.graph.replace import vectorize_graph
from pytensor.raise_op import Assert
from pytensor.tensor.math import eq
from pytensor.tensor.random import normal
from pytensor.tensor.random.basic import NormalRV
from pytensor.tensor.random.op import RandomVariable, default_rng
from pytensor.tensor.shape import specify_shape
from pytensor.tensor.type import iscalar, tensor


@pytest.fixture(scope="function", autouse=False)
def strict_test_value_flags():
    with config.change_flags(cxx="", compute_test_value="raise"):
        yield


def test_RandomVariable_basics(strict_test_value_flags):
    str_res = str(
        RandomVariable(
            "normal",
            signature="(),()->()",
            dtype="float32",
            inplace=False,
        )
    )

    assert str_res == 'normal_rv{"(),()->()"}'

    # `ndims_params` should be a `Sequence` type
    with pytest.raises(TypeError, match="^Parameter ndims_params*"):
        RandomVariable(
            "normal",
            0,
            0,
            config.floatX,
            inplace=True,
        )

    # `size` should be a `Sequence` type
    with pytest.raises(TypeError, match="^Parameter size*"):
        RandomVariable(
            "normal",
            0,
            [0, 0],
            config.floatX,
            inplace=True,
        )(0, 1, size={1, 2})

    # Confirm that `inplace` works
    rv = RandomVariable(
        "normal",
        signature="(),()->()",
        inplace=True,
    )

    assert rv.inplace
    assert rv.destroy_map == {0: [0]}

    # A no-params `RandomVariable`
    rv = RandomVariable(name="test_rv", signature="->()")

    with pytest.raises(TypeError):
        rv.make_node(rng=1)

    # `RandomVariable._infer_shape` should handle no parameters
    rv_shape = rv._infer_shape(pt.constant([]), (), [])
    assert rv_shape == ()

    # `dtype` is respected
    rv = RandomVariable("normal", signature="(),()->()", dtype="int32")
    with config.change_flags(compute_test_value="off"):
        rv_out = rv(0, 0)
        assert rv_out.dtype == "int32"
        rv_out = rv(0, 0, dtype="int64")
        assert rv_out.dtype == "int64"

        with pytest.raises(
            ValueError,
            match="Cannot change the dtype of a normal RV from int32 to float32",
        ):
            assert rv(0, 0, dtype="float32").dtype == "float32"


def test_RandomVariable_bcast(strict_test_value_flags):
    rv = RandomVariable("normal", 0, [0, 0], config.floatX, inplace=True)

    mu = tensor(dtype=config.floatX, shape=(1, None, None))
    mu.tag.test_value = np.zeros((1, 2, 3)).astype(config.floatX)
    sd = tensor(dtype=config.floatX, shape=(None, None))
    sd.tag.test_value = np.ones((2, 3)).astype(config.floatX)

    s1 = iscalar()
    s1.tag.test_value = 1
    s2 = iscalar()
    s2.tag.test_value = 2
    s3 = iscalar()
    s3.tag.test_value = 3
    s3 = Assert("testing")(s3, eq(s1, 1))

    res = rv(mu, sd, size=(s1, s2, s3))
    assert res.broadcastable == (False,) * 3

    size = pt.as_tensor((1, 2, 3), dtype=np.int32).astype(np.int64)
    res = rv(mu, sd, size=size)
    assert res.broadcastable == (True, False, False)

    res = rv(0, 1, size=pt.as_tensor(1, dtype=np.int64))
    assert res.broadcastable == (True,)

    res = rv(0, 1, size=(pt.as_tensor(1, dtype=np.int32), s3))
    assert res.broadcastable == (True, False)


def test_RandomVariable_bcast_specify_shape(strict_test_value_flags):
    rv = RandomVariable("normal", 0, [0, 0], config.floatX, inplace=True)

    s1 = pt.as_tensor(1, dtype=np.int64)
    s2 = iscalar()
    s2.tag.test_value = 2
    s3 = iscalar()
    s3.tag.test_value = 3
    s3 = Assert("testing")(s3, eq(s1, 1))

    size = specify_shape(pt.as_tensor([s1, s3, s2, s2, s1]), (5,))
    mu = tensor(dtype=config.floatX, shape=(None, None, 1))
    mu.tag.test_value = np.random.normal(size=(2, 2, 1)).astype(config.floatX)

    std = tensor(dtype=config.floatX, shape=(None, 1, 1))
    std.tag.test_value = np.ones((2, 1, 1)).astype(config.floatX)

    res = rv(mu, std, size=size)
    assert res.type.shape == (1, None, None, None, 1)


def test_RandomVariable_floatX(strict_test_value_flags):
    test_rv_op = RandomVariable(
        "normal",
        0,
        [0, 0],
        "floatX",
        inplace=True,
    )

    assert test_rv_op.dtype == "floatX"

    assert test_rv_op(0, 1).dtype == config.floatX

    new_floatX = "float64" if config.floatX == "float32" else "float32"

    with config.change_flags(floatX=new_floatX):
        assert test_rv_op(0, 1).dtype == new_floatX


@pytest.mark.parametrize(
    "seed, maker_op, numpy_res",
    [
        (3, default_rng, np.random.default_rng(3)),
    ],
)
def test_random_maker_op(strict_test_value_flags, seed, maker_op, numpy_res):
    seed = pt.as_tensor_variable(seed)
    z = function(inputs=[], outputs=[maker_op(seed)])()
    aes_res = z[0]
    assert maker_op.random_type.values_eq(aes_res, numpy_res)


def test_random_maker_ops_no_seed(strict_test_value_flags):
    # Testing the initialization when seed=None
    # Since internal states randomly generated,
    # we just check the output classes
    z = function(inputs=[], outputs=[default_rng()])()
    aes_res = z[0]
    assert isinstance(aes_res, np.random.Generator)


def test_RandomVariable_incompatible_size(strict_test_value_flags):
    rv_op = RandomVariable("normal", 0, [0, 0], config.floatX, inplace=True)
    with pytest.raises(
        ValueError, match="Size length is incompatible with batched dimensions"
    ):
        rv_op(np.zeros((1, 3)), 1, size=(3,))

    rv_op = RandomVariable("dirichlet", 0, [1], config.floatX, inplace=True)
    with pytest.raises(
        ValueError, match="Size length is incompatible with batched dimensions"
    ):
        rv_op(np.zeros((2, 4, 3)), 1, size=(4,))


class MultivariateRandomVariable(RandomVariable):
    name = "MultivariateRandomVariable"
    ndim_supp = 1
    ndims_params = (1, 2)
    dtype = "floatX"

    def _supp_shape_from_params(self, dist_params, param_shapes=None):
        return [dist_params[0].shape[-1]]


def test_multivariate_rv_infer_static_shape():
    """Test that infer shape for multivariate random variable works when a parameter must be broadcasted."""
    mv_op = MultivariateRandomVariable()

    param1 = tensor(shape=(10, 2, 3))
    param2 = tensor(shape=(10, 2, 3, 3))
    assert mv_op(param1, param2).type.shape == (10, 2, 3)

    param1 = tensor(shape=(2, 3))
    param2 = tensor(shape=(10, 2, 3, 3))
    assert mv_op(param1, param2).type.shape == (10, 2, 3)

    param1 = tensor(shape=(10, 2, 3))
    param2 = tensor(shape=(2, 3, 3))
    assert mv_op(param1, param2).type.shape == (10, 2, 3)

    param1 = tensor(shape=(10, 1, 3))
    param2 = tensor(shape=(2, 3, 3))
    assert mv_op(param1, param2).type.shape == (10, 2, 3)

    param1 = tensor(shape=(2, 3))
    param2 = tensor(shape=(2, 3, 3))
    assert mv_op(param1, param2, size=(10, 2)).type.shape == (10, 2, 3)


def test_vectorize():
    vec = tensor(shape=(None,))
    mat = tensor(shape=(None, None))

    # Test without size
    out = normal(vec)
    vect_node = vectorize_graph(out, {vec: mat}).owner
    assert isinstance(vect_node.op, NormalRV)
    assert vect_node.op.dist_params(vect_node)[0] is mat

    # Test with size, new size provided
    size = pt.as_tensor(np.array((3,), dtype="int64"))
    out = normal(vec, size=size)
    vect_node = vectorize_graph(out, {vec: mat, size: (2, 3)}).owner
    assert isinstance(vect_node.op, NormalRV)
    assert tuple(vect_node.op.size_param(vect_node).eval()) == (2, 3)
    assert vect_node.op.dist_params(vect_node)[0] is mat

    # Test with size, new size not provided
    out = normal(vec, size=(3,))
    vect_node = vectorize_graph(out, {vec: mat}).owner
    assert isinstance(vect_node.op, NormalRV)
    assert vect_node.op.dist_params(vect_node)[0] is mat
    assert tuple(
        vect_node.op.size_param(vect_node).eval(
            {mat: np.zeros((2, 3), dtype=config.floatX)}
        )
    ) == (2, 3)

    # Test parameter broadcasting
    mu = vec
    sigma = pt.as_tensor(np.array(1.0))
    out = normal(mu, sigma)
    new_mu = tensor("mu", shape=(10, 5))
    new_sigma = tensor("sigma", shape=(10,))
    vect_node = vectorize_graph(out, {mu: new_mu, sigma: new_sigma}).owner
    assert isinstance(vect_node.op, NormalRV)
    assert vect_node.default_output().type.shape == (10, 5)

    # Test parameter broadcasting with non-expanding size
    mu = vec
    sigma = pt.as_tensor(np.array(1.0))
    out = normal(mu, sigma, size=(5,))
    new_mu = tensor("mu", shape=(10, 5))
    new_sigma = tensor("sigma", shape=(10,))
    vect_node = vectorize_graph(out, {mu: new_mu, sigma: new_sigma}).owner
    assert isinstance(vect_node.op, NormalRV)
    assert vect_node.default_output().type.shape == (10, 5)

    mu = vec
    sigma = pt.as_tensor(np.array(1.0))
    out = normal(mu, sigma, size=(5,))
    new_mu = tensor("mu", shape=(1, 5))  # mu
    new_sigma = tensor("sigma", shape=(10,))  # sigma
    vect_node = vectorize_graph(out, {mu: new_mu, sigma: new_sigma}).owner
    assert isinstance(vect_node.op, NormalRV)
    assert vect_node.default_output().type.shape == (10, 5)

    # Test parameter broadcasting with expanding size
    mu = vec
    sigma = pt.as_tensor(np.array(1.0))
    out = normal(mu, sigma, size=(2, 5))
    new_mu = tensor("mu", shape=(1, 5))
    new_sigma = tensor("sigma", shape=(10,))
    vect_node = vectorize_graph(out, {mu: new_mu, sigma: new_sigma}).owner
    assert isinstance(vect_node.op, NormalRV)
    assert vect_node.default_output().type.shape == (10, 2, 5)


def test_vectorize_empty_size():
    scalar_mu = pt.scalar("scalar_mu")
    scalar_x = pt.random.normal(loc=scalar_mu, size=())
    assert scalar_x.type.shape == ()

    vector_mu = pt.vector("vector_mu", shape=(5,))
    vector_x = vectorize_graph(scalar_x, {scalar_mu: vector_mu})
    assert vector_x.type.shape == (5,)


def test_size_none_vs_empty():
    rv = RandomVariable(
        "normal",
        signature="(),()->()",
    )
    assert rv([0], [1], size=None).type.shape == (1,)

    with pytest.raises(
        ValueError, match="Size length is incompatible with batched dimensions"
    ):
        rv([0], [1], size=())
