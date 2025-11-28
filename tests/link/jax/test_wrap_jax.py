import numpy as np
import pytest

from pytensor import config, grad, wrap_jax
from pytensor.compile.sharedvalue import shared
from pytensor.link.jax.ops import JAXOp
from pytensor.scalar import all_types
from pytensor.tensor import TensorType, tensor
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


def test_two_inputs_single_output():
    rng = np.random.default_rng(1)
    x = tensor("x", shape=(2,))
    y = tensor("y", shape=(2,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    def f(x, y):
        return jax.nn.sigmoid(x + y)

    # Test with wrap_jax decorator
    out = wrap_jax(f)(x, y)
    grad_out = grad(out.sum(), [x, y])

    compare_jax_and_py([x, y], [out, *grad_out], test_values)
    with jax.disable_jit():
        compare_jax_and_py([x, y], [out, *grad_out], test_values)

    def f(x, y):
        return (jax.nn.sigmoid(x + y),)

    # Test direct JAXOp usage
    jax_op = JAXOp(
        [x.type, y.type],
        [TensorType(config.floatX, shape=(2,))],
        f,
    )
    out = jax_op(x, y)
    grad_out = grad(out.sum(), [x, y])
    compare_jax_and_py([x, y], [out, *grad_out], test_values)


def test_op_returns_list():
    x = tensor("x", shape=(2,))
    y = tensor("y", shape=(2,))

    test_values = [np.ones((2,)).astype(config.floatX) for inp in (x, y)]

    def f(x, y):
        return jax.nn.sigmoid(x + y)

    # Test direct JAXOp usage
    jax_op = JAXOp(
        [x.type, y.type],
        [TensorType(config.floatX, shape=(2,))],
        f,
    )

    with pytest.raises(TypeError, match="tuple of outputs"):
        out = jax_op(x, y)
        grad_out = grad(out.sum(), [x, y])
        compare_jax_and_py([x, y], [out, *grad_out], test_values)


def test_two_inputs_tuple_output():
    rng = np.random.default_rng(2)
    x = tensor("x", shape=(2,))
    y = tensor("y", shape=(2,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    def f(x, y):
        return jax.nn.sigmoid(x + y), y * 2

    # Test with wrap_jax decorator
    out1, out2 = wrap_jax(f)(x, y)
    grad_out = grad((out1 + out2).sum(), [x, y])

    compare_jax_and_py([x, y], [out1, out2, *grad_out], test_values)
    with jax.disable_jit():
        # must_be_device_array is False, because the with disabled jit compilation,
        # inputs are not automatically transformed to jax.Array anymore
        compare_jax_and_py(
            [x, y], [out1, out2, *grad_out], test_values, must_be_device_array=False
        )

    # Test direct JAXOp usage
    jax_op = JAXOp(
        [x.type, y.type],
        [TensorType(config.floatX, shape=(2,)), TensorType(config.floatX, shape=(2,))],
        f,
    )
    out1, out2 = jax_op(x, y)
    grad_out = grad((out1 + out2).sum(), [x, y])
    compare_jax_and_py([x, y], [out1, out2, *grad_out], test_values)


def test_two_inputs_list_output_one_unused_output():
    # One output is unused, to test whether the wrapper can handle DisconnectedType
    rng = np.random.default_rng(3)
    x = tensor("x", shape=(2,))
    y = tensor("y", shape=(2,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    def f(x, y):
        return (jax.nn.sigmoid(x + y), y * 2)

    # Test with wrap_jax decorator
    out, _ = wrap_jax(f)(x, y)
    grad_out = grad(out.sum(), [x, y])

    compare_jax_and_py([x, y], [out, *grad_out], test_values)
    with jax.disable_jit():
        compare_jax_and_py([x, y], [out, *grad_out], test_values)

    # Test direct JAXOp usage
    jax_op = JAXOp(
        [x.type, y.type],
        [TensorType(config.floatX, shape=(2,)), TensorType(config.floatX, shape=(2,))],
        f,
    )
    out, _ = jax_op(x, y)
    grad_out = grad(out.sum(), [x, y])
    compare_jax_and_py([x, y], [out, *grad_out], test_values)


def test_single_input_tuple_output():
    rng = np.random.default_rng(4)
    x = tensor("x", shape=(2,))
    test_values = [rng.normal(size=(x.type.shape)).astype(config.floatX)]

    def f(x):
        return jax.nn.sigmoid(x), x * 2

    # Test with wrap_jax decorator
    out1, out2 = wrap_jax(f)(x)
    grad_out = grad(out1.sum(), [x])

    compare_jax_and_py([x], [out1, out2, *grad_out], test_values)
    with jax.disable_jit():
        compare_jax_and_py(
            [x], [out1, out2, *grad_out], test_values, must_be_device_array=False
        )

    # Test direct JAXOp usage
    jax_op = JAXOp(
        [x.type],
        [TensorType(config.floatX, shape=(2,)), TensorType(config.floatX, shape=(2,))],
        f,
    )
    out1, out2 = jax_op(x)
    grad_out = grad(out1.sum(), [x])
    compare_jax_and_py([x], [out1, out2, *grad_out], test_values)


def test_scalar_input_tuple_output():
    rng = np.random.default_rng(5)
    x = tensor("x", shape=())
    test_values = [rng.normal(size=(x.type.shape)).astype(config.floatX)]

    def f(x):
        return jax.nn.sigmoid(x), x

    # Test with wrap_jax decorator
    out1, out2 = wrap_jax(f)(x)
    grad_out = grad(out1.sum(), [x])

    compare_jax_and_py([x], [out1, out2, *grad_out], test_values)
    with jax.disable_jit():
        compare_jax_and_py(
            [x], [out1, out2, *grad_out], test_values, must_be_device_array=False
        )

    # Test direct JAXOp usage
    jax_op = JAXOp(
        [x.type],
        [TensorType(config.floatX, shape=()), TensorType(config.floatX, shape=())],
        f,
    )
    out1, out2 = jax_op(x)
    grad_out = grad(out1.sum(), [x])
    compare_jax_and_py([x], [out1, out2, *grad_out], test_values)


def test_single_input_list_output():
    rng = np.random.default_rng(6)
    x = tensor("x", shape=(2,))
    test_values = [rng.normal(size=(x.type.shape)).astype(config.floatX)]

    def f(x):
        return (jax.nn.sigmoid(x), 2 * x)

    # Test with wrap_jax decorator
    out1, out2 = wrap_jax(f)(x)
    grad_out = grad(out1.sum(), [x])

    compare_jax_and_py([x], [out1, out2, *grad_out], test_values)
    with jax.disable_jit():
        compare_jax_and_py(
            [x], [out1, out2, *grad_out], test_values, must_be_device_array=False
        )

    # Test direct JAXOp usage, with unspecified output shapes
    jax_op = JAXOp(
        [x.type],
        [
            TensorType(config.floatX, shape=(None,)),
            TensorType(config.floatX, shape=(None,)),
        ],
        f,
    )
    out1, out2 = jax_op(x)
    grad_out = grad(out1.sum(), [x])
    compare_jax_and_py([x], [out1, out2, *grad_out], test_values)


def test_pytree_input_tuple_output():
    rng = np.random.default_rng(7)
    x = tensor("x", shape=(2,))
    y = tensor("y", shape=(2,))
    y_tmp = {"y": y, "y2": [y**2]}
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    @wrap_jax
    def f(x, y):
        return jax.nn.sigmoid(x), 2 * x + y["y"] + y["y2"][0]

    # Test with wrap_jax decorator
    out = f(x, y_tmp)
    grad_out = grad(out[1].sum(), [x, y])

    compare_jax_and_py([x, y], [out[0], out[1], *grad_out], test_values)

    with jax.disable_jit():
        compare_jax_and_py(
            [x, y], [out[0], out[1], *grad_out], test_values, must_be_device_array=False
        )


def test_pytree_input_pytree_output():
    rng = np.random.default_rng(8)
    x = tensor("x", shape=(3,))
    y = tensor("y", shape=(1,))
    y_tmp = {"a": y, "b": [y**2]}
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    @wrap_jax
    def f(x, y):
        return x, jax.tree_util.tree_map(lambda x: jax.numpy.exp(x), y)

    # Test with wrap_jax decorator
    out = f(x, y_tmp)
    grad_out = grad(out[1]["b"][0].sum(), [x, y])

    compare_jax_and_py([x, y], [out[0], out[1]["a"], *grad_out], test_values)

    with jax.disable_jit():
        compare_jax_and_py(
            [x, y],
            [out[0], out[1]["a"], *grad_out],
            test_values,
            must_be_device_array=False,
        )


def test_pytree_input_with_non_graph_args():
    rng = np.random.default_rng(9)
    x = tensor("x", shape=(3,))
    y = tensor("y", shape=(1,))
    y_tmp = {"a": y, "b": [y**2]}
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    @wrap_jax
    def f(x, y, depth, which_variable):
        if which_variable == "x":
            var = x
        elif which_variable == "y":
            var = y["a"] + y["b"][0]
        else:
            return "Unsupported argument"
        for _ in range(depth):
            var = jax.nn.sigmoid(var)
        return var

    # Test with wrap_jax decorator
    # arguments depth and which_variable are not part of the graph
    out = f(x, y_tmp, depth=3, which_variable="x")
    grad_out = grad(out.sum(), [x])
    compare_jax_and_py([x, y], [out[0], *grad_out], test_values)
    with jax.disable_jit():
        compare_jax_and_py([x, y], [out[0], *grad_out], test_values)

    out = f(x, y_tmp, depth=7, which_variable="y")
    grad_out = grad(out.sum(), [x])
    compare_jax_and_py([x, y], [out[0], *grad_out], test_values)
    with jax.disable_jit():
        compare_jax_and_py([x, y], [out[0], *grad_out], test_values)

    out = f(x, y_tmp, depth=10, which_variable="z")
    assert out == "Unsupported argument"


def test_unused_matrix_product():
    # A matrix output is unused, to test whether the wrapper can handle a
    # DisconnectedType with a larger dimension.

    rng = np.random.default_rng(10)
    x = tensor("x", shape=(3,))
    y = tensor("y", shape=(3,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    def f(x, y):
        return x[:, None] @ y[None], jax.numpy.exp(x)

    # Test with wrap_jax decorator
    out = wrap_jax(f)(x, y)
    grad_out = grad(out[1].sum(), [x])

    compare_jax_and_py([x, y], [out[1], *grad_out], test_values)

    with jax.disable_jit():
        compare_jax_and_py([x, y], [out[1], *grad_out], test_values)

    # Test direct JAXOp usage
    jax_op = JAXOp(
        [x.type, y.type],
        [
            TensorType(config.floatX, shape=(3, 3)),
            TensorType(config.floatX, shape=(3,)),
        ],
        f,
    )
    out = jax_op(x, y)
    grad_out = grad(out[1].sum(), [x])
    compare_jax_and_py([x, y], [out[1], *grad_out], test_values)


def test_unknown_static_shape():
    rng = np.random.default_rng(11)
    x = tensor("x", shape=(3,))
    y = tensor("y", shape=(3,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    x_cumsum = x.cumsum()  # Now x_cumsum has an unknown shape

    def f(x, y):
        return (x * jax.numpy.ones(3),)

    (out,) = wrap_jax(f)(x_cumsum, y)
    grad_out = grad(out.sum(), [x])

    compare_jax_and_py([x, y], [out, *grad_out], test_values)

    with jax.disable_jit():
        compare_jax_and_py([x, y], [out, *grad_out], test_values)

    # Test direct JAXOp usage
    jax_op = JAXOp(
        [x.type, y.type],
        [TensorType(config.floatX, shape=(None,))],
        f,
    )
    out = jax_op(x_cumsum, y)
    grad_out = grad(out.sum(), [x])
    compare_jax_and_py([x, y], [out, *grad_out], test_values)


def test_nn():
    eqx = pytest.importorskip("equinox")
    nn = pytest.importorskip("equinox.nn")

    rng = np.random.default_rng(13)
    x = tensor("x", shape=(3,))
    y = tensor("y", shape=(3,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    x = tensor("x", shape=(3,))
    y = tensor("y", shape=(3,))
    mlp = nn.MLP(3, 3, 3, depth=2, activation=jax.numpy.tanh, key=jax.random.key(0))
    mlp = eqx.tree_at(lambda m: m.layers[0].bias, mlp, y)

    @wrap_jax
    def f(x, mlp):
        return mlp(x)

    out = f(x, mlp)
    grad_out = grad(out.sum(), [x])

    compare_jax_and_py([x, y], [out, *grad_out], test_values)

    with jax.disable_jit():
        compare_jax_and_py([x, y], [out, *grad_out], test_values)


def test_no_inputs():
    def f():
        return jax.numpy.array(42.0)

    out = wrap_jax(f)()
    assert out.eval() == 42.0


def test_unknown_shape():
    x = tensor("x", shape=(None,))

    def f(x):
        return x * 2

    with pytest.raises(ValueError, match="Please provide inputs"):
        wrap_jax(f)(x)


def test_unknown_shape_with_eval():
    x = shared(np.ones(3))
    assert x.type.shape == (None,)

    def f(x):
        return x * 2

    out = wrap_jax(f)(x)
    grad_out = grad(out.sum(), [x])

    compare_jax_and_py([], [out, *grad_out], [])

    with jax.disable_jit():
        compare_jax_and_py([], [out, *grad_out], [], must_be_device_array=False)

    with pytest.raises(ValueError, match="Please provide inputs"):
        wrap_jax(f, allow_eval=False)(x)


def test_decorator_forms():
    x = tensor("x", shape=(3,))
    y = tensor("y", shape=(3,))

    @wrap_jax
    def the_name1(x, y):
        return (x + y).sum()

    @wrap_jax(allow_eval=True)
    def the_name2(x, y):
        return (x + y).sum()

    the_name1(x, y)
    the_name2(x, y)


def test_repr():
    x = tensor("x", shape=(3,))
    y = tensor("y", shape=(3,))

    def the_name(x, y):
        return (x + y).sum()

    jax_op = wrap_jax(the_name)
    assert "the_name" in repr(jax_op(x, y).owner.op)

    (grad_x, _) = grad(jax_op(x, y), [x, y])
    assert "vjp_the_name" in repr(grad_x.owner.op)


class TestDtypes:
    @pytest.mark.parametrize("in_dtype", list(map(str, all_types)))
    @pytest.mark.parametrize("out_dtype", list(map(str, all_types)))
    def test_different_in_output(self, in_dtype, out_dtype):
        x = tensor("x", shape=(3,), dtype=in_dtype)
        y = tensor("y", shape=(3,), dtype=in_dtype)

        if "int" in in_dtype:
            test_values = [
                np.random.randint(0, 10, size=(inp.type.shape)).astype(inp.type.dtype)
                for inp in (x, y)
            ]
        else:
            test_values = [
                np.random.normal(size=(inp.type.shape)).astype(inp.type.dtype)
                for inp in (x, y)
            ]

        @wrap_jax
        def f(x, y):
            out = jax.numpy.add(x, y)
            return jax.numpy.real(out).astype(out_dtype)

        out = f(x, y)
        assert out.dtype == out_dtype

        if "float" in in_dtype and "float" in out_dtype:
            grad_out = grad(out[0], [x, y])
            assert grad_out[0].dtype == in_dtype
            compare_jax_and_py([x, y], [out, *grad_out], test_values)
        else:
            compare_jax_and_py([x, y], [out], test_values)

        with jax.disable_jit():
            if "float" in in_dtype and "float" in out_dtype:
                compare_jax_and_py([x, y], [out, *grad_out], test_values)
            else:
                compare_jax_and_py([x, y], [out], test_values)

    @pytest.mark.parametrize("in1_dtype", list(map(str, all_types)))
    @pytest.mark.parametrize("in2_dtype", list(map(str, all_types)))
    def test_test_different_inputs(self, in1_dtype, in2_dtype):
        x = tensor("x", shape=(3,), dtype=in1_dtype)
        y = tensor("y", shape=(3,), dtype=in2_dtype)

        if "int" in in1_dtype:
            test_values = [np.random.randint(0, 10, size=(3,)).astype(x.type.dtype)]
        else:
            test_values = [np.random.normal(size=(3,)).astype(x.type.dtype)]
        if "int" in in2_dtype:
            test_values.append(np.random.randint(0, 10, size=(3,)).astype(y.type.dtype))
        else:
            test_values.append(np.random.normal(size=(3,)).astype(y.type.dtype))

        @wrap_jax
        def f(x, y):
            out = jax.numpy.add(x, y)
            return jax.numpy.real(out).astype(in1_dtype)

        out = f(x, y)
        assert out.dtype == in1_dtype

        if "float" in in1_dtype and "float" in in2_dtype:
            # In principle, the gradient should also be defined if the second input is
            # an integer, but it doesn't work for some reason.
            grad_out = grad(out[0], [x])
            assert grad_out[0].dtype == in1_dtype
            inputs = [x, y]
            outputs = [out, *grad_out]
        else:
            inputs = [x, y]
            outputs = [out]

        compare_jax_and_py(inputs, outputs, test_values)

        with jax.disable_jit():
            if "float" in in1_dtype and "float" in in2_dtype:
                compare_jax_and_py([x, y], [out, *grad_out], test_values)
            else:
                compare_jax_and_py([x, y], [out], test_values)


def test_mixed_static_shape():
    x_unknown = shared(np.ones((3,)))
    x_known = shared(np.ones((4,)), shape=(4,))

    def f(x1, x2):
        return jax.numpy.concatenate([x1, x2])

    assert wrap_jax(f)(x_known, x_known).type.shape == (8,)
    assert wrap_jax(f)(x_known, x_unknown).type.shape == (None,)
    assert wrap_jax(f)(x_unknown, x_known).type.shape == (None,)
