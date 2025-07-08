import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import as_jax_op, config, grad
from pytensor.graph.fg import FunctionGraph
from pytensor.link.jax.ops import JAXOp
from pytensor.scalar import all_types
from pytensor.tensor import TensorType, tensor
from tests.link.jax.test_basic import compare_jax_and_py


def test_two_inputs_single_output():
    rng = np.random.default_rng(1)
    x = tensor("x", shape=(2,))
    y = tensor("y", shape=(2,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    def f(x, y):
        return jax.nn.sigmoid(x + y)

    # Test with as_jax_op decorator
    out = as_jax_op(f)(x, y)
    grad_out = grad(pt.sum(out), [x, y])

    fg = FunctionGraph([x, y], [out, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)
    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values)

    # Test direct JAXOp usage
    jax_op = JAXOp(
        [x.type, y.type],
        [TensorType(config.floatX, shape=(2,))],
        f,
    )
    out = jax_op(x, y)
    grad_out = grad(pt.sum(out), [x, y])
    fg = FunctionGraph([x, y], [out, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)


def test_two_inputs_tuple_output():
    rng = np.random.default_rng(2)
    x = tensor("x", shape=(2,))
    y = tensor("y", shape=(2,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    def f(x, y):
        return jax.nn.sigmoid(x + y), y * 2

    # Test with as_jax_op decorator
    out1, out2 = as_jax_op(f)(x, y)
    grad_out = grad(pt.sum(out1 + out2), [x, y])

    fg = FunctionGraph([x, y], [out1, out2, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)
    with jax.disable_jit():
        # must_be_device_array is False, because the with disabled jit compilation,
        # inputs are not automatically transformed to jax.Array anymore
        fn, _ = compare_jax_and_py(fg, test_values, must_be_device_array=False)

    # Test direct JAXOp usage
    jax_op = JAXOp(
        [x.type, y.type],
        [TensorType(config.floatX, shape=(2,)), TensorType(config.floatX, shape=(2,))],
        f,
    )
    out1, out2 = jax_op(x, y)
    grad_out = grad(pt.sum(out1 + out2), [x, y])
    fg = FunctionGraph([x, y], [out1, out2, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)


def test_two_inputs_list_output_one_unused_output():
    # One output is unused, to test whether the wrapper can handle DisconnectedType
    rng = np.random.default_rng(3)
    x = tensor("x", shape=(2,))
    y = tensor("y", shape=(2,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    def f(x, y):
        return [jax.nn.sigmoid(x + y), y * 2]

    # Test with as_jax_op decorator
    out, _ = as_jax_op(f)(x, y)
    grad_out = grad(pt.sum(out), [x, y])

    fg = FunctionGraph([x, y], [out, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)
    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values)

    # Test direct JAXOp usage
    jax_op = JAXOp(
        [x.type, y.type],
        [TensorType(config.floatX, shape=(2,)), TensorType(config.floatX, shape=(2,))],
        f,
    )
    out, _ = jax_op(x, y)
    grad_out = grad(pt.sum(out), [x, y])
    fg = FunctionGraph([x, y], [out, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)


def test_single_input_tuple_output():
    rng = np.random.default_rng(4)
    x = tensor("x", shape=(2,))
    test_values = [rng.normal(size=(x.type.shape)).astype(config.floatX)]

    def f(x):
        return jax.nn.sigmoid(x), x * 2

    # Test with as_jax_op decorator
    out1, out2 = as_jax_op(f)(x)
    grad_out = grad(pt.sum(out1), [x])

    fg = FunctionGraph([x], [out1, out2, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)
    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values, must_be_device_array=False)

    # Test direct JAXOp usage
    jax_op = JAXOp(
        [x.type],
        [TensorType(config.floatX, shape=(2,)), TensorType(config.floatX, shape=(2,))],
        f,
    )
    out1, out2 = jax_op(x)
    grad_out = grad(pt.sum(out1), [x])
    fg = FunctionGraph([x], [out1, out2, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)


def test_scalar_input_tuple_output():
    rng = np.random.default_rng(5)
    x = tensor("x", shape=())
    test_values = [rng.normal(size=(x.type.shape)).astype(config.floatX)]

    def f(x):
        return jax.nn.sigmoid(x), x

    # Test with as_jax_op decorator
    out1, out2 = as_jax_op(f)(x)
    grad_out = grad(pt.sum(out1), [x])

    fg = FunctionGraph([x], [out1, out2, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)
    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values, must_be_device_array=False)

    # Test direct JAXOp usage
    jax_op = JAXOp(
        [x.type],
        [TensorType(config.floatX, shape=()), TensorType(config.floatX, shape=())],
        f,
    )
    out1, out2 = jax_op(x)
    grad_out = grad(pt.sum(out1), [x])
    fg = FunctionGraph([x], [out1, out2, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)


def test_single_input_list_output():
    rng = np.random.default_rng(6)
    x = tensor("x", shape=(2,))
    test_values = [rng.normal(size=(x.type.shape)).astype(config.floatX)]

    def f(x):
        return [jax.nn.sigmoid(x), 2 * x]

    # Test with as_jax_op decorator
    out1, out2 = as_jax_op(f)(x)
    grad_out = grad(pt.sum(out1), [x])

    fg = FunctionGraph([x], [out1, out2, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)
    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values, must_be_device_array=False)

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
    grad_out = grad(pt.sum(out1), [x])
    fg = FunctionGraph([x], [out1, out2, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)


def test_pytree_input_tuple_output():
    rng = np.random.default_rng(7)
    x = tensor("x", shape=(2,))
    y = tensor("y", shape=(2,))
    y_tmp = {"y": y, "y2": [y**2]}
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    @as_jax_op
    def f(x, y):
        return jax.nn.sigmoid(x), 2 * x + y["y"] + y["y2"][0]

    # Test with as_jax_op decorator
    out = f(x, y_tmp)
    grad_out = grad(pt.sum(out[1]), [x, y])

    fg = FunctionGraph([x, y], [out[0], out[1], *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)

    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values, must_be_device_array=False)


def test_pytree_input_pytree_output():
    rng = np.random.default_rng(8)
    x = tensor("x", shape=(3,))
    y = tensor("y", shape=(1,))
    y_tmp = {"a": y, "b": [y**2]}
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    @as_jax_op
    def f(x, y):
        return x, jax.tree_util.tree_map(lambda x: jnp.exp(x), y)

    # Test with as_jax_op decorator
    out = f(x, y_tmp)
    grad_out = grad(pt.sum(out[1]["b"][0]), [x, y])

    fg = FunctionGraph([x, y], [out[0], out[1]["a"], *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)

    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values, must_be_device_array=False)


def test_pytree_input_with_non_graph_args():
    rng = np.random.default_rng(9)
    x = tensor("x", shape=(3,))
    y = tensor("y", shape=(1,))
    y_tmp = {"a": y, "b": [y**2]}
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    @as_jax_op
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

    # Test with as_jax_op decorator
    # arguments depth and which_variable are not part of the graph
    out = f(x, y_tmp, depth=3, which_variable="x")
    grad_out = grad(pt.sum(out), [x])
    fg = FunctionGraph([x, y], [out[0], *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)
    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values)

    out = f(x, y_tmp, depth=7, which_variable="y")
    grad_out = grad(pt.sum(out), [x])
    fg = FunctionGraph([x, y], [out[0], *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)
    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values)

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
        return x[:, None] @ y[None], jnp.exp(x)

    # Test with as_jax_op decorator
    out = as_jax_op(f)(x, y)
    grad_out = grad(pt.sum(out[1]), [x])

    fg = FunctionGraph([x, y], [out[1], *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)

    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values)

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
    grad_out = grad(pt.sum(out[1]), [x])
    fg = FunctionGraph([x, y], [out[1], *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)


def test_unknown_static_shape():
    rng = np.random.default_rng(11)
    x = tensor("x", shape=(3,))
    y = tensor("y", shape=(3,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    x_cumsum = pt.cumsum(x)  # Now x_cumsum has an unknown shape

    def f(x, y):
        return x * jnp.ones(3)

    out = as_jax_op(f)(x_cumsum, y)
    grad_out = grad(pt.sum(out), [x])

    fg = FunctionGraph([x, y], [out, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)

    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values)

    # Test direct JAXOp usage
    jax_op = JAXOp(
        [x.type, y.type],
        [TensorType(config.floatX, shape=(None,))],
        f,
    )
    out = jax_op(x_cumsum, y)
    grad_out = grad(pt.sum(out), [x])
    fg = FunctionGraph([x, y], [out, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)


def test_nested_functions():
    rng = np.random.default_rng(13)
    x = tensor("x", shape=(3,))
    y = tensor("y", shape=(3,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    @as_jax_op
    def f_internal(y):
        def f_ret(t):
            return y + t

        def f_ret2(t):
            return f_ret(t) + t**2

        return f_ret, y**2 * jnp.ones(1), f_ret2

    f, y_pow, f2 = f_internal(y)

    @as_jax_op
    def f_outer(x, dict_other):
        f, y_pow = dict_other["func"], dict_other["y"]
        return x * jnp.ones(3), f(x) * y_pow

    out = f_outer(x, {"func": f, "y": y_pow})
    grad_out = grad(pt.sum(out[1]), [x])

    fg = FunctionGraph([x, y], [out[1], *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)

    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values)


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

        @as_jax_op
        def f(x, y):
            out = jnp.add(x, y)
            return jnp.real(out).astype(out_dtype)

        out = f(x, y)
        assert out.dtype == out_dtype

        if "float" in in_dtype and "float" in out_dtype:
            grad_out = grad(out[0], [x, y])
            assert grad_out[0].dtype == in_dtype
            fg = FunctionGraph([x, y], [out, *grad_out])
        else:
            fg = FunctionGraph([x, y], [out])

        fn, _ = compare_jax_and_py(fg, test_values)

        with jax.disable_jit():
            fn, _ = compare_jax_and_py(fg, test_values)

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

        @as_jax_op
        def f(x, y):
            out = jnp.add(x, y)
            return jnp.real(out).astype(in1_dtype)

        out = f(x, y)
        assert out.dtype == in1_dtype

        if "float" in in1_dtype and "float" in in2_dtype:
            # In principle, the gradient should also be defined if the second input is
            # an integer, but it doesn't work for some reason.
            grad_out = grad(out[0], [x])
            assert grad_out[0].dtype == in1_dtype
            fg = FunctionGraph([x, y], [out, *grad_out])
        else:
            fg = FunctionGraph([x, y], [out])

        fn, _ = compare_jax_and_py(fg, test_values)

        with jax.disable_jit():
            fn, _ = compare_jax_and_py(fg, test_values)
