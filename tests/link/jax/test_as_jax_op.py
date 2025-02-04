import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import as_jax_op, config, grad
from pytensor.graph.fg import FunctionGraph
from pytensor.scalar import all_types
from pytensor.tensor import tensor
from tests.link.jax.test_basic import compare_jax_and_py


def test_2in_1out():
    rng = np.random.default_rng(1)
    x = tensor("a", shape=(2,))
    y = tensor("b", shape=(2,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    @as_jax_op
    def f(x, y):
        return jax.nn.sigmoid(x + y)

    out = f(x, y)
    grad_out = grad(pt.sum(out), [x, y])

    fg = FunctionGraph([x, y], [out, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)
    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values)


def test_2in_tupleout():
    rng = np.random.default_rng(2)
    x = tensor("a", shape=(2,))
    y = tensor("b", shape=(2,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    @as_jax_op
    def f(x, y):
        return jax.nn.sigmoid(x + y), y * 2

    out, _ = f(x, y)
    grad_out = grad(pt.sum(out), [x, y])

    fg = FunctionGraph([x, y], [out, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)
    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values)


def test_2in_listout():
    rng = np.random.default_rng(3)
    x = tensor("a", shape=(2,))
    y = tensor("b", shape=(2,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    @as_jax_op
    def f(x, y):
        return [jax.nn.sigmoid(x + y), y * 2]

    out, _ = f(x, y)
    grad_out = grad(pt.sum(out), [x, y])

    fg = FunctionGraph([x, y], [out, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)
    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values)


def test_1din_tupleout():
    rng = np.random.default_rng(4)
    x = tensor("a", shape=(2,))
    test_values = [rng.normal(size=(x.type.shape)).astype(config.floatX)]

    @as_jax_op
    def f(x):
        return jax.nn.sigmoid(x), x * 2

    out, _ = f(x)
    grad_out = grad(pt.sum(out), [x])

    fg = FunctionGraph([x], [out, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)
    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values)


def test_0din_tupleout():
    rng = np.random.default_rng(5)
    x = tensor("a", shape=())
    test_values = [rng.normal(size=(x.type.shape)).astype(config.floatX)]

    @as_jax_op
    def f(x):
        return jax.nn.sigmoid(x), x

    out, _ = f(x)
    grad_out = grad(pt.sum(out), [x])

    fg = FunctionGraph([x], [out, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)
    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values)


def test_1in_listout():
    rng = np.random.default_rng(6)
    x = tensor("a", shape=(2,))
    test_values = [rng.normal(size=(x.type.shape)).astype(config.floatX)]

    @as_jax_op
    def f(x):
        return [jax.nn.sigmoid(x), 2 * x]

    out, _ = f(x)
    grad_out = grad(pt.sum(out), [x])

    fg = FunctionGraph([x], [out, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)

    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values)


def test_pytreein_tupleout():
    rng = np.random.default_rng(7)
    x = tensor("a", shape=(2,))
    y = tensor("b", shape=(2,))
    y_tmp = {"y": y, "y2": [y**2]}
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    @as_jax_op
    def f(x, y):
        return jax.nn.sigmoid(x), 2 * x + y["y"] + y["y2"][0]

    out = f(x, y_tmp)
    grad_out = grad(pt.sum(out[1]), [x, y])

    fg = FunctionGraph([x, y], [out[0], out[1], *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)

    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values)


def test_pytreein_pytreeout():
    rng = np.random.default_rng(8)
    x = tensor("a", shape=(3,))
    y = tensor("b", shape=(1,))
    y_tmp = {"a": y, "b": [y**2]}
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    @as_jax_op
    def f(x, y):
        return x, jax.tree_util.tree_map(lambda x: jnp.exp(x), y)

    out = f(x, y_tmp)
    grad_out = grad(pt.sum(out[1]["b"][0]), [x, y])

    fg = FunctionGraph([x, y], [out[0], out[1]["a"], *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)


def test_pytreein_pytreeout_w_nongraphargs():
    rng = np.random.default_rng(9)
    x = tensor("a", shape=(3,))
    y = tensor("b", shape=(1,))
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


def test_as_jax_op10():
    # Use "None" in shape specification and have a non-used output of higher rank
    rng = np.random.default_rng(10)
    x = tensor("a", shape=(3,))
    y = tensor("b", shape=(3,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    @as_jax_op
    def f(x, y):
        return x[:, None] @ y[None], jnp.exp(x)

    out = f(x, y)
    grad_out = grad(pt.sum(out[1]), [x])

    fg = FunctionGraph([x, y], [out[1], *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)

    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values)


def test_as_jax_op11():
    # Test unknown static shape
    rng = np.random.default_rng(11)
    x = tensor("a", shape=(3,))
    y = tensor("b", shape=(3,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    x = pt.cumsum(x)  # Now x has an unknown shape

    @as_jax_op
    def f(x, y):
        return x * jnp.ones(3)

    out = f(x, y)
    grad_out = grad(pt.sum(out), [x])

    fg = FunctionGraph([x, y], [out, *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)

    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values)


def test_as_jax_op12():
    # Test non-array return values
    rng = np.random.default_rng(12)
    x = tensor("a", shape=(3,))
    y = tensor("b", shape=(3,))
    test_values = [
        rng.normal(size=(inp.type.shape)).astype(config.floatX) for inp in (x, y)
    ]

    @as_jax_op
    def f(x, y, message):
        return x * jnp.ones(3), "Success: " + message

    out = f(x, y, "Hi")
    grad_out = grad(pt.sum(out[0]), [x])

    fg = FunctionGraph([x, y], [out[0], *grad_out])
    fn, _ = compare_jax_and_py(fg, test_values)

    with jax.disable_jit():
        fn, _ = compare_jax_and_py(fg, test_values)


def test_as_jax_op13():
    # Test nested functions
    rng = np.random.default_rng(13)
    x = tensor("a", shape=(3,))
    y = tensor("b", shape=(3,))
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
        x = tensor("a", shape=(3,), dtype=in_dtype)
        y = tensor("b", shape=(3,), dtype=in_dtype)

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
        x = tensor("a", shape=(3,), dtype=in1_dtype)
        y = tensor("b", shape=(3,), dtype=in2_dtype)

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
