import re

import numpy as np
import pytest

from pytensor import In, Mode, function
from pytensor.compile import get_default_mode
from pytensor.scalar import (
    Composite,
    as_scalar,
    cos,
    exp,
    float16,
    float32,
    float64,
    identity,
    int64,
    sin,
)
from pytensor.scalar.loop import ScalarLoop
from pytensor.tensor import exp as tensor_exp
from pytensor.tensor import lvector
from pytensor.tensor.elemwise import Elemwise


mode = pytest.mark.parametrize(
    "mode",
    [
        Mode(optimizer="fast_compile", linker="py"),
        Mode(optimizer="fast_compile", linker="cvm"),
    ],
)


@mode
def test_single_output(mode):
    n_steps = int64("n_steps")
    x0 = float64("x0")
    const = float64("const")
    x = x0 + const

    op = ScalarLoop(init=[x0], constant=[const], update=[x])
    x = op(n_steps, x0, const)

    fn = function([n_steps, x0, const], x, mode=mode)
    np.testing.assert_allclose(fn(5, 0, 1), 5)
    np.testing.assert_allclose(fn(5, 0, 2), 10)
    np.testing.assert_allclose(fn(4, 3, -1), -1)


@mode
def test_multiple_output(mode):
    n_steps = int64("n_steps")
    x0 = float64("x0")
    y0 = int64("y0")
    const = float64("const")
    x = x0 + const
    y = y0 + 1

    op = ScalarLoop(init=[x0, y0], constant=[const], update=[x, y])
    x, y = op(n_steps, x0, y0, const)

    fn = function([n_steps, x0, y0, const], [x, y], mode=mode)

    res_x, res_y = fn(n_steps=5, x0=0, y0=0, const=1)
    np.testing.assert_allclose(res_x, 5)
    np.testing.assert_allclose(res_y, 5)

    res_x, res_y = fn(n_steps=5, x0=0, y0=0, const=2)
    np.testing.assert_allclose(res_x, 10)
    np.testing.assert_allclose(res_y, 5)

    res_x, res_y = fn(n_steps=4, x0=3, y0=2, const=-1)
    np.testing.assert_allclose(res_x, -1)
    np.testing.assert_allclose(res_y, 6)


@mode
def test_input_not_aliased_to_update(mode):
    n_steps = int64("n_steps")
    x0 = float64("x0")
    y0 = float64("y0")
    const = float64("const")

    def update(x_prev, y_prev):
        x = x_prev + const
        # y depends on x_prev, so x_prev should not be overriden by x!
        y = y_prev + x_prev
        return [x, y]

    op = ScalarLoop(init=[x0, y0], constant=[const], update=update(x0, y0))
    _x, y = op(n_steps, x0, y0, const)

    fn = function([n_steps, x0, y0, const], y, mode=mode)
    np.testing.assert_allclose(fn(n_steps=1, x0=0, y0=0, const=1), 0.0)
    np.testing.assert_allclose(fn(n_steps=2, x0=0, y0=0, const=1), 1.0)
    np.testing.assert_allclose(fn(n_steps=3, x0=0, y0=0, const=1), 3.0)
    np.testing.assert_allclose(fn(n_steps=4, x0=0, y0=0, const=1), 6.0)
    np.testing.assert_allclose(fn(n_steps=5, x0=0, y0=0, const=1), 10.0)


@mode
def test_until(mode):
    n_steps = int64("n_steps")
    x0 = float64("x0")
    x = x0 + 1
    until = x >= 10

    op = ScalarLoop(init=[x0], update=[x], until=until)
    fn = function([n_steps, x0], op(n_steps, x0), mode=mode)
    np.testing.assert_allclose(fn(n_steps=20, x0=0), [10, True])
    np.testing.assert_allclose(fn(n_steps=20, x0=1), [10, True])
    np.testing.assert_allclose(fn(n_steps=5, x0=1), [6, False])


def test_update_missing_error():
    x0 = float64("x0")
    const = float64("const")
    with pytest.raises(
        ValueError, match="An update must be given for each init variable"
    ):
        ScalarLoop(init=[x0], constant=[const], update=[])


def test_init_update_type_error():
    x0 = float32("x0")
    const = float64("const")
    x = x0 + const
    assert x.type.dtype == "float64"
    with pytest.raises(TypeError, match="Init and update types must be the same"):
        ScalarLoop(init=[x0], constant=[const], update=[x])


def test_rebuild_dtype():
    n_steps = int64("n_steps")
    x0 = float64("x0")
    const = float64("const")
    x = x0 + const
    op = ScalarLoop(init=[x0], constant=[const], update=[x])

    # If x0 is float32 but const is still float64, the output type will not be able to match
    x0_float32 = float32("x0_float32")
    with pytest.raises(TypeError, match="Init and update types must be the same"):
        op(n_steps, x0_float32, const)

    # Now it should be fine
    const_float32 = float32("const_float32")
    y = op(n_steps, x0_float32, const_float32)
    assert y.dtype == "float32"


def test_non_scalar_error():
    x0 = float64("x0")
    x = as_scalar(tensor_exp(x0))

    with pytest.raises(
        TypeError,
        match="The fgraph of ScalarLoop must be exclusively composed of scalar operations",
    ):
        ScalarLoop(init=[x0], constant=[], update=[x])


def test_n_steps_type_error():
    x0 = float64("x0")
    const = float64("const")
    x = x0 + const

    op = ScalarLoop(init=[x0], constant=[const], update=[x])
    with pytest.raises(
        TypeError, match=re.escape("(n_steps) must be of integer type. Got float64")
    ):
        op(float64("n_steps"), x0, const)


def test_same_out_as_inp_error():
    xtm2 = float64("xtm2")
    xtm1 = float64("xtm1")
    x = xtm2 + xtm1

    with pytest.raises(
        ValueError, match="Some inputs and outputs are the same variable"
    ):
        ScalarLoop(init=[xtm2, xtm1], update=[xtm1, x])


@mode
def test_lags(mode):
    n_steps = int64("n_steps")
    xtm2 = float64("xtm2")
    xtm1 = float64("xtm1")
    x = xtm2 + xtm1

    op = ScalarLoop(init=[xtm2, xtm1], update=[identity(xtm1), x])
    _, x = op(n_steps, xtm2, xtm1)

    fn = function([n_steps, xtm2, xtm1], x, mode=mode)
    np.testing.assert_allclose(fn(n_steps=5, xtm2=0, xtm1=1), 8)


@mode
def test_inner_composite(mode):
    n_steps = int64("n_steps")
    x = float64("x")

    one = Composite([x], [cos(exp(x)) ** 2 + sin(exp(x)) ** 2])(x)

    op = ScalarLoop(init=[x], update=[one + x])
    y = op(n_steps, x)

    fn = function([n_steps, x], y, mode=mode)
    np.testing.assert_allclose(fn(n_steps=5, x=2.53), 2.53 + 5)

    # Now with a dtype that must be rebuilt
    x16 = float16("x16")
    y16 = op(n_steps, x16)
    assert y16.type.dtype == "float16"

    fn16 = function([n_steps, x16], y16, mode=mode)
    out16 = fn16(n_steps=3, x16=np.array(4.73, dtype="float16"))
    np.testing.assert_allclose(
        out16,
        4.73 + 3,
        rtol=1e-3,
    )
    out16overflow = fn16(n_steps=9, x16=np.array(4.73, dtype="float16"))
    assert out16overflow.dtype == "float16"
    # with this dtype overflow happens
    assert np.isnan(out16overflow)


@mode
def test_inner_loop(mode):
    n_steps = int64("n_steps")
    x = float64("x")

    x_in = float64("x_in")
    inner_loop_op = ScalarLoop(init=[x_in], update=[x_in + 1])

    outer_loop_op = ScalarLoop(
        init=[x], update=[inner_loop_op(n_steps, x)], constant=[n_steps]
    )
    y = outer_loop_op(n_steps, x, n_steps)

    fn = function([n_steps, x], y, mode=mode)
    np.testing.assert_allclose(fn(n_steps=5, x=0), 5**2)
    np.testing.assert_allclose(fn(n_steps=7, x=0), 7**2)
    np.testing.assert_allclose(fn(n_steps=7, x=1), 7**2 + 1)

    # Now with a dtype that must be rebuilt
    x16 = float16("x16")
    y16 = outer_loop_op(n_steps, x16, n_steps)
    assert y16.type.dtype == "float16"

    fn16 = function([n_steps, x16], y16, mode=mode)
    out16 = fn16(n_steps=3, x16=np.array(2.5, dtype="float16"))
    assert out16.dtype == "float16"
    np.testing.assert_allclose(
        out16,
        3**2 + 2.5,
    )


@pytest.mark.parametrize("mutate_arg_idx", (0, 1, 2, 3))
def test_elemwise_inplace(mutate_arg_idx):
    x0 = int64("x0")
    y0 = int64("y0")
    c = int64("c")
    x = x0 - y0 + c
    y = y0 - x0 + c
    op = Elemwise(ScalarLoop(init=[x0, y0], constant=[c], update=[x, y]))

    n_steps = lvector("n_steps")
    x0v = lvector("x0")
    y0v = lvector("y0")
    cv = lvector("c")
    xv, yv = op(n_steps, x0v, y0v, cv)

    inputs = [
        In(inp, mutable=i == mutate_arg_idx)
        for i, inp in enumerate([n_steps, x0v, y0v, cv])
    ]

    fn = function(
        inputs,
        [xv, yv],
        mode=get_default_mode().including("inplace"),
    )
    fn.dprint()
    elem_op = fn.maker.fgraph.outputs[0].owner.op
    assert isinstance(elem_op, Elemwise) and isinstance(elem_op.scalar_op, ScalarLoop)
    destroy_map = elem_op.destroy_map
    assert destroy_map == {0: [mutate_arg_idx]}

    n_test = np.array([1, 4, 8], dtype="int64")
    x0v_test = np.array([0, 0, 0], dtype="int64")
    y0v_test = np.array([1, 1, 1], dtype="int64")
    cv_test = np.array([0, 0, 0], dtype="int64")

    xv_res, yv_res = fn(n_test, x0v_test, y0v_test, cv_test)
    assert xv_res is (n_test, x0v_test, y0v_test, cv_test)[mutate_arg_idx]
    np.testing.assert_allclose(xv_res, [-1, -8, -128])
    np.testing.assert_allclose(yv_res, [1, 8, 128])


def test_identical_loops_share_inner_graph():
    x0 = float64("x0")
    c = float64("c")

    op1 = ScalarLoop(init=[x0], constant=[c], update=[x0 + c])
    op2 = ScalarLoop(init=[x0], constant=[c], update=[x0 + c])

    assert op1 == op2
    assert hash(op1) == hash(op2)
    assert op1.fgraph == op2.fgraph

    # Two loops with the same structure but different outer inputs.
    # MergeOptimizer can't collapse the Apply nodes (different inputs),
    # but both should reference the same inner Op after merging.
    n = int64("n")
    a, b, c_val, d = float64("a"), float64("b"), float64("c_val"), float64("d")
    y1 = op1(n, a, b)
    y2 = op2(n, c_val, d)

    fn = function(
        [n, a, b, c_val, d], [y1, y2], mode=Mode(optimizer="merge", linker="py")
    )
    nodes = fn.maker.fgraph.toposort()
    loop_nodes = [nd for nd in nodes if isinstance(nd.op, ScalarLoop)]
    assert len(loop_nodes) == 2
    assert loop_nodes[0].op is loop_nodes[1].op
