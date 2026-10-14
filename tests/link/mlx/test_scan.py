import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import function
from pytensor.scan import until
from pytensor.scan.basic import scan
from tests.link.mlx.test_basic import compare_mlx_and_py, mlx_mode
from tests.scan.test_basic import ScanCompatibilityTests


mx = pytest.importorskip("mlx.core")
from pytensor.link.mlx.dispatch.scan import _functional_set


@pytest.mark.parametrize("view", [None, (-1,), slice(-2, None, None)])
def test_scan_sit_sot(view):
    x0 = pt.scalar("x0", dtype="float32")
    xs = scan(
        lambda xtm1: xtm1 + 1,
        outputs_info=[x0],
        n_steps=10,
        return_updates=False,
    )
    if view:
        xs = xs[view]
    compare_mlx_and_py([x0], [xs], [np.float32(np.e)])


@pytest.mark.parametrize("view", [None, (-1,), slice(-4, -1, None)])
def test_scan_mit_sot(view):
    x0 = pt.vector("x0", dtype="float32", shape=(3,))
    xs = scan(
        lambda xtm3, xtm1: xtm3 + xtm1 + 1,
        outputs_info=[{"initial": x0, "taps": [-3, -1]}],
        n_steps=10,
        return_updates=False,
    )
    if view:
        xs = xs[view]
    compare_mlx_and_py([x0], [xs], [np.full((3,), np.e, dtype="float32")])


def test_scan_with_sequence_and_non_seq():
    # RNN-style recurrence over a sequence with a shared weight matrix.
    xs = pt.matrix("xs", dtype="float32")
    h0 = pt.vector("h0", dtype="float32", shape=(3,))
    W = pt.matrix("W", dtype="float32", shape=(3, 3))
    hs = scan(
        lambda x_t, h_tm1, W: pt.tanh(x_t + h_tm1 @ W),
        sequences=[xs],
        outputs_info=[h0],
        non_sequences=[W],
        return_updates=False,
    )
    rng = np.random.default_rng(0)
    compare_mlx_and_py(
        [xs, h0, W],
        [hs],
        [
            rng.standard_normal((5, 3)).astype("float32"),
            np.zeros(3, dtype="float32"),
            (0.1 * rng.standard_normal((3, 3))).astype("float32"),
        ],
    )


def test_scan_multiple_outputs():
    # One recurring (sit_sot) and one mapped (nit_sot) output.
    s = pt.vector("s", dtype="float32")

    def step(s_t, acc):
        return acc + s_t, s_t * s_t

    acc, sq = scan(
        step,
        sequences=[s],
        outputs_info=[pt.zeros((), dtype="float32"), None],
        return_updates=False,
    )
    compare_mlx_and_py([s], [acc, sq], [np.arange(1, 5, dtype="float32")])


def test_scan_nit_sot_only():
    # Pure tiling/map from a non-sequence with an explicit ``n_steps`` (no
    # recurring buffer or sequence to infer the step count from).
    w = pt.scalar("w", dtype="float32")
    ys = scan(
        lambda w: w * 2,
        outputs_info=[None],
        non_sequences=[w],
        n_steps=5,
        return_updates=False,
    )
    compare_mlx_and_py([w], [ys], [np.float32(3.0)])


def test_scan_multiple_recurring_states():
    # MIT-SOT (taps -2, -1) and SIT-SOT and a NIT-SOT map in one scan.
    x0 = pt.vector("x0", dtype="float32", shape=(2,))

    def step(xtm2, xtm1, stm1):
        x_t = xtm2 + xtm1
        return x_t, stm1 + x_t, x_t * 2

    xs, ss, ys = scan(
        step,
        outputs_info=[{"initial": x0, "taps": [-2, -1]}, pt.zeros((), "float32"), None],
        n_steps=6,
        return_updates=False,
    )
    compare_mlx_and_py([x0], [xs, ss, ys], [np.array([1.0, 1.0], dtype="float32")])


def test_scan_int_dtype_preserved():
    # Integer recurrence: dtype must be preserved (no float upcast).
    x0 = pt.scalar("x0", dtype="int32")
    xs = scan(
        lambda xtm1: xtm1 + 1,
        outputs_info=[x0],
        n_steps=5,
        return_updates=False,
    )

    def assert_int(mlx_res, py_res):
        np.testing.assert_array_equal(mlx_res, py_res)
        assert np.asarray(mlx_res).dtype == np.asarray(py_res).dtype == np.int32

    compare_mlx_and_py([x0], [xs], [np.int32(0)], assert_fn=assert_int)


@pytest.mark.parametrize("dtype", ["float32", "int64"])
def test_scan_zero_steps(dtype):
    x0 = pt.scalar("x0", dtype=dtype)
    xs = scan(
        lambda xtm1: xtm1 + 1,
        outputs_info=[x0],
        n_steps=0,
        return_updates=False,
    )

    def assert_empty(mlx_res, py_res):
        np.testing.assert_array_equal(mlx_res, py_res)
        assert mlx_res.shape == py_res.shape == (0,)
        assert np.asarray(mlx_res).dtype == py_res.dtype == np.dtype(dtype)

    compare_mlx_and_py([x0], [xs], [np.asarray(3, dtype=dtype)], assert_fn=assert_empty)


@pytest.mark.parametrize("n_steps, static_shape", [(0, True), (0, False), (3, False)])
def test_scan_nit_sot_shape(n_steps, static_shape):
    w = pt.vector("w", dtype="float32", shape=(3,) if static_shape else (None,))
    ys = scan(
        lambda w: w * 2,
        outputs_info=[None],
        non_sequences=[w],
        n_steps=n_steps,
        return_updates=False,
    )
    w_val = np.arange(3, dtype="float32")
    if n_steps == 0 and not static_shape:
        with pytest.raises(
            NotImplementedError, match=r"static trailing shape.*zero-step NIT-SOT"
        ):
            function([w], ys, mode=mlx_mode)(w_val)
        return

    if n_steps == 0:
        # The Python reference incorrectly gives vector NIT-SOTs shape (0, 0).
        result = function([w], ys, mode=mlx_mode)(w_val)
    else:
        _, [result] = compare_mlx_and_py([w], [ys], [w_val])
    assert result.shape == (n_steps, 3)
    assert result.dtype == mx.float32


def test_functional_set_exact():
    buffer_np = np.array(
        [
            [2**100, np.nan, np.inf, -np.inf],
            [1.0, -2.0, 3.0, -4.0],
            [-np.inf, 2**100, np.nan, np.inf],
        ],
        dtype="float32",
    )
    vals_np = np.array(
        [[1.0, -2.0, 0.0, -0.0], [-0.0, 0.0, 3.0, -4.0]], dtype="float32"
    )
    buffer, vals = mx.array(buffer_np), mx.array(vals_np)
    set_rows = mx.compile(lambda buffer, vals: _functional_set(buffer, [0, 2], vals))
    result = np.asarray(set_rows(buffer, vals))
    expected = buffer_np.copy()
    expected[[0, 2]] = vals_np
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
    np.testing.assert_array_equal(buffer, buffer_np)
    np.testing.assert_array_equal(vals, vals_np)


def test_scan_while_not_implemented():
    x0 = pt.scalar("x0", dtype="float32")
    xs = scan(
        lambda xtm1: (xtm1 + 1, until(xtm1 > 5)),
        outputs_info=[x0],
        n_steps=100,
        return_updates=False,
    )
    with pytest.raises(NotImplementedError, match="While Scan"):
        function([x0], xs, mode=mlx_mode)


def test_scan_grad_non_sequence():
    # Gradient w.r.t. a non-sequence through a pure recurrence (no input
    # sequences to reverse), which exercises the MIT-MOT backward Scan.
    w = pt.scalar("w", dtype="float32")
    xs = scan(
        lambda x_tm1, w: x_tm1 * w,
        outputs_info=[pt.ones((), dtype="float32")],
        non_sequences=[w],
        n_steps=4,
        return_updates=False,
    )
    g = pt.grad(xs[-1], w)
    compare_mlx_and_py([w], [g], [np.float32(2.0)])


def _rnn_grad_over_sequence():
    xs = pt.matrix("xs", dtype="float32")
    h0 = pt.vector("h0", dtype="float32", shape=(3,))
    W = pt.matrix("W", dtype="float32", shape=(3, 3))
    hs = scan(
        lambda x_t, h_tm1, W: pt.tanh(x_t + h_tm1 @ W),
        sequences=[xs],
        outputs_info=[h0],
        non_sequences=[W],
        return_updates=False,
    )
    gW = pt.grad((hs**2).sum(), W)
    rng = np.random.default_rng(0)
    test_inputs = [
        rng.standard_normal((5, 3)).astype("float32"),
        np.zeros(3, dtype="float32"),
        (0.1 * rng.standard_normal((3, 3))).astype("float32"),
    ]
    return [xs, h0, W], [gW], test_inputs


def test_scan_grad_over_sequence():
    # The backward Scan (MIT-MOT + reversed forward trace) is correct under the
    # base MLX optimizer query.
    inputs, outputs, test_inputs = _rnn_grad_over_sequence()
    compare_mlx_and_py(inputs, outputs, test_inputs)


def test_scan_grad_over_sequence_default_mode():
    inputs, outputs, test_inputs = _rnn_grad_over_sequence()
    compare_mlx_and_py(inputs, outputs, test_inputs, mlx_mode="MLX")


def test_higher_order_derivatives():
    ScanCompatibilityTests.check_higher_order_derivative(mode="MLX")


@pytest.mark.parametrize("static_shape", [True, False])
def test_aliased_inner_outputs(static_shape):
    ScanCompatibilityTests.check_aliased_inner_outputs(
        static_shape=static_shape, mode="MLX"
    )
