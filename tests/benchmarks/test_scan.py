import numpy as np
import pytest

from pytensor import Mode, config, function, ifelse, scan, shared
from pytensor.gradient import hessian
from pytensor.scan.op import Scan
from pytensor.tensor import (
    arange,
    as_tensor,
    dot,
    eye,
    gammaln,
    grad,
    horizontal_stack,
    linalg,
    log,
    matrix,
    scalar,
    vector,
    vertical_stack,
    zeros,
)


def SEIR_model_logp():
    def binomln(n, k):
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

    def binom_log_prob(n, p, value):
        return binomln(n, value) + value * log(p) + (n - value) * log(1 - p)

    # sequences
    C_t = vector("C_t", dtype="int32", shape=(1200,))
    D_t = vector("D_t", dtype="int32", shape=(1200,))
    # outputs_info (initial conditions)
    st0 = scalar("s_t0")
    et0 = scalar("e_t0")
    it0 = scalar("i_t0")
    # non_sequences
    beta = scalar("beta")
    gamma = scalar("gamma")
    delta = scalar("delta")

    def seir_one_step(ct0, dt0, st0, et0, it0, beta, gamma, delta):
        bt0 = st0 * beta
        bt0 = bt0.astype(st0.dtype)

        logp_c1 = binom_log_prob(et0, gamma, ct0)
        logp_d1 = binom_log_prob(it0, delta, dt0)

        st1 = st0 - bt0
        et1 = et0 + bt0 - ct0
        it1 = it0 + ct0 - dt0
        return st1, et1, it1, logp_c1, logp_d1

    (st, et, it, logp_c_all, logp_d_all) = scan(
        fn=seir_one_step,
        sequences=[C_t, D_t],
        outputs_info=[st0, et0, it0, None, None],
        non_sequences=[beta, gamma, delta],
        return_updates=False,
    )
    st.name = "S_t"
    et.name = "E_t"
    it.name = "I_t"
    logp_c_all.name = "C_t_logp"
    logp_d_all.name = "D_t_logp"

    st0_val, et0_val, it0_val = np.array(100.0), np.array(50.0), np.array(25.0)
    beta_val, gamma_val, delta_val = (
        np.array(0.277792),
        np.array(0.135330),
        np.array(0.108753),
    )
    C_t_val = np.array([3, 5, 8, 13, 21, 26, 10, 3] * 150, dtype=np.int32)
    D_t_val = np.array([1, 2, 3, 7, 9, 11, 5, 1] * 150, dtype=np.int32)
    assert C_t_val.shape == D_t_val.shape == C_t.type.shape == D_t.type.shape

    test_input_vals = [
        C_t_val,
        D_t_val,
        st0_val,
        et0_val,
        it0_val,
        beta_val,
        gamma_val,
        delta_val,
    ]

    loss_graph = logp_c_all.sum() + logp_d_all.sum()

    return dict(
        graph_inputs=[C_t, D_t, st0, et0, it0, beta, gamma, delta],
        differentiable_vars=[st0, et0, it0, beta, gamma, delta],
        test_input_vals=test_input_vals,
        loss_graph=loss_graph,
    )


def _test_scan_pregreedy_optimizer_benchmark(mode, benchmark):
    W = zeros((5, 4))
    bv = zeros((5,))
    bh = zeros((4,))
    v = matrix("v")
    (bv_t, bh_t) = scan(
        lambda _: [bv, bh],
        sequences=v,
        outputs_info=[None, None],
        return_updates=False,
    )
    chain = scan(
        lambda x: dot(dot(x, W) + bh_t, W.T) + bv_t,
        outputs_info=v,
        n_steps=2,
        return_updates=False,
    )
    chain_fn = function([v], chain, mode=mode, trust_input=True)

    benchmark(chain_fn, np.zeros((3, 5), dtype=config.floatX))


def test_scan_pregreedy_optimizer_benchmark_c(benchmark):
    _test_scan_pregreedy_optimizer_benchmark("CVM", benchmark)


def test_scan_pregreedy_optimizer_benchmark_numba(benchmark):
    _test_scan_pregreedy_optimizer_benchmark("NUMBA", benchmark)


def _test_vector_taps_benchmark(mode, benchmark):
    n_steps = 1000

    seq1 = vector("seq1", dtype="float64", shape=(n_steps,))
    seq2 = vector("seq2", dtype="float64", shape=(n_steps,))
    mitsot_init = vector("mitsot_init", dtype="float64", shape=(2,))
    sitsot_init = scalar("sitsot_init", dtype="float64")

    def step(seq1, seq2, mitsot1, mitsot2, sitsot1):
        mitsot3 = (mitsot1 + seq2 + mitsot2 + seq1) / np.sqrt(4)
        sitsot2 = (sitsot1 + mitsot3) / np.sqrt(2)
        return mitsot3, sitsot2

    outs = scan(
        fn=step,
        sequences=[seq1, seq2],
        outputs_info=[
            dict(initial=mitsot_init, taps=[-2, -1]),
            dict(initial=sitsot_init, taps=[-1]),
        ],
        return_updates=False,
    )

    rng = np.random.default_rng(474)
    test = {
        seq1: rng.normal(size=n_steps),
        seq2: rng.normal(size=n_steps),
        mitsot_init: rng.normal(size=(2,)),
        sitsot_init: rng.normal(),
    }

    fn = function(list(test), outs, mode=mode, trust_input=False)
    scan_nodes = [
        node for node in fn.maker.fgraph.apply_nodes if isinstance(node.op, Scan)
    ]
    assert len(scan_nodes) == 1
    res = fn(*test.values())

    ref_fn = function(list(test), outs, mode=Mode(linker="py", optimizer=None))
    ref_res = ref_fn(*test.values())
    for r, ref_r in zip(res, ref_res, strict=True):
        np.testing.assert_array_almost_equal(r, ref_r)

    benchmark(fn, *test.values())


def test_vector_taps_benchmark_c(benchmark):
    _test_vector_taps_benchmark("CVM", benchmark)


def test_vector_taps_benchmark_numba(benchmark):
    _test_vector_taps_benchmark("NUMBA", benchmark)


def _test_sit_sot_buffer_benchmark(
    mode, n_steps, op_size, buffer_size, mode_preallocs_output, benchmark
):
    x0 = vector(shape=(op_size,), dtype="float64")
    xs = scan(
        fn=lambda xtm1: (xtm1 + 1),
        outputs_info=[x0],
        n_steps=n_steps - 1,
        return_updates=False,
    )
    if buffer_size == "unit":
        xs_kept = xs[-1]
        expected_buffer_size = 1 + mode_preallocs_output
    elif buffer_size == "aligned":
        xs_kept = xs[-2:]
        expected_buffer_size = 2
    elif buffer_size == "misaligned":
        xs_kept = xs[-3:]
        expected_buffer_size = 3
    elif buffer_size == "whole":
        xs_kept = xs
        expected_buffer_size = n_steps
    elif buffer_size == "whole+init":
        xs_kept = xs.owner.inputs[0]
        expected_buffer_size = n_steps
    else:
        raise ValueError(f"{buffer_size=} not understood")

    x_test = np.zeros(x0.type.shape)
    fn = function([x0], xs_kept, mode=mode, trust_input=True)
    ref_fn = function([x0], xs_kept, mode=Mode(linker="py", optimizer=None))
    np.testing.assert_allclose(fn(x_test), ref_fn(x_test))

    [scan_node] = [
        node for node in fn.maker.fgraph.toposort() if isinstance(node.op, Scan)
    ]
    buffer = scan_node.inputs[1]
    assert buffer.type.shape[0] == expected_buffer_size
    benchmark(fn, x_test)


@pytest.mark.parametrize(
    "buffer_size", ("unit", "aligned", "misaligned", "whole", "whole+init")
)
@pytest.mark.parametrize("n_steps, op_size", [(10, 2), (512, 2), (512, 256)])
def test_sit_sot_buffer_benchmark_c(n_steps, op_size, buffer_size, benchmark):
    _test_sit_sot_buffer_benchmark(
        "CVM", n_steps, op_size, buffer_size, True, benchmark
    )


@pytest.mark.parametrize(
    "buffer_size", ("unit", "aligned", "misaligned", "whole", "whole+init")
)
@pytest.mark.parametrize("n_steps, op_size", [(10, 2), (512, 2), (512, 256)])
def test_sit_sot_buffer_benchmark_numba(n_steps, op_size, buffer_size, benchmark):
    _test_sit_sot_buffer_benchmark(
        "NUMBA", n_steps, op_size, buffer_size, False, benchmark
    )


def _test_mit_sot_buffer_benchmark(
    mode, constant_n_steps, n_steps_val, mode_preallocs_output, benchmark
):
    def f_pow2(x_tm2, x_tm1):
        return 2 * x_tm1 + x_tm2

    init_x = vector("init_x", shape=(2,))
    n_steps = scalar("n_steps", dtype="int32")
    output = scan(
        f_pow2,
        sequences=[],
        outputs_info=[{"initial": init_x, "taps": [-2, -1]}],
        non_sequences=[],
        n_steps=n_steps_val if constant_n_steps else n_steps,
        return_updates=False,
    )

    init_x_val = np.array([1.0, 2.0], dtype=init_x.type.dtype)
    test_vals = (
        [init_x_val]
        if constant_n_steps
        else [init_x_val, np.asarray(n_steps_val, dtype=n_steps.type.dtype)]
    )
    inputs = [init_x] if constant_n_steps else [init_x, n_steps]
    fn = function(inputs, output[-1], mode=mode, trust_input=True)
    ref_fn = function(
        inputs, output[-1], mode=Mode(linker="py", optimizer=None), trust_input=True
    )
    np.testing.assert_allclose(fn(*test_vals), ref_fn(*test_vals))

    if n_steps_val == 1 and constant_n_steps:
        # There's no Scan in the graph when nsteps=constant(1)
        pytest.skip("No Scan in graph when n_steps=constant(1)")

    # Check the buffer size has been optimized
    [scan_node] = [
        node for node in fn.maker.fgraph.toposort() if isinstance(node.op, Scan)
    ]
    [mitsot_buffer] = scan_node.op.outer_mitsot(scan_node.inputs)
    mitsot_buffer_shape = mitsot_buffer.shape.eval(
        {init_x: init_x_val, n_steps: n_steps_val},
        accept_inplace=True,
        on_unused_input="ignore",
    )
    assert tuple(mitsot_buffer_shape) == (2 + mode_preallocs_output,)
    benchmark(fn, *test_vals)


@pytest.mark.parametrize("constant_n_steps", [False, True])
@pytest.mark.parametrize("n_steps_val", [1, 1000])
def test_mit_sot_buffer_benchmark_c(constant_n_steps, n_steps_val, benchmark):
    _test_mit_sot_buffer_benchmark(
        "CVM", constant_n_steps, n_steps_val, True, benchmark
    )


@pytest.mark.parametrize("constant_n_steps", [False, True])
@pytest.mark.parametrize("n_steps_val", [1, 1000])
def test_mit_sot_buffer_benchmark_numba(constant_n_steps, n_steps_val, benchmark):
    _test_mit_sot_buffer_benchmark(
        "NUMBA", constant_n_steps, n_steps_val, False, benchmark
    )


def test_scan_multiple_outs_taps_benchmark(benchmark):
    l = 5
    rng = np.random.default_rng(171)

    vW_in2 = rng.uniform(-2.0, 2.0, size=(2,))
    vW = rng.uniform(-2.0, 2.0, size=(2, 2))
    vWout = rng.uniform(-2.0, 2.0, size=(2,))
    vW_in1 = rng.uniform(-2.0, 2.0, size=(2, 2))
    v_u1 = rng.uniform(-2.0, 2.0, size=(l, 2))
    v_u2 = rng.uniform(-2.0, 2.0, size=(l + 2, 2))
    v_x0 = rng.uniform(-2.0, 2.0, size=(2,))
    v_y0 = rng.uniform(size=(3,))

    W_in2 = shared(vW_in2, name="win2")
    W = shared(vW, name="w")
    W_out = shared(vWout, name="wout")
    W_in1 = matrix("win")
    u1 = matrix("u1")
    u2 = matrix("u2")
    x0 = vector("x0")
    y0 = vector("y0")

    def f_rnn_cmpl(u1_t, u2_tm1, u2_t, u2_tp1, x_tm1, y_tm1, y_tm3, W_in1):
        return [
            dot(u1_t, W_in1) + (u2_t + u2_tm1 * u2_tp1) * W_in2 + dot(x_tm1, W),
            (y_tm1 + y_tm3) * dot(x_tm1, W_out),
            dot(u1_t, W_in1),
        ]

    outputs = scan(
        f_rnn_cmpl,
        [u1, dict(input=u2, taps=[-1, 0, 1])],
        [x0, dict(initial=y0, taps=[-1, -3]), None],
        W_in1,
        n_steps=None,
        truncate_gradient=-1,
        go_backwards=False,
        return_updates=False,
    )

    f = function([u1, u2, x0, y0, W_in1], outputs, allow_input_downcast=True)

    ny0 = np.zeros((5, 2))
    ny1 = np.zeros((5,))
    ny2 = np.zeros((5, 2))
    ny0[0] = (
        np.dot(v_u1[0], vW_in1)
        + (v_u2[1] + v_u2[0] * v_u2[2]) * vW_in2
        + np.dot(v_x0, vW)
    )

    ny1[0] = (v_y0[2] + v_y0[0]) * np.dot(v_x0, vWout)
    ny2[0] = np.dot(v_u1[0], vW_in1)

    ny0[1] = (
        np.dot(v_u1[1], vW_in1)
        + (v_u2[2] + v_u2[1] * v_u2[3]) * vW_in2
        + np.dot(ny0[0], vW)
    )

    ny1[1] = (ny1[0] + v_y0[1]) * np.dot(ny0[0], vWout)
    ny2[1] = np.dot(v_u1[1], vW_in1)

    ny0[2] = (
        np.dot(v_u1[2], vW_in1)
        + (v_u2[3] + v_u2[2] * v_u2[4]) * vW_in2
        + np.dot(ny0[1], vW)
    )
    ny1[2] = (ny1[1] + v_y0[2]) * np.dot(ny0[1], vWout)
    ny2[2] = np.dot(v_u1[2], vW_in1)

    ny0[3] = (
        np.dot(v_u1[3], vW_in1)
        + (v_u2[4] + v_u2[3] * v_u2[5]) * vW_in2
        + np.dot(ny0[2], vW)
    )

    ny1[3] = (ny1[2] + ny1[0]) * np.dot(ny0[2], vWout)
    ny2[3] = np.dot(v_u1[3], vW_in1)

    ny0[4] = (
        np.dot(v_u1[4], vW_in1)
        + (v_u2[5] + v_u2[4] * v_u2[6]) * vW_in2
        + np.dot(ny0[3], vW)
    )

    ny1[4] = (ny1[3] + ny1[1]) * np.dot(ny0[3], vWout)
    ny2[4] = np.dot(v_u1[4], vW_in1)

    res = f(v_u1, v_u2, v_x0, v_y0, vW_in1)
    np.testing.assert_almost_equal(res[0], ny0)
    np.testing.assert_almost_equal(res[1], ny1)
    np.testing.assert_almost_equal(res[2], ny2)

    benchmark(f, v_u1, v_u2, v_x0, v_y0, vW_in1)


def _test_SEIR_model_benchmark(mode, benchmark):
    model = SEIR_model_logp()
    graph_inputs = model["graph_inputs"]
    test_input_vals = model["test_input_vals"]
    loss_graph = model["loss_graph"]

    fn = function(graph_inputs, loss_graph, mode=mode, trust_input=True)
    ref_fn = function(graph_inputs, loss_graph, mode=Mode(linker="py", optimizer=None))
    np.testing.assert_allclose(fn(*test_input_vals), ref_fn(*test_input_vals))

    benchmark(fn, *test_input_vals)


def test_SEIR_model_benchmark_c(benchmark):
    _test_SEIR_model_benchmark("CVM", benchmark)


def test_SEIR_model_benchmark_numba(benchmark):
    _test_SEIR_model_benchmark("NUMBA", benchmark)


def _test_scan_hessian_benchmark(mode, benchmark):
    # Bug reported by Bitton Tenessi
    W = vector(name="W", dtype="float32")
    n_steps = scalar(name="Nb_steps", dtype="int32")

    def loss_outer(sum_outer, W):
        def loss_inner(sum_inner, W):
            return sum_inner + (W**2).sum()

        result_inner = scan(
            fn=loss_inner,
            outputs_info=as_tensor(np.asarray(0, dtype=np.float32)),
            non_sequences=[W],
            n_steps=1,
            return_updates=False,
        )
        return sum_outer + result_inner[-1]

    result_outer = scan(
        fn=loss_outer,
        outputs_info=as_tensor(np.asarray(0, dtype=np.float32)),
        non_sequences=[W],
        n_steps=n_steps,
        return_list=True,
        return_updates=False,
    )

    cost = result_outer[0][-1]
    H = hessian(cost, W)
    f = function([W, n_steps], H, mode=mode)
    benchmark(f, np.ones((8,), dtype="float32"), 1)


def test_scan_hessian_benchmark_c(benchmark):
    _test_scan_hessian_benchmark("CVM", benchmark)


def test_scan_hessian_benchmark_numba(benchmark):
    _test_scan_hessian_benchmark("NUMBA", benchmark)


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_scan_cython_benchmark(benchmark):
    # This implicitly confirms that the Cython version is being used
    from pytensor.scan import scan_perform_ext  # noqa: F401

    # Python usually out-performs PyTensor below 100 iterations
    N = 200
    M = -1 / np.arange(1, 11).astype(config.floatX)
    r = np.arange(N * 10).astype(config.floatX).reshape(N, 10)

    def f_py():
        py_out = np.empty((N, 10), dtype=config.floatX)
        py_out[0] = r[0]
        for i in range(1, py_out.shape[0]):
            py_out[i] = r[i] + M * py_out[i - 1]
        return py_out[1:]

    py_res = f_py()

    s_r = as_tensor(r, dtype=config.floatX)
    s_y = scan(
        fn=lambda ri, rii, M: ri + M * rii,
        sequences=[s_r[1:]],
        non_sequences=[as_tensor(M, dtype=config.floatX)],
        outputs_info=s_r[0],
        mode="CVM",
        return_updates=False,
    )

    f_cvm = function([], s_y, mode="CVM")
    f_cvm.trust_input = True

    # Make sure we're actually computing a `Scan`
    assert any(isinstance(node.op, Scan) for node in f_cvm.maker.fgraph.apply_nodes)

    cvm_res = benchmark(f_cvm)

    # Make sure the results are the same between the two implementations
    assert np.allclose(cvm_res, py_res)


def test_scan_reordering_benchmark(benchmark):
    rng = np.random.default_rng(51)

    vW_in2 = rng.uniform(-0.5, 0.5, size=(2,))
    vW = rng.uniform(-0.5, 0.5, size=(2, 2))
    vWout = rng.uniform(-0.5, 0.5, size=(2,))
    vW_in1 = rng.uniform(-0.5, 0.5, size=(2, 2))
    v_u1 = rng.uniform(-0.5, 0.5, size=(3, 2))
    v_u2 = rng.uniform(-0.5, 0.5, size=(3,))
    v_x0 = rng.uniform(-0.5, 0.5, size=(2,))
    v_y0 = rng.uniform(size=(3,))

    W_in2 = shared(vW_in2, name="win2")
    W = shared(vW, name="w")
    W_out = shared(vWout, name="wout")
    W_in1 = matrix("win")
    u1 = matrix("u1")
    u2 = vector("u2")
    x0 = vector("x0")
    y0 = vector("y0")

    def f_rnn_cmpl(u1_t, u2_t, x_tm1, y_tm1, y_tm3, W_in1):
        return [
            y_tm3 + 1,
            y_tm3 + 2,
            dot(u1_t, W_in1) + u2_t * W_in2 + dot(x_tm1, W),
            y_tm1 + dot(x_tm1, W_out),
        ]

    outputs = scan(
        f_rnn_cmpl,
        [u1, u2],
        [None, None, x0, dict(initial=y0, taps=[-1, -3])],
        W_in1,
        n_steps=None,
        truncate_gradient=-1,
        go_backwards=False,
        return_updates=False,
    )

    f4 = function(
        [u1, u2, x0, y0, W_in1], outputs, mode="CVM", allow_input_downcast=True
    )

    # compute the values in numpy
    v_x = np.zeros((3, 2), dtype=config.floatX)
    v_y = np.zeros((3,), dtype=config.floatX)
    v_x[0] = np.dot(v_u1[0], vW_in1) + v_u2[0] * vW_in2 + np.dot(v_x0, vW)
    v_y[0] = np.dot(v_x0, vWout) + v_y0[2]
    for i in range(1, 3):
        v_x[i] = np.dot(v_u1[i], vW_in1) + v_u2[i] * vW_in2 + np.dot(v_x[i - 1], vW)
        v_y[i] = np.dot(v_x[i - 1], vWout) + v_y[i - 1]

    (_pytensor_dump1, _pytensor_dump2, pytensor_x, pytensor_y) = benchmark(
        f4, v_u1, v_u2, v_x0, v_y0, vW_in1
    )

    np.testing.assert_allclose(pytensor_x, v_x)
    np.testing.assert_allclose(pytensor_y, v_y)


def _jax_cyclical_reduction():
    def stabilize(x, jitter=1e-16):
        return x + jitter * eye(x.shape[0])

    def step(A0, A1, A2, A1_hat, norm, step_num, tol):
        def cycle_step(A0, A1, A2, A1_hat, _norm, step_num):
            tmp = dot(
                vertical_stack(A0, A2),
                linalg.solve(
                    stabilize(A1),
                    horizontal_stack(A0, A2),
                    assume_a="gen",
                    check_finite=False,
                ),
            )

            n = A0.shape[0]
            idx_0 = arange(n)
            idx_1 = idx_0 + n
            A1 = A1 - tmp[idx_0, :][:, idx_1] - tmp[idx_1, :][:, idx_0]
            A0 = -tmp[idx_0, :][:, idx_0]
            A2 = -tmp[idx_1, :][:, idx_1]
            A1_hat = A1_hat - tmp[idx_1, :][:, idx_0]

            A0_L1_norm = linalg.norm(A0, ord=1)

            return A0, A1, A2, A1_hat, A0_L1_norm, step_num + 1

        return ifelse(
            norm < tol,
            (A0, A1, A2, A1_hat, norm, step_num),
            cycle_step(A0, A1, A2, A1_hat, norm, step_num),
        )

    A = matrix("A", shape=(20, 20))
    B = matrix("B", shape=(20, 20))
    C = matrix("C", shape=(20, 20))

    norm = np.array(1e9, dtype="float64")
    step_num = zeros((), dtype="int32")
    max_iter = 100
    tol = 1e-7

    (*_, A1_hat, norm, _n_steps) = scan(
        step,
        outputs_info=[A, B, C, B, norm, step_num],
        non_sequences=[tol],
        n_steps=max_iter,
        return_updates=False,
    )
    A1_hat = A1_hat[-1]

    T = -linalg.solve(stabilize(A1_hat), A, assume_a="gen", check_finite=False)

    rng = np.random.default_rng(sum(map(ord, "cycle_reduction")))
    n = A.type.shape[0]
    A_test = rng.standard_normal(size=(n, n))
    C_test = rng.standard_normal(size=(n, n))
    # B must be invertible, so we make it symmetric positive-definite
    B_rand = rng.standard_normal(size=(n, n))
    B_test = B_rand @ B_rand.T + np.eye(n) * 1e-3

    return dict(
        graph_inputs=[A, B, C],
        differentiable_vars=[A, B, C],
        test_input_vals=[A_test, B_test, C_test],
        loss_graph=T.sum(),
    )


@pytest.mark.parametrize("gradient_backend", ["PYTENSOR", "JAX"])
@pytest.mark.parametrize("mode", ("0forward", "1backward", "2both"))
@pytest.mark.parametrize("model", ["cyclical_reduction", "SEIR_model_logp"])
def test_jax_scan_benchmark(model, mode, gradient_backend, benchmark):
    jax = pytest.importorskip("jax")
    from tests.link.jax.test_basic import compare_jax_and_py

    model_builders = {
        "cyclical_reduction": _jax_cyclical_reduction,
        "SEIR_model_logp": SEIR_model_logp,
    }
    model_dict = model_builders[model]()
    graph_inputs = model_dict["graph_inputs"]
    differentiable_vars = model_dict["differentiable_vars"]
    loss_graph = model_dict["loss_graph"]
    test_input_vals = model_dict["test_input_vals"]

    if gradient_backend == "PYTENSOR":
        backward_loss = grad(
            loss_graph,
            wrt=differentiable_vars,
        )

        match mode:
            case "0forward":
                graph_outputs = [loss_graph]
            case "1backward":
                graph_outputs = backward_loss
            case "2both":
                graph_outputs = [loss_graph, *backward_loss]
            case _:
                raise ValueError(f"Unknown mode: {mode}")

        jax_fn, _ = compare_jax_and_py(
            graph_inputs,
            graph_outputs,
            test_input_vals,
            jax_mode="JAX",
        )
        jax_fn.trust_input = True

    else:  # gradient_backend == "JAX"
        loss_fn_tuple = function(graph_inputs, loss_graph, mode="JAX").vm.jit_fn

        def loss_fn(*args):
            return loss_fn_tuple(*args)[0]

        match mode:
            case "0forward":
                jax_fn = jax.jit(loss_fn_tuple)
            case "1backward":
                jax_fn = jax.jit(
                    jax.grad(loss_fn, argnums=tuple(range(len(graph_inputs))[2:]))
                )
            case "2both":
                value_and_grad_fn = jax.value_and_grad(
                    loss_fn, argnums=tuple(range(len(graph_inputs))[2:])
                )

                @jax.jit
                def jax_fn(*args):
                    loss, grads = value_and_grad_fn(*args)
                    return loss, *grads

            case _:
                raise ValueError(f"Unknown mode: {mode}")

    def block_until_ready(*inputs, jax_fn=jax_fn):
        return [o.block_until_ready() for o in jax_fn(*inputs)]

    block_until_ready(*test_input_vals)  # Warmup

    benchmark.pedantic(block_until_ready, test_input_vals, rounds=200, iterations=1)
