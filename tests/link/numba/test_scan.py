from itertools import chain

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import config, function, grad
from pytensor.compile.mode import Mode, get_mode
from pytensor.scalar import Log1p
from pytensor.scan.basic import scan
from pytensor.scan.op import Scan
from pytensor.scan.utils import until
from pytensor.tensor import log, scalar, vector
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.utils import RandomStream
from tests import unittest_tools as utt
from tests.link.numba.test_basic import compare_numba_and_py
from tests.scan.test_basic import ScanCompatibilityTests


@pytest.mark.parametrize(
    "fn, sequences, outputs_info, non_sequences, n_steps, input_vals, output_vals, op_check",
    [
        # sequences
        (
            lambda a_t: 2 * a_t,
            [pt.dvector("a")],
            [{}],
            [],
            None,
            [np.arange(10)],
            None,
            lambda op: op.info.n_seqs > 0,
        ),
        # nit-sot
        (
            lambda: pt.as_tensor(2.0),
            [],
            [{}],
            [],
            3,
            [],
            None,
            lambda op: op.info.n_nit_sot > 0,
        ),
        # nit-sot, non_seq
        (
            lambda c: pt.as_tensor(2.0) * c,
            [],
            [{}],
            [pt.dscalar("c")],
            3,
            [1.0],
            None,
            lambda op: op.info.n_nit_sot > 0 and op.info.n_non_seqs > 0,
        ),
        # sit-sot
        (
            lambda a_tm1: 2 * a_tm1,
            [],
            [{"initial": pt.as_tensor(0.0, dtype="floatX"), "taps": [-1]}],
            [],
            3,
            [],
            None,
            lambda op: op.info.n_sit_sot > 0,
        ),
        # sit-sot, while
        (
            lambda a_tm1: (a_tm1 + 1, until(a_tm1 > 2)),
            [],
            [{"initial": pt.as_tensor(1, dtype=np.int64), "taps": [-1]}],
            [],
            3,
            [],
            None,
            lambda op: op.info.n_sit_sot > 0,
        ),
        # nit-sot, shared input/output
        (
            lambda: RandomStream(seed=1930).normal(0, 1, name="a"),
            [],
            [{}],
            [],
            3,
            [],
            [np.array([0.50100236, 2.16822932, 1.36326596])],
            lambda op: op.info.n_untraced_sit_sot_outs > 0,
        ),
        # mit-sot (that's also a type of sit-sot)
        (
            lambda a_tm1: 2 * a_tm1,
            [],
            [{"initial": pt.as_tensor([0.0, 1.0], dtype="floatX"), "taps": [-2]}],
            [],
            6,
            [],
            None,
            lambda op: op.info.n_mit_sot > 0,
        ),
        # mit-sot
        (
            lambda a_tm1, b_tm1: (2 * a_tm1, 2 * b_tm1),
            [],
            [
                {"initial": pt.as_tensor(0.0, dtype="floatX"), "taps": [-1]},
                {"initial": pt.as_tensor(0.0, dtype="floatX"), "taps": [-1]},
            ],
            [],
            10,
            [],
            None,
            lambda op: op.info.n_mit_sot > 0,
        ),
    ],
)
def test_xit_xot_types(
    fn,
    sequences,
    outputs_info,
    non_sequences,
    n_steps,
    input_vals,
    output_vals,
    op_check,
):
    """Test basic xit-xot configurations."""
    res, updates = scan(
        fn,
        sequences=sequences,
        outputs_info=outputs_info,
        non_sequences=non_sequences,
        n_steps=n_steps,
        strict=True,
        mode=Mode(linker="py", optimizer=None),
    )

    if not isinstance(res, list):
        res = [res]

    # Get rid of any `Subtensor` indexing on the `Scan` outputs
    res = [r.owner.inputs[0] if not isinstance(r.owner.op, Scan) else r for r in res]

    scan_op = res[0].owner.op
    assert isinstance(scan_op, Scan)

    _ = op_check(scan_op)

    if output_vals is None:
        compare_numba_and_py(
            sequences + non_sequences, res, input_vals, updates=updates
        )
    else:
        numba_mode = get_mode("NUMBA")
        numba_fn = function(
            sequences + non_sequences, res, mode=numba_mode, updates=updates
        )
        res_val = numba_fn(*input_vals)
        assert np.allclose(res_val, output_vals)


def test_scan_multiple_output(benchmark):
    """Test a scan implementation of a SEIR model.

    SEIR model definition:

        S[t+1] = S[t] - B[t]
        E[t+1] = E[t] + B[t] - C[t]
        I[t+1] = I[t+1] + C[t] - D[t]

        B[t] ~ Binom(S[t], beta)
        C[t] ~ Binom(E[t], gamma)
        D[t] ~ Binom(I[t], delta)

    """

    def binomln(n, k):
        return pt.exp(n + 1) - pt.exp(k + 1) - pt.exp(n - k + 1)

    def binom_log_prob(n, p, value):
        return binomln(n, value) + value * pt.exp(p) + (n - value) * pt.exp(1 - p)

    # sequences
    pt_C = pt.ivector("C_t")
    pt_D = pt.ivector("D_t")
    # outputs_info (initial conditions)
    st0 = pt.lscalar("s_t0")
    et0 = pt.lscalar("e_t0")
    it0 = pt.lscalar("i_t0")
    logp_c = pt.scalar("logp_c")
    logp_d = pt.scalar("logp_d")
    # non_sequences
    beta = pt.scalar("beta")
    gamma = pt.scalar("gamma")
    delta = pt.scalar("delta")

    def seir_one_step(ct0, dt0, st0, et0, it0, logp_c, logp_d, beta, gamma, delta):
        bt0 = st0 * beta
        bt0 = bt0.astype(st0.dtype)

        logp_c1 = binom_log_prob(et0, gamma, ct0).astype(logp_c.dtype)
        logp_d1 = binom_log_prob(it0, delta, dt0).astype(logp_d.dtype)

        st1 = st0 - bt0
        et1 = et0 + bt0 - ct0
        it1 = it0 + ct0 - dt0
        return st1, et1, it1, logp_c1, logp_d1

    (st, et, it, logp_c_all, logp_d_all) = scan(
        fn=seir_one_step,
        sequences=[pt_C, pt_D],
        outputs_info=[st0, et0, it0, logp_c, logp_d],
        non_sequences=[beta, gamma, delta],
        return_updates=False,
    )
    st.name = "S_t"
    et.name = "E_t"
    it.name = "I_t"
    logp_c_all.name = "C_t_logp"
    logp_d_all.name = "D_t_logp"

    out = [st, et, it, logp_c_all, logp_d_all]

    s0, e0, i0 = 100, 50, 25
    logp_c0 = np.array(0.0, dtype=config.floatX)
    logp_d0 = np.array(0.0, dtype=config.floatX)
    beta_val, gamma_val, delta_val = (
        np.array(val, dtype=config.floatX) for val in [0.277792, 0.135330, 0.108753]
    )
    C = np.array([3, 5, 8, 13, 21, 26, 10, 3], dtype=np.int32)
    D = np.array([1, 2, 3, 7, 9, 11, 5, 1], dtype=np.int32)

    test_input_vals = [
        C,
        D,
        s0,
        e0,
        i0,
        logp_c0,
        logp_d0,
        beta_val,
        gamma_val,
        delta_val,
    ]
    scan_fn, _ = compare_numba_and_py(
        [pt_C, pt_D, st0, et0, it0, logp_c, logp_d, beta, gamma, delta],
        out,
        test_input_vals,
    )

    benchmark(scan_fn, *test_input_vals)


def test_scan_tap_output():
    a_pt = pt.scalar("a")

    b_pt = pt.vector("b")

    c_pt = pt.vector("c")

    def input_step_fn(b, b2, c, x_tm1, y_tm1, y_tm3, a):
        x_tm1.name = "x_tm1"
        y_tm1.name = "y_tm1"
        y_tm3.name = "y_tm3"
        y_t = (y_tm1 + y_tm3) * a + b + b2
        z_t = y_t * c
        x_t = x_tm1 + 1
        x_t.name = "x_t"
        y_t.name = "y_t"
        return x_t, y_t, pt.fill((10,), z_t)

    scan_res = scan(
        fn=input_step_fn,
        sequences=[
            {
                "input": b_pt,
                "taps": [-1, -2],
            },
            {
                "input": c_pt,
                "taps": [-2],
            },
        ],
        outputs_info=[
            {
                "initial": pt.as_tensor_variable(0.0, dtype=config.floatX),
                "taps": [-1],
            },
            {
                "initial": pt.as_tensor_variable(
                    np.r_[-1.0, 1.3, 0.0].astype(config.floatX)
                ),
                "taps": [-1, -3],
            },
            None,
        ],
        non_sequences=[a_pt],
        n_steps=5,
        name="yz_scan",
        strict=True,
        return_updates=False,
    )

    test_input_vals = [
        np.array(10.0).astype(config.floatX),
        np.arange(11, dtype=config.floatX),
        np.arange(20, 31, dtype=config.floatX),
    ]
    compare_numba_and_py([a_pt, b_pt, c_pt], scan_res, test_input_vals)


def test_scan_while():
    def power_of_2(previous_power, max_value):
        return previous_power * 2, until(previous_power * 2 > max_value)

    max_value = pt.scalar()
    values = scan(
        power_of_2,
        outputs_info=pt.constant(1.0),
        non_sequences=max_value,
        n_steps=1024,
        return_updates=False,
    )

    test_input_vals = [
        np.array(45).astype(config.floatX),
    ]
    compare_numba_and_py([max_value], [values], test_input_vals)


def test_scan_multiple_none_output():
    A = pt.dvector("A")

    def power_step(prior_result, x):
        return prior_result * x, prior_result * x * x, prior_result * x * x * x

    result = scan(
        power_step,
        non_sequences=[A],
        outputs_info=[pt.ones_like(A), None, None],
        n_steps=3,
        return_updates=False,
    )
    test_input_vals = (np.array([1.0, 2.0]),)
    compare_numba_and_py([A], result, test_input_vals)


def test_grad_sitsot():
    def get_sum_of_grad(inp):
        scan_outputs = scan(
            fn=lambda x: x * 2,
            outputs_info=[inp],
            n_steps=5,
            mode="NUMBA",
            return_updates=False,
        )
        return grad(scan_outputs.sum(), inp).sum()

    floatX = config.floatX
    inputs_test_values = [
        np.random.default_rng(utt.fetch_seed()).random(3).astype(floatX)
    ]
    utt.verify_grad(get_sum_of_grad, inputs_test_values, mode="NUMBA")


def test_mitmots_basic():
    init_x = pt.dvector()
    seq = pt.dvector()

    def inner_fct(seq, state_old, state_current):
        return state_old * 2 + state_current + seq

    out = scan(
        inner_fct,
        sequences=seq,
        outputs_info={"initial": init_x, "taps": [-2, -1]},
        return_updates=False,
    )

    g_outs = grad(out.sum(), [seq, init_x])

    numba_mode = get_mode("NUMBA").including("scan_save_mem")
    py_mode = Mode("py").including("scan_save_mem")

    seq_val = np.arange(3)
    init_x_val = np.r_[-2, -1]
    test_input_vals = (seq_val, init_x_val)

    compare_numba_and_py(
        [seq, init_x], g_outs, test_input_vals, numba_mode=numba_mode, py_mode=py_mode
    )


def test_inner_graph_optimized():
    """Test that inner graph of Scan is optimized"""
    xs = vector("xs")
    seq = scan(
        fn=lambda x: log(1 + x),
        sequences=[xs],
        mode=get_mode("NUMBA"),
        return_updates=False,
    )

    # Disable scan pushout, in which case the whole scan is replaced by an Elemwise
    f = function([xs], seq, mode=get_mode("NUMBA").excluding("scan_pushout"))
    (scan_node,) = (
        node for node in f.maker.fgraph.apply_nodes if isinstance(node.op, Scan)
    )
    inner_scan_nodes = scan_node.op.fgraph.apply_nodes
    assert len(inner_scan_nodes) == 1
    (inner_scan_node,) = scan_node.op.fgraph.apply_nodes
    assert isinstance(inner_scan_node.op, Elemwise) and isinstance(
        inner_scan_node.op.scalar_op, Log1p
    )


def test_vector_taps_benchmark(benchmark):
    """Test vector taps performance.

    Vector taps get indexed into numeric types, that must be wrapped back into
    scalar arrays. The numba Scan implementation has an optimization to reuse
    these scalar arrays instead of allocating them in every iteration.
    """
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

    numba_fn = pytensor.function(list(test), outs, mode=get_mode("NUMBA"))
    scan_nodes = [
        node for node in numba_fn.maker.fgraph.apply_nodes if isinstance(node.op, Scan)
    ]
    assert len(scan_nodes) == 1
    numba_res = numba_fn(*test.values())

    ref_fn = pytensor.function(list(test), outs, mode=get_mode("FAST_COMPILE"))
    ref_res = ref_fn(*test.values())
    for numba_r, ref_r in zip(numba_res, ref_res, strict=True):
        np.testing.assert_array_almost_equal(numba_r, ref_r)

    benchmark(numba_fn, *test.values())


@pytest.mark.parametrize("n_steps_constant", (True, False))
def test_inplace_taps(n_steps_constant):
    """Test that numba will inplace in the inner_function of the oldest sit-sot, mit-sot taps."""
    n_steps = 10 if n_steps_constant else scalar("n_steps", dtype=int)
    a = scalar("a")
    x0 = scalar("x0")
    y0 = vector("y0", shape=(2,))
    z0 = vector("z0", shape=(3,))

    def step(ztm3, ztm1, xtm1, ytm1, ytm2, a):
        z = ztm1 + 1 + ztm3 + a
        x = xtm1 + 1
        y = ytm1 + 1 + ytm2 + a
        return z, x, z + x + y, y

    [zs, xs, ws, ys] = scan(
        fn=step,
        outputs_info=[
            dict(initial=z0, taps=[-3, -1]),
            dict(initial=x0, taps=[-1]),
            None,
            dict(initial=y0, taps=[-1, -2]),
        ],
        non_sequences=[a],
        n_steps=n_steps,
        return_updates=False,
    )
    numba_fn, _ = compare_numba_and_py(
        [n_steps] * (not n_steps_constant) + [a, x0, y0, z0],
        [zs[-1], xs[-1], ws[-1], ys[-1]],
        [10] * (not n_steps_constant) + [np.pi, np.e, [1, np.euler_gamma], [0, 1, 2]],
        numba_mode="NUMBA",
        eval_obj_mode=False,
    )
    [scan_op] = [
        node.op
        for node in numba_fn.maker.fgraph.toposort()
        if isinstance(node.op, Scan)
    ]

    # Collect inner inputs we expect to be destroyed by the step function
    # Scan reorders inputs internally, so we need to check its ordering
    inner_inps = scan_op.fgraph.inputs
    mit_sot_inps = scan_op.inner_mitsot(inner_inps)
    oldest_mit_sot_inps = [
        # Implicitly assume that the first mit-sot input is the one with 3 taps
        # This is not a required behavior and the test can change if we need to change Scan.
        mit_sot_inps[:2][scan_op.info.mit_sot_in_slices[0].index(-3)],
        mit_sot_inps[2:][scan_op.info.mit_sot_in_slices[1].index(-2)],
    ]
    [sit_sot_inp] = scan_op.inner_sitsot(inner_inps)

    destroyed_inputs = []
    for inner_out in scan_op.fgraph.outputs:
        node = inner_out.owner
        dm = node.op.destroy_map
        if dm:
            destroyed_inputs.extend(
                node.inputs[idx] for idx in chain.from_iterable(dm.values())
            )

    if n_steps_constant:
        assert len(destroyed_inputs) == 3
        assert set(destroyed_inputs) == {*oldest_mit_sot_inps, sit_sot_inp}
    else:
        # This is not a feature, but a current limitation
        # https://github.com/pymc-devs/pytensor/issues/1283
        assert not destroyed_inputs


@pytest.mark.parametrize(
    "buffer_size", ("unit", "aligned", "misaligned", "whole", "whole+init")
)
@pytest.mark.parametrize("n_steps, op_size", [(10, 2), (512, 2), (512, 256)])
class TestScanSITSOTBuffer:
    def buffer_tester(self, n_steps, op_size, buffer_size, benchmark=None):
        x0 = pt.vector(shape=(op_size,), dtype="float64")
        xs = pytensor.scan(
            fn=lambda xtm1: (xtm1 + 1),
            outputs_info=[x0],
            n_steps=n_steps - 1,  # 1- makes it easier to align/misalign
            return_updates=False,
        )
        if buffer_size == "unit":
            xs_kept = xs[-1]  # Only last state is used
            expected_buffer_size = 1
        elif buffer_size == "aligned":
            xs_kept = xs[-2:]  # The buffer will be aligned at the end of the 9 steps
            expected_buffer_size = 2
        elif buffer_size == "misaligned":
            xs_kept = xs[-3:]  # The buffer will be misaligned at the end of the 9 steps
            expected_buffer_size = 3
        elif buffer_size == "whole":
            xs_kept = xs  # What users think is the whole buffer
            expected_buffer_size = n_steps
        elif buffer_size == "whole+init":
            xs_kept = xs.owner.inputs[0]  # Whole buffer actually used by Scan
            expected_buffer_size = n_steps

        x_test = np.zeros(x0.type.shape)
        numba_fn, _ = compare_numba_and_py(
            [x0],
            [xs_kept],
            test_inputs=[x_test],
            numba_mode="NUMBA",  # Default doesn't include optimizations
            eval_obj_mode=False,
        )
        [scan_node] = [
            node
            for node in numba_fn.maker.fgraph.toposort()
            if isinstance(node.op, Scan)
        ]
        buffer = scan_node.inputs[1]
        assert buffer.type.shape[0] == expected_buffer_size

        if benchmark is not None:
            numba_fn.trust_input = True
            benchmark(numba_fn, x_test)

    def test_sit_sot_buffer(self, n_steps, op_size, buffer_size):
        self.buffer_tester(n_steps, op_size, buffer_size, benchmark=None)

    def test_sit_sot_buffer_benchmark(self, n_steps, op_size, buffer_size, benchmark):
        self.buffer_tester(n_steps, op_size, buffer_size, benchmark=benchmark)


@pytest.mark.parametrize("constant_n_steps", [False, True])
@pytest.mark.parametrize("n_steps_val", [1, 1000])
class TestScanMITSOTBuffer:
    def buffer_tester(self, constant_n_steps, n_steps_val, benchmark=None):
        """Make sure we can handle storage changes caused by the `scan_save_mem` rewrite."""

        def f_pow2(x_tm2, x_tm1):
            return 2 * x_tm1 + x_tm2

        init_x = pt.vector("init_x", shape=(2,))
        n_steps = pt.iscalar("n_steps")
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
        numba_fn, _ = compare_numba_and_py(
            [init_x] if constant_n_steps else [init_x, n_steps],
            [output[-1]],
            test_vals,
            numba_mode="NUMBA",
            eval_obj_mode=False,
        )

        if n_steps_val == 1 and constant_n_steps:
            # There's no Scan in the graph when nsteps=constant(1)
            return

        # Check the buffer size as been optimized
        [scan_node] = [
            node
            for node in numba_fn.maker.fgraph.toposort()
            if isinstance(node.op, Scan)
        ]
        [mitsot_buffer] = scan_node.op.outer_mitsot(scan_node.inputs)
        mitsot_buffer_shape = mitsot_buffer.shape.eval(
            {init_x: init_x_val, n_steps: n_steps_val},
            accept_inplace=True,
            on_unused_input="ignore",
        )
        assert tuple(mitsot_buffer_shape) == (2,)
        if benchmark is not None:
            numba_fn.trust_input = True
            benchmark(numba_fn, *test_vals)

    def test_mit_sot_buffer(self, constant_n_steps, n_steps_val):
        self.buffer_tester(constant_n_steps, n_steps_val, benchmark=None)

    def test_mit_sot_buffer_benchmark(self, constant_n_steps, n_steps_val, benchmark):
        self.buffer_tester(constant_n_steps, n_steps_val, benchmark=benchmark)


def test_higher_order_derivatives():
    ScanCompatibilityTests.check_higher_order_derivative(mode="NUMBA")


def test_grad_until_and_truncate_sequence_taps():
    ScanCompatibilityTests.check_grad_until_and_truncate_sequence_taps(mode="NUMBA")
