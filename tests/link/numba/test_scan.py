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
            lambda op: op.info.n_untraced_sit_sot > 0,
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


@pytest.mark.parametrize("n_steps_constant", (True, False))
def test_inplace_taps(n_steps_constant):
    """No inner output may be computed in place on a sit-sot/mit-sot tap.

    The scan loop overwrites those tap buffers in place each iteration, so an output
    aliasing one would be corrupted. Only an ``untraced_sit_sot`` input (rebound by
    reference, never loop-overwritten) may be destroyed in place by its own output.
    See #2252.
    """
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

    # Collect the inner inputs destroyed in place by the output-producing nodes.
    # Scan reorders inputs internally, so we go through its accessors.
    inner_inps = scan_op.fgraph.inputs
    mit_sot_inps = scan_op.inner_mitsot(inner_inps)
    sit_sot_inps = scan_op.inner_sitsot(inner_inps)
    untraced_sit_sot_inps = scan_op.inner_untraced_sit_sot(inner_inps)

    destroyed_inputs = []
    for inner_out in scan_op.fgraph.outputs:
        node = inner_out.owner
        dm = node.op.destroy_map
        if dm:
            destroyed_inputs.extend(
                node.inputs[idx] for idx in chain.from_iterable(dm.values())
            )

    # ``local_subtensor_merge_integer`` + ``scan_reduce_buffer`` reduce the buffers
    # the same way for both constant and symbolic ``n_steps`` (xs collapses to an
    # untraced sit_sot), so the result is identical in both cases. No mit_sot tap may
    # be destroyed by an output (the loop overwrites those slots in place); only the
    # untraced_sit_sot input is destroyed in place by its own output.
    assert len(sit_sot_inps) == 0
    assert len(untraced_sit_sot_inps) == 1
    assert not any(tap in destroyed_inputs for tap in mit_sot_inps)
    assert set(destroyed_inputs) == {untraced_sit_sot_inps[0]}


def test_mitmot_inplace():
    """A mit-mot tap read may be destroyed in place by an intermediate inner node.

    Gradients of scans produce mit-mot taps whose read slot is overwritten by the
    accumulator write-back the same iteration, so scan grants ``mutable=True`` on the
    certainly-overwritten reads. The destroy is realized only when a read feeds an
    *intermediate* (non-output) op that can run in place -- the ``Dot`` here breaks
    Elemwise fusion, leaving an intermediate that consumes a gradient accumulator. The
    same-iteration overwrite still forbids an *output* from carrying the read, so the
    result stays correct.
    """
    k = 3
    W = pt.dmatrix("W")
    init = pt.dmatrix("init")  # two initial taps, each a length-k state
    seq = pt.dvector("seq")

    def step(seq_t, x2, x1):
        return pt.tanh(W @ x1 + x2 + seq_t)

    out = scan(
        step,
        sequences=seq,
        outputs_info={"initial": init, "taps": [-2, -1]},
        return_updates=False,
    )
    g = grad(out.sum(), [init, W])

    rng = np.random.default_rng(0)
    test_vals = [
        rng.standard_normal(5),
        rng.standard_normal((2, k)),
        rng.standard_normal((k, k)),
    ]
    numba_fn, _ = compare_numba_and_py(
        [seq, init, W], g, test_vals, numba_mode="NUMBA", eval_obj_mode=False
    )

    mitmot_scan_ops = [
        node.op
        for node in numba_fn.maker.fgraph.toposort()
        if isinstance(node.op, Scan) and node.op.info.n_mit_mot
    ]
    assert mitmot_scan_ops, "expected the gradient to produce a mit-mot scan"
    destroyed_mitmot_reads = []
    for scan_op in mitmot_scan_ops:
        mitmot_reads = set(scan_op.inner_mitmot(scan_op.fgraph.inputs))
        for node in scan_op.fgraph.toposort():
            destroyed_mitmot_reads.extend(
                node.inputs[idx]
                for idx in chain.from_iterable(node.op.destroy_map.values())
                if node.inputs[idx] in mitmot_reads
            )
    assert destroyed_mitmot_reads, "expected a mit-mot read destroyed in place"


@pytest.mark.parametrize(
    "buffer_size", ("unit", "aligned", "misaligned", "whole", "whole+init")
)
@pytest.mark.parametrize("n_steps, op_size", [(10, 2), (512, 2), (512, 256)])
class TestScanSITSOTBuffer:
    def buffer_tester(self, n_steps, op_size, buffer_size, benchmark=None):
        x0 = pt.vector(shape=(op_size,), dtype="float64")
        xs = pytensor.scan(
            fn=lambda xtm1: xtm1 + 1,
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
        if expected_buffer_size == 1:
            # sit_sot_to_untraced converts unit-buffer sit_sot to untraced_sit_sot
            assert scan_node.op.info.n_sit_sot == 0
            assert scan_node.op.info.n_untraced_sit_sot == 1
        else:
            buffer = scan_node.inputs[1]
            assert buffer.type.shape[0] == expected_buffer_size

        if benchmark is not None:
            numba_fn.trust_input = True
            benchmark(numba_fn, x_test)

    def test_sit_sot_buffer(self, n_steps, op_size, buffer_size):
        self.buffer_tester(n_steps, op_size, buffer_size, benchmark=None)


@pytest.mark.parametrize("constant_n_steps", [False, True])
@pytest.mark.parametrize("n_steps_val", [1, 1000])
class TestScanMITSOTBuffer:
    def buffer_tester(self, constant_n_steps, n_steps_val, benchmark=None):
        """Make sure we can handle storage changes caused by the `scan_reduce_trace` rewrite."""

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


def test_higher_order_derivatives():
    ScanCompatibilityTests.check_higher_order_derivative(mode="NUMBA")


def test_grad_until_and_truncate_sequence_taps():
    ScanCompatibilityTests.check_grad_until_and_truncate_sequence_taps(mode="NUMBA")


@pytest.mark.parametrize("static_shape", (True, False))
def test_aliased_inner_outputs(static_shape):
    ScanCompatibilityTests.check_aliased_inner_outputs(static_shape, mode="NUMBA")


class TestUntracedSitSotAliasedInnerOutput:
    """Regressions for #2252.

    A sit_sot whose trace is unused is lowered to untraced_sit_sot and carried by reference
    across iterations. When its recurrence aliases an inner input or output, the numba
    backend must keep the borrowed view only when sound and copy it otherwise; a wrong
    choice corrupts the carry or aliases an outer buffer, so matching the reference is the
    check.
    """

    @pytest.mark.parametrize(
        "echo",
        [
            "seq",
            "non_seq",
            "tap_input",
            "tap_output",
            "other_untraced_input",
            "other_untraced_output",
        ],
    )
    def test_echo(self, echo):
        # One general scan: traced accumulator `y` and traced sit_sot `s` keep the loop
        # alive and give `s` a tap buffer; untraced states `u` (independent) and `m` are
        # carried by reference. `m`'s recurrence echoes whichever inner value `echo`
        # selects, exercising each borrow/copy decision.
        x = pt.matrix("x")
        c = pt.vector("c")

        def step(x_t, y_prev, s_prev, u_prev, m_prev, c):
            s_next = s_prev + x_t  # traced sit_sot -> provides a tap buffer
            u_next = x_t * 2.0  # independent untraced value
            m_next = {
                "seq": x_t,
                "non_seq": c,
                "tap_input": s_prev,
                "tap_output": s_next,
                "other_untraced_input": u_prev,
                "other_untraced_output": u_next,
            }[echo]
            # Accumulate the *carried* values u_prev and m_prev. A wrong borrow corrupts
            # the carry buffer between iterations, so the bug only surfaces if a later step
            # reads it back: reading m_next/u_next instead would pass even on buggy code.
            y_next = y_prev + m_prev + u_prev
            return y_next, s_next, u_next, m_next

        outs = scan(
            step,
            sequences=[x],
            outputs_info=[pt.zeros(3), pt.zeros(3), pt.zeros(3), pt.zeros(3)],
            non_sequences=[c],
            return_updates=False,
        )
        # Only y and s traces are used, so u and m are lowered to untraced_sit_sot.
        compare_numba_and_py(
            [x, c],
            [outs[0], outs[1]],
            [np.arange(15.0).reshape(5, 3), np.arange(3.0)],
            numba_mode="NUMBA",
        )

    def test_echo_self_in_place(self):
        # untraced state updates itself in place -> own previous value, kept by reference
        x = pt.matrix("x")

        def step(x_t, acc_prev):
            return acc_prev[:1].inc(x_t[:1])

        outs = scan(
            step,
            sequences=[x],
            outputs_info=[pt.zeros(3)],
            return_updates=False,
        )
        compare_numba_and_py(
            [x], [outs[-1]], [np.arange(15.0).reshape(5, 3)], numba_mode="NUMBA"
        )

    def test_two_untraced_states_sharing_inner_root(self):
        # Two untraced states whose recurrences are the *same* fresh inner value must each
        # get their own carry buffer (invariant B): borrowing both onto the shared root
        # would carry them in one buffer and alias the two outer outputs. The reversed read
        # of one carry surfaces the corruption when they alias. Guards the
        # ``seen_untraced_roots`` borrow check in the numba backend.
        x = pt.matrix("x")

        def step(x_t, y_prev, a_prev, b_prev):
            shared = x_t * 2.0  # single fresh root echoed by both untraced states
            y_next = y_prev + a_prev[::-1] - b_prev
            return y_next, shared, shared

        outs = scan(
            step,
            sequences=[x],
            outputs_info=[pt.zeros(3), pt.zeros(3), pt.zeros(3)],
            return_updates=False,
        )
        # Only y's trace is used, so a and b are lowered to untraced_sit_sot.
        compare_numba_and_py(
            [x], [outs[0]], [np.arange(15.0).reshape(5, 3)], numba_mode="NUMBA"
        )


@pytest.mark.parametrize("case", ["cross_store", "untraced_on_tap"])
def test_no_foreign_inplace_on_tap(case):
    """A recurrence may reuse only its own tap buffer in place.

    The numba inner optimizer may compute an output in place on another state's
    (destroyable) tap. That tap slot is overwritten by its own recurrence, so any
    foreign output landing there is corrupted: a tapped cross-store (both outputs
    ``>=1``-d) or an untraced output that keeps a reference to the tap. The
    ``NoOutputInplaceOnInput`` feature must reject those inplaces.
    """
    if case == "cross_store":
        # Two vector mit_sots each computed in place on the *other*'s oldest tap.
        n = pt.iscalar("n")

        def step(a_tm2, a_tm1, b_tm2, b_tm1, y):
            return b_tm2 + 1.0, a_tm2 + 1.0, y + a_tm1.sum() + b_tm1.sum()

        outs = scan(
            step,
            n_steps=n,
            outputs_info=[
                {"initial": pt.as_tensor(np.ones((2, 3))), "taps": [-2, -1]},
                {"initial": pt.as_tensor(np.ones((2, 3)) * 5), "taps": [-2, -1]},
                np.float64(0.0),
            ],
            return_updates=False,
        )
        graph_inputs, graph_outputs, test_inputs = [n], [outs[2]], [6]
    else:  # untraced_on_tap: an untraced output computed in place on a mit_sot tap
        x = pt.dvector("x")

        def step(x_t, z_tm2, z_tm1, y, m):
            return z_tm1 + z_tm2 + 0.0 * x_t, y + m, z_tm2 + 1.0

        outs = scan(
            step,
            sequences=[x],
            outputs_info=[
                {"initial": pt.as_tensor([1.0, 2.0]), "taps": [-2, -1]},
                np.float64(0.0),
                np.float64(0.0),
            ],
            return_updates=False,
        )
        graph_inputs, graph_outputs, test_inputs = [x], [outs[1]], [np.arange(1.0, 9.0)]

    compare_numba_and_py(graph_inputs, graph_outputs, test_inputs, numba_mode="NUMBA")
