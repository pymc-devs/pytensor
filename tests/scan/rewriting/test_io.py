import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import function, scan
from pytensor.compile.mode import Mode, get_default_mode
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import in2out
from pytensor.scan.op import Scan
from pytensor.scan.rewriting import scan_remove_unused
from pytensor.scan.utils import until
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.math import dot
from pytensor.tensor.shape import specify_shape
from pytensor.tensor.type import ivector, matrix, vector
from tests.scan.test_basic import scan_nodes_from_fct


class TestScanInputAndOutputCleanup:
    """Integration tests for the combined ``scan_input_and_output_cleanup``
    pass: ``scan_remove_unused`` + ``scan_inline_invariant_constants`` +
    ``scan_merge_duplicate_inputs``.
    """

    mode = get_default_mode().including("scan")

    def test_non_seqs(self):
        """Duplicate + unused non-sequences are collapsed / dropped."""
        W = matrix(name="W")
        v = ivector(name="v")
        y1 = scan(
            lambda i, W: W[i],
            sequences=v,
            outputs_info=None,
            non_sequences=[W],
            return_updates=False,
        )
        y2 = scan(
            lambda i, _, W: W[i],
            sequences=v,
            outputs_info=None,
            non_sequences=[W[0], W],
            return_updates=False,
        )
        y3 = scan(
            lambda i, W, _: W[i],
            sequences=v,
            outputs_info=None,
            non_sequences=[W, W[0]],
            return_updates=False,
        )
        y4 = scan(
            lambda i, _, _2, W: W[i],
            sequences=v,
            outputs_info=None,
            non_sequences=[W[0], W[0], W],
            return_updates=False,
        )
        y5 = scan(
            lambda i, _, W, _2: W[i],
            sequences=v,
            outputs_info=None,
            non_sequences=[W[0], W, W[0]],
            return_updates=False,
        )
        y6 = scan(
            lambda i, W, _, _2: W[i],
            sequences=v,
            outputs_info=None,
            non_sequences=[W, W[0], W[0]],
            return_updates=False,
        )
        # TODO: y7 have problem during run time. I think it should
        # raise an error during the scan construction.
        # y7, _ = scan(lambda i, W, _, _2: W[i], sequences=v,
        #                    outputs_info=None, non_sequences=[v, W[0], W])

        W_val = np.random.normal(size=(3, 3)).astype(config.floatX)
        exp_val = W_val[np.r_[1, 2]]

        for out in [y1, y2, y3, y4, y5, y6]:
            f = function([W, v], out, mode=self.mode)

            res = f(W_val, [1, 2])
            assert np.array_equal(res, exp_val)

            scan_nodes = scan_nodes_from_fct(f)
            assert len(scan_nodes) == 1

            scan_node = scan_nodes[0]
            assert len(scan_node.inputs[1:]) == len(set(scan_node.inputs[1:]))
            inp = scan_node.op.inner_non_seqs(scan_node.op.inner_inputs)
            assert len(inp) == 1
            assert len(inp) == len(set(inp))

            inp = scan_node.op.outer_non_seqs(scan_node.inputs)
            assert len(inp) == 1
            assert len(inp) == len(set(inp))

    def test_seqs(self):
        """Duplicate + unused sequences are collapsed / dropped."""
        W = matrix(name="W")
        v = ivector(name="v")
        vv = matrix(name="vv")
        y1 = scan(
            lambda i, W: W[i],
            sequences=v,
            outputs_info=None,
            non_sequences=[W],
            return_updates=False,
        )
        y2 = scan(
            lambda i, _, W: W[i],
            sequences=[v, v],
            outputs_info=None,
            non_sequences=W,
            return_updates=False,
        )
        y3 = scan(
            lambda i, _, W: W[i],
            sequences=[v, vv[0]],
            outputs_info=None,
            non_sequences=W,
            return_updates=False,
        )
        y4 = scan(
            lambda _, i, W: W[i],
            sequences=[vv[0], v],
            outputs_info=None,
            non_sequences=W,
            return_updates=False,
        )
        y5 = scan(
            lambda _, i, _2, W: W[i],
            sequences=[vv, v, vv[0]],
            outputs_info=None,
            non_sequences=W,
            return_updates=False,
        )
        y6 = scan(
            lambda _, _2, i, W: W[i],
            sequences=[vv[0], vv, v],
            outputs_info=None,
            non_sequences=W,
            return_updates=False,
        )
        y7 = scan(
            lambda i, _, _2, W: W[i],
            sequences=[v, vv[0], vv[0]],
            outputs_info=None,
            non_sequences=W,
            return_updates=False,
        )
        y8 = scan(
            lambda _, i, W, _2, _3: W[i],
            sequences=[vv[0], v],
            outputs_info=None,
            non_sequences=[W, W[0], W[0]],
            return_updates=False,
        )

        W_val = np.random.normal(size=(3, 3)).astype(config.floatX)
        exp_val = W_val[np.r_[1, 2]]

        for out in [y1, y2, y3, y4, y5, y6, y7, y8]:
            f = function(
                [W, v, vv],
                out,
                on_unused_input="ignore",
                mode=self.mode,
            )

            res = f(W_val, [1, 2], W_val)
            assert np.array_equal(res, exp_val)

            scan_nodes = scan_nodes_from_fct(f)
            assert len(scan_nodes) == 1
            scan_node = scan_nodes[0]

            assert len(scan_node.inputs[1:]) == len(set(scan_node.inputs[1:]))
            inp = scan_node.op.inner_seqs(scan_node.op.inner_inputs)
            assert len(inp) == 1
            inp = scan_node.op.outer_seqs(scan_node.inputs)
            assert len(inp) == 1
            inp = scan_node.op.inner_non_seqs(scan_node.op.inner_inputs)
            assert len(inp) == 1
            inp = scan_node.op.outer_non_seqs(scan_node.inputs)
            assert len(inp) == 1


class TestRemoveUnused:
    """Tests for ``scan_remove_unused``.

    Each test:

        1. Builds a minimal scan and collects its outputs.
        2. Clones the graph and applies only ``scan_remove_unused`` to the
           clone (isolated from every other rewrite).
        3. Asserts structural changes on the rewritten ``Scan`` op
           (``ScanInfo`` counts including ``n_seqs`` and ``n_non_seqs``).
        4. Compiles both the original and rewritten graphs under
           ``Mode(linker="py", optimizer=None)`` and verifies identical output.
    """

    in2out_scan_remove_unused = in2out(scan_remove_unused, ignore_newtrees=True)
    NO_OPT = Mode(linker="py", optimizer=None)

    @classmethod
    def rewrite(cls, inputs, outputs):
        """Clone ``inputs``/``outputs`` into two fresh FunctionGraphs and apply
        ``scan_remove_unused`` to one of them.

        Returns ``((orig_fg, *orig_scans), (rewr_fg, *rewr_scans))`` with
        scans in topological order. ``orig_fg`` is untouched.
        """
        inputs = list(inputs)
        outputs = list(outputs)
        orig_fg = FunctionGraph(inputs=inputs, outputs=outputs, clone=True)
        rewr_fg = FunctionGraph(inputs=inputs, outputs=outputs, clone=True)
        cls.in2out_scan_remove_unused.rewrite(rewr_fg)
        orig_scans = [n for n in orig_fg.toposort() if isinstance(n.op, Scan)]
        rewr_scans = [n for n in rewr_fg.toposort() if isinstance(n.op, Scan)]
        return (orig_fg, *orig_scans), (rewr_fg, *rewr_scans)

    @staticmethod
    def assert_structure(
        scan,
        *,
        n_seqs=0,
        n_mit_mot=0,
        n_mit_sot=0,
        n_sit_sot=0,
        n_nit_sot=0,
        n_untraced_sit_sot=0,
        n_non_seqs=0,
        as_while=False,
    ):
        info = scan.op.info
        expected = {
            "n_seqs": n_seqs,
            "n_mit_mot": n_mit_mot,
            "n_mit_sot": n_mit_sot,
            "n_sit_sot": n_sit_sot,
            "n_nit_sot": n_nit_sot,
            "n_untraced_sit_sot": n_untraced_sit_sot,
            "n_non_seqs": n_non_seqs,
            "as_while": as_while,
        }
        actual = {k: getattr(info, k) for k in expected}
        assert actual == expected, "\n".join(
            map(str, zip(actual.keys(), actual.values(), expected.values()))
        )

    @classmethod
    def assert_numerical_match(cls, orig_fg, rewr_fg, input_vals):
        orig_fn = function(
            orig_fg.inputs,
            orig_fg.outputs,
            mode=cls.NO_OPT,
            on_unused_input="ignore",
        )
        rewr_fn = function(
            rewr_fg.inputs,
            rewr_fg.outputs,
            mode=cls.NO_OPT,
            on_unused_input="ignore",
        )
        orig_out = orig_fn(*input_vals)
        rewr_out = rewr_fn(*input_vals)
        if not isinstance(orig_out, list | tuple):
            orig_out = [orig_out]
        if not isinstance(rewr_out, list | tuple):
            rewr_out = [rewr_out]
        for a, b in zip(orig_out, rewr_out, strict=True):
            np.testing.assert_almost_equal(a, b)

    def test_seq_non_seq(self):
        x0 = pt.scalar("x0")
        s = pt.vector("s", shape=(4,))
        ns = pt.scalar("ns")

        def step(s_t, x_t, ns):
            # only x_t is used
            return x_t**2

        xs = scan(
            step,
            sequences=[s],
            outputs_info=[x0],
            non_sequences=[ns],
            n_steps=4,
            return_updates=False,
        )

        (orig_fg, orig_scan), (rewr_fg, rewr_scan) = self.rewrite([s, x0, ns], [xs])
        self.assert_structure(orig_scan, n_seqs=1, n_sit_sot=1, n_non_seqs=1)
        self.assert_structure(rewr_scan, n_seqs=0, n_sit_sot=1, n_non_seqs=0)
        self.assert_numerical_match(
            orig_fg,
            rewr_fg,
            [np.arange(4, dtype="float64"), np.array(1.0), np.array(np.pi)],
        )

    def test_nit_sot(self):
        s = pt.vector("s", shape=(4,))
        x = pt.vector("x", shape=(5,))

        def step(s_t, x_ns, x_nsp1):
            return (x_ns + 1.0), (x_ns * x_nsp1 * s_t * 2.0)

        xs, _ys = scan(
            step,
            sequences=[s],
            outputs_info=[None, None],
            non_sequences=[x, x + 1],
            n_steps=4,
            return_updates=False,
        )

        (orig_fg, orig_scan), (rewr_fg, rewr_scan) = self.rewrite([s, x], [xs])
        self.assert_structure(orig_scan, n_seqs=1, n_nit_sot=2, n_non_seqs=2)
        self.assert_structure(rewr_scan, n_seqs=0, n_nit_sot=1, n_non_seqs=1)
        self.assert_numerical_match(
            orig_fg,
            rewr_fg,
            [np.arange(4, dtype="float64"), np.arange(5, dtype="float64")],
        )

    def test_sit_sot(self):
        x0 = pt.vector("x0", shape=(5,))
        y0 = pt.vector("y0", shape=(5,))

        def step(x_prev, y_prev):
            return x_prev + 1.0, y_prev * 0.5

        xs, _ys = scan(
            step,
            outputs_info=[x0, y0],
            n_steps=4,
            return_updates=False,
        )

        (orig_fg, orig_scan), (rewr_fg, rewr_scan) = self.rewrite([x0, y0], [xs])
        self.assert_structure(orig_scan, n_sit_sot=2)
        self.assert_structure(rewr_scan, n_sit_sot=1)
        self.assert_numerical_match(
            orig_fg,
            rewr_fg,
            [np.zeros(5, dtype="float64"), np.ones(5, dtype="float64")],
        )

    def test_direct_dependency(self):
        """Each inner output reads BOTH taps; only x consumed -> BOTH stay."""
        x0 = pt.vector("x0", shape=(5,))
        y0 = pt.vector("y0", shape=(5,))

        def step(x_prev, y_prev):
            return 0.5 * x_prev + 0.3 * y_prev, 0.2 * x_prev + 0.8 * y_prev

        xs, _ys = scan(
            step,
            outputs_info=[x0, y0],
            n_steps=4,
            return_updates=False,
        )

        (orig_fg, orig_scan), (rewr_fg, rewr_scan) = self.rewrite([x0, y0], [xs])
        self.assert_structure(orig_scan, n_sit_sot=2)
        self.assert_structure(rewr_scan, n_sit_sot=2)
        self.assert_numerical_match(
            orig_fg,
            rewr_fg,
            [
                np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
            ],
        )

    def test_dependency_cycle(self):
        """Two sit_sot candidates cross-read each other's tap; a nit_sot
        pins one -> BOTH stay. ``b_prev`` only enters
        ``surviving_ancestors`` after A un-confirms, so the fixpoint must
        grow the set in-pass, else B drops and the rebuilt Scan dangles
        on ``b_prev``.
        """
        a0 = pt.dscalar("a0")
        b0 = pt.dscalar("b0")

        def step(a_prev, b_prev):
            a_next = b_prev * 0.5
            b_next = a_prev * 0.5
            c = a_prev + 1.0
            return a_next, b_next, c

        _as, _bs, cs = scan(
            step,
            outputs_info=[a0, b0, None],
            n_steps=4,
            return_updates=False,
        )

        fg = FunctionGraph(inputs=[a0, b0], outputs=[cs], clone=True)
        scan_node = next(n for n in fg.toposort() if isinstance(n.op, Scan))
        assert scan_remove_unused.fn(fg, scan_node) is None

    def test_dependency_chain(self):
        """Linear chain ``b`` reads ``a``, ``c`` reads ``b``, nit_sot
        ``d`` reads ``c``. Two cases:

        Case 1: only ``ds`` observed. The chain is pinned tail-first --
        ``c_prev`` is reached directly, ``b_prev`` only after C
        un-confirms, ``a_prev`` only after B does. All four states stay.

        Case 2: only ``bs`` and ``cs`` observed. ``a_prev`` and
        ``b_prev`` are reached directly, so ``as`` is pinned via the
        chain. ``ds`` (a nit_sot with no inner inputs and no outer
        client) has nothing pinning it -- dropped.
        """
        a0 = pt.dscalar("a0")
        b0 = pt.dscalar("b0")
        c0 = pt.dscalar("c0")

        def step(a_prev, b_prev, c_prev):
            a_next = a_prev * 0.7
            b_next = a_prev * 0.5
            c_next = b_prev * 0.5
            d = c_prev + 1.0
            return a_next, b_next, c_next, d

        _as, bs, cs, ds = scan(
            step,
            outputs_info=[a0, b0, c0, None],
            n_steps=4,
            return_updates=False,
        )

        input_vals = [np.array(1.0), np.array(2.0), np.array(3.0)]

        (orig_fg, orig_scan), (rewr_fg, rewr_scan) = self.rewrite([a0, b0, c0], [ds])
        self.assert_structure(orig_scan, n_sit_sot=3, n_nit_sot=1)
        self.assert_structure(rewr_scan, n_sit_sot=3, n_nit_sot=1)
        self.assert_numerical_match(orig_fg, rewr_fg, input_vals)

        (orig_fg, orig_scan), (rewr_fg, rewr_scan) = self.rewrite(
            [a0, b0, c0], [bs, cs]
        )
        self.assert_structure(orig_scan, n_sit_sot=3, n_nit_sot=1)
        self.assert_structure(rewr_scan, n_sit_sot=3, n_nit_sot=0)
        self.assert_numerical_match(orig_fg, rewr_fg, input_vals)

    def test_untraced_sit_sot(self):
        rng_a = pt.random.rng("rng_a")
        rng_b = pt.random.rng("rng_b")
        # Exclude random_unsafe so unused draws don't drop the rng pre-test.
        scan_mode = get_default_mode().excluding("random_unsafe")

        def step(prev_rng_a, prev_rng_b):
            next_rng_a, draw_a = prev_rng_a.normal()
            next_rng_b, draw_b = prev_rng_b.normal()
            return (next_rng_a, next_rng_b, draw_a, draw_b)

        final_rng_a, _final_rng_b, _draws_a, draws_b = scan(
            step,
            outputs_info=[rng_a, rng_b, None, None],
            n_steps=4,
            return_updates=False,
            mode=scan_mode,
        )

        rng_a_test = np.random.default_rng(200)
        rng_b_test = np.random.default_rng(201)

        # Case 1: only draws_b kept -> a-side entirely dropped
        (orig_fg, orig_scan), (rewr_fg, rewr_scan) = self.rewrite(
            [rng_a, rng_b], [draws_b]
        )
        self.assert_structure(orig_scan, n_nit_sot=2, n_untraced_sit_sot=2)
        self.assert_structure(rewr_scan, n_nit_sot=1, n_untraced_sit_sot=1)
        self.assert_numerical_match(orig_fg, rewr_fg, [rng_a_test, rng_b_test])

        # Case 2: final_rng_a pinned by external use + draws_b kept -> drops
        # draws_a (but not rng_a, pinned via its outer client).
        _, draws_a_external = final_rng_a.normal()
        (orig_fg, orig_scan), (rewr_fg, rewr_scan) = self.rewrite(
            [rng_a, rng_b], [draws_b, draws_a_external]
        )
        self.assert_structure(orig_scan, n_nit_sot=2, n_untraced_sit_sot=2)
        self.assert_structure(rewr_scan, n_nit_sot=1, n_untraced_sit_sot=2)
        self.assert_numerical_match(orig_fg, rewr_fg, [rng_a_test, rng_b_test])

    def test_pullback_disconnected_output(self):
        # Tests unused mit-mot
        x0 = pt.vector("x0", shape=(5,))
        y0 = pt.vector("y0", shape=(5,))

        xs, ys = scan(
            lambda x, y: (x**2, y * 1.1),
            outputs_info=[x0, y0],
            n_steps=4,
            return_updates=False,
        )

        # Case 1: Cost depends only on xs. L_op's eager cleanup already drops
        # the pullback's dead mit_mot, so the orig pullback is already at
        # ``n_mit_mot=1``. Our rewrite then drops the now-clientless forward
        # ys, bringing the forward scan to ``n_sit_sot=1``.
        cost = xs[-1].sum()
        gx = pt.grad(cost, x0)

        (
            (orig_fg, orig_forward_scan, orig_pullback_scan),
            (rewr_fg, rewr_forward_scan, rewr_pullback_scan),
        ) = self.rewrite([x0, y0], [cost, gx])
        self.assert_structure(orig_forward_scan, n_sit_sot=2)
        self.assert_structure(orig_pullback_scan, n_seqs=1, n_mit_mot=1)
        self.assert_structure(rewr_forward_scan, n_sit_sot=1)
        self.assert_structure(rewr_pullback_scan, n_seqs=1, n_mit_mot=1)
        self.assert_numerical_match(
            orig_fg,
            rewr_fg,
            [np.ones(5, dtype="float64"), np.ones(5, dtype="float64")],
        )

        # Case 2: Cost depends on both outputs -> nothing to clean.
        cost = xs[-1].sum() + ys[-1].sum()
        gx = pt.grad(cost, x0)
        (
            (orig_fg, orig_forward_scan, orig_pullback_scan),
            (rewr_fg, rewr_forward_scan, rewr_pullback_scan),
        ) = self.rewrite([x0, y0], [cost, gx])
        self.assert_structure(orig_forward_scan, n_sit_sot=2)
        self.assert_structure(orig_pullback_scan, n_seqs=1, n_mit_mot=1)
        self.assert_structure(rewr_forward_scan, n_sit_sot=2)
        self.assert_structure(rewr_pullback_scan, n_seqs=1, n_mit_mot=1)
        self.assert_numerical_match(
            orig_fg,
            rewr_fg,
            [np.ones(5, dtype="float64"), np.ones(5, dtype="float64")],
        )

    def test_shared_inner_output(self):
        x = pt.vector("x", shape=(5,))

        def step(x_ns):
            y = x_ns + 1.0
            return y, y  # same Variable in two output slots

        kept, _unused = scan(
            step,
            outputs_info=[None, None],
            non_sequences=[x],
            n_steps=4,
            return_updates=False,
        )

        (orig_fg, orig_scan), (rewr_fg, rewr_scan) = self.rewrite([x], [kept])
        self.assert_structure(orig_scan, n_nit_sot=2, n_non_seqs=1)
        self.assert_structure(rewr_scan, n_nit_sot=1, n_non_seqs=1)
        self.assert_numerical_match(orig_fg, rewr_fg, [np.arange(5, dtype="float64")])

    def test_while_scan(self):
        x0 = pt.scalar("x0", dtype="float64")
        y0 = pt.scalar("y0", dtype="float64")

        def step(x_prev, y_prev):
            return (x_prev + 1.0, y_prev * 0.5), until(x_prev > 5.0)

        xs, ys = scan(
            step,
            outputs_info=[x0, y0],
            n_steps=20,
            return_updates=False,
        )

        # Should drop state if condition doesn't depend on it.
        (orig_fg, orig_scan), (rewr_fg, rewr_scan) = self.rewrite([x0, y0], [xs])
        self.assert_structure(orig_scan, n_sit_sot=2, as_while=True)
        self.assert_structure(rewr_scan, n_sit_sot=1, as_while=True)
        self.assert_numerical_match(
            orig_fg,
            rewr_fg,
            [np.float64(0.0), np.float64(1.0)],
        )

        # But not if the condition depends on it.
        (orig_fg, orig_scan), (rewr_fg, rewr_scan) = self.rewrite([x0, y0], [ys])
        self.assert_structure(orig_scan, n_sit_sot=2, as_while=True)
        self.assert_structure(rewr_scan, n_sit_sot=2, as_while=True)
        self.assert_numerical_match(
            orig_fg,
            rewr_fg,
            [np.float64(0.0), np.float64(1.0)],
        )


def test_alloc_inputs1():
    W1 = matrix("W1")
    W2 = matrix("W2")
    h0 = vector("h0")

    def lambda_fn(h, W1, W2):
        return dot(h, W1 * W2)

    o = scan(
        lambda_fn,
        outputs_info=h0,
        non_sequences=[W1, pt.zeros_like(W2)],
        n_steps=5,
        return_updates=False,
    )

    f = function([h0, W1, W2], o, mode=get_default_mode().including("scan"))
    scan_node = next(x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan))
    assert (
        len(
            [
                x
                for x in scan_node.op.fn.maker.fgraph.toposort()
                if isinstance(x.op, Elemwise)
            ]
        )
        == 0
    )


@pytest.mark.skip(
    reason="This tests depends on an optimization for "
    "scan that has not been implemented yet."
)
def test_alloc_inputs2():
    W1 = matrix()
    W2 = matrix()
    h0 = vector()

    def lambda_fn(W1, h, W2):
        return W1 * dot(h, W2)

    o = scan(
        lambda_fn,
        sequences=pt.zeros_like(W1),
        outputs_info=h0,
        non_sequences=[pt.zeros_like(W2)],
        n_steps=5,
        return_updates=False,
    )

    f = function([h0, W1, W2], o, mode=get_default_mode().including("scan"))
    scan_node = next(x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan))

    assert (
        len(
            [
                x
                for x in scan_node.op.fn.maker.fgraph.toposort()
                if isinstance(x.op, Elemwise)
            ]
        )
        == 0
    )


def test_alloc_inputs3():
    _W1 = matrix()
    _W2 = matrix()
    _h0 = vector()

    W1 = specify_shape(_W1, (3, 3))
    W2 = specify_shape(_W2, (3, 3))
    h0 = specify_shape(_h0, (3,))

    def lambda_fn(W1, h, W2):
        return W1 * dot(h, W2)

    o = scan(
        lambda_fn,
        sequences=pt.zeros_like(W1),
        outputs_info=h0,
        non_sequences=[pt.zeros_like(W2)],
        n_steps=5,
        return_updates=False,
    )

    # TODO FIXME: This result depends on unrelated rewrites in the "fast" mode.
    f = function([_h0, _W1, _W2], o, mode="FAST_RUN")
    scan_node = next(x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan))

    assert len(scan_node.op.inner_inputs) == 1
