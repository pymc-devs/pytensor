import numpy as np

import pytensor
from pytensor import function, scan
from pytensor.compile.executor import Function
from pytensor.compile.mode import get_default_mode, get_mode
from pytensor.configdefaults import config
from pytensor.graph.destroyhandler import _contains_cycle
from pytensor.graph.fg import FunctionGraph
from pytensor.scan.op import Scan
from pytensor.scan.rewriting import ScanMerge
from pytensor.scan.utils import until
from pytensor.tensor import stack
from pytensor.tensor.type import scalar, vector
from tests import unittest_tools as utt


mode = get_mode(config.mode)


class TestScanMerge:
    mode = get_default_mode().including("scan").excluding("scan_pushout_seqs_ops")

    @staticmethod
    def count_scans(fn):
        if isinstance(fn, Function):
            nodes = fn.maker.fgraph.apply_nodes
        else:
            nodes = fn.apply_nodes
        scans = [node for node in nodes if isinstance(node.op, Scan)]
        return len(scans)

    def test_basic(self):
        x = vector()
        y = vector()

        def sum(s):
            return s + 1

        sx, _upx = scan(sum, sequences=[x])
        sy, _upy = scan(sum, sequences=[y])

        f = function([x, y], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 2

        sx, _upx = scan(sum, sequences=[x], n_steps=2)
        sy, _upy = scan(sum, sequences=[y], n_steps=3)

        f = function([x, y], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 2

        sx, _upx = scan(sum, sequences=[x], n_steps=4)
        sy, _upy = scan(sum, sequences=[y], n_steps=4)

        f = function([x, y], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 1

        sx, _upx = scan(sum, sequences=[x])
        sy, _upy = scan(sum, sequences=[x])

        f = function([x], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 1

        sx, _upx = scan(sum, sequences=[x])
        sy, _upy = scan(sum, sequences=[x], mode="FAST_COMPILE")

        f = function([x], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 1

        sx, _upx = scan(sum, sequences=[x])
        sy, _upy = scan(sum, sequences=[x], truncate_gradient=1)

        f = function([x], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 2

    def test_three_scans(self):
        r"""
        This test checks a case where we have three `Scan`\s, two of them
        cannot be merged together, but the third one can be merged with
        either.
        """
        x = vector()
        y = vector()

        def sum(s):
            return s + 1

        sx, _upx = scan(sum, sequences=[x], n_steps=4, name="X")
        # We need to use an expression of y rather than y so the toposort
        # comes up with the 'Y' scan last.
        sy, _upy = scan(sum, sequences=[2 * y + 2], n_steps=4, name="Y")
        sz, _upz = scan(sum, sequences=[sx], n_steps=4, name="Z")

        f = function([x, y], [sy, sz], mode=self.mode)
        assert self.count_scans(f) == 2

        rng = np.random.default_rng(utt.fetch_seed())
        x_val = rng.uniform(size=(4,)).astype(config.floatX)
        y_val = rng.uniform(size=(4,)).astype(config.floatX)
        # Run it so DebugMode can detect optimization problems.
        f(x_val, y_val)

    def test_no_cyclic_merge(self):
        r"""Merging must not contract `Scan`\s into a cyclic graph.

        Regression test for #2221. `ScanMerge` groups `Scan`\s that share
        ``n_steps`` into mutually-independent sets, but contracting two sets that
        depend on each other in both directions creates a cycle that later hangs
        ``toposort``. Here set A = ``{a_out, a_in}`` (n_steps=15) and set
        B = ``{b_fwd, b_sm}`` (n_steps=14), with ``a_in`` consuming ``b_fwd`` and
        ``b_sm`` consuming ``a_out`` — so merging both bundles would deadlock.
        """
        K = 3
        x = vector("x")

        a_out = scan(
            lambda p: 0.9 * p + 1.0,
            outputs_info=[x],
            n_steps=15,
            return_updates=False,
        )
        b_fwd = scan(
            lambda p: 0.8 * p + 0.5,
            outputs_info=[x],
            n_steps=14,
            return_updates=False,
        )
        b_glue = stack([x, *[b_fwd[i] for i in range(14)]])
        a_in = scan(
            lambda s, p: 0.5 * p + s,
            sequences=[b_glue],
            outputs_info=[pytensor.tensor.zeros(K)],
            n_steps=15,
            return_updates=False,
        )
        b_sm = scan(
            lambda s, p: 0.5 * p + s,
            sequences=[a_out[:-1]],
            outputs_info=[pytensor.tensor.zeros(K)],
            n_steps=14,
            return_updates=False,
        )
        out = a_in[-1] + b_sm[-1]

        # Run `ScanMerge` directly: it must not commit a cyclic graph. This
        # asserts at the graph level so a regression fails fast instead of
        # hanging the later compilation `toposort`.
        fgraph = FunctionGraph([x], [out], clone=True)
        n_before = sum(isinstance(n.op, Scan) for n in fgraph.apply_nodes)
        ScanMerge().apply(fgraph)
        n_after = sum(isinstance(n.op, Scan) for n in fgraph.apply_nodes)
        assert not _contains_cycle(fgraph, {})
        # The safe bundle is still merged; only the cycle-forming one is skipped.
        assert n_before == 4
        assert n_after == 3

        # End-to-end the function compiles and matches a scan_merge-free build.
        f = function([x], out, mode=self.mode)
        f_ref = function([x], out, mode=self.mode.excluding("scan_merge"))
        x_val = np.array([1.0, 2.0, 3.0], dtype=config.floatX)
        np.testing.assert_allclose(f(x_val), f_ref(x_val))

    def test_belongs_to_set(self):
        """
        Test the method belongs_to of this class. Specifically see if it
        detects the two `Scan` nodes as not being similar.
        """
        inps = vector()
        state = scalar()
        y1 = scan(
            lambda x, y: x * y,
            sequences=inps,
            outputs_info=state,
            n_steps=5,
            return_updates=False,
        )

        y2 = scan(
            lambda x, y: (x + y, until(x > 0)),
            sequences=inps,
            outputs_info=state,
            n_steps=5,
            return_updates=False,
        )
        scan_node1 = y1.owner.inputs[0].owner
        assert isinstance(scan_node1.op, Scan)
        scan_node2 = y2.owner.inputs[0].owner
        assert isinstance(scan_node2.op, Scan)
        opt_obj = ScanMerge()
        assert not opt_obj.belongs_to_set(scan_node1, [scan_node2])
        assert not opt_obj.belongs_to_set(scan_node2, [scan_node1])

    @config.change_flags(cxx="")  # Just for faster compilation
    def test_while_scan(self):
        x = vector("x")
        y = vector("y")

        def add(s):
            return s + 1, until(s > 5)

        def sub(s):
            return s - 1, until(s > 5)

        def sub_alt(s):
            return s - 1, until(s > 4)

        sx, _upx = scan(add, sequences=[x])
        sy, _upy = scan(sub, sequences=[y])

        f = function([x, y], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 2

        sx, _upx = scan(add, sequences=[x])
        sy, _upy = scan(sub, sequences=[x])

        f = function([x], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 1

        sx, _upx = scan(add, sequences=[x])
        sy, _upy = scan(sub_alt, sequences=[x])

        f = function([x], [sx, sy], mode=self.mode)
        assert self.count_scans(f) == 2

    @config.change_flags(cxx="")  # Just for faster compilation
    def test_while_scan_nominal_dependency(self):
        """Test case where condition depends on nominal variables.

        This is a regression test for #509
        """
        c1 = scalar("c1")
        c2 = scalar("c2")
        x = vector("x", shape=(5,))
        y = vector("y", shape=(5,))
        z = vector("z", shape=(5,))

        def add(s1, s2, const):
            return s1 + 1, until(s2 > const)

        def sub(s1, s2, const):
            return s1 - 1, until(s2 > const)

        sx = scan(add, sequences=[x, z], non_sequences=[c1], return_updates=False)
        sy = scan(sub, sequences=[y, -z], non_sequences=[c1], return_updates=False)

        f = pytensor.function(inputs=[x, y, z, c1], outputs=[sx, sy], mode=self.mode)
        assert self.count_scans(f) == 2
        res_sx, res_sy = f(
            x=[0, 0, 0, 0, 0],
            y=[0, 0, 0, 0, 0],
            z=[0, 1, 2, 3, 4],
            c1=0,
        )
        np.testing.assert_array_equal(res_sx, [1, 1])
        np.testing.assert_array_equal(res_sy, [-1, -1, -1, -1, -1])

        sx = scan(add, sequences=[x, z], non_sequences=[c1], return_updates=False)
        sy = scan(sub, sequences=[y, z], non_sequences=[c2], return_updates=False)

        f = pytensor.function(
            inputs=[x, y, z, c1, c2], outputs=[sx, sy], mode=self.mode
        )
        assert self.count_scans(f) == 2
        res_sx, res_sy = f(
            x=[0, 0, 0, 0, 0],
            y=[0, 0, 0, 0, 0],
            z=[0, 1, 2, 3, 4],
            c1=3,
            c2=1,
        )
        np.testing.assert_array_equal(res_sx, [1, 1, 1, 1, 1])
        np.testing.assert_array_equal(res_sy, [-1, -1, -1])

        sx = scan(add, sequences=[x, z], non_sequences=[c1], return_updates=False)
        sy = scan(sub, sequences=[y, z], non_sequences=[c1], return_updates=False)

        f = pytensor.function(inputs=[x, y, z, c1], outputs=[sx, sy], mode=self.mode)
        assert self.count_scans(f) == 1

        def nested_scan(c, x, z):
            sx = scan(add, sequences=[x, z], non_sequences=[c], return_updates=False)
            sy = scan(sub, sequences=[x, z], non_sequences=[c], return_updates=False)
            return sx.sum() + sy.sum()

        sz = scan(
            nested_scan,
            sequences=[stack([c1, c2])],
            non_sequences=[x, z],
            mode=self.mode,
            return_updates=False,
        )

        f = pytensor.function(inputs=[x, z, c1, c2], outputs=sz, mode=mode)
        [scan_node] = [
            node for node in f.maker.fgraph.apply_nodes if isinstance(node.op, Scan)
        ]
        inner_f = scan_node.op.fgraph
        assert self.count_scans(inner_f) == 1
