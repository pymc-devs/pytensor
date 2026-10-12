import numpy as np
import pytest

import pytensor
from pytensor import function
from pytensor.conditional.op import IfElse, ifelse
from pytensor.conditional.rewriting import cond_make_inplace
from pytensor.tensor.type import iscalar, scalar, vector
from tests import unittest_tools as utt


@pytest.mark.parametrize("large_graph", [False, True])
def test_cond_make_inplace(large_graph):
    c, x, y = iscalar("c"), scalar("x"), scalar("y")
    branch = y
    # Scalar graphs with at least 500 nodes skip views to reduce rewrite cost.
    for index in range(500 if large_graph else 0):
        branch = branch + index
    out = ifelse(c, x, branch)
    result = utt.RewriteTester(
        [c, x, y], [out], include=None, custom_rewrite=cond_make_inplace
    )
    expected = IfElse(1, as_view=not large_graph)(c, x, branch)
    result.assert_graph(expected)


class TestIfelseRewrites:
    @pytest.mark.skip(reason="Optimization temporarily disabled")
    def test_merge(self):
        x = vector("x")
        y = vector("y")
        c = iscalar("c")
        z1 = ifelse(c, x + 1, y + 1)
        z2 = ifelse(c, x + 2, y + 2)
        z = z1 + z2
        f = function([c, x, y], z)
        assert (
            len([n for n in f.maker.fgraph.toposort() if isinstance(n.op, IfElse)]) == 1
        )

    @pytest.mark.skip(reason="Optimization temporarily disabled")
    def test_remove_useless_inputs1(self):
        x = vector("x")
        y = vector("y")
        c = iscalar("c")
        z = ifelse(c, (x, x), (y, y))
        f = function([c, x, y], z)

        ifnode = next(n for n in f.maker.fgraph.toposort() if isinstance(n.op, IfElse))
        assert len(ifnode.inputs) == 3

    @pytest.mark.skip(reason="Optimization temporarily disabled")
    def test_remove_useless_inputs2(self):
        x1 = vector("x1")
        x2 = vector("x2")
        y1 = vector("y1")
        y2 = vector("y2")
        c = iscalar("c")
        z = ifelse(c, (x1, x1, x1, x2, x2), (y1, y1, y2, y2, y2))
        f = function([c, x1, x2, y1, y2], z)

        ifnode = next(x for x in f.maker.fgraph.toposort() if isinstance(x.op, IfElse))
        assert len(ifnode.outputs) == 3

    @pytest.mark.skip(reason="Optimization temporarily disabled")
    def test_pushout1(self):
        x1 = scalar("x1")
        x2 = scalar("x2")
        y1 = scalar("y1")
        y2 = scalar("y2")
        w1 = scalar("w1")
        w2 = scalar("w2")
        c = iscalar("c")
        x, y = ifelse(c, (x1, y1), (x2, y2), name="f1")
        z = ifelse(c, w1, w2, name="f2")
        out = x * z * y

        f = function([x1, x2, y1, y2, w1, w2, c], out, allow_input_downcast=True)
        assert isinstance(f.maker.fgraph.toposort()[-1].op, IfElse)
        rng = np.random.default_rng(utt.fetch_seed())
        vx1 = rng.uniform()
        vx2 = rng.uniform()
        vy1 = rng.uniform()
        vy2 = rng.uniform()
        vw1 = rng.uniform()
        vw2 = rng.uniform()

        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 1), vx1 * vy1 * vw1)
        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 0), vx2 * vy2 * vw2)

    @pytest.mark.skip(reason="Optimization temporarily disabled")
    def test_pushout3(self):
        x1 = scalar("x1")
        y1 = scalar("x2")
        y2 = scalar("y2")
        c = iscalar("c")
        two = np.asarray(2, dtype=pytensor.config.floatX)
        x, y = ifelse(c, (x1, y1), (two, y2), name="f1")
        o3 = np.asarray(0.3, dtype=pytensor.config.floatX)
        o2 = np.asarray(0.2, dtype=pytensor.config.floatX)
        z = ifelse(c, o3, o2, name="f2")
        out = x * z * y

        f = function([x1, y1, y2, c], out, allow_input_downcast=True)
        assert isinstance(f.maker.fgraph.toposort()[-1].op, IfElse)
        rng = np.random.default_rng(utt.fetch_seed())
        vx1 = rng.uniform()
        vy1 = rng.uniform()
        vy2 = rng.uniform()

        assert np.allclose(f(vx1, vy1, vy2, 1), vx1 * vy1 * 0.3)
        assert np.allclose(f(vx1, vy1, vy2, 0), 2 * vy2 * 0.2)

    @pytest.mark.skip(reason="Optimization temporarily disabled")
    def test_pushout2(self):
        x1 = scalar("x1")
        x2 = scalar("x2")
        y1 = scalar("y1")
        y2 = scalar("y2")
        w1 = scalar("w1")
        w2 = scalar("w2")
        c = iscalar("c")
        x, y = ifelse(c, (x1, y1), (x2, y2), name="f1")
        z = ifelse(x > y, w1, w2, name="f2")
        out = x * z * y

        f = function([x1, x2, y1, y2, w1, w2, c], out, allow_input_downcast=True)
        assert isinstance(f.maker.fgraph.toposort()[-1].op, IfElse)
        rng = np.random.default_rng(utt.fetch_seed())
        vx1 = rng.uniform()
        vx2 = rng.uniform()
        vy1 = rng.uniform()
        vy2 = rng.uniform()
        vw1 = rng.uniform()
        vw2 = rng.uniform()
        if vx1 > vy1:
            vw = vw1
        else:
            vw = vw2
        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 1), vx1 * vy1 * vw)

        if vx2 > vy2:
            vw = vw1
        else:
            vw = vw2
        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 0), vx2 * vy2 * vw)

    @pytest.mark.skip(reason="Optimization temporarily disabled")
    def test_merge_ifs_true_false(self):
        x1 = scalar("x1")
        x2 = scalar("x2")
        y1 = scalar("y1")
        y2 = scalar("y2")
        w1 = scalar("w1")
        w2 = scalar("w2")
        c = iscalar("c")

        out = ifelse(
            c,
            ifelse(c, x1, x2) + ifelse(c, y1, y2) + w1,
            ifelse(c, x1, x2) + ifelse(c, y1, y2) + w2,
        )
        f = function([x1, x2, y1, y2, w1, w2, c], out, allow_input_downcast=True)
        assert (
            len([x for x in f.maker.fgraph.toposort() if isinstance(x.op, IfElse)]) == 1
        )

        rng = np.random.default_rng(utt.fetch_seed())
        vx1 = rng.uniform()
        vx2 = rng.uniform()
        vy1 = rng.uniform()
        vy2 = rng.uniform()
        vw1 = rng.uniform()
        vw2 = rng.uniform()
        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 1), vx1 + vy1 + vw1)
        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 0), vx2 + vy2 + vw2)
