from itertools import product

import numpy as np
import pytest

from pytensor import function
from pytensor import tensor as pt
from pytensor.compile.mode import get_mode
from pytensor.conditional.op import LazyIfElse, ifelse
from pytensor.raise_op import CheckAndRaise
from pytensor.tensor.subtensor import inc_subtensor
from pytensor.tensor.type import iscalar, vector


class IfElseCompatibilityTests:
    """Shared linker tests; subclasses supply a mode and optional check/gradient hooks."""

    mode = None
    check_aliasing = True

    @staticmethod
    def check(x, message):
        return CheckAndRaise(ValueError, message)(x, pt.all(x > 0))

    @staticmethod
    def assert_invalid(fn, *args, message):
        with pytest.raises(ValueError, match=message):
            fn(*args)

    @staticmethod
    def gradient(fn, argnums):
        return None

    def test_lazy_ifelse_nested(self):
        a, b, c = (pt.scalar(name, dtype="bool") for name in "abc")
        x = vector("x")
        checked = self.check(x, "selected invalid branch")
        # Both outer branches share checked, but either inner condition can avoid it.
        out = ifelse(a, ifelse(b, checked, x * 2), ifelse(c, checked, x * 3))
        fn = function([a, b, c, x], out, mode=self.mode)
        [node] = fn.maker.fgraph.toposort()
        assert isinstance(node.op, LazyIfElse)
        grad_fn = self.gradient(fn, argnums=3)

        value = np.array([-1.0, 2.0], dtype=x.dtype)
        for av, bv, cv in product((False, True), repeat=3):
            if bv if av else cv:
                self.assert_invalid(
                    fn, av, bv, cv, value, message="selected invalid branch"
                )
            else:
                scale = 2 if av else 3
                np.testing.assert_array_equal(fn(av, bv, cv, value), value * scale)
                if grad_fn is not None:
                    np.testing.assert_array_equal(
                        grad_fn(av, bv, cv, value), np.full_like(value, scale)
                    )
        np.testing.assert_array_equal(fn(True, True, False, -value[:1]), -value[:1])

    def test_lazy_ifelse_merge_siblings(self):
        cond = pt.scalar("cond", dtype="bool")
        x = vector("x")
        checked = self.check(x, "shared")
        left = ifelse(cond, checked + 1, x + 2)
        right = ifelse(cond, checked * 2, x + 3)
        # Addition needs both siblings, so their shared check stays in one true branch.
        fn = function([cond, x], left + right, mode=self.mode)
        [lazy] = [
            n.op for n in fn.maker.fgraph.toposort() if isinstance(n.op, LazyIfElse)
        ]
        assert len(lazy.output_types) == 2
        assert (
            sum(n.op == checked.owner_op for n in lazy.true_branch.fgraph.toposort())
            == 1
        )
        np.testing.assert_array_equal(fn(True, [1.0, 2.0]), [4.0, 7.0])
        np.testing.assert_array_equal(fn(False, [-1.0, -2.0]), [3.0, 1.0])
        grad_fn = self.gradient(fn, argnums=1)
        if grad_fn is not None:
            value = np.array([-1.0, -2.0], dtype=x.dtype)
            np.testing.assert_array_equal(grad_fn(False, value), [2.0, 2.0])

    def test_lazy_ifelse_dont_merge_exclusive_branches(self):
        outer, inner = (pt.scalar(name, dtype="bool") for name in ("outer", "inner"))
        x = vector("x")
        checked = self.check(x, "unselected sibling")
        left = ifelse(inner, x + 1, x * 2)
        right = ifelse(inner, checked, x * 3)
        # The outer condition selects one sibling. Merging them would run right's
        # check even when outer selects left and inner is true.
        fn = function([outer, inner, x], ifelse(outer, left, right), mode=self.mode)
        np.testing.assert_array_equal(fn(True, True, [-1.0]), [0.0])
        np.testing.assert_array_equal(fn(False, False, [-1.0]), [-3.0])
        self.assert_invalid(fn, False, True, [-1.0], message="unselected sibling")
        grad_fn = self.gradient(fn, argnums=2)
        if grad_fn is not None:
            value = np.array([-1.0], dtype=x.dtype)
            np.testing.assert_array_equal(grad_fn(True, True, value), [1.0])

    @pytest.mark.parametrize("external_use", [False, True])
    def test_lazy_ifelse_shared(self, external_use):
        cond = pt.scalar("cond", dtype="bool")
        x = vector("x")
        shared = self.check(x, "shared")
        true = [shared, pt.sin(shared)]
        identical = self.check(x, "shared")
        false = [identical, pt.cos(identical)] if not external_use else [x, pt.cos(x)]
        outputs = ifelse(cond, true, false)
        if external_use:
            # This output needs shared even when the false branch is selected.
            outputs.append(shared)
        fn = function([cond, x], outputs, mode=get_mode(self.mode).excluding("fusion"))
        nodes = fn.maker.fgraph.toposort()
        assert sum(n.op == shared.owner_op for n in nodes) == 1
        [lazy] = [n.op for n in nodes if isinstance(n.op, LazyIfElse)]
        for branch in (lazy.true_branch, lazy.false_branch):
            assert not any(n.op == shared.owner_op for n in branch.fgraph.toposort())

        value = np.array([1.0, 2.0], dtype=x.dtype)
        for condition in (False, True):
            result = fn(condition, value)
            np.testing.assert_array_equal(result[0], value)
            np.testing.assert_allclose(
                result[1], np.sin(value) if condition else np.cos(value)
            )
            if self.check_aliasing:
                assert not np.shares_memory(result[0], value)
                if external_use:
                    assert not np.shares_memory(result[0], result[2])

    @pytest.mark.parametrize("views", [False, True])
    def test_lazy_ifelse_output_aliases(self, views):
        self.check_output_aliases(self.mode, views, expect_inplace=self.check_aliasing)

    @staticmethod
    def check_output_aliases(mode, views, *, expect_inplace=True):
        c, x = iscalar("c"), vector("x")
        t, f = x + 1, x - 1
        true = [t[:2], t[1:]] if views else [t, t]
        false = [f[:2], f[1:]] if views else [f, f]
        left, right = ifelse(c, true, false)
        # These outputs share a fresh internal buffer. Updating left must not
        # leak into right through duplicate outputs or overlapping views.
        fn = function([c, x], [inc_subtensor(left[1], 100), right], mode=mode)
        if expect_inplace:
            assert any(n.op.destroy_map for n in fn.maker.fgraph.toposort())
        value = np.zeros(3, dtype=x.dtype)
        for condition in (False, True):
            result = fn(condition, value)
            expected = np.full(2 if views else 3, 1 if condition else -1)
            np.testing.assert_array_equal(result[1], expected)
            expected[1] += 100
            np.testing.assert_array_equal(result[0], expected)
