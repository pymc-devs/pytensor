from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.reachability import (
    ancestor_bitsets,
    greedy_independent_subset,
    update_ancestors_after_contraction,
    update_ancestors_after_contraction_bits,
)
from tests.graph.utils import MyVariable, op1, op2, op3, op4, op5


def _bitsets(inputs, outputs):
    # clone=False keeps the original Apply nodes so tests can reference them.
    fgraph = FunctionGraph(inputs, outputs, clone=False)
    return ancestor_bitsets(fgraph)


def _is_ancestor(ancestors, bitflags, a, b):
    """Whether ``a`` is a (transitive) ancestor of ``b``."""
    return bool(ancestors[b] & bitflags[a])


def _independent(ancestors, bitflags, a, b):
    """Whether neither node is an ancestor of the other."""
    return not (ancestors[a] & bitflags[b] or ancestors[b] & bitflags[a])


class TestAncestorBitsets:
    def test_reachability(self):
        # x0 -> A -> C <- B <- x1 ;  A -> D
        x0 = MyVariable("x0")
        x1 = MyVariable("x1")
        a = op1(x0)
        b = op2(x1)
        c = op3(a, b)
        d = op4(a)
        ancestors, bitflags = _bitsets([x0, x1], [c, d])
        A, B, C, D = a.owner, b.owner, c.owner, d.owner

        assert _is_ancestor(ancestors, bitflags, A, C)
        assert _is_ancestor(ancestors, bitflags, B, C)
        assert _is_ancestor(ancestors, bitflags, A, D)
        assert not _is_ancestor(ancestors, bitflags, B, D)
        assert not _is_ancestor(ancestors, bitflags, C, A)
        assert _is_ancestor(ancestors, bitflags, A, A)  # own ancestor

        # Siblings and diamond-tips are pairwise independent; a node and its
        # ancestor are not.
        assert _independent(ancestors, bitflags, A, B)
        assert _independent(ancestors, bitflags, C, D)
        assert not _independent(ancestors, bitflags, A, C)
        assert not _independent(ancestors, bitflags, A, D)

    def test_root_and_none_handling(self):
        x = MyVariable("x")
        a = op1(x)
        ancestors, bitflags = _bitsets([x], [a])
        # Root variables have `None` owner -> bitset 0; never an ancestor.
        assert bitflags[None] == 0
        assert ancestors[None] == 0
        assert not _is_ancestor(ancestors, bitflags, None, a.owner)


class TestGreedyIndependentSubset:
    def test_basic(self):
        x0 = MyVariable("x0")
        x1 = MyVariable("x1")
        a = op1(x0)
        b = op2(x1)
        c = op3(a, b)
        d = op4(a)
        ancestors, bitflags = _bitsets([x0, x1], [c, d])
        A, B, C, D = a.owner, b.owner, c.owner, d.owner

        # A full antichain is kept entirely.
        assert greedy_independent_subset(ancestors, bitflags, [A, B]) == [A, B]
        assert greedy_independent_subset(ancestors, bitflags, [C, D]) == [C, D]
        # A dependent node is dropped; input order is preserved (first kept).
        assert greedy_independent_subset(ancestors, bitflags, [A, C]) == [A]
        assert greedy_independent_subset(ancestors, bitflags, [C, A]) == [C]
        assert greedy_independent_subset(ancestors, bitflags, [A, B, C, D]) == [A, B]


class TestUpdateAncestorsAfterContraction:
    def test_2221_shape(self):
        # Mirrors issue #2221: two independent bundles A={aout, ain} and
        # B={bfwd, bsm} where ain consumes bfwd and bsm consumes aout. Merging A
        # links bfwd and bsm (previously independent), so they must no longer be
        # mergeable together.
        x = MyVariable("x")
        bfwd = op1(x)
        ain = op2(bfwd)
        aout = op3(x)
        bsm = op4(aout)
        ancestors, bitflags = _bitsets([x], [ain, bsm])
        AOUT, AIN, BFWD, BSM = aout.owner, ain.owner, bfwd.owner, bsm.owner

        # Each bundle is independent to start.
        assert _independent(ancestors, bitflags, AOUT, AIN)
        assert _independent(ancestors, bitflags, BFWD, BSM)

        # Contract bundle A = {aout, ain}.
        update_ancestors_after_contraction(ancestors, bitflags, [AOUT, AIN])

        # bsm now (transitively) depends on bfwd through the merged A node.
        assert _is_ancestor(ancestors, bitflags, BFWD, BSM)
        assert not _independent(ancestors, bitflags, BFWD, BSM)
        assert greedy_independent_subset(ancestors, bitflags, [BFWD, BSM]) == [BFWD]

    def test_only_touches_dependents(self):
        # A node that does not depend on the contracted group is unaffected.
        x = MyVariable("x")
        a = op1(x)
        b = op2(x)
        c = op3(a)  # depends on a only
        unrelated = op5(x)  # depends on x only
        ancestors, bitflags = _bitsets([x], [b, c, unrelated])
        A, B, C, U = a.owner, b.owner, c.owner, unrelated.owner

        before = dict(ancestors)
        update_ancestors_after_contraction(ancestors, bitflags, [A])

        # C depended on A -> its ancestors may grow; B and U did not -> unchanged.
        assert ancestors[B] == before[B]
        assert ancestors[U] == before[U]
        assert ancestors[C] & bitflags[A]

    def test_bits_form_matches_node_form(self):
        # The `_bits` fast-path and the node-based form agree.
        x = MyVariable("x")
        a = op1(x)
        b = op2(a)  # depends on a
        c = op3(x)  # independent of a, b
        ancestors, bitflags = _bitsets([x], [b, c])
        A, B, C = a.owner, b.owner, c.owner

        member_bits = bitflags[A] | bitflags[B]
        combined = ancestors[A] | ancestors[B]
        update_ancestors_after_contraction_bits(ancestors, member_bits, combined)

        assert ancestors[B] & bitflags[A]  # B already depended on A
        assert not (ancestors[C] & member_bits)  # C is unrelated, untouched
