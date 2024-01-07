from typing import Sequence

import pytest

import pytensor.xtensor.spaces as xsp


class TestDims:
    def test_str_is_dimlike(self):
        assert isinstance("d", xsp.DimLike)

    def test_dim(self):
        d0 = xsp.Dim("d0")
        assert isinstance(d0, xsp.DimLike)
        assert str(d0) == "d0"
        assert "d0" in d0.__repr__()
        # Dims can compare with strings
        assert d0 == "d0"
        # They must be hashable to be used as keys
        assert isinstance(hash(d0), int)


class TestBaseSpace:
    def test_type(self):
        assert issubclass(xsp.BaseSpace, frozenset)
        s1 = xsp.BaseSpace({"d0", "d1"})
        assert "Space" in s1.__repr__()
        assert "d0" in s1.__repr__()
        assert "d1" in s1.__repr__()
        # Spaces are frozensets which makes them convenient to use
        assert isinstance(s1, frozenset)
        # But they can't be sets, because .add(newdim) would mess up things
        assert not isinstance(s1, set)
        assert "d0" in s1
        assert "d1" in s1
        assert len(s1) == 2
        assert s1 == {"d1", "d0"}
        # Can't index an unordered space
        assert not hasattr(s1, "index")
        with pytest.raises(TypeError, match="not subscriptable"):
            s1[1]

    def test_spacemath(self):
        assert xsp.BaseSpace("ab") == {"a", "b"}
        # Set logic operations result in spaces
        union = xsp.BaseSpace("ab") | {"b", "c"}
        assert isinstance(union, xsp.BaseSpace)
        assert union == {"a", "b", "c"}

        intersection = xsp.BaseSpace("ab") & {"b", "c"}
        assert isinstance(intersection, xsp.BaseSpace)
        assert intersection == {"b"}

        xor = xsp.BaseSpace("ab") ^ {"b", "c"}
        assert isinstance(xor, xsp.BaseSpace)
        assert xor == {"a", "c"}

    def test_tensormath(self):
        # Tensors and spaces follow the same basic math rules
        addition = xsp.BaseSpace("ab") + {"c"}
        assert isinstance(addition, xsp.BaseSpace)
        assert addition == {"a", "b", "c"}

        subtraction = xsp.BaseSpace("ab") - {"b", "c"}
        assert isinstance(subtraction, xsp.BaseSpace)
        assert subtraction == {"a", "b", "c"}

        multiplication = xsp.BaseSpace("ab") * {"c"}
        assert isinstance(multiplication, xsp.BaseSpace)
        assert multiplication == {"a", "b", "c"}

        division = xsp.BaseSpace("ab") / {"b", "c"}
        assert isinstance(division, xsp.BaseSpace)
        assert division == {"a", "b", "c"}

        power = xsp.BaseSpace("ba") ** 3
        assert isinstance(power, xsp.BaseSpace)
        assert power == {"a", "b"}


class TestOrderedSpace:
    def test_type(self):
        o1 = xsp.OrderedSpace(["b", "a"])
        assert o1.__repr__() == "OrderedSpace('b', 'a')"
        assert isinstance(o1, Sequence)
        assert len(o1) == 2
        # Addition/multiplication is different compare to tuples
        assert not isinstance(o1, tuple)
        # And lists would be mutable, but ordered spaces are not
        assert not isinstance(o1, list)

    def test_comparison(self):
        # Ordered spaces can only be equal to other ordered things
        assert xsp.OrderedSpace("a") != {"a"}
        assert xsp.OrderedSpace("a") == ("a",)
        assert xsp.OrderedSpace("a") == ["a"]
        assert xsp.OrderedSpace("a") == xsp.OrderedSpace("a")
        # Except for strings, because they could be a dim
        assert not xsp.OrderedSpace("a") == "a"
        assert xsp.OrderedSpace("a") != "a"

    def test_indexing(self):
        b = xsp.Dim("b")
        o1 = xsp.OrderedSpace([b, "a"])
        # Ordered spaces can be indexed
        assert o1.index("b") == 0
        assert o1[0] is b
        sliced = o1[::-1]
        assert isinstance(sliced, xsp.OrderedSpace)
        assert sliced == ("a", "b")


def test_space_function():
    usp = xsp.Space({"a", "b"})
    assert isinstance(usp, xsp.BaseSpace)
    assert not isinstance(usp, xsp.OrderedSpace)

    assert isinstance(xsp.Space(["a", "b"]), xsp.OrderedSpace)
    assert isinstance(xsp.Space(("a", "b")), xsp.OrderedSpace)
    assert isinstance(xsp.Space("ab"), xsp.OrderedSpace)
