import sys
from collections.abc import Iterable, Iterator, Sequence
from typing import (
    FrozenSet,
    Protocol,
    SupportsIndex,
    Union,
    cast,
    overload,
    runtime_checkable,
)


@runtime_checkable
class DimLike(Protocol):
    """Most basic signature of a dimension."""

    def __str__(self) -> str:
        ...

    def __hash__(self) -> int:
        ...


class Dim(DimLike):
    """The most common type of dimension."""

    _name: str

    def __init__(self, name: str) -> None:
        self._name = name
        super().__init__()

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"Dim('{self._name}')"

    def __eq__(self, __value: object) -> bool:
        return self._name == str(__value)

    def __hash__(self) -> int:
        return self._name.__hash__()


class BaseSpace(FrozenSet[DimLike]):
    """The most generic type of space is an unordered frozen set of dimensions.

    It implements broadcasting rules for the following tensor operators:
    * Addition → Unordered union
    * Subtraction → Unordered union
    * Multiplication → Unordered union
    * Power → Identity

    The logic operators (AND &, OR |, XOR ^) do space math with the frozenset.
    """

    def __add__(self, other: Iterable[DimLike]) -> "BaseSpace":
        try:
            other = Space(other)
            return BaseSpace({*self, *other})
        except Exception as ex:
            raise TypeError(f"Can't {other} to space.") from ex

    def __sub__(self, other: Iterable[DimLike]) -> "BaseSpace":
        try:
            other = Space(other)
            return BaseSpace({*self, *other})
        except Exception as ex:
            raise TypeError(f"Can't subtract {other} from space.") from ex

    def __mul__(self, other) -> "BaseSpace":
        try:
            other = Space(other)
            return BaseSpace({*self, *other})
        except Exception as ex:
            raise TypeError(f"Can't multiply space by {other}.") from ex

    def __truediv__(self, other) -> "BaseSpace":
        try:
            other = Space(other)
            return BaseSpace({*self, *other})
        except Exception as ex:
            raise TypeError(f"Can't divide space by {other}.") from ex

    def __pow__(self, other: Iterable[DimLike]) -> "BaseSpace":
        return self

    def __and__(self, other: Iterable[DimLike]) -> "BaseSpace":
        try:
            other = Space(other)
            return BaseSpace(set(self) & other)
        except Exception as ex:
            raise TypeError(f"Can't AND space with {other}.") from ex

    def __or__(self, other: Iterable[DimLike]) -> "BaseSpace":
        try:
            other = Space(other)
            return BaseSpace(set(self) | other)
        except Exception as ex:
            raise TypeError(f"Can't OR space with {other}.") from ex

    def __xor__(self, other: Iterable[DimLike]) -> "BaseSpace":
        try:
            other = Space(other)
            return BaseSpace(set(self) ^ other)
        except Exception as ex:
            raise TypeError(f"Can't XOR space with {other}.") from ex

    def __repr__(self) -> str:
        return "Space{" + ", ".join(f"'{d}'" for d in self) + "}"


class OrderedSpace(BaseSpace, Sequence[DimLike]):
    """A very tidied-up space, with a known order of dimensions."""

    def __init__(self, dims: Sequence[DimLike]) -> None:
        self._order = tuple(dims)
        super().__init__()

    def __eq__(self, __value) -> bool:
        if not isinstance(__value, Sequence) or isinstance(__value, str):
            return False
        return self._order == tuple(__value)

    def __ne__(self, __value) -> bool:
        if not isinstance(__value, Sequence) or isinstance(__value, str):
            return True
        return self._order != tuple(__value)

    def __iter__(self) -> Iterator[DimLike]:
        yield from self._order

    def __reversed__(self) -> Iterator[DimLike]:
        yield from reversed(self._order)

    def index(
        self,
        __value: DimLike,
        __start: SupportsIndex = 0,
        __stop: SupportsIndex = sys.maxsize,
    ) -> int:
        return self._order.index(__value, __start, __stop)

    @overload
    def __getitem__(self, __key: SupportsIndex) -> DimLike:
        """Indexing gives a dim"""

    @overload
    def __getitem__(self, __key: slice) -> "OrderedSpace":
        """Slicing preserves order"""

    def __getitem__(self, __key) -> Union[DimLike, "OrderedSpace"]:
        if isinstance(__key, slice):
            return OrderedSpace(self._order[__key])
        return cast(DimLike, self._order[__key])

    def __repr__(self) -> str:
        return "OrderedSpace(" + ", ".join(f"'{d}'" for d in self) + ")"


@overload
def Space(dims: Sequence[DimLike]) -> OrderedSpace:
    """Sequences of dims give an ordered space."""
    ...


@overload
def Space(dims: Iterable[DimLike]) -> BaseSpace:
    ...


def Space(dims: Iterable[DimLike]) -> Union[OrderedSpace, BaseSpace]:
    if isinstance(dims, Sequence):
        return OrderedSpace(dims)
    return BaseSpace(dims)
