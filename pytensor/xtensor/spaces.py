from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterator, Sequence, Collection, Sized, Iterable, Container, Set, Reversible
import sys
from typing import FrozenSet, Protocol, Tuple, Union, Iterator, overload, SupportsIndex


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
    
    It implements the following calculation operators:
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


class OrderedSpace(BaseSpace, Reversible[DimLike]):
    """A very tidied-up space, with a known order of dimensions."""

    def __init__(self, dims: Sequence[DimLike]) -> None:
        self._order = tuple(dims)
        super().__init__()

    def __iter__(self) -> Iterator[DimLike]:
        for d in self._order:
            yield d

    def __reversed__(self) -> Iterator[DimLike]:
        for d in reversed(self._order):
            yield d

    def index(self, __value: DimLike, __start: SupportsIndex = 0, __stop: SupportsIndex = sys.maxsize) -> int:
        return self._order.index(__value, __start, __stop)
    
    @overload
    def __getitem__(self, __key: slice) -> "OrderedSpace":
        """Slicing an ordered space results in an ordered space."""
        return OrderedSpace(self._order[__key])

    @overload
    def __getitem__(self, __key: SupportsIndex) -> DimLike:
        return self._order[__key]

    def __getitem__(self, __key) -> DimLike:
        return self._order[__key]


    def __repr__(self) -> str:
        return "OrderedSpace(" + ", ".join(f"'{d}'" for d in self) + ")"


@overload
def Space(dims: Sequence[DimLike]) -> OrderedSpace:
    """Sequences of dims give an ordered space."""
    ...

@overload
def Space(dims: Set[DimLike]) -> BaseSpace:
    ...

def Space(dims: Iterable[DimLike]) -> Union[OrderedSpace, BaseSpace]:
    if isinstance(dims, Sequence):
        return OrderedSpace(dims)
    return BaseSpace(dims)
