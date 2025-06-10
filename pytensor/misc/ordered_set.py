from collections.abc import Iterable, Iterator, MutableSet
from typing import Any


class OrderedSet(MutableSet):
    values: dict[Any, None]

    def __init__(self, iterable: Iterable | None = None) -> None:
        if iterable is None:
            self.values = {}
        else:
            self.values = dict.fromkeys(iterable)

    def __contains__(self, value) -> bool:
        return value in self.values

    def __iter__(self) -> Iterator:
        yield from self.values

    def __len__(self) -> int:
        return len(self.values)

    def add(self, value) -> None:
        self.values[value] = None

    def discard(self, value) -> None:
        if value in self.values:
            del self.values[value]

    def copy(self) -> "OrderedSet":
        return OrderedSet(self)

    def update(self, other: Iterable) -> None:
        for value in other:
            self.add(value)

    def union(self, other: Iterable) -> "OrderedSet":
        new_set = self.copy()
        new_set.update(other)
        return new_set

    def difference_update(self, other: Iterable) -> None:
        for value in other:
            self.discard(value)
