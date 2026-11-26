"""Static layout facts about arrays.

A `Layout` says what is known, before the program runs, about the byte strides of an
array value. Every axis is

* `PACKED`: its stride is the itemsize times the lengths of the packed axes inside it
  in memory order (`order`, outermost first);
* `Static(n)`, a plain int: its stride is `n` bytes (0 for a broadcast axis);
* `DYNAMIC`: only known at run time (the stride is an argument).

Non-packed axes don't take part in the product: a broadcast axis (stride 0) inside an
otherwise contiguous array keeps the others packed. Axes whose length is statically 1
don't matter for addressing: they are always PACKED. C-contiguous is "all PACKED in the
identity order" (`Layout.c(ndim)`); numpy's other contiguity rules hold as well
(length-1 axes and empty arrays are ignored when checking).

Layouts are claims the generated code relies on: a value claimed PACKED is read without
its strides. Where the layout of each value of a graph comes from is in
analysis/layout.py.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from pytensor.tensor.type import TensorType


class StrideKind(enum.Enum):
    PACKED = "P"
    DYNAMIC = "?"

    def __repr__(self):
        return self.value


PACKED = StrideKind.PACKED
DYNAMIC = StrideKind.DYNAMIC

Stride = int | StrideKind


def _is_one(n) -> bool:
    return n is not None and int(n) == 1


@dataclass(frozen=True)
class Layout:
    #: all axes, outermost first; canonical: the non-packed axes in ascending order,
    #: then the block of packed axes in their memory order, with length-1 axes inserted
    #: where they keep the order ascending
    order: tuple[int, ...]
    #: per axis
    strides: tuple[Stride, ...]

    # ------------------------------------------------------------------ construction

    @staticmethod
    def c(ndim: int) -> Layout:
        """C-contiguous."""
        return Layout(tuple(range(ndim)), (PACKED,) * ndim)

    @staticmethod
    def of_const(data) -> Layout:
        """How a constant is stored: packed in its own order if it has one, else C."""
        a, nd = np.asarray(data), np.ndim(data)
        order = sorted(range(nd), key=lambda i: -a.strides[i])
        if nd < 2 or a.flags.c_contiguous or not a.transpose(order).flags.c_contiguous:
            return Layout.c(nd)
        return Layout.make(a.shape, order, (PACKED,) * nd)

    @staticmethod
    def dynamic(ndim: int) -> Layout:
        """Nothing known."""
        return Layout(tuple(range(ndim)), (DYNAMIC,) * ndim)

    @staticmethod
    def unknown(shape) -> Layout:
        """Nothing known but what the static lengths `shape` (None where dynamic) fix:
        `dynamic`, canonical for `shape`."""
        nd = len(shape)
        return Layout.make(shape, range(nd), (DYNAMIC,) * nd)

    @staticmethod
    def make(shape, order, strides) -> Layout:
        """The canonical layout for static lengths `shape` (None where dynamic):
        length-1 axes become PACKED; only the order of the packed axes matters."""
        nd = len(shape)
        order = list(order)
        assert sorted(order) == list(range(nd)) and len(strides) == nd
        strides = list(strides)
        for a in range(nd):
            if _is_one(shape[a]):
                strides[a] = PACKED
        # packed axes of nonzero length, innermost first
        block = [
            a for a in reversed(order) if strides[a] is PACKED and not _is_one(shape[a])
        ]
        ones = [a for a in range(nd) if _is_one(shape[a])]
        rest = sorted(a for a in range(nd) if a not in block and a not in ones)
        order = [*rest, *reversed(block)]
        # length-1 axes go where they keep the order ascending (C stays the identity)
        for a in ones:
            pos = next((i for i, b in enumerate(order) if b > a), len(order))
            order.insert(pos, a)
        return Layout(tuple(order), tuple(strides))

    def for_shape(self, shape) -> Layout:
        """This layout, canonical for the static lengths `shape` (`make`)."""
        return Layout.make(shape, self.order, self.strides)

    # ------------------------------------------------------------------ queries

    @property
    def ndim(self) -> int:
        return len(self.strides)

    def packed_block(self, shape) -> list[int]:
        """The packed axes (not of static length 1), innermost first."""
        return [
            a
            for a in reversed(self.order)
            if self.strides[a] is PACKED and not _is_one(shape[a])
        ]

    def fully_packed(self, shape) -> bool:
        """Contiguous in some axis order (`order`)?"""
        return all(s is PACKED for s in self.strides)

    def is_c(self, shape) -> bool:
        """C-contiguous?"""
        if any(s is not PACKED for s in self.strides):
            return False
        real = [a for a in self.order if not _is_one(shape[a])]
        return real == sorted(real)

    @property
    def dynamic_axes(self) -> tuple[int, ...]:
        """Axes whose stride is passed at run time."""
        return tuple(a for a in range(self.ndim) if self.strides[a] is DYNAMIC)

    def static_stride(
        self, a: int, shape: Sequence[int | None], itemsize: int
    ) -> int | None:
        """Axis `a`'s stride if it is known statically."""
        s = self.strides[a]
        if isinstance(s, int):
            return s
        if s is DYNAMIC:
            return None
        n = itemsize
        for b in self.packed_block(shape):
            if b == a:
                return n
            length = shape[b]
            if length is None:
                return None
            n *= int(length)
        return n  # length-1 axis

    def outer_in_block(self, axis: int, shape) -> list[int]:
        """The packed axes outside `axis` in the packed block, innermost first. Empty
        if `axis` isn't in the block."""
        block = self.packed_block(shape)
        if axis not in block:
            return []
        return block[block.index(axis) + 1 :]

    def demoted_strides(self, axes, shape, itemsize: int) -> list[Stride]:
        """The strides, with the packed `axes` turned into their static value, or
        DYNAMIC where that isn't known. Used when an axis inside them changes length,
        so that they no longer follow from the lengths of the axes inside them."""
        strides = list(self.strides)
        for a in axes:
            if strides[a] is PACKED and not _is_one(shape[a]):
                s = self.static_stride(a, shape, itemsize)
                strides[a] = DYNAMIC if s is None else s
        return strides

    def remove_axis(self, axis: int, shape, itemsize: int) -> Layout:
        """The layout of `x[..., i, ...]`, an integer index on `axis`, for an array `x`
        with this layout and `shape`."""
        strides = self.demoted_strides(
            self.outer_in_block(axis, shape), shape, itemsize
        )
        keep = [a for a in range(self.ndim) if a != axis]
        renumber = {a: k for k, a in enumerate(keep)}
        order = [renumber[a] for a in self.order if a in renumber]
        return Layout.make([shape[a] for a in keep], order, [strides[a] for a in keep])

    def satisfied_by(self, shape, strides, itemsize) -> bool:
        """Do the run-time `shape` and byte `strides` of an array satisfy the layout?"""
        if any(n == 0 for n in shape):
            return True
        acc = itemsize
        for a in reversed(self.order):
            s = self.strides[a]
            if shape[a] != 1:
                if s is PACKED and strides[a] != acc:
                    return False
                if (
                    isinstance(s, int)
                    and not isinstance(s, StrideKind)
                    and strides[a] != s
                ):
                    return False
            if s is PACKED:
                acc *= shape[a]
        return True

    def implies(self, other: Layout, shape, itemsize) -> bool:
        """Does every array with this layout have layout `other`?"""
        if self == other:
            return True
        mine = self.packed_block(shape)
        theirs = other.packed_block(shape)
        if mine[: len(theirs)] != theirs:
            return False
        for a in range(self.ndim):
            s = other.strides[a]
            if (
                isinstance(s, int)
                and not isinstance(s, StrideKind)
                and not _is_one(shape[a])
            ):
                if self.static_stride(a, shape, itemsize) != s:
                    return False
        return True

    def join(self, other: Layout, shape, itemsize) -> Layout:
        """The most specific layout both imply (for a value that has one or the other)."""
        if self == other:
            return self
        mine, theirs = self.packed_block(shape), other.packed_block(shape)
        common = []
        for x, y in zip(mine, theirs):
            if x != y:
                break
            common.append(x)
        strides: list[Stride] = []
        for a in range(self.ndim):
            if a in common or _is_one(shape[a]):
                strides.append(PACKED)
                continue
            s1, s2 = (
                self.static_stride(a, shape, itemsize),
                other.static_stride(a, shape, itemsize),
            )
            strides.append(s1 if s1 is not None and s1 == s2 else DYNAMIC)
        rest = [a for a in range(self.ndim) if a not in common]
        return Layout.make(shape, (*rest, *reversed(common)), strides)

    def to_spec(self):
        """`(order, strides)` with strides "P", "?" or bytes: how Rust exchanges layouts."""
        return (
            list(self.order),
            [
                int(s) if not isinstance(s, StrideKind) else "P" if s is PACKED else "?"
                for s in self.strides
            ],
        )

    @staticmethod
    def from_spec(spec, shape) -> Layout:
        order, strides = spec
        return Layout.make(
            shape,
            order,
            [PACKED if s == "P" else DYNAMIC if s == "?" else int(s) for s in strides],
        )

    def describe(self, shape) -> str:
        """Like str(), with static length-1 axes shown as `1`."""
        if self.is_c(shape):
            return "C"
        block = self.packed_block(shape)
        parts = []
        for a in range(self.ndim):
            s = self.strides[a]
            if _is_one(shape[a]):
                parts.append("1")
            else:
                parts.append(
                    f"P{block.index(a)}"
                    if s is PACKED
                    else ("?" if s is DYNAMIC else str(s))
                )
        return "[" + ", ".join(parts) + "]"

    def __str__(self):
        if self.ndim == 0:
            return "C"
        if all(s is PACKED for s in self.strides) and list(self.order) == sorted(
            self.order
        ):
            return "C"
        block = [a for a in reversed(self.order) if self.strides[a] is PACKED]
        parts = []
        for a in range(self.ndim):
            s = self.strides[a]
            parts.append(
                f"P{block.index(a)}"
                if s is PACKED
                else ("?" if s is DYNAMIC else str(s))
            )
        return "[" + ", ".join(parts) + "]"

    __repr__ = __str__


def common_order(layouts, shape) -> list[int] | None:
    """The memory order (outermost first) most of `layouts` are packed in over the axes
    of `shape` that aren't statically 1, if one covers them all (else None)."""
    real = [a for a in range(len(shape)) if not _is_one(shape[a])]
    votes: dict[tuple[int, ...], int] = {}
    for layout in layouts:
        if layout is None or layout.ndim != len(shape):
            continue
        block = tuple(a for a in layout.packed_block(shape) if a in real)
        if block:
            votes[block] = votes.get(block, 0) + len(block)
    if not votes:
        return None
    block = max(votes, key=votes.__getitem__)
    if sorted(block) != real:
        return None
    rest = [a for a in range(len(shape)) if a not in block]
    return rest + list(reversed(block))


def new_memory_layout(declared: Layout | None, t: TensorType) -> Layout:
    """`declared` if new memory can have it: packed (allocated in it), or implied by
    C-contiguous memory; else DYNAMIC. The layout of a value that may be new memory or
    not (an output of a compiled unit, an output that may overwrite an input)."""
    if declared is None:
        return Layout.dynamic(t.ndim)
    shape = list(t.shape)
    if declared.fully_packed(shape) or Layout.c(t.ndim).implies(
        declared, shape, np.dtype(t.dtype).itemsize
    ):
        return declared
    return Layout.dynamic(t.ndim)


__all__ = [
    "DYNAMIC",
    "PACKED",
    "Layout",
    "Stride",
    "StrideKind",
    "common_order",
    "new_memory_layout",
]
