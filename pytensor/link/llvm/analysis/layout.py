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
its strides. They come from how a value is produced:

* memory the planner provides (arena, fresh outputs) and buffers that generated code,
  Rust kernels and Scan allocate are C-contiguous;
* views get theirs from their input's (as the op declares, with the rules below);
* numba kernels allocate their outputs packed, in the order asked for (default C);
* graph inputs are DYNAMIC unless checked.

The layout of each output follows from what its op says: `llvm_layout`
(codegen/interface.py), and its `view_map`. It is binding: the
`LayoutFeature` of the graph holds the layout of every value (`node_outputs`), and the
planner makes whatever runs a node give its outputs those layouts.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pytensor.graph.basic import Constant, Variable
from pytensor.graph.features import AlreadyThere, Feature
from pytensor.graph.rewriting.basic import GraphRewriter


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

    def stride_values(self, b, itemsize: int, shape_vals, passed: dict):
        """i64 stride values: constants, products of lengths (packed) or `passed[axis]`."""

        out = [None] * self.ndim
        acc = b.i64(itemsize)
        for a in reversed(self.order):
            s = self.strides[a]
            if s is PACKED:
                out[a] = acc
                acc = b.mul(acc, shape_vals[a], nsw=True)
            elif s is DYNAMIC:
                out[a] = passed[a]
            else:
                out[a] = b.i64(s)
        return out

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
                "P" if s is PACKED else "?" if s is DYNAMIC else int(s)
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


def shape_of(var) -> tuple:
    return tuple(None if s is None else int(s) for s in var.type.shape)


def itemsize_of(var) -> int:
    dtype: np.dtype = np.dtype(var.type.dtype)
    return dtype.itemsize


def demote(layout: Layout, axes, shape, itemsize) -> list[Stride]:
    """Strides with packed `axes` turned into their static value or DYNAMIC."""
    strides = list(layout.strides)
    for a in axes:
        if strides[a] is PACKED and not _is_one(shape[a]):
            s = layout.static_stride(a, shape, itemsize)
            strides[a] = DYNAMIC if s is None else s
    return strides


def outer_in_block(layout: Layout, axis, shape) -> list[int]:
    """Packed axes outside `axis` in the packed block (empty if `axis` isn't in it)."""
    block = layout.packed_block(shape)
    if axis not in block:
        return []
    return block[block.index(axis) + 1 :]


def remove_axis(layout: Layout, axis, shape, itemsize) -> Layout:
    """The layout of `x[..., i, ...]` (integer index on `axis`) for `x` with `layout`."""
    strides = demote(layout, outer_in_block(layout, axis, shape), shape, itemsize)
    keep = [a for a in range(layout.ndim) if a != axis]
    renumber = {a: k for k, a in enumerate(keep)}
    order = [renumber[a] for a in layout.order if a in renumber]
    return Layout.make([shape[a] for a in keep], order, [strides[a] for a in keep])


# ------------------------------------------------------------------ layouts of views
# (used by the ops' `llvm_layout`, next to their emitters)


def same_layout(node, x: Layout) -> Layout:
    """A view that is its input (ViewOp, SpecifyShape, ...): the input's layout, for the
    output's (possibly more static) shape."""
    return x.for_shape(shape_of(node.outputs[0]))


def dimshuffle_layout(op, node, x: Layout) -> Layout:
    out_of = {}  # input axis -> output axis
    strides: list[Stride] = []
    for j, entry in enumerate(op.new_order):
        if entry == "x":
            strides.append(PACKED)
        else:
            out_of[int(entry)] = j
            strides.append(x.strides[int(entry)])
    order = [j for j, entry in enumerate(op.new_order) if entry == "x"] + [
        out_of[a] for a in x.order if a in out_of
    ]
    return Layout.make(shape_of(node.outputs[0]), order, strides)


def _static_int(var):
    from pytensor.graph.basic import Constant

    if var is None:
        return None
    if isinstance(var, Constant):
        return int(np.asarray(var.data))
    return "dynamic"


def subtensor_layout(op, node, x: Layout) -> Layout:
    from pytensor.tensor.subtensor import unflatten_index_variables

    xv = node.inputs[0]
    shape, isz = shape_of(xv), itemsize_of(xv)
    out_shape = shape_of(node.outputs[0])
    entries = unflatten_index_variables(node.inputs[1:], op.idx_list)
    demoted, removed = set(), []
    stepped: dict[int, Stride] = {}
    for a, e in enumerate(entries):
        if not isinstance(e, slice):
            # the axis disappears: packed axes outside it now skip its length
            removed.append(a)
            demoted.update(outer_in_block(x, a, shape))
            continue
        step = _static_int(e.step)
        if e.start is None and e.stop is None and step in (None, 1):
            continue
        n_out = out_shape[a - len(removed)]
        if not (shape[a] is not None and n_out == shape[a]):
            demoted.update(outer_in_block(x, a, shape))
        if step not in (None, 1):
            st = x.static_stride(a, shape, isz)
            stepped[a] = DYNAMIC if step == "dynamic" or st is None else st * step
            demoted.update(outer_in_block(x, a, shape))
    strides = demote(x, demoted, shape, isz)
    for a, stride in stepped.items():
        strides[a] = stride
    keep = [a for a in range(x.ndim) if a not in removed]
    renumber = {a: k for k, a in enumerate(keep)}
    order = [renumber[a] for a in x.order if a in renumber]
    return Layout.make(out_shape, order, [strides[a] for a in keep])


def split_layouts(op, node, x: Layout) -> list[Layout]:
    xv = node.inputs[0]
    shape, isz = shape_of(xv), itemsize_of(xv)
    a = int(op.axis) % x.ndim
    outs = []
    for var in node.outputs:
        os = shape_of(var)
        if os[a] is not None and os[a] == shape[a]:
            outs.append(x.for_shape(os))
        else:
            outs.append(
                Layout.make(
                    os, x.order, demote(x, outer_in_block(x, a, shape), shape, isz)
                )
            )
    return outs


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


# ------------------------------------------------------------------ graph analysis


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


def _is_tensor(var) -> bool:
    from pytensor.link.llvm.codegen.types import is_absent

    return hasattr(var.type, "ndim") and not is_absent(var.type)


def _default_layout(var, inputs_c=False):
    if not _is_tensor(var):
        return None
    # (a sparse constant's data isn't an ndarray: the backend rejects it later)
    if isinstance(var, Constant) and isinstance(var.data, np.ndarray):
        return Layout.of_const(var.data)
    if inputs_c:
        return Layout.c(var.type.ndim)
    return Layout.dynamic(var.type.ndim)


def infer_layouts(fgraph, input_layouts=None) -> dict:
    """{variable: Layout} for the variables of `fgraph` (`node_outputs`; None for
    those that aren't tensors). `input_layouts`: per graph input (default: nothing
    known)."""
    return graph_layouts(fgraph.inputs, fgraph.outputs, input_layouts)


def graph_layouts(inputs, outputs, input_layouts=None, inlined=False) -> dict:
    """Like `infer_layouts`, for the graph from the variables `inputs` to `outputs`.

    With `inlined`, every OpFromGraph is treated as inlined, which is what generated
    code does with them: its outputs get the layouts its inner graph gives them, even
    if its `execution` says it is compiled on its own. (Except a FusedElemwise
    generated as one loop, a node of its own: `fusable`.)"""
    from pytensor.compile.builders import OpFromGraph
    from pytensor.graph.traversal import io_toposort
    from pytensor.link.llvm.codegen.signature import node_outputs
    from pytensor.link.llvm.dispatch.fused import fusable

    layouts = {}
    for k, v in enumerate(inputs):
        layout = input_layouts[k] if input_layouts is not None else None
        layouts[v] = layout if layout is not None else _default_layout(v)
    for node in io_toposort(inputs, outputs):
        ins = [layouts[i] if i in layouts else _default_layout(i) for i in node.inputs]
        if inlined and isinstance(node.op, OpFromGraph) and not fusable(node.op, node):
            inner = node.op.fgraph
            inner_lays = graph_layouts(inner.inputs, inner.outputs, ins, inlined=True)
            layouts.update(
                zip(node.outputs, (inner_lays[var] for var in inner.outputs))
            )
            continue
        outs = node_outputs(node, ins)
        layouts.update(
            (v, var.layout) for v, var in zip(node.outputs, outs, strict=True)
        )
    for var in outputs:
        if var not in layouts:
            layouts[var] = _default_layout(var)
    return layouts


class LayoutFeature(Feature):
    """The layout of every variable of a FunctionGraph, as the LLVM backend gives it:
    `fgraph.llvm_layout(var)` (None for values that aren't tensors).

    This is where the planner gets the layouts of the graph's values from: whatever
    runs a node gives its outputs these layouts (`node_outputs`), and code is
    generated for them. `input_layout`: what is known about the graph's inputs, "any"
    strides or "c" (the linker's option of that name).

    Layouts are computed lazily and cached. Replacing an input of a node invalidates
    everything downstream: a replacement can have a weaker layout, and then facts
    derived from the old one would be wrong, not just stale.
    """

    provides = ("llvm_layout",)

    cache: dict[Variable, Layout | None]

    def __init__(self, input_layout="any"):
        self.input_layout = input_layout
        self.cache = {}

    def clone(self):
        return type(self)(self.input_layout)

    def on_attach(self, fgraph):
        if "llvm_layout" in fgraph._feature_methods:
            raise AlreadyThere("a LayoutFeature is already attached")

    def on_detach(self, fgraph):
        self.cache.clear()

    def on_prune(self, fgraph, node, reason):
        for var in node.outputs:
            self.cache.pop(var, None)

    def on_change_input(self, fgraph, node, i, var, new_var, reason=None):
        todo = list(node.outputs)
        while todo:
            v = todo.pop()
            if self.cache.pop(v, None) is None:
                continue  # never computed: neither is anything downstream
            for client, _ in fgraph.clients.get(v, ()):
                todo.extend(client.outputs)

    def set_input_layout(self, input_layout):
        if input_layout != self.input_layout:
            self.input_layout = input_layout
            self.cache.clear()

    def llvm_layout(self, fgraph, var) -> Layout | None:
        """The layout of `var`."""
        from pytensor.link.llvm.codegen.signature import node_outputs

        cache = self.cache
        if var in cache:
            return cache[var]
        stack = [var]
        while stack:
            v = stack[-1]
            if v in cache:
                stack.pop()
                continue
            node = v.owner
            if node is None:
                cache[v] = _default_layout(
                    v, self.input_layout == "c" and v in fgraph.inputs
                )
                stack.pop()
                continue
            missing = [i for i in node.inputs if i not in cache]
            if missing:
                stack.extend(missing)
                continue
            outs = node_outputs(node, [cache[i] for i in node.inputs])
            cache.update(
                (v, output.layout) for v, output in zip(node.outputs, outs, strict=True)
            )
            stack.pop()
        return cache[var]


def layout_feature(fgraph) -> LayoutFeature:
    """The `LayoutFeature` of `fgraph`, attached if it has none."""
    feature: LayoutFeature | None = fgraph._feature_methods.get("llvm_layout")
    if feature is None:
        feature = LayoutFeature()
        fgraph.attach_feature(feature)
    return feature


class AttachLayoutFeature(GraphRewriter):
    """Attaches a `LayoutFeature` to the graph, for the rewrites after it (the LLVM
    linker requires it)."""

    def add_requirements(self, fgraph):
        layout_feature(fgraph)

    def apply(self, fgraph):
        pass


__all__ = [
    "DYNAMIC",
    "PACKED",
    "AttachLayoutFeature",
    "Layout",
    "LayoutFeature",
    "common_order",
    "dimshuffle_layout",
    "graph_layouts",
    "infer_layouts",
    "layout_feature",
    "new_memory_layout",
    "same_layout",
    "split_layouts",
    "subtensor_layout",
]
