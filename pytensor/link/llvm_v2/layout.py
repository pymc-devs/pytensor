"""Static layout facts about arrays (DESIGN.md, section 1a).

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
* views get theirs from their input's (`llvm_view_layout`, per op);
* graph inputs and values from numba fall-backs are DYNAMIC unless checked.

`infer_output_layouts` is the one place that decides; the planner and the
`LayoutFeature` (for rewrites and inspection) both use it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from functools import singledispatch

import numpy as np


class Kind(enum.Enum):
    PACKED = "P"
    DYNAMIC = "?"

    def __repr__(self):
        return self.value


PACKED = Kind.PACKED
DYNAMIC = Kind.DYNAMIC

Stride = int | Kind


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
    def dynamic(ndim: int) -> Layout:
        """Nothing known."""
        return Layout(tuple(range(ndim)), (DYNAMIC,) * ndim)

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
        block = [a for a in reversed(order) if strides[a] is PACKED and not _is_one(shape[a])]
        ones = [a for a in range(nd) if _is_one(shape[a])]
        rest = sorted(a for a in range(nd) if a not in block and a not in ones)
        order = [*rest, *reversed(block)]
        # length-1 axes go where they keep the order ascending (C stays the identity)
        for a in ones:
            pos = next((i for i, b in enumerate(order) if b > a), len(order))
            order.insert(pos, a)
        return Layout(tuple(order), tuple(strides))

    # ------------------------------------------------------------------ queries

    @property
    def ndim(self) -> int:
        return len(self.strides)

    def packed_block(self, shape) -> list[int]:
        """The packed axes (not of static length 1), innermost first."""
        return [a for a in reversed(self.order) if self.strides[a] is PACKED and not _is_one(shape[a])]

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

    def static_stride(self, a, shape, itemsize) -> int | None:
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
            if shape[b] is None:
                return None
            n *= int(shape[b])
        return n  # length-1 axis

    def stride_values(self, b, itemsize: int, shape_vals, passed: dict):
        """i64 stride values: constants, products of lengths (packed) or `passed[axis]`."""
        from pytensor.link.llvm_v2.codegen.emit import mul
        from pytensor.link.llvm_v2.codegen.types import ir

        m = ir()
        out = [None] * self.ndim
        acc = b.const_int(m.i64, itemsize)
        for a in reversed(self.order):
            s = self.strides[a]
            if s is PACKED:
                out[a] = acc
                acc = mul(b, acc, shape_vals[a])
            elif s is DYNAMIC:
                out[a] = passed[a]
            else:
                out[a] = b.const_int(m.i64, s)
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
                if isinstance(s, int) and not isinstance(s, Kind) and strides[a] != s:
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
            if isinstance(s, int) and not isinstance(s, Kind) and not _is_one(shape[a]):
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
        strides = []
        for a in range(self.ndim):
            if a in common or _is_one(shape[a]):
                strides.append(PACKED)
                continue
            s1, s2 = self.static_stride(a, shape, itemsize), other.static_stride(a, shape, itemsize)
            strides.append(s1 if s1 is not None and s1 == s2 else DYNAMIC)
        rest = [a for a in range(self.ndim) if a not in common]
        return Layout.make(shape, (*rest, *reversed(common)), strides)

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
                parts.append(f"P{block.index(a)}" if s is PACKED else ("?" if s is DYNAMIC else str(s)))
        return "[" + ", ".join(parts) + "]"

    def __str__(self):
        if self.ndim == 0:
            return "C"
        if all(s is PACKED for s in self.strides) and list(self.order) == sorted(self.order):
            return "C"
        block = [a for a in reversed(self.order) if self.strides[a] is PACKED]
        parts = []
        for a in range(self.ndim):
            s = self.strides[a]
            parts.append(f"P{block.index(a)}" if s is PACKED else ("?" if s is DYNAMIC else str(s)))
        return "[" + ", ".join(parts) + "]"

    __repr__ = __str__


def shape_of(var) -> tuple:
    return tuple(None if s is None else int(s) for s in var.type.shape)


def itemsize_of(var) -> int:
    return np.dtype(var.type.dtype).itemsize


def demote(layout: Layout, axes, shape, itemsize) -> list:
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


# ------------------------------------------------------------------ view rules


@singledispatch
def llvm_view_layout(op, node, in_layouts) -> list[Layout | None]:
    """Layouts of the outputs of `node` that are views of an input (None for the
    others). The default knows nothing about a view."""
    return [Layout.dynamic(o.type.ndim) if hasattr(o.type, "ndim") else None for o in node.outputs]


def _register_rules():
    from pytensor.compile.ops import ViewOp
    from pytensor.graph.basic import Constant
    from pytensor.raise_op import CheckAndRaise
    from pytensor.tensor.basic import Split
    from pytensor.tensor.elemwise import DimShuffle
    from pytensor.tensor.shape import Reshape, SpecifyShape
    from pytensor.tensor.subtensor import Subtensor, unflatten_index_variables

    @llvm_view_layout.register(ViewOp)
    @llvm_view_layout.register(SpecifyShape)
    @llvm_view_layout.register(CheckAndRaise)
    def _(op, node, in_layouts):
        x = in_layouts[0]
        return [Layout.make(shape_of(node.outputs[0]), x.order, x.strides)]

    @llvm_view_layout.register(Reshape)
    def _(op, node, in_layouts):
        # a view of a C-contiguous input, else a C-contiguous copy
        return [Layout.c(node.outputs[0].type.ndim)]

    @llvm_view_layout.register(DimShuffle)
    def _(op, node, in_layouts):
        x = in_layouts[0]
        new_order = op.new_order
        out_of = {}  # input axis -> output axis
        strides = []
        for j, o in enumerate(new_order):
            if o == "x":
                strides.append(PACKED)
            else:
                out_of[int(o)] = j
                strides.append(x.strides[int(o)])
        order = [j for j, o in enumerate(new_order) if o == "x"] + [out_of[a] for a in x.order if a in out_of]
        return [Layout.make(shape_of(node.outputs[0]), order, strides)]

    def static_int(var):
        if var is None:
            return None
        if isinstance(var, Constant):
            return int(np.asarray(var.data))
        return "dynamic"

    @llvm_view_layout.register(Subtensor)
    def _(op, node, in_layouts):
        x = in_layouts[0]
        xv = node.inputs[0]
        shape, isz = shape_of(xv), itemsize_of(xv)
        out_shape = shape_of(node.outputs[0])
        entries = unflatten_index_variables(node.inputs[1:], op.idx_list)
        demoted, removed, stepped = set(), [], {}
        for a, e in enumerate(entries):
            if not isinstance(e, slice):
                # the axis disappears: packed axes outside it now skip its length
                removed.append(a)
                demoted.update(outer_in_block(x, a, shape))
                continue
            step = static_int(e.step)
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
        for a, st in stepped.items():
            strides[a] = st
        keep = [a for a in range(x.ndim) if a not in removed]
        renumber = {a: k for k, a in enumerate(keep)}
        order = [renumber[a] for a in x.order if a in renumber]
        return [Layout.make(out_shape, order, [strides[a] for a in keep])]

    @llvm_view_layout.register(Split)
    def _(op, node, in_layouts):
        x = in_layouts[0]
        xv = node.inputs[0]
        shape, isz = shape_of(xv), itemsize_of(xv)
        a = int(op.axis) % x.ndim
        outs = []
        for o in node.outputs:
            os = shape_of(o)
            if os[a] is not None and os[a] == shape[a]:
                outs.append(Layout.make(os, x.order, x.strides))
            else:
                outs.append(Layout.make(os, x.order, demote(x, outer_in_block(x, a, shape), shape, isz)))
        return outs


_register_rules()


# ------------------------------------------------------------------ inference


def infer_output_layouts(node, modes, in_layouts, provided=None, allocated=None) -> list[Layout | None]:
    """Layouts of `node`'s outputs given how they are produced.

    * `modes`: `OutMode` per output (PROVIDED, ALLOCATED, VIEW, SCALAR);
    * `in_layouts`: per input (None for non-tensors);
    * `provided`: the layout of the memory provided per output (default C-contiguous,
      what the planner provides);
    * `allocated`: the layout of ALLOCATED outputs: C-contiguous for generated code,
      Rust kernels and Scan (the default), DYNAMIC for numba fall-backs.
    """
    from pytensor.link.llvm_v2.codegen.function import OutMode

    views = None
    out = []
    for k, (o, mode) in enumerate(zip(node.outputs, modes, strict=True)):
        if mode is OutMode.SCALAR or not hasattr(o.type, "ndim"):
            out.append(None)
        elif mode is OutMode.PROVIDED:
            lay = provided[k] if provided is not None else None
            out.append(lay if lay is not None else Layout.c(o.type.ndim))
        elif mode is OutMode.ALLOCATED:
            out.append(Layout.c(o.type.ndim) if allocated is None else allocated)
        else:
            if views is None:
                in_layouts = [
                    lay if lay is not None or not hasattr(i.type, "ndim") else Layout.dynamic(i.type.ndim)
                    for i, lay in zip(node.inputs, in_layouts, strict=True)
                ]
                views = llvm_view_layout(node.op, node, in_layouts)
            out.append(views[k] if views[k] is not None else Layout.dynamic(o.type.ndim))
    return out


def tensor_layouts(variables, default) -> list[Layout | None]:
    return [default(v.type.ndim) if hasattr(v.type, "ndim") else None for v in variables]




# ------------------------------------------------------------------ graph analysis


def predict_layouts(node, in_layouts) -> list[Layout | None]:
    """The layouts the planner will (most likely) give `node`'s outputs.

    The planner is authoritative (it knows which steps fall back to numba, for one);
    this is its prediction for graph rewrites and inspection."""
    from pytensor.compile.builders import OpFromGraph, UnitKind
    from pytensor.link.llvm_v2.codegen.function import llvm_output_modes

    op = node.op
    try:
        if isinstance(op, OpFromGraph):
            if op.unit is UnitKind.COMPILED:
                return [Layout.c(o.type.ndim) if hasattr(o.type, "ndim") else None for o in node.outputs]
            inner = infer_layouts(op.fgraph, in_layouts)
            outs = [inner[o] for o in op.fgraph.outputs]
            if op.unit is UnitKind.INTERPRETED:
                # an allocated output of a nested program: its value or a C copy
                outs = [
                    None if lay is None else lay.join(Layout.c(o.type.ndim), shape_of(o), itemsize_of(o))
                    for lay, o in zip(outs, node.outputs, strict=True)
                ]
            return outs
        modes = [m for m, _ in llvm_output_modes(op, node)]
        return infer_output_layouts(node, modes, in_layouts)
    except NotImplementedError:
        return [Layout.dynamic(o.type.ndim) if hasattr(o.type, "ndim") else None for o in node.outputs]


def _default_layout(var, inputs_c=False):
    from pytensor.graph.basic import Constant

    if not hasattr(var.type, "ndim"):
        return None
    if isinstance(var, Constant) or inputs_c:
        return Layout.c(var.type.ndim)
    return Layout.dynamic(var.type.ndim)


def infer_layouts(fgraph, input_layouts=None) -> dict:
    """{variable: Layout} for every tensor variable of `fgraph` (predicted, see
    `predict_layouts`); `input_layouts` per graph input (default: nothing known)."""
    lays = {}
    for k, v in enumerate(fgraph.inputs):
        lay = input_layouts[k] if input_layouts is not None else None
        lays[v] = lay if lay is not None else _default_layout(v)
    for node in fgraph.toposort():
        ins = [lays[i] if i in lays else _default_layout(i) for i in node.inputs]
        for o, lay in zip(node.outputs, predict_layouts(node, ins), strict=True):
            lays[o] = lay
    return lays


def _feature_base():
    from pytensor.graph.features import Feature

    return Feature


class LayoutFeature(_feature_base()):
    """Keeps the (predicted) layout of every variable of a FunctionGraph, like the
    DestroyHandler keeps its view and destroy maps: `fgraph.llvm_layout(var)`.

    Layouts are computed lazily and cached. Replacing an input of a node invalidates
    everything downstream: a replacement can have a weaker layout, and then facts
    derived from the old one would be wrong, not just stale.
    """

    def __init__(self, inputs_c=False):
        self.inputs_c = inputs_c
        self.cache = {}

    def on_attach(self, fgraph):
        if hasattr(fgraph, "llvm_layout"):
            from pytensor.graph.features import AlreadyThere

            raise AlreadyThere("a LayoutFeature is already attached")
        self.fgraph = fgraph
        fgraph.llvm_layout = self.layout

    def on_detach(self, fgraph):
        del fgraph.llvm_layout
        self.cache.clear()

    def on_import(self, fgraph, node, reason):
        pass  # computed when asked

    def on_prune(self, fgraph, node, reason):
        for o in node.outputs:
            self.cache.pop(o, None)

    def on_change_input(self, fgraph, node, i, var, new_var, reason=None):
        todo = list(node.outputs)
        while todo:
            v = todo.pop()
            if self.cache.pop(v, None) is None:
                continue  # never computed: neither is anything downstream
            for client, _ in fgraph.clients.get(v, ()):
                todo.extend(client.outputs)

    def layout(self, var) -> Layout | None:
        """The predicted layout of `var`."""
        if var in self.cache:
            return self.cache[var]
        stack = [var]
        while stack:
            v = stack[-1]
            if v in self.cache:
                stack.pop()
                continue
            node = v.owner
            if node is None:
                self.cache[v] = _default_layout(v, self.inputs_c and v in self.fgraph.inputs)
                stack.pop()
                continue
            missing = [i for i in node.inputs if i not in self.cache]
            if missing:
                stack.extend(missing)
                continue
            outs = predict_layouts(node, [self.cache[i] for i in node.inputs])
            for o, lay in zip(node.outputs, outs, strict=True):
                self.cache[o] = lay
            stack.pop()
        return self.cache[var]


__all__ = [
    "DYNAMIC",
    "PACKED",
    "Layout",
    "LayoutFeature",
    "infer_layouts",
    "infer_output_layouts",
    "llvm_view_layout",
    "predict_layouts",
]
