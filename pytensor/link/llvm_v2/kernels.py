"""Precompiled Rust kernels for nodes the interpreter runs directly.

`llvm_kernel(op, node, modes, in_layouts)` returns a `pytensor_llvm.rt.Kernel` (function
address + environment) or None when the node has no Rust kernel. `modes` are the output
modes the planner chose (`OutMode` per output); a kernel that can't produce them returns
None. `in_layouts` are the inputs' layouts (layout.py; None where unknown).

Kernels allocate C-contiguous outputs (layout.py relies on it).
"""

from __future__ import annotations

from functools import singledispatch

import numpy as np

from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.link.llvm_v2.codegen.function import OutMode, Output, llvm_outputs
from pytensor.link.llvm_v2.codegen.types import repr_of
from pytensor.link.llvm_v2.layout import Layout
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import Dot
from pytensor.tensor.shape import SpecifyShape


def rt():
    from pytensor_llvm import rt

    return rt


def rt_mode(mode: OutMode):
    m = rt().OutputMode
    return {OutMode.PROVIDED: m.Provided, OutMode.ALLOCATED: m.Allocated, OutMode.VIEW: m.View}[mode]


@singledispatch
def llvm_kernel(op, node, modes, in_layouts=None):
    return None


@llvm_kernel.register(DimShuffle)
def _(op, node, modes, in_layouts=None):
    if modes[0] is not OutMode.VIEW:
        return None
    return rt().DimShuffle([None if o == "x" else int(o) for o in op.new_order])


@llvm_kernel.register(ViewOp)
@llvm_kernel.register(SpecifyShape)
def _(op, node, modes, in_layouts=None):
    from pytensor.link.llvm_v2.codegen.types import is_absent

    if modes[0] is not OutMode.VIEW:
        return None
    if any(not is_absent(i.type) for i in node.inputs[1:]):
        return None  # lengths to check: generated code
    return rt().DimShuffle(list(range(node.outputs[0].type.ndim)))


@llvm_kernel.register(DeepCopyOp)
def _(op, node, modes, in_layouts=None):
    if modes[0] not in (OutMode.PROVIDED, OutMode.ALLOCATED):
        return None
    return rt().Copy(rt_mode(modes[0]), np.dtype(node.outputs[0].type.dtype))


@llvm_kernel.register(Dot)
def _(op, node, modes, in_layouts=None):
    a, b = node.inputs
    out = node.outputs[0]
    if not (1 <= a.type.ndim <= 2 and 1 <= b.type.ndim <= 2):
        return None
    if not (a.type.dtype == b.type.dtype == out.type.dtype in ("float64", "float32")):
        return None
    if modes[0] not in (OutMode.PROVIDED, OutMode.ALLOCATED):
        return None
    return rt().Dot(rt_mode(modes[0]), np.dtype(out.type.dtype))


def _float_mode(node, modes):
    """The output's dtype if the kernel can produce mode `modes[0]` with all inputs of the
    output's float dtype, else None."""
    out = node.outputs[0]
    if modes[0] not in (OutMode.PROVIDED, OutMode.ALLOCATED):
        return None
    if out.type.dtype not in ("float64", "float32") or any(i.type.dtype != out.type.dtype for i in node.inputs):
        return None
    return np.dtype(out.type.dtype)


def _register_more():
    from pytensor.scan.op import Scan
    from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
    from pytensor.tensor.linalg.solvers.triangular import SolveTriangular
    from pytensor.tensor.sort import SortOp

    @llvm_kernel.register(SortOp)
    def _(op, node, modes, in_layouts=None):
        x = node.inputs[0]
        if x.type.ndim == 0 or x.type.dtype not in ("float64", "float32", "int64"):
            return None
        if modes[0] not in (OutMode.PROVIDED, OutMode.ALLOCATED):
            return None
        axis = int(op.axis) % x.type.ndim
        return rt().Sort(rt_mode(modes[0]), np.dtype(x.type.dtype), axis, op.kind in ("mergesort", "stable"))

    @llvm_kernel.register(Cholesky)
    def _(op, node, modes, in_layouts=None):
        dt = _float_mode(node, modes)
        if dt is None or node.inputs[0].type.ndim != 2:
            return None
        return rt().Cholesky(rt_mode(modes[0]), dt, bool(op.lower))

    @llvm_kernel.register(SolveTriangular)
    def _(op, node, modes, in_layouts=None):
        dt = _float_mode(node, modes)
        if dt is None or node.inputs[0].type.ndim != 2 or node.inputs[1].type.ndim not in (1, 2):
            return None
        return rt().SolveTriangular(rt_mode(modes[0]), dt, bool(op.lower), bool(op.unit_diagonal))

    @llvm_outputs.register(Scan)
    def _(op, node, in_layouts):
        # the Scan kernel allocates every output, C-contiguous
        return [Output(repr_of(o.type, Layout.c(o.type.ndim)), OutMode.ALLOCATED) for o in node.outputs]

    @llvm_kernel.register(Scan)
    def _(op, node, modes, in_layouts=None):
        import dataclasses

        from pytensor.link.llvm_v2.program import Planner, current_options

        info = op.info
        # the step's inputs aren't graph inputs: their layouts come from the kernel
        planner = Planner(dataclasses.replace(current_options(), input_layout="any"))
        # the step program returns every output allocated; static ones use memory the
        # Scan kernel provides
        step_modes = [
            OutMode.PROVIDED if all(s is not None for s in getattr(o.type, "shape", (None,))) else OutMode.ALLOCATED
            for o in op.fgraph.outputs
        ]
        step = planner.plan(op.fgraph, output_modes=step_modes, input_layouts=_step_input_layouts(op, node, in_layouts))

        def taps(slices):
            return [[int(t) for t in s] for s in slices]

        return rt().Scan(
            info.n_seqs,
            taps(info.mit_mot_in_slices),
            taps(info.mit_mot_out_slices),
            taps(info.mit_sot_in_slices),
            taps(info.sit_sot_in_slices),
            info.n_untraced_sit_sot,
            info.n_nit_sot,
            info.n_non_seqs,
            bool(info.as_while),
            step,
            [np.dtype(i.type.dtype) for i in node.inputs],
            [np.dtype(o.type.dtype) for o in op.fgraph.outputs],
        )


def _step_input_layouts(op, node, in_layouts):
    """What the Scan kernel passes the step program: rows of the sequences, rows of its
    own C-contiguous state buffers, whole untraced buffers (C-contiguous) and the
    non-sequences as they are."""
    from pytensor.link.llvm_v2.layout import Layout, remove_axis, shape_of, itemsize_of

    info = op.info
    if in_layouts is None:
        in_layouts = [None] * len(node.inputs)
    inner = op.fgraph.inputs
    lays = []
    for j in range(info.n_seqs):
        outer, lay = node.inputs[1 + j], in_layouts[1 + j]
        lays.append(None if lay is None else remove_axis(lay, 0, shape_of(outer), itemsize_of(outer)))
    n_taps = (
        sum(len(t) for t in info.mit_mot_in_slices)
        + sum(len(t) for t in info.mit_sot_in_slices)
        + len(info.sit_sot_in_slices)
        + info.n_untraced_sit_sot
    )
    for k in range(n_taps):
        lays.append(Layout.c(inner[len(lays)].type.ndim))
    first_ns = len(node.inputs) - info.n_non_seqs
    lays += list(in_layouts[first_ns:])
    if len(lays) != len(inner):
        return None
    return lays


_register_more()
