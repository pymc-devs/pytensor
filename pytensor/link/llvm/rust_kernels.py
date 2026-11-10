"""Precompiled Rust kernels for nodes the interpreter runs directly.

`rust_kernel(op, node, in_layouts)` returns a `pytensor_llvm.interp.RustKernel` (function
address + config) or None when the node has no Rust kernel; `in_layouts` are the
inputs' layouts (layout.py; None where unknown). A kernel declares how it produces its
outputs and in which layouts (`Kernel.outputs`, in Rust next to the kernel; the
counterpart of `llvm_outputs` for generated code): `rust_kernel_outputs` reads that.
"""

from __future__ import annotations

from functools import singledispatch

import numpy as np

from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.link.llvm.codegen.function import OutputMode, Output
from pytensor.link.llvm.codegen.types import repr_of
from pytensor.link.llvm.layout import Layout
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import Dot
from pytensor.tensor.shape import SpecifyShape


def interp():
    from pytensor_llvm import interp

    return interp


def fresh_mode(out):
    """New memory for `out`: provided for a static shape, else allocated by the kernel."""
    m = interp().OutputMode
    return m.Provided if all(s is not None for s in out.type.shape) else m.Allocated


@singledispatch
def rust_kernel(op, node, in_layouts=None):
    """A Rust kernel for `node`, or None. The kernel chooses how its outputs are
    produced and declares it (`Kernel.outputs`, next to the kernel in Rust)."""
    return None


def rust_kernel_outputs(kernel, node, in_layouts) -> list[Output]:
    """What `kernel` declares about its outputs, as the `Output`s of a signature."""
    from pytensor.link.llvm.codegen.function import known_layouts

    ins = [None if lay is None else lay.to_spec() for lay in known_layouts(node, in_layouts)]
    decls = kernel.outputs(ins, [o.type.ndim for o in node.outputs])
    modes = {"provided": OutputMode.PROVIDED, "allocated": OutputMode.ALLOCATED, "view": OutputMode.VIEW}
    outs = []
    for o, (mode, view_of, spec) in zip(node.outputs, decls, strict=True):
        shape = tuple(None if s is None else int(s) for s in o.type.shape)
        outs.append(Output(repr_of(o.type, Layout.from_spec(spec, shape)), modes[mode], view_of))
    return outs


@rust_kernel.register(DimShuffle)
def _(op, node, in_layouts=None):
    return interp().DimShuffle([None if o == "x" else int(o) for o in op.new_order])


@rust_kernel.register(ViewOp)
@rust_kernel.register(SpecifyShape)
def _(op, node, in_layouts=None):
    from pytensor.link.llvm.codegen.types import is_absent

    if any(not is_absent(i.type) for i in node.inputs[1:]):
        return None  # lengths to check: generated code
    return interp().DimShuffle(list(range(node.outputs[0].type.ndim)))


@rust_kernel.register(DeepCopyOp)
def _(op, node, in_layouts=None):
    out = node.outputs[0]
    return interp().Copy(fresh_mode(out), np.dtype(out.type.dtype))


@rust_kernel.register(Dot)
def _(op, node, in_layouts=None):
    a, b = node.inputs
    out = node.outputs[0]
    if not (1 <= a.type.ndim <= 2 and 1 <= b.type.ndim <= 2):
        return None
    if not (a.type.dtype == b.type.dtype == out.type.dtype in ("float64", "float32")):
        return None
    return interp().Dot(fresh_mode(out), np.dtype(out.type.dtype))


def _float_dtype(node):
    """The output's dtype if all inputs have it and it is a float, else None."""
    out = node.outputs[0]
    if out.type.dtype not in ("float64", "float32") or any(i.type.dtype != out.type.dtype for i in node.inputs):
        return None
    return np.dtype(out.type.dtype)


def _register_more():
    from pytensor.scan.op import Scan
    from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
    from pytensor.tensor.linalg.solvers.triangular import SolveTriangular
    from pytensor.tensor.sort import SortOp

    @rust_kernel.register(SortOp)
    def _(op, node, in_layouts=None):
        x = node.inputs[0]
        if x.type.ndim == 0 or x.type.dtype not in ("float64", "float32", "int64"):
            return None
        axis = int(op.axis) % x.type.ndim
        return interp().Sort(fresh_mode(node.outputs[0]), np.dtype(x.type.dtype), axis, op.kind in ("mergesort", "stable"))

    @rust_kernel.register(Cholesky)
    def _(op, node, in_layouts=None):
        dt = _float_dtype(node)
        if dt is None or node.inputs[0].type.ndim != 2:
            return None
        return interp().Cholesky(fresh_mode(node.outputs[0]), dt, bool(op.lower))

    @rust_kernel.register(SolveTriangular)
    def _(op, node, in_layouts=None):
        dt = _float_dtype(node)
        if dt is None or node.inputs[0].type.ndim != 2 or node.inputs[1].type.ndim not in (1, 2):
            return None
        return interp().SolveTriangular(fresh_mode(node.outputs[0]), dt, bool(op.lower), bool(op.unit_diagonal))

    @rust_kernel.register(Scan)
    def _(op, node, in_layouts=None):
        import dataclasses

        from pytensor.link.llvm.program import Planner, current_options

        info = op.info
        # the body's inputs aren't graph inputs: their layouts come from the kernel
        planner = Planner(dataclasses.replace(current_options(), input_layout="any"))
        # the loop body returns every output allocated; static ones use memory the
        # Scan kernel provides
        body_modes = [
            OutputMode.PROVIDED if all(s is not None for s in getattr(o.type, "shape", (None,))) else OutputMode.ALLOCATED
            for o in op.fgraph.outputs
        ]
        body = planner.plan(op.fgraph, output_modes=body_modes, input_layouts=_body_input_layouts(op, node, in_layouts))

        def taps(slices):
            return [[int(t) for t in s] for s in slices]

        return interp().Scan(
            info.n_seqs,
            taps(info.mit_mot_in_slices),
            taps(info.mit_mot_out_slices),
            taps(info.mit_sot_in_slices),
            taps(info.sit_sot_in_slices),
            info.n_untraced_sit_sot,
            info.n_nit_sot,
            info.n_non_seqs,
            bool(info.as_while),
            body,
            [np.dtype(i.type.dtype) for i in node.inputs],
            [np.dtype(o.type.dtype) for o in op.fgraph.outputs],
        )


def _body_input_layouts(op, node, in_layouts):
    """What the Scan kernel passes the loop body: rows of the sequences, rows of its
    own C-contiguous state buffers, whole untraced buffers (C-contiguous) and the
    non-sequences as they are."""
    from pytensor.link.llvm.layout import Layout, remove_axis, shape_of, itemsize_of

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
