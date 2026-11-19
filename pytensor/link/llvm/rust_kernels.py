"""Precompiled Rust kernels: what generated code calls for nodes it doesn't generate.

`rust_kernel(op, node, in_layouts)` returns a `pytensor_llvm.interp.RustKernel` (function
address + config) or None when the node has no Rust kernel; `in_layouts` are the
inputs' layouts (layout.py; None where unknown). A kernel declares how it produces its
outputs and in which layouts (`Kernel.outputs`, in Rust next to the kernel; the
counterpart of `node_outputs` for generated code): `rust_kernel_outputs` reads that.
"""

from __future__ import annotations

from functools import singledispatch

import numpy as np

from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.link.llvm.codegen.function import Output, OutputMode
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

    ins = [
        None if lay is None else lay.to_spec()
        for lay in known_layouts(node, in_layouts)
    ]
    decls = kernel.outputs(ins, [o.type.ndim for o in node.outputs])
    outs = []
    for o, (mode, view_of, spec) in zip(node.outputs, decls, strict=True):
        shape = tuple(None if s is None else int(s) for s in o.type.shape)
        r = repr_of(o.type, Layout.from_spec(spec, shape))
        # (a kernel says what it does: it allocates, or returns a view of one input)
        if mode == "provided":
            outs.append(Output(r, OutputMode.PROVIDED))
        else:
            outs.append(
                Output(r, OutputMode.RETURNED, (view_of,) if mode == "view" else ())
            )
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
    return interp().Dot(fresh_mode(out), np.dtype(out.type.dtype), linalg_parallel())


def linalg_parallel():
    """May faer kernels use the thread pool (the linker's `linalg_parallel`)?"""
    from pytensor.link.llvm.program import current_options

    return current_options().linalg_parallel


def fresh_modes(node):
    return [fresh_mode(o) for o in node.outputs]


def _linalg_dtype(node, int_outputs=()):
    """The inputs' dtype if they all have the same float dtype, and so do the outputs
    (except those at `int_outputs`, which are int32), else None."""
    dt = node.inputs[0].type.dtype
    if dt not in ("float64", "float32") or any(i.type.dtype != dt for i in node.inputs):
        return None
    for k, o in enumerate(node.outputs):
        if o.type.dtype != ("int32" if k in int_outputs else dt):
            return None
    return np.dtype(dt)


def _float_dtype(node):
    """The output's dtype if all inputs have it and it is a float, else None."""
    out = node.outputs[0]
    if out.type.dtype not in ("float64", "float32") or any(
        i.type.dtype != out.type.dtype for i in node.inputs
    ):
        return None
    return np.dtype(out.type.dtype)


def _register_more():
    from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
    from pytensor.tensor.linalg.decomposition.eigen import Eigh, Eigvalsh
    from pytensor.tensor.linalg.decomposition.lu import (
        LU,
        LUFactor,
        PivotToPermutations,
    )
    from pytensor.tensor.linalg.decomposition.qr import QR
    from pytensor.tensor.linalg.decomposition.svd import SVD
    from pytensor.tensor.linalg.inverse import MatrixInverse, MatrixPinv
    from pytensor.tensor.linalg.solvers.general import Solve
    from pytensor.tensor.linalg.solvers.psd import CholeskySolve
    from pytensor.tensor.linalg.solvers.triangular import SolveTriangular
    from pytensor.tensor.linalg.summary import Det, SLogDet
    from pytensor.tensor.sort import SortOp

    @rust_kernel.register(SortOp)
    def _(op, node, in_layouts=None):
        x = node.inputs[0]
        if x.type.ndim == 0 or x.type.dtype not in ("float64", "float32", "int64"):
            return None
        axis = int(op.axis) % x.type.ndim
        return interp().Sort(
            fresh_mode(node.outputs[0]),
            np.dtype(x.type.dtype),
            axis,
            op.kind in ("mergesort", "stable"),
        )

    @rust_kernel.register(Cholesky)
    def _(op, node, in_layouts=None):
        dt = _float_dtype(node)
        if dt is None or node.inputs[0].type.ndim != 2:
            return None
        return interp().Cholesky(
            fresh_mode(node.outputs[0]), dt, bool(op.lower), linalg_parallel()
        )

    @rust_kernel.register(SolveTriangular)
    def _(op, node, in_layouts=None):
        dt = _float_dtype(node)
        if (
            dt is None
            or node.inputs[0].type.ndim != 2
            or node.inputs[1].type.ndim not in (1, 2)
        ):
            return None
        return interp().SolveTriangular(
            fresh_mode(node.outputs[0]),
            dt,
            bool(op.lower),
            bool(op.unit_diagonal),
            linalg_parallel(),
        )

    @rust_kernel.register(Solve)
    def _(op, node, in_layouts=None):
        # (real matrices: hermitian is symmetric)
        assume_a = {"gen": "gen", "sym": "sym", "her": "sym", "pos": "pos"}.get(
            op.assume_a
        )
        dt = _linalg_dtype(node)
        if dt is None or assume_a is None:
            return None
        return interp().Solve(
            fresh_mode(node.outputs[0]), dt, assume_a, bool(op.lower), linalg_parallel()
        )

    @rust_kernel.register(CholeskySolve)
    def _(op, node, in_layouts=None):
        dt = _linalg_dtype(node)
        if dt is None:
            return None
        return interp().CholeskySolve(
            fresh_mode(node.outputs[0]), dt, bool(op.lower), linalg_parallel()
        )

    @rust_kernel.register(MatrixInverse)
    def _(op, node, in_layouts=None):
        dt = _linalg_dtype(node)
        return (
            None
            if dt is None
            else interp().MatrixInverse(
                fresh_mode(node.outputs[0]), dt, linalg_parallel()
            )
        )

    @rust_kernel.register(Det)
    def _(op, node, in_layouts=None):
        dt = _linalg_dtype(node)
        return (
            None
            if dt is None
            else interp().Det(fresh_mode(node.outputs[0]), dt, linalg_parallel())
        )

    @rust_kernel.register(SLogDet)
    def _(op, node, in_layouts=None):
        dt = _linalg_dtype(node)
        return (
            None
            if dt is None
            else interp().SLogDet(fresh_modes(node), dt, linalg_parallel())
        )

    @rust_kernel.register(LUFactor)
    def _(op, node, in_layouts=None):
        dt = _linalg_dtype(node, int_outputs=(1,))
        return (
            None
            if dt is None
            else interp().LUFactor(fresh_modes(node), dt, linalg_parallel())
        )

    @rust_kernel.register(LU)
    def _(op, node, in_layouts=None):
        if op.permute_l:
            form, ints = "permute_l", ()
        elif op.p_indices:
            form, ints = "indices", (0,)
        else:
            form, ints = "matrix", ()
        dt = _linalg_dtype(node, int_outputs=ints)
        return (
            None
            if dt is None
            else interp().LU(fresh_modes(node), dt, form, linalg_parallel())
        )

    @rust_kernel.register(PivotToPermutations)
    def _(op, node, in_layouts=None):
        dt = node.inputs[0].type.dtype
        if dt not in ("int32", "int64") or node.outputs[0].type.dtype != "int64":
            return None
        return interp().PivotToPermutations(
            fresh_mode(node.outputs[0]), np.dtype(dt), bool(op.inverse)
        )

    @rust_kernel.register(Eigh)
    @rust_kernel.register(Eigvalsh)
    def _(op, node, in_layouts=None):
        dt = _linalg_dtype(node)
        if dt is None or len(node.inputs) != 1:
            return None  # the generalized problem (b): numba
        return interp().Eigh(
            fresh_modes(node),
            dt,
            bool(op.lower),
            isinstance(op, Eigh),
            linalg_parallel(),
        )

    @rust_kernel.register(SVD)
    def _(op, node, in_layouts=None):
        dt = _linalg_dtype(node)
        if dt is None:
            return None
        return interp().SVD(
            fresh_modes(node),
            dt,
            bool(op.full_matrices),
            bool(op.compute_uv),
            linalg_parallel(),
        )

    @rust_kernel.register(QR)
    def _(op, node, in_layouts=None):
        dt = _linalg_dtype(node)
        if dt is None or op.pivoting or op.mode not in ("full", "economic", "r"):
            return None
        return interp().QR(fresh_modes(node), dt, op.mode, linalg_parallel())

    @rust_kernel.register(MatrixPinv)
    def _(op, node, in_layouts=None):
        dt = _linalg_dtype(node)
        if dt is None:
            return None
        return interp().MatrixPinv(
            fresh_mode(node.outputs[0]), dt, bool(op.hermitian), linalg_parallel()
        )


_register_more()
