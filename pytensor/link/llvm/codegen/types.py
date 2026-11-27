"""The LLVM types of PyTensor values, and the kernel ABI.

The representations of values (types.py) say what is passed; this module says as which
LLVM types: `value_type` and `memory_type` for elements, `ir_type` for a whole
representation. For example, the `TensorRepr` of a float64 matrix with a dynamic first
length and C-contiguous strides is `{ptr, [1 x i64], [0 x i64]}`.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

import pytensor.scalar as ps
from pytensor.link.llvm.types import (
    SUPPORTED_TYPES,
    Representation,
    ScalarRepr,
    TensorRepr,
    element_type,
)
from pytensor.scalar.basic import ScalarType


class _IR:
    """`pytensor_llvm.ir`, imported when first used (the extension is optional, and
    PyTensor must import without it)."""

    def __getattr__(self, name: str) -> Any:
        from pytensor_llvm import ir as module

        value = getattr(module, name)
        setattr(self, name, value)
        return value


if TYPE_CHECKING:
    from pytensor_llvm import ir
else:
    #: the IR builder's module: `ir.i64`, `ir.Module(...)`, `ir.Libm.Erf`
    ir = _IR()


def dtype_code(t: ScalarType) -> int:
    """The code of `t` in the kernel ABI (`PtArray.dtype`)."""
    return SUPPORTED_TYPES.index(element_type(t))


def value_type(t: ScalarType) -> ir.Type:
    """The LLVM type of a value of type `t` in registers (bool is i1)."""
    return _value_types()[element_type(t)]


@cache
def _value_types() -> dict[ScalarType, ir.Type]:
    return {
        ps.float64: ir.f64,
        ps.float32: ir.f32,
        ps.int64: ir.i64,
        ps.int32: ir.i32,
        ps.int16: ir.i16,
        ps.int8: ir.i8,
        ps.uint8: ir.i8,
        ps.uint16: ir.i16,
        ps.uint32: ir.i32,
        ps.uint64: ir.i64,
        ps.bool: ir.i1,
    }


def value_bits(t: ScalarType) -> int:
    """The number of bits of a value of type `t` in registers (bool: 1)."""
    return 1 if t == ps.bool else 8 * np.dtype(t.dtype).itemsize


def memory_type(t: ScalarType) -> ir.Type:
    """The LLVM type of an element of type `t` in memory (bool is a byte)."""
    return ir.i8 if t == ps.bool else value_type(t)


def ir_type(representation: Representation) -> ir.Type:
    """The LLVM type of a value passed as `representation`: the scalar by value, `ptr`
    for a 0-d tensor, else `{ptr, [d x i64] lengths, [s x i64] strides}` with the
    dynamic lengths and the strides the layout doesn't fix. Values without data have
    no LLVM type."""
    match representation:
        case ScalarRepr(scalar_type):
            return value_type(scalar_type)
        case TensorRepr():
            return _tensor_ir_type(
                representation.ndim,
                len(representation.dynamic_axes),
                len(representation.stride_axes),
            )
    raise TypeError(f"no LLVM type for {representation}")


@cache
def _tensor_ir_type(ndim: int, n_lengths: int, n_strides: int) -> ir.Type:
    if ndim == 0:
        return ir.ptr
    return ir.Type.struct_(
        [ir.ptr, ir.Type.array(ir.i64, n_lengths), ir.Type.array(ir.i64, n_strides)]
    )


class PtArray:
    """Fields of the kernel ABI array (`ptarray_type`)."""

    DATA, SHAPE, STRIDES, OWNER, NDIM, DTYPE = range(6)


#: The kernel ABI array: `{data, shape*, strides*, owner, i32 ndim, i32 dtype}` (40
#: bytes). The caller of a kernel sets `ndim` and `dtype` (`dtype_code`) of every
#: argument; kernels don't write them.
def ptarray_type() -> ir.Type:
    return ir.Type.struct_([ir.ptr, ir.ptr, ir.ptr, ir.ptr, ir.i32, ir.i32])


#: The arguments of a kernel: `(PtCtx* ctx, PtArray* args, i32 n_inputs, i32 n_outputs,
#: const void* config)`.
def kernel_params() -> list[ir.Type]:
    return [ir.ptr, ir.ptr, ir.i32, ir.i32, ir.ptr]


PTARRAY_SIZE = 40


__all__ = [
    "ir",
    "ir_type",
    "memory_type",
    "value_bits",
    "value_type",
]
