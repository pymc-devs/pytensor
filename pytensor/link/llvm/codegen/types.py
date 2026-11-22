"""LLVM representations of PyTensor types."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

import pytensor.scalar as ps
from pytensor.graph.type import Type
from pytensor.link.llvm.analysis.layout import Layout
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.type import TensorType


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


#: The element types the backend supports, in the order of their codes in the kernel
#: ABI (`PT_FLOAT64` etc. in the C header, `DType::code` in Rust).
SUPPORTED_TYPES: tuple[ScalarType, ...] = (
    ps.float64,
    ps.float32,
    ps.int64,
    ps.int32,
    ps.int16,
    ps.int8,
    ps.uint8,
    ps.bool,
    ps.uint16,
    ps.uint32,
    ps.uint64,
)


def element_type(t: TensorType | ScalarType | str) -> ScalarType:
    """Return the type of the elements of `t`: a TensorType, a ScalarType, or a dtype
    string (where one comes from outside the backend: numpy, a kernel's spec).

    Raises NotImplementedError for types the backend doesn't support."""
    scalar_type: ScalarType
    if isinstance(t, TensorType):
        scalar_type = t.to_scalar_type()
    elif isinstance(t, str):
        scalar_type = ps.get_scalar_type(t)
    else:
        scalar_type = t
    if scalar_type not in SUPPORTED_TYPES:
        raise NotImplementedError(
            f"LLVM backend: dtype {scalar_type.dtype} is not supported"
        )
    return scalar_type


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


def _check_scalar_type(t: object) -> None:
    if not isinstance(t, ScalarType):
        raise TypeError(f"expected a ScalarType, got {t!r}")


@dataclass(frozen=True)
class ScalarRepr:
    """A ScalarType value: the scalar, by value."""

    scalar_type: ScalarType

    def __post_init__(self) -> None:
        _check_scalar_type(self.scalar_type)

    @property
    def ir_type(self) -> ir.Type:
        return value_type(self.scalar_type)


@dataclass(frozen=True)
class TensorRepr:
    """A TensorType value: `ptr` for 0-d, else `{ptr, [d x i64] lengths, [s x i64] strides}`
    with only the dynamic lengths and the strides the layout doesn't fix."""

    #: the type of the elements
    scalar_type: ScalarType
    shape: tuple[int | None, ...]
    #: what is known about the strides (canonical for `shape`, see `repr_of`)
    layout: Layout

    def __post_init__(self) -> None:
        _check_scalar_type(self.scalar_type)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dynamic_axes(self) -> tuple[int, ...]:
        return tuple(a for a, s in enumerate(self.shape) if s is None)

    @property
    def stride_axes(self) -> tuple[int, ...]:
        """Axes whose strides are passed."""
        return self.layout.dynamic_axes

    @property
    def static(self) -> bool:
        return all(s is not None for s in self.shape)

    @property
    def fixed_shape(self) -> tuple[int, ...]:
        """The shape of a `static` tensor, as ints (ValueError for any other)."""
        lengths = tuple(s for s in self.shape if s is not None)
        if len(lengths) != len(self.shape):
            raise ValueError(f"the shape {self.shape} isn't static")
        return lengths

    @property
    def ir_type(self) -> ir.Type:
        return _tensor_ir_type(self.ndim, len(self.dynamic_axes), len(self.stride_axes))


@cache
def _tensor_ir_type(ndim: int, n_lengths: int, n_strides: int) -> ir.Type:
    if ndim == 0:
        return ir.ptr
    return ir.Type.struct_(
        [ir.ptr, ir.Type.array(ir.i64, n_lengths), ir.Type.array(ir.i64, n_strides)]
    )


@dataclass(frozen=True)
class NoneRepr:
    """No value at all (NoneTypeT, random generators): no argument."""

    @property
    def ndim(self) -> None:
        return None


#: How a value of a PyTensor type is passed in generated code.
type Representation = ScalarRepr | TensorRepr | NoneRepr


def is_absent(t: Type) -> bool:
    """Types whose values aren't passed at all: NoneTypeT, and random generators (random
    numbers come from the call context: the graph's rng values are tokens without
    data)."""
    from pytensor.tensor.random.type import RandomType
    from pytensor.tensor.type_other import NoneTypeT

    return isinstance(t, NoneTypeT | RandomType)


def repr_of(t: Type, layout: Layout | None = None) -> Representation:
    """The representation of a PyTensor type; tensors with `layout` (default: nothing
    known about the strides)."""
    if is_absent(t):
        return NoneRepr()
    if isinstance(t, ScalarType):
        return ScalarRepr(element_type(t))
    if isinstance(t, TensorType):
        shape = tuple(None if s is None else int(s) for s in t.shape)
        if layout is None:
            layout = Layout.unknown(shape)
        else:
            layout = Layout.make(shape, layout.order, layout.strides)
        return TensorRepr(element_type(t), shape, layout)
    raise NotImplementedError(f"LLVM backend: no representation for {t}")


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
    "NoneRepr",
    "Representation",
    "ScalarRepr",
    "TensorRepr",
    "element_type",
    "ir",
    "is_absent",
    "memory_type",
    "repr_of",
    "value_bits",
    "value_type",
]
