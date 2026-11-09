"""LLVM representations of PyTensor types (DESIGN.md, section 1)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from pytensor.link.llvm_v2.layout import DYNAMIC, Layout
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.type import TensorType


def ir():
    from pytensor_llvm import ir

    return ir


SUPPORTED_DTYPES = ("float64", "float32", "int64", "int32", "int16", "int8", "uint8", "bool")


def check_dtype(dtype: str) -> str:
    if dtype not in SUPPORTED_DTYPES:
        raise NotImplementedError(f"LLVM backend: dtype {dtype} is not supported")
    return dtype


def value_type(dtype: str):
    """The LLVM type of a value of `dtype` in registers (bool is i1)."""
    m = ir()
    return {
        "float64": m.f64,
        "float32": m.f32,
        "int64": m.i64,
        "int32": m.i32,
        "int16": m.i16,
        "int8": m.i8,
        "uint8": m.i8,
        "bool": m.i1,
    }[check_dtype(dtype)]


def memory_type(dtype: str):
    """The LLVM type of an element of `dtype` in memory (bool is a byte)."""
    return ir().i8 if dtype == "bool" else value_type(dtype)


def itemsize(dtype: str) -> int:
    return np.dtype(check_dtype(dtype)).itemsize


def is_signed(dtype: str) -> bool:
    return dtype.startswith("int")


def is_float(dtype: str) -> bool:
    return dtype.startswith("float")


@dataclass(frozen=True)
class ScalarRepr:
    """A ScalarType value: the scalar, by value."""

    dtype: str

    @property
    def ir_type(self):
        return value_type(self.dtype)


@dataclass(frozen=True)
class TensorRepr:
    """A TensorType value: `ptr` for 0-d, else `{ptr, [d x i64] lengths, [s x i64] strides}`
    with only the dynamic lengths and the strides the layout doesn't fix."""

    dtype: str
    shape: tuple[int | None, ...]
    #: what is known about the strides (canonical for `shape`, see `repr_of`)
    layout: Layout = None

    def __post_init__(self):
        if self.layout is None:
            object.__setattr__(self, "layout", Layout.make(self.shape, range(len(self.shape)), (DYNAMIC,) * len(self.shape)))

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
    def contiguous(self) -> bool:
        return self.layout.is_c(self.shape)

    @property
    def ir_type(self):
        return _tensor_ir_type(self.ndim, len(self.dynamic_axes), len(self.stride_axes))

    @property
    def slot_type(self):
        """The output slot a kernel fills: `{ptr, [n x i64] lengths, [n x i64] strides, ptr owner}`."""
        return slot_type(self.ndim)


@cache
def _tensor_ir_type(ndim, n_lengths, n_strides):
    m = ir()
    if ndim == 0:
        return m.ptr
    return m.Type.struct_([m.ptr, m.Type.array(m.i64, n_lengths), m.Type.array(m.i64, n_strides)])


@cache
def slot_type(ndim):
    m = ir()
    return m.Type.struct_([m.ptr, m.Type.array(m.i64, ndim), m.Type.array(m.i64, ndim), m.ptr])


@dataclass(frozen=True)
class NoneRepr:
    """No value at all (NoneTypeT, random generators): no argument."""

    @property
    def ndim(self):
        return None


def is_absent(t) -> bool:
    """Types whose values aren't passed at all: NoneTypeT, and random generators (random
    numbers come from the call context, see DESIGN.md: the graph's rng values are
    tokens without data)."""
    from pytensor.tensor.random.type import RandomType
    from pytensor.tensor.type_other import NoneTypeT

    return isinstance(t, NoneTypeT | RandomType)


def repr_of(t, layout: Layout | None = None):
    """The representation of a PyTensor type; tensors with `layout` (default: nothing
    known about the strides)."""
    if is_absent(t):
        return NoneRepr()
    if isinstance(t, ScalarType):
        return ScalarRepr(check_dtype(t.dtype))
    if isinstance(t, TensorType):
        shape = tuple(None if s is None else int(s) for s in t.shape)
        if layout is not None:
            layout = Layout.make(shape, layout.order, layout.strides)
        return TensorRepr(check_dtype(t.dtype), shape, layout)
    raise NotImplementedError(f"LLVM backend: no representation for {t}")


#: The kernel ABI array: `{data, shape*, strides*, owner, ndim}` (40 bytes).
def ptarray_type():
    m = ir()
    return m.Type.struct_([m.ptr, m.ptr, m.ptr, m.ptr, m.i64])


PTARRAY_SIZE = 40
