"""The PyTensor types the LLVM backend handles, and how their values are passed.

Planning and code generation share these: `element_type` says which element types the
backend supports, `is_absent` which values aren't passed at all, and `repr_of` how a
value of a type is passed between the functions of a program (`Representation`). For
example, a float64 matrix with a static second length and C-contiguous strides is a
`TensorRepr` that passes its data and its first length, and no strides. The LLVM types
of these representations are in codegen/types.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytensor.scalar as ps
from pytensor.link.llvm.layout import Layout
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.type import DenseTensorType


if TYPE_CHECKING:
    from pytensor.graph.type import HasDataType, Type


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


def element_type(t: HasDataType | ScalarType | str) -> ScalarType:
    """Return the type of the elements of `t`: a ScalarType, a type with a dtype
    (TensorType, XTensorType), or a dtype string (where one comes from outside the
    backend: numpy, a kernel's spec).

    Raises NotImplementedError for types the backend doesn't support."""
    scalar_type: ScalarType
    if isinstance(t, ScalarType):
        scalar_type = t
    elif isinstance(t, str):
        scalar_type = ps.get_scalar_type(t)
    else:
        scalar_type = ps.get_scalar_type(t.dtype)
    if scalar_type not in SUPPORTED_TYPES:
        raise NotImplementedError(
            f"LLVM backend: dtype {scalar_type.dtype} is not supported"
        )
    return scalar_type


def _check_scalar_type(t: object) -> None:
    if not isinstance(t, ScalarType):
        raise TypeError(f"expected a ScalarType, got {t!r}")


@dataclass(frozen=True)
class ScalarRepr:
    """A ScalarType value: the scalar, by value."""

    scalar_type: ScalarType

    def __post_init__(self) -> None:
        _check_scalar_type(self.scalar_type)


@dataclass(frozen=True)
class TensorRepr:
    """A TensorType value: its data, with the lengths of only its dynamic axes and the
    strides only of the axes the layout doesn't fix."""

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
    # A SparseTensorType is a TensorType subclass, but its values aren't ndarrays.
    if not isinstance(t, DenseTensorType):
        from pytensor.xtensor.type import XTensorType

        if not isinstance(t, XTensorType):
            raise NotImplementedError(f"LLVM backend: no representation for {t}")
        # An XTensorType value is an ndarray, like a TensorType of its dtype and shape.
    shape = tuple(None if s is None else int(s) for s in t.shape)
    if layout is None:
        layout = Layout.unknown(shape)
    else:
        layout = layout.for_shape(shape)
    return TensorRepr(element_type(t), shape, layout)


__all__ = [
    "SUPPORTED_TYPES",
    "NoneRepr",
    "Representation",
    "ScalarRepr",
    "TensorRepr",
    "element_type",
    "is_absent",
    "repr_of",
]
