"""Numbers in generated code: an IR value together with its type (a `ScalarType`).

LLVM integers don't know whether they are signed, and floats and integers have
different instructions. A `Num` carries the type, so emitters write the arithmetic
and the operators pick the instruction:

    i = Num(b, value, ps.int64)
    i = (i < 0).select(i + n, i)
    ok = (i >= 0) & (i < n)

The operators follow numpy for the type: `+` of booleans is `or`, integer `//` and `%`
round to the floor and give 0 for a zero divisor, comparisons of floats are false for
NaN. (Float `//` is `floor(x / y)`, like PyTensor's C code; numpy differs for infinite
operands.)
Both operands must have the same type: convert with `astype`. Python numbers take the
type of the `Num` they meet.

A `Num` is not a Python number: `==` compares identity (use `x.eq(y)`), and using one
as a condition (`if x < y:`) raises; that decision belongs to the generated code
(`cond.select(a, b)`, `with b.if_(cond.value):`).
"""

from __future__ import annotations

import pytensor.scalar as ps
from pytensor.link.llvm.codegen.types import (
    ir,
    memory_type,
    value_bits,
    value_type,
)
from pytensor.link.llvm.types import element_type
from pytensor.scalar.basic import ScalarType


#: what arithmetic takes as the other operand of a `Num`: a `Num` of the same type, or
#: a Python number (as a constant of that type)
type NumLike = Num | bool | int | float


class Num:
    __slots__ = ("b", "scalar_type", "value")

    def __init__(self, b: ir.Builder, value: ir.Value, scalar_type: ScalarType):
        self.b = b
        self.value = value
        if not isinstance(scalar_type, ScalarType):
            raise TypeError(f"expected a ScalarType, got {scalar_type!r}")
        self.scalar_type = element_type(scalar_type)

    @classmethod
    def const(cls, b: ir.Builder, scalar_type: ScalarType, value: float) -> Num:
        """The constant `value` of type `scalar_type`."""
        return cls(b, constant(b, scalar_type, value), scalar_type)

    def __repr__(self) -> str:
        return f"Num({self.scalar_type}, {self.value!r})"

    def __bool__(self) -> bool:
        raise TypeError(
            "a Num is a value of the generated code, not a Python condition: "
            "use cond.select(a, b) or `with b.if_(cond.value):`"
        )

    @property
    def const_value(self) -> bool | int | float | None:
        """The Python value if this is a constant, else None."""
        if self.scalar_type in ps.float_types:
            return self.value.const_float
        c = self.value.const_int
        if c is not None and self.scalar_type == ps.bool:
            return bool(c)
        if c is not None and self.scalar_type in ps.uint_types:
            return c % (1 << value_bits(self.scalar_type))
        return c

    # ------------------------------------------------------------ plumbing

    @property
    def _float(self) -> bool:
        return self.scalar_type in ps.float_types

    @property
    def _bool(self) -> bool:
        return self.scalar_type == ps.bool

    def _new(self, value: ir.Value, scalar_type: ScalarType | None = None) -> Num:
        return Num(
            self.b, value, self.scalar_type if scalar_type is None else scalar_type
        )

    def _raw(self, other: NumLike, what: str) -> ir.Value:
        """The IR value of the other operand of `what`, of this type."""
        if isinstance(other, Num):
            if other.scalar_type != self.scalar_type:
                raise TypeError(
                    f"{what} of {self.scalar_type} and {other.scalar_type}: convert one with astype"
                )
            return other.value
        if isinstance(other, bool | int | float):
            if isinstance(other, float) and not self._float:
                raise TypeError(
                    f"{what} of {self.scalar_type} and the float {other}: convert with astype"
                )
            return constant(self.b, self.scalar_type, other)
        raise TypeError(
            f"{what} of a Num and {type(other).__name__}: wrap IR values as Num(b, value, scalar_type)"
        )

    def _no(self, what: str) -> NotImplementedError:
        # (NotImplementedError: the planner then looks for another implementation)
        return NotImplementedError(
            f"LLVM backend: {what} is not defined for {self.scalar_type}"
        )

    # ------------------------------------------------------------ conversion

    def astype(self, scalar_type: ScalarType) -> Num:
        """Converted like numpy's astype."""
        return Num(
            self.b, cast(self.b, self.value, self.scalar_type, scalar_type), scalar_type
        )

    def select(self, a: NumLike, b: NumLike) -> Num:
        """`a` where this boolean is true, else `b` (both are evaluated)."""
        if not self._bool:
            raise TypeError(
                f"select on a condition of dtype {self.scalar_type}: compare it first"
            )
        like = a if isinstance(a, Num) else b
        if not isinstance(like, Num):
            raise TypeError("select needs at least one Num to choose from")
        return like._new(
            self.b.select(self.value, like._raw(a, "select"), like._raw(b, "select"))
        )

    # ------------------------------------------------------------ arithmetic

    def _arith(
        self,
        other: NumLike,
        what: str,
        float_op: str,
        int_op: str | None,
        bool_op: str | None = None,
        swap: bool = False,
    ) -> Num:
        b, x, y = self.b, self.value, self._raw(other, what)
        if swap:
            x, y = y, x
        if self._float:
            return self._new(getattr(b, float_op)(x, y))
        if self._bool:
            if bool_op is None:
                raise self._no(what)
            return self._new(getattr(b, bool_op)(x, y))
        if int_op is None:
            raise self._no(what)
        return self._new(getattr(b, int_op)(x, y))

    def __add__(self, other: NumLike) -> Num:
        return self._arith(other, "+", "fadd", "add", "or_")

    __radd__ = __add__

    def __sub__(self, other: NumLike) -> Num:
        return self._arith(other, "-", "fsub", "sub")

    def __rsub__(self, other: NumLike) -> Num:
        return self._arith(other, "-", "fsub", "sub", swap=True)

    def __mul__(self, other: NumLike) -> Num:
        return self._arith(other, "*", "fmul", "mul", "and_")

    __rmul__ = __mul__

    def __truediv__(self, other: NumLike) -> Num:
        if not self._float:
            raise self._no("/ (convert to a float dtype, or use //)")
        return self._arith(other, "/", "fdiv", None)

    def __rtruediv__(self, other: NumLike) -> Num:
        if not self._float:
            raise self._no("/ (convert to a float dtype, or use //)")
        return self._arith(other, "/", "fdiv", None, swap=True)

    def _divisor(self, y: ir.Value) -> tuple[ir.Value, ir.Value]:
        """(y with 0 replaced by 1, is-zero flag): integer division by zero is undefined
        in LLVM; numpy gives 0."""
        b = self.b
        is_zero = b.eq(y, 0)
        return b.select(is_zero, 1, y), is_zero

    def _floor_differs(self, remainder: ir.Value, y: ir.Value) -> ir.Value:
        """For the C remainder `r` of a division by `y`: does floor rounding differ?"""
        b = self.b
        return b.and_(b.ne(remainder, 0), b.xor(b.slt(remainder, 0), b.slt(y, 0)))

    def _divmod(
        self, other: NumLike, what: str, want_quotient: bool, swap: bool = False
    ) -> Num:
        b, x, y = self.b, self.value, self._raw(other, what)
        if swap:
            x, y = y, x
        if self._float:
            q = b.intrinsic(ir.Intrinsic.Floor, [b.fdiv(x, y)])
            return self._new(q if want_quotient else b.fsub(x, b.fmul(q, y)))
        y, is_zero = self._divisor(y)
        if self.scalar_type not in ps.int_types:
            return self._new(
                b.select(is_zero, 0, b.udiv(x, y) if want_quotient else b.urem(x, y))
            )
        # a divisor of -1 is handled apart: MIN / -1 overflows (numpy wraps to MIN)
        minus_one = b.eq(y, -1)
        y = b.select(minus_one, 1, y)
        remainder = b.srem(x, y)
        differs = self._floor_differs(remainder, y)
        if want_quotient:
            q = b.sdiv(x, y)
            q = b.select(differs, b.sub(q, 1), q)
            res = b.select(minus_one, b.neg(x), q)
        else:
            res = b.select(
                differs, b.add(remainder, y), remainder
            )  # x % -1 == 0 comes out of y == 1
        return self._new(b.select(is_zero, 0, res))

    def __floordiv__(self, other: NumLike) -> Num:
        """Floor division: integers like numpy (0 for a zero divisor), floats `floor(x / y)`."""
        return self._divmod(other, "//", True)

    def __rfloordiv__(self, other: NumLike) -> Num:
        return self._divmod(other, "//", True, swap=True)

    def __mod__(self, other: NumLike) -> Num:
        """Modulo with the sign of the divisor, like numpy: 0 for a zero divisor."""
        return self._divmod(other, "%", False)

    def __rmod__(self, other: NumLike) -> Num:
        return self._divmod(other, "%", False, swap=True)

    def __neg__(self) -> Num:
        if self._bool:
            raise self._no("negation")
        return self._new(
            self.b.fneg(self.value) if self._float else self.b.neg(self.value)
        )

    def __abs__(self) -> Num:
        b, x = self.b, self.value
        if self._float:
            return self._new(b.intrinsic(ir.Intrinsic.Fabs, [x]))
        if self.scalar_type in ps.int_types:
            return self._new(b.select(b.slt(x, 0), b.neg(x), x))
        return self

    # ------------------------------------------------------------ bits and logic

    def _bits(self, other: NumLike, what: str, method: str) -> Num:
        if self._float:
            raise self._no(what)
        return self._new(getattr(self.b, method)(self.value, self._raw(other, what)))

    def __and__(self, other: NumLike) -> Num:
        return self._bits(other, "&", "and_")

    __rand__ = __and__

    def __or__(self, other: NumLike) -> Num:
        return self._bits(other, "|", "or_")

    __ror__ = __or__

    def __xor__(self, other: NumLike) -> Num:
        return self._bits(other, "^", "xor")

    __rxor__ = __xor__

    def __invert__(self) -> Num:
        if self._float:
            raise self._no("~")
        return self._new(self.b.not_(self.value))

    # ------------------------------------------------------------ comparisons

    def _compare(
        self, other: NumLike, what: str, fpred: str, signed: str, unsigned: str
    ) -> Num:
        b, x, y = self.b, self.value, self._raw(other, what)
        if self._float:
            return self._new(b.fcmp(getattr(ir.FCmp, fpred), x, y), ps.bool)
        return self._new(
            getattr(b, signed if self.scalar_type in ps.int_types else unsigned)(x, y),
            ps.bool,
        )

    def __lt__(self, other: NumLike) -> Num:
        return self._compare(other, "<", "OLT", "slt", "ult")

    def __le__(self, other: NumLike) -> Num:
        return self._compare(other, "<=", "OLE", "sle", "ule")

    def __gt__(self, other: NumLike) -> Num:
        return self._compare(other, ">", "OGT", "sgt", "ugt")

    def __ge__(self, other: NumLike) -> Num:
        return self._compare(other, ">=", "OGE", "sge", "uge")

    def eq(self, other: NumLike) -> Num:
        """Equal (false if either is NaN)."""
        return self._compare(other, "eq", "OEQ", "eq", "eq")

    def ne(self, other: NumLike) -> Num:
        """Not equal (true if either is NaN)."""
        return self._compare(other, "ne", "UNE", "ne", "ne")

    def _extreme(self, other: NumLike, what: str, is_max: bool) -> Num:
        b, x, y = self.b, self.value, self._raw(other, what)
        if self._float:
            return self._new(
                b.intrinsic(
                    ir.Intrinsic.Maximum if is_max else ir.Intrinsic.Minimum, [x, y]
                )
            )
        if self._bool:
            return self._new(b.or_(x, y) if is_max else b.and_(x, y))
        if self.scalar_type in ps.int_types:
            return self._new(b.smax(x, y) if is_max else b.smin(x, y))
        return self._new(b.umax(x, y) if is_max else b.umin(x, y))

    def maximum(self, other: NumLike) -> Num:
        """The larger one, like numpy.maximum (NaN if either is NaN)."""
        return self._extreme(other, "maximum", True)

    def minimum(self, other: NumLike) -> Num:
        """The smaller one, like numpy.minimum (NaN if either is NaN)."""
        return self._extreme(other, "minimum", False)


# ------------------------------------------------------------------ loading, storing and converting elements


def load(b: ir.Builder, scalar_type: ScalarType, ptr: ir.Value) -> ir.Value:
    """Load an element; bools become i1."""
    v = b.load(memory_type(scalar_type), ptr)
    if scalar_type == ps.bool:
        return b.ne(v, 0)
    return v


def store(
    b: ir.Builder, scalar_type: ScalarType, value: ir.Value, ptr: ir.Value
) -> None:
    """Store an element; bools are stored as bytes."""
    if scalar_type == ps.bool:
        value = b.zext(value, ir.i8)
    b.store(value, ptr)


def constant(b: ir.Builder, scalar_type: ScalarType, value: float) -> ir.Value:
    """The constant `value` as a value of type `scalar_type`."""
    if scalar_type in ps.float_types:
        return b.const_float(value_type(scalar_type), float(value))
    if scalar_type == ps.bool:
        return b.const_bool(bool(value))
    return b.const_int(value_type(scalar_type), int(value))


def _float_to_int(b: ir.Builder, v: ir.Value, dst: ScalarType) -> ir.Value:
    """Float to integer, saturating; NaN gives 0. (`fptosi`/`fptoui` are poison for
    values out of range.)"""
    dt, bits = value_type(dst), value_bits(dst)
    if dst in ps.int_types:
        lo, hi = -(2.0 ** (bits - 1)), 2.0 ** (bits - 1)
        low_end, high_end = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        below = b.fcmp(ir.FCmp.OLT, v, lo)
        lower_ok = b.fcmp(ir.FCmp.OGE, v, lo)
        convert = b.fptosi
    else:
        # truncation takes (-1, 0) to 0, which is in range
        lo, hi = -1.0, 2.0**bits
        low_end, high_end = 0, 2**bits - 1
        below = b.fcmp(ir.FCmp.OLE, v, lo)
        lower_ok = b.fcmp(ir.FCmp.OGT, v, lo)
        convert = b.fptoui
    above = b.fcmp(ir.FCmp.OGE, v, hi)
    # ordered compares: NaN is neither below, above nor in range, and converts 0.0
    in_range = b.and_(lower_ok, b.fcmp(ir.FCmp.OLT, v, hi))
    return b.select(
        above,
        high_end,
        b.select(below, low_end, convert(b.select(in_range, v, 0.0), dt)),
    )


def cast(b: ir.Builder, v: ir.Value, src: ScalarType, dst: ScalarType) -> ir.Value:
    """Convert a value like numpy's astype."""
    if src == dst:
        return v
    dt = value_type(dst)
    if dst == ps.bool:
        if src in ps.float_types:
            return b.fcmp(ir.FCmp.UNE, v, 0.0)
        return b.ne(v, 0)
    if src in ps.float_types and dst in ps.float_types:
        return b.fpext(v, dt) if value_bits(dst) > value_bits(src) else b.fptrunc(v, dt)
    if dst in ps.float_types:
        return b.sitofp(v, dt) if src in ps.int_types else b.uitofp(v, dt)
    if src in ps.float_types:
        return _float_to_int(b, v, dst)
    # integer to integer (bool is i1, unsigned)
    if value_bits(dst) > value_bits(src):
        return b.sext(v, dt) if src in ps.int_types else b.zext(v, dt)
    if value_bits(dst) < value_bits(src):
        return b.trunc(v, dt)
    return v  # int8 <-> uint8


__all__ = [
    "Num",
    "NumLike",
    "cast",
    "constant",
    "load",
    "store",
]
