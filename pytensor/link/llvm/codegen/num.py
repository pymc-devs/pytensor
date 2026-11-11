"""Numbers in generated code: an IR value together with its dtype.

LLVM integers don't know whether they are signed, and floats and integers have
different instructions. A `Num` carries the dtype, so emitters write the arithmetic
and the operators pick the instruction:

    i = Num(b, value, "int64")
    i = (i < 0).select(i + n, i)
    ok = (i >= 0) & (i < n)

The operators follow numpy for the dtype: `+` of booleans is `or`, integer `//` and `%`
round to the floor and give 0 for a zero divisor, comparisons of floats are false for
NaN. (Float `//` is `floor(x / y)`, like PyTensor's C code; numpy differs for infinite
operands.)
Both operands must have the same dtype: convert with `astype`. Python numbers take the
dtype of the `Num` they meet.

A `Num` is not a Python number: `==` compares identity (use `x.eq(y)`), and using one
as a condition (`if x < y:`) raises; that decision belongs to the generated code
(`cond.select(a, b)`, `with b.if_(cond.value):`).
"""

from __future__ import annotations

from pytensor.link.llvm.codegen.emit import cast, constant
from pytensor.link.llvm.codegen.types import check_dtype, ir, is_float, is_signed


class Num:
    __slots__ = ("b", "dtype", "value")

    def __init__(self, b, value, dtype: str):
        self.b = b
        self.value = value
        self.dtype = check_dtype(dtype)

    @classmethod
    def const(cls, b, dtype: str, value) -> Num:
        """The constant `value` of `dtype`."""
        return cls(b, constant(b, dtype, value), dtype)

    def __repr__(self):
        return f"Num({self.dtype}, {self.value!r})"

    def __bool__(self):
        raise TypeError(
            "a Num is a value of the generated code, not a Python condition: "
            "use cond.select(a, b) or `with b.if_(cond.value):`"
        )

    @property
    def const_value(self):
        """The Python value if this is a constant, else None."""
        if is_float(self.dtype):
            return self.value.const_float
        c = self.value.const_int
        if c is not None and self.dtype == "bool":
            return bool(c)
        if c is not None and self.dtype == "uint8":
            return c % 256
        return c

    # ------------------------------------------------------------ plumbing

    @property
    def _float(self) -> bool:
        return is_float(self.dtype)

    @property
    def _bool(self) -> bool:
        return self.dtype == "bool"

    def _new(self, value, dtype: str | None = None) -> Num:
        return Num(self.b, value, self.dtype if dtype is None else dtype)

    def _raw(self, other, what: str):
        """The IR value of the other operand of `what`, in this dtype."""
        if isinstance(other, Num):
            if other.dtype != self.dtype:
                raise TypeError(
                    f"{what} of {self.dtype} and {other.dtype}: convert one with astype"
                )
            return other.value
        if isinstance(other, bool | int | float):
            if isinstance(other, float) and not self._float:
                raise TypeError(
                    f"{what} of {self.dtype} and the float {other}: convert with astype"
                )
            return constant(self.b, self.dtype, other)
        raise TypeError(
            f"{what} of a Num and {type(other).__name__}: wrap IR values as Num(b, value, dtype)"
        )

    def _no(self, what: str):
        # (NotImplementedError: the planner then looks for another implementation)
        return NotImplementedError(
            f"LLVM backend: {what} is not defined for {self.dtype}"
        )

    # ------------------------------------------------------------ conversion

    def astype(self, dtype: str) -> Num:
        """Converted like numpy's astype."""
        return Num(self.b, cast(self.b, self.value, self.dtype, dtype), dtype)

    def select(self, a, b) -> Num:
        """`a` where this boolean is true, else `b` (both are evaluated)."""
        if not self._bool:
            raise TypeError(
                f"select on a condition of dtype {self.dtype}: compare it first"
            )
        like = a if isinstance(a, Num) else b
        if not isinstance(like, Num):
            raise TypeError("select needs at least one Num to choose from")
        return like._new(
            self.b.select(self.value, like._raw(a, "select"), like._raw(b, "select"))
        )

    # ------------------------------------------------------------ arithmetic

    def _arith(self, other, what, float_op, int_op, bool_op=None, swap=False):
        b, x, y = self.b, self.value, self._raw(other, what)
        if swap:
            x, y = y, x
        if self._float:
            return self._new(getattr(b, float_op)(x, y))
        if self._bool:
            if bool_op is None:
                raise self._no(what)
            return self._new(getattr(b, bool_op)(x, y))
        return self._new(getattr(b, int_op)(x, y))

    def __add__(self, other):
        return self._arith(other, "+", "fadd", "add", "or_")

    __radd__ = __add__

    def __sub__(self, other):
        return self._arith(other, "-", "fsub", "sub")

    def __rsub__(self, other):
        return self._arith(other, "-", "fsub", "sub", swap=True)

    def __mul__(self, other):
        return self._arith(other, "*", "fmul", "mul", "and_")

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not self._float:
            raise self._no("/ (convert to a float dtype, or use //)")
        return self._arith(other, "/", "fdiv", None)

    def __rtruediv__(self, other):
        if not self._float:
            raise self._no("/ (convert to a float dtype, or use //)")
        return self._arith(other, "/", "fdiv", None, swap=True)

    def _divisor(self, y):
        """(y with 0 replaced by 1, is-zero flag): integer division by zero is undefined
        in LLVM; numpy gives 0."""
        b = self.b
        is_zero = b.eq(y, 0)
        return b.select(is_zero, 1, y), is_zero

    def _floor_differs(self, r, y):
        """For the C remainder `r` of a division by `y`: does floor rounding differ?"""
        b = self.b
        return b.and_(b.ne(r, 0), b.xor(b.slt(r, 0), b.slt(y, 0)))

    def _divmod(self, other, what: str, want_quotient: bool, swap: bool = False) -> Num:
        b, x, y = self.b, self.value, self._raw(other, what)
        if swap:
            x, y = y, x
        if self._float:
            q = b.intrinsic(ir.Intrinsic.Floor, [b.fdiv(x, y)])
            return self._new(q if want_quotient else b.fsub(x, b.fmul(q, y)))
        y, is_zero = self._divisor(y)
        if not is_signed(self.dtype):
            return self._new(
                b.select(is_zero, 0, b.udiv(x, y) if want_quotient else b.urem(x, y))
            )
        # a divisor of -1 is handled apart: MIN / -1 overflows (numpy wraps to MIN)
        minus_one = b.eq(y, -1)
        y = b.select(minus_one, 1, y)
        r = b.srem(x, y)
        differs = self._floor_differs(r, y)
        if want_quotient:
            q = b.sdiv(x, y)
            q = b.select(differs, b.sub(q, 1), q)
            res = b.select(minus_one, b.neg(x), q)
        else:
            res = b.select(differs, b.add(r, y), r)  # x % -1 == 0 comes out of y == 1
        return self._new(b.select(is_zero, 0, res))

    def __floordiv__(self, other):
        """Floor division: integers like numpy (0 for a zero divisor), floats `floor(x / y)`."""
        return self._divmod(other, "//", True)

    def __rfloordiv__(self, other):
        return self._divmod(other, "//", True, swap=True)

    def __mod__(self, other):
        """Modulo with the sign of the divisor, like numpy: 0 for a zero divisor."""
        return self._divmod(other, "%", False)

    def __rmod__(self, other):
        return self._divmod(other, "%", False, swap=True)

    def __neg__(self):
        if self._bool:
            raise self._no("negation")
        return self._new(
            self.b.fneg(self.value) if self._float else self.b.neg(self.value)
        )

    def __abs__(self):
        b, x = self.b, self.value
        if self._float:
            return self._new(b.intrinsic(ir.Intrinsic.Fabs, [x]))
        if is_signed(self.dtype):
            return self._new(b.select(b.slt(x, 0), b.neg(x), x))
        return self

    # ------------------------------------------------------------ bits and logic

    def _bits(self, other, what, method):
        if self._float:
            raise self._no(what)
        return self._new(getattr(self.b, method)(self.value, self._raw(other, what)))

    def __and__(self, other):
        return self._bits(other, "&", "and_")

    __rand__ = __and__

    def __or__(self, other):
        return self._bits(other, "|", "or_")

    __ror__ = __or__

    def __xor__(self, other):
        return self._bits(other, "^", "xor")

    __rxor__ = __xor__

    def __invert__(self):
        if self._float:
            raise self._no("~")
        return self._new(self.b.not_(self.value))

    # ------------------------------------------------------------ comparisons

    def _compare(self, other, what, fpred, signed, unsigned) -> Num:
        b, x, y = self.b, self.value, self._raw(other, what)
        if self._float:
            return self._new(b.fcmp(getattr(ir.FCmp, fpred), x, y), "bool")
        return self._new(
            getattr(b, signed if is_signed(self.dtype) else unsigned)(x, y), "bool"
        )

    def __lt__(self, other):
        return self._compare(other, "<", "OLT", "slt", "ult")

    def __le__(self, other):
        return self._compare(other, "<=", "OLE", "sle", "ule")

    def __gt__(self, other):
        return self._compare(other, ">", "OGT", "sgt", "ugt")

    def __ge__(self, other):
        return self._compare(other, ">=", "OGE", "sge", "uge")

    def eq(self, other) -> Num:
        """Equal (false if either is NaN)."""
        return self._compare(other, "eq", "OEQ", "eq", "eq")

    def ne(self, other) -> Num:
        """Not equal (true if either is NaN)."""
        return self._compare(other, "ne", "UNE", "ne", "ne")

    def _extreme(self, other, what: str, is_max: bool) -> Num:
        b, x, y = self.b, self.value, self._raw(other, what)
        if self._float:
            return self._new(
                b.intrinsic(
                    ir.Intrinsic.Maximum if is_max else ir.Intrinsic.Minimum, [x, y]
                )
            )
        if self._bool:
            return self._new(b.or_(x, y) if is_max else b.and_(x, y))
        if is_signed(self.dtype):
            return self._new(b.smax(x, y) if is_max else b.smin(x, y))
        return self._new(b.umax(x, y) if is_max else b.umin(x, y))

    def maximum(self, other) -> Num:
        """The larger one, like numpy.maximum (NaN if either is NaN)."""
        return self._extreme(other, "maximum", True)

    def minimum(self, other) -> Num:
        """The smaller one, like numpy.minimum (NaN if either is NaN)."""
        return self._extreme(other, "minimum", False)


__all__ = ["Num"]
