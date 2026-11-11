"""How much work a node does: estimates for decisions like running a loop in parallel.

`llvm_work(op, node) -> Work` per op: a number of elements (a product of lengths, some only
known at run time) times a cost per element. Costs are in units of one simple float
operation (an add or a multiply, about a nanosecond or less): an `exp` is about 20, a
`gammaln` about 60. They are rough on purpose: they decide between "a few microseconds"
and "milliseconds", not between similar alternatives.

`scalar_cost(scalar_op)` is the cost of one scalar operation (Composite: its inner graph).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import singledispatch

import pytensor.scalar.basic as ps
import pytensor.scalar.math as psm


@dataclass(frozen=True)
class Work:
    #: cost of one element
    per_element: float
    #: the number of elements as a product of factors; None where only known at run time
    elements: tuple = ()

    @property
    def n_elements(self) -> int | None:
        if any(f is None for f in self.elements):
            return None
        return math.prod(self.elements)

    @property
    def total(self) -> float | None:
        """The estimate, if the shapes are static."""
        n = self.n_elements
        return None if n is None else n * self.per_element

    def describe(self) -> str:
        elements = (
            " x ".join("?" if f is None else str(f) for f in self.elements) or "1"
        )
        return f"{elements} x {self.per_element:g}"


def lengths(var) -> tuple:
    return tuple(getattr(var.type, "shape", ()))


# ------------------------------------------------------------------ scalar operations

#: cost of one call, by scalar op class
SCALAR_COSTS = {}


def _set(cost, *classes):
    for c in classes:
        SCALAR_COSTS[c] = cost


_set(
    1,
    ps.Add,
    ps.Sub,
    ps.Mul,
    ps.Neg,
    ps.Abs,
    ps.Identity,
    ps.Cast,
    ps.Second,
    ps.Sqr,
    ps.Sign,
    ps.Maximum,
    ps.Minimum,
    ps.Switch,
    ps.Clip,
    ps.EQ,
    ps.NEQ,
    ps.LT,
    ps.LE,
    ps.GT,
    ps.GE,
    ps.AND,
    ps.OR,
    ps.XOR,
    ps.Invert,
    ps.Floor,
    ps.Ceil,
    ps.Trunc,
    ps.RoundHalfToEven,
    ps.RoundHalfAwayFromZero,
    ps.IsNan,
    ps.IsInf,
)
_set(4, ps.TrueDiv, ps.Reciprocal, ps.Sqrt)
_set(10, ps.IntDiv, ps.Mod)
_set(
    20,
    ps.Exp,
    ps.Exp2,
    ps.Expm1,
    ps.Log,
    ps.Log1p,
    ps.Log2,
    ps.Log10,
    ps.Sin,
    ps.Cos,
    ps.Tan,
    ps.Sinh,
    ps.Cosh,
    ps.Tanh,
    psm.Sigmoid,
    psm.Erf,
    psm.Erfc,
)
_set(25, psm.Softplus, psm.Log1mexp, ps.ArcSin, ps.ArcCos, ps.ArcTan)
_set(30, ps.Pow, ps.ArcTan2, ps.ArcSinh, ps.ArcCosh, ps.ArcTanh)
_set(60, psm.Gamma, psm.GammaLn, psm.Psi)


@singledispatch
def scalar_cost(op) -> float:
    for cls in type(op).__mro__:
        if cls in SCALAR_COSTS:
            return SCALAR_COSTS[cls]
    return 4.0


@scalar_cost.register(ps.Composite)
def _(op) -> float:
    return sum(scalar_cost(n.op) for n in op.fgraph.toposort()) or 1.0


def _register_loop():
    from pytensor.scalar.loop import ScalarLoop

    @scalar_cost.register(ScalarLoop)
    def _(op) -> float:
        # the number of steps is data: assume a few dozen
        return 30 * (sum(scalar_cost(n.op) for n in op.fgraph.toposort()) or 1.0)


_register_loop()


# ------------------------------------------------------------------ nodes


@singledispatch
def llvm_work(op, node) -> Work:
    """The default: producing the outputs, one unit per element."""
    elements = ()
    for o in node.outputs:
        if hasattr(o.type, "shape"):
            elements = lengths(o)
            break
    return Work(1.0, elements)


def _register():
    from pytensor.compile.ops import DeepCopyOp, ViewOp
    from pytensor.scan.op import Scan
    from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
    from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
    from pytensor.tensor.linalg.solvers.triangular import SolveTriangular
    from pytensor.tensor.math import Dot
    from pytensor.tensor.random.op import RandomVariable
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise
    from pytensor.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape
    from pytensor.tensor.sort import SortOp
    from pytensor.tensor.subtensor import Subtensor

    @llvm_work.register(Elemwise)
    def _(op, node):
        return Work(scalar_cost(op.scalar_op), lengths(node.outputs[0]))

    @llvm_work.register(FusedElemwise)
    def _(op, node):
        biggest = max(
            (i for i in node.inputs if hasattr(i.type, "shape")),
            key=lambda v: v.type.ndim,
            default=None,
        )
        elements = lengths(biggest) if biggest is not None else lengths(node.outputs[0])
        return Work(scalar_cost(op.scalar_op) + 1, elements)

    @llvm_work.register(CAReduce)
    def _(op, node):
        return Work(scalar_cost(op.scalar_op), lengths(node.inputs[0]))

    @llvm_work.register(DimShuffle)
    @llvm_work.register(Subtensor)
    @llvm_work.register(ViewOp)
    @llvm_work.register(SpecifyShape)
    @llvm_work.register(Shape)
    @llvm_work.register(Shape_i)
    def _(op, node):
        return Work(0.0)

    @llvm_work.register(Reshape)
    def _(op, node):
        # a view, or a copy when the input isn't contiguous
        return Work(0.5, lengths(node.outputs[0]))

    @llvm_work.register(DeepCopyOp)
    def _(op, node):
        return Work(1.0, lengths(node.outputs[0]))

    @llvm_work.register(Dot)
    def _(op, node):
        a, b = (lengths(v) for v in node.inputs)
        # (m, k) @ (k, n): m * k * n multiply-adds
        m = a[0] if len(a) == 2 else 1
        k = a[-1]
        n = b[-1] if len(b) == 2 else 1
        return Work(2.0, (m, k, n))

    @llvm_work.register(Cholesky)
    def _(op, node):
        n = lengths(node.inputs[0])[0]
        return Work(1 / 3, (n, n, n))

    @llvm_work.register(SolveTriangular)
    def _(op, node):
        n = lengths(node.inputs[0])[0]
        b = lengths(node.inputs[1])
        return Work(1.0, (n, n, b[1] if len(b) == 2 else 1))

    @llvm_work.register(SortOp)
    def _(op, node):
        x = lengths(node.inputs[0])
        n = x[int(op.axis) % len(x)] if x else 1
        return Work(3.0 * math.log2(n) if n else 30.0, x)

    @llvm_work.register(RandomVariable)
    def _(op, node):
        heavy = type(op).__name__ in (
            "GammaRV",
            "InvGammaRV",
            "BetaRV",
            "PoissonRV",
            "BinomialRV",
        )
        return Work(60.0 if heavy else 20.0, lengths(node.outputs[1]))

    @llvm_work.register(Scan)
    def _(op, node):
        # the step's work times the number of steps (data)
        inner = sum((llvm_work(n.op, n).total or 0) for n in op.fgraph.toposort())
        return Work(max(inner, 1.0), (None,))


_register()


__all__ = ["Work", "llvm_work", "scalar_cost"]
