from __future__ import annotations

from egglog import (
    EGraph,
    Expr,
    PyObject,
    String,
    StringLike,
    convert,
    converter,
    i64,
    i64Like,
)

from pytensor import Variable
from pytensor.graph import FunctionGraph


egraph = EGraph()


tensorify_ruleset = egraph.ruleset("tensorify")


@egraph.class_
class Int(Expr):
    def __init__(self, value: i64Like) -> None:
        ...

    @classmethod
    def var(cls, name: StringLike) -> Int:
        ...

    def __add__(self, other: Int) -> Int:
        ...

    def __sub__(self, other: Int) -> Int:
        ...

    def __eq__(self, i: Int) -> Int:
        ...

    # Egglog doesn't allow to override __ne__ for now
    # # def __ne__(self, i: Int) -> Int: ...

    def __gt__(self, i: Int) -> Int:
        ...

    def __ge__(self, i: Int) -> Int:
        ...

    def __lt__(self, i: Int) -> Int:
        ...

    def __le__(self, i: Int) -> Int:
        ...

    @property
    def tensorify(self) -> PyObject:
        ...


converter(i64, Int, Int)


@egraph.class_
class IntTuple(Expr):
    def __init__(self, head: Int) -> None:
        ...

    @classmethod
    def empty(cls) -> IntTuple:
        ...

    @egraph.method(cost=1000)
    @classmethod
    def from_range(cls, i: Int, n: Int) -> IntTuple:
        ...

    def __add__(self, other: IntTuple) -> IntTuple:
        ...

    def __getitem__(self, i: Int) -> Int:
        ...

    @egraph.method(cost=1000)
    def length(self) -> Int:
        ...

    def insert(self, idx: Int, value: Int) -> IntTuple:
        ...

    def pop(self, idx: Int) -> IntTuple:
        ...

    @property
    def tensorify(self) -> PyObject:
        ...


converter(int, IntTuple, lambda i: IntTuple(Int(i64(i))))
converter(i64, IntTuple, lambda i: IntTuple(Int(i)))
converter(Int, IntTuple, lambda i: IntTuple(i))
converter(
    tuple,
    IntTuple,
    lambda x: (
        IntTuple(convert(x[0], Int)) + convert(x[1:], IntTuple)
        if len(x) > 1
        else (IntTuple(convert(x[0], Int)) if x else IntTuple.empty())
    ),
)
# converter(list, IntTuple, lambda x: convert(tuple(x), IntTuple))  # Not working!


@egraph.class_
class Tensor(Expr):
    def __init__(self, name: StringLike, shape: IntTuple = IntTuple.empty()) -> None:
        ...

    @classmethod
    def constant(cls, value: Int, shape: IntTuple = IntTuple.empty()) -> Tensor:
        ...

    @property
    def tensorify(self) -> PyObject:
        ...

    def __add__(self, other: Tensor) -> Tensor:
        ...

    def __sub__(self, other: Tensor) -> Tensor:
        ...

    def __mul__(self, other: Tensor) -> Tensor:
        ...

    def __pow__(self, other: Tensor) -> Tensor:
        ...

    def __neg__(self) -> Tensor:
        ...


@egraph.class_
class TensorTuple(Expr):
    def __init__(self, value: Tensor) -> None:
        ...

    def __add__(self, other: TensorTuple) -> TensorTuple:
        ...

    @classmethod
    def empty(cls) -> TensorTuple:
        ...

    def __add__(self, other: TensorTuple) -> TensorTuple:
        ...

    def __getitem__(self, i: Int) -> Tensor:
        ...

    # __xor__ is used as a shorcut for broadcasting shape tuples
    def __xor__(self, other: TensorTuple) -> TensorTuple:
        ...

    @egraph.method(cost=1000)
    def length(self) -> Int:
        ...

    def insert(self, idx: Int, value: Tensor) -> TensorTuple:
        ...

    def pop(self, idx: Int) -> TensorTuple:
        ...

    @property
    def tensorify(self) -> PyObject:
        ...

    @egraph.method(cost=1000)
    @classmethod
    def from_int_tuple(cls, int_tuple: IntTuple) -> TensorTuple:
        ...

    @egraph.method(cost=1000)
    @classmethod
    def from_tensor_shape(
        cls, sh: TensorTuple, static_sh: IntTuple, idx: Int
    ) -> TensorTuple:
        ...

    @property
    def tensorify(self) -> PyObject:
        ...


converter(i64, Tensor, lambda i: Tensor.constant(Int(i)))
converter(int, Tensor, lambda i: Tensor.constant(Int(i64(i))))
converter(i64, TensorTuple, lambda i: TensorTuple(Tensor.constant(Int(i))))
converter(int, TensorTuple, lambda i: TensorTuple(Tensor.constant(Int(i64(i)))))
converter(
    tuple,
    TensorTuple,
    lambda x: (
        TensorTuple(convert(x[0], Tensor)) + convert(x[1:], TensorTuple)
        if len(x) > 1
        else (TensorTuple(convert(x[0], Tensor)) if x else TensorTuple.empty())
    ),
)


@egraph.class_
class UnaryInOp(Expr):
    def __call__(self, x: Tensor) -> Tensor:
        ...

    @property
    def tensorify(self) -> PyObject:
        ...


@egraph.class_
class BinaryInOp(Expr):
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        ...

    @property
    def tensorify(self) -> PyObject:
        ...


@egraph.class_
class VariadicInOp(Expr):
    def __call__(self, vars: TensorTuple) -> Tensor:
        ...

    @property
    def tensorify(self) -> PyObject:
        ...


@egraph.class_
class VariadicInOutOp(Expr):
    def __call__(self, vars: TensorTuple) -> TensorTuple:
        ...

    @property
    def tensorify(self) -> PyObject:
        ...


@egraph.class_
class ScalarOp(Expr):
    ...

    @property
    def tensorify(self) -> PyObject:
        ...


def eggify(*vars: Variable | FunctionGraph) -> tuple[Expr]:
    from pytensor.sandbox.scrambled.eggify.basic import eggify_fg

    if len(vars) > 1 or isinstance(vars[0], Variable):
        fg = FunctionGraph(outputs=vars, clone=False)
    else:
        [fg] = vars
    return eggify_fg(fg)


def rewrite_exprs(*exprs: Expr, epochs=100, verbose=False) -> tuple[Expr]:
    with egraph:
        initial_costs = []
        for expr in exprs:
            egraph.register(expr)
            initial_costs.append(egraph.extract(expr, include_cost=True)[1])

        egraph.run(epochs)

        new_exprs = []
        for expr, initial_cost in zip(exprs, initial_costs):
            new_expr, final_cost = egraph.extract(expr, include_cost=True)
            new_exprs.append(new_expr)
            if verbose:
                print(f"Cost: {initial_cost} -> {final_cost}")
                print(new_expr)
                print("")
    return tuple(new_exprs)


def tensorify(*exprs: Expr) -> tuple[Variable]:
    with egraph:
        for expr in exprs:
            egraph.register(expr)
        egraph.run(100, ruleset=tensorify_ruleset)
        return tuple(egraph.eval(expr.tensorify) for expr in exprs)


def egg_rewrite(
    *variables: Variable, epochs: int = 100, verbose: bool = False
) -> tuple[Variable]:
    var_exprs = eggify(*variables)
    new_var_exprs = rewrite_exprs(*var_exprs, epochs=epochs, verbose=verbose)
    # TODO: Assert all root variables where present in fg
    return tensorify(*new_var_exprs)
