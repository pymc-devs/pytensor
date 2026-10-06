"""Scalar Op dispatch for JavaScript expressions."""

from functools import singledispatch

import numpy as np

from pytensor.graph.basic import Constant
from pytensor.scalar.basic import (
    AND,
    EQ,
    GE,
    GT,
    LE,
    LT,
    NEQ,
    Abs,
    Add,
    Cast,
    Composite,
    Cos,
    Exp,
    Expm1,
    Identity,
    IntDiv,
    Log,
    Log1p,
    Maximum,
    Minimum,
    Mul,
    Neg,
    Pow,
    Reciprocal,
    Second,
    Sin,
    Sqr,
    Sqrt,
    Sub,
    Switch,
    Tanh,
    TrueDiv,
)
from pytensor.scalar.math import GammaLn, Psi, Sigmoid, Softplus


def literal(value):
    value = float(value)
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "Infinity"
    if np.isneginf(value):
        return "-Infinity"
    return repr(value)


@singledispatch
def js_scalar(op, args):
    raise NotImplementedError(f"JS scalar Op {op} ({type(op).__name__}) is unsupported")


@js_scalar.register(Add)
def js_scalar_add(op, args):
    return "(" + " + ".join(args) + ")"


@js_scalar.register(Mul)
def js_scalar_mul(op, args):
    return "(" + " * ".join(args) + ")"


for op_type, operator in {
    AND: "&&",
    Sub: "-",
    TrueDiv: "/",
    Pow: "**",
    LT: "<",
    LE: "<=",
    GT: ">",
    GE: ">=",
    EQ: "===",
    NEQ: "!==",
}.items():
    js_scalar.register(op_type)(
        lambda op, args, operator=operator: f"({args[0]} {operator} {args[1]})"
    )


for op_type, function in {
    Abs: "abs",
    Sqrt: "sqrt",
    Exp: "exp",
    Log: "log",
    Log1p: "log1p",
    Expm1: "expm1",
    Sin: "sin",
    Cos: "cos",
    Tanh: "tanh",
    Maximum: "max",
    Minimum: "min",
}.items():
    js_scalar.register(op_type)(
        lambda op, args, function=function: f"Math.{function}({', '.join(args)})"
    )


@js_scalar.register(IntDiv)
def js_scalar_int_div(op, args):
    return f"Math.floor({args[0]} / {args[1]})"


@js_scalar.register(Cast)
def js_scalar_cast(op, args):
    dtype = op.o_type.dtype
    if dtype == "float64":
        return args[0]
    if dtype == "float32":
        return f"Math.fround({args[0]})"
    if dtype == "bool":
        return f"Boolean({args[0]})"
    if dtype in ("int8", "int16", "int32", "uint8", "uint16", "uint32"):
        bits = int(dtype.lstrip("uint"))
        if dtype.startswith("uint"):
            return f"(({args[0]} >>> 0) {'& ' + str((1 << bits) - 1) if bits < 32 else ''})"
        shift = 32 - bits
        return f"(({args[0]} << {shift}) >> {shift})"
    raise NotImplementedError(f"JS scalar cast to {dtype} is unsupported")


@js_scalar.register(Neg)
def js_scalar_neg(op, args):
    return f"(-{args[0]})"


@js_scalar.register(Sqr)
def js_scalar_sqr(op, args):
    return f"({args[0]} * {args[0]})"


@js_scalar.register(Reciprocal)
def js_scalar_reciprocal(op, args):
    return f"(1 / {args[0]})"


@js_scalar.register(Sigmoid)
def js_scalar_sigmoid(op, args):
    return f"(1 / (1 + Math.exp(-({args[0]}))))"


@js_scalar.register(Softplus)
def js_scalar_softplus(op, args):
    return f"(Math.log1p(Math.exp(-Math.abs({args[0]}))) + Math.max({args[0]}, 0))"


@js_scalar.register(GammaLn)
def js_scalar_gamma_ln(op, args):
    return f"scalar_lgamma({args[0]})"


@js_scalar.register(Psi)
def js_scalar_psi(op, args):
    return f"scalar_digamma({args[0]})"


@js_scalar.register(Switch)
def js_scalar_switch(op, args):
    return f"({args[0]} ? {args[1]} : {args[2]})"


@js_scalar.register(Identity)
def js_scalar_identity(op, args):
    return args[0]


@js_scalar.register(Second)
def js_scalar_second(op, args):
    return args[1]


def scalar_program(op, inputs):
    """Emit statements and expressions for a scalar Op or Composite."""
    if not isinstance(op, Composite):
        return [], [js_scalar(op, inputs)]

    names = dict(zip(op.inputs, inputs, strict=True))

    def operand(variable):
        if variable not in names and isinstance(variable, Constant):
            names[variable] = literal(variable.data)
        return names[variable]

    statements = []
    for index, node in enumerate(op.fgraph.toposort()):
        if len(node.outputs) != 1:
            raise NotImplementedError("JS Composite scalar Ops with multiple outputs")
        name = f"q{index}"
        expression = js_scalar(node.op, [operand(arg) for arg in node.inputs])
        statements.append(f"const {name} = {expression};")
        names[node.outputs[0]] = name
    return statements, [operand(output) for output in op.outputs]
