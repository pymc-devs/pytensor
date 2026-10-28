"""Scalar Op dispatch for JavaScript expressions."""

from collections import Counter
from functools import singledispatch

import numpy as np

from pytensor.graph.basic import Constant
from pytensor.graph.traversal import ancestors
from pytensor.link.js.rewriting import MulLog, switch_branch
from pytensor.scalar.basic import (
    AND,
    EQ,
    GE,
    GT,
    LE,
    LT,
    NEQ,
    OR,
    Abs,
    Add,
    Cast,
    Clip,
    Composite,
    Cos,
    Exp,
    Expm1,
    Identity,
    IntDiv,
    Invert,
    IsInf,
    IsNan,
    Log,
    Log1p,
    Maximum,
    Minimum,
    Mul,
    Neg,
    Pow,
    Reciprocal,
    Second,
    Sign,
    Sin,
    Sqr,
    Sqrt,
    Sub,
    Switch,
    Tanh,
    TrueDiv,
    identity,
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


@js_scalar.register(MulLog)
def js_scalar_mul_log(op, args):
    x, y = args
    return (
        f"({x} === 0 && {y} > 0 && {y} < Infinity "
        f"? {x} * ({y} < 1 ? -1 : 1) : {x} * Math.log({y}))"
    )


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


@js_scalar.register(Clip)
def js_scalar_clip(op, args):
    value, lower, upper = args
    return f"({value} < {lower} ? {lower} : {value} > {upper} ? {upper} : {value})"


@js_scalar.register(Sign)
def js_scalar_sign(op, args):
    value = args[0]
    return f"({value} === 0 ? 0 : Math.sign({value}))"


@js_scalar.register(IsInf)
def js_scalar_isinf(op, args):
    return f"(Math.abs({args[0]}) === Infinity)"


@js_scalar.register(IsNan)
def js_scalar_isnan(op, args):
    return f"Number.isNaN({args[0]})"


@js_scalar.register(OR)
def js_scalar_or(op, args):
    return f"Number(BigInt({args[0]}) | BigInt({args[1]}))"


@js_scalar.register(Invert)
def js_scalar_invert(op, args):
    return f"Number(~BigInt({args[0]}))"


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
    return f"(({args[0]} !== 0 && {args[0]} !== false) ? {args[1]} : {args[2]})"


@js_scalar.register(Identity)
def js_scalar_identity(op, args):
    return args[0]


@js_scalar.register(Second)
def js_scalar_second(op, args):
    return args[1]


def scalar_program(
    op, inputs, switch_choices=None, input_dtypes=None, _prefix="", input_loads=None
):
    """Emit statements and expressions for a scalar Op or Composite."""
    if not isinstance(op, Composite):
        loads = [
            f"const {name} = {expression};"
            for name, expression in (input_loads or {}).items()
        ]
        if isinstance(op, Invert) and input_dtypes == ["bool"]:
            return loads, [f"(!{inputs[0]})"]
        if (
            isinstance(op, OR)
            and input_dtypes
            and all(dtype == "bool" for dtype in input_dtypes)
        ):
            return loads, [f"({inputs[0]} || {inputs[1]})"]
        return loads, [js_scalar(op, inputs)]

    names = dict(zip(op.inputs, inputs, strict=True))
    nodes = op.fgraph.toposort()
    aliases = {}

    def resolve(variable):
        return aliases.get(variable, variable)

    for node in nodes:
        if not isinstance(node.op, Switch):
            continue
        condition, left, right = [resolve(arg) for arg in node.inputs]
        if switch_choices and condition in switch_choices:
            aliases[node.outputs[0]] = left if switch_choices[condition] else right
        elif isinstance(condition, Constant):
            aliases[node.outputs[0]] = left if bool(condition.data) else right
        elif left is right or (
            isinstance(left, Constant)
            and isinstance(right, Constant)
            and literal(left.data) == literal(right.data)
        ):
            aliases[node.outputs[0]] = left

    resolved_outputs = [resolve(output) for output in op.outputs]
    conditions = Counter(
        resolve(output.owner.inputs[0])
        for output in resolved_outputs
        if output.owner and isinstance(output.owner.op, Switch)
    )
    if conditions:
        condition, count = conditions.most_common(1)[0]
        cheap = all(
            variable.owner is None or variable.dtype == "bool"
            for variable in ancestors([condition])
        )
        if count >= 3 and condition.dtype == "bool" and cheap:

            def composite(outputs):
                return Composite(
                    op.inputs,
                    [
                        identity(output) if output in op.inputs else output
                        for output in outputs
                    ],
                )

            def leaves(outputs):
                visited = set()
                pending = list(outputs)
                result = set()
                while pending:
                    variable = resolve(pending.pop())
                    if variable in visited:
                        continue
                    visited.add(variable)
                    if variable.owner is None:
                        result.add(variable)
                    else:
                        pending.extend(variable.owner.inputs)
                return result

            branches = [
                [
                    switch_branch(output, truth)
                    if output.owner
                    and isinstance(output.owner.op, Switch)
                    and resolve(output.owner.inputs[0]) is condition
                    else output
                    for output in resolved_outputs
                ]
                for truth in (True, False)
            ]
            condition_inputs = leaves([condition])
            only_condition = condition_inputs - leaves(
                [output for branch in branches for output in branch]
            )
            condition_arguments = [
                input_loads[name]
                if variable in only_condition and input_loads and name in input_loads
                else name
                for variable, name in zip(op.inputs, inputs, strict=True)
            ]
            branch_inputs = [
                f"{_prefix}c_{name}"
                if variable in condition_inputs - only_condition
                and input_loads
                and name in input_loads
                else name
                for variable, name in zip(op.inputs, inputs, strict=True)
            ]
            statements, [predicate] = scalar_program(
                composite([condition]),
                condition_arguments,
                switch_choices,
                _prefix=f"{_prefix}c_",
                input_loads=input_loads,
            )
            names = [f"{_prefix}branch{index}" for index in range(len(op.outputs))]
            statements.append(f"let {', '.join(names)};")
            for truth, label, selected in zip(
                (True, False), ("t_", "f_"), branches, strict=True
            ):
                body, expressions = scalar_program(
                    composite(selected),
                    branch_inputs,
                    switch_choices,
                    _prefix=f"{_prefix}{label}",
                    input_loads=input_loads,
                )
                statements.append(f"if ({predicate}) {{" if truth else "} else {")
                statements.extend(f"    {statement}" for statement in body)
                statements.extend(
                    f"    {name} = {expression};"
                    for name, expression in zip(names, expressions, strict=True)
                )
            statements.append("}")
            return statements, names

    needed = set()
    pending = list(op.outputs)
    while pending:
        variable = resolve(pending.pop())
        node = variable.owner
        if node is not None and node not in needed:
            needed.add(node)
            pending.extend(node.inputs)

    softplus_bases = {}
    for node in needed:
        if isinstance(node.op, Softplus):
            argument = resolve(node.inputs[0])
            base = (
                resolve(argument.owner.inputs[0])
                if argument.owner and isinstance(argument.owner.op, Neg)
                else argument
            )
            softplus_bases[node] = base
    counts = Counter(softplus_bases.values())
    shared_softplus = {}

    def operand(variable):
        variable = resolve(variable)
        if variable not in names and isinstance(variable, Constant):
            names[variable] = literal(variable.data)
        return names[variable]

    statements = []
    used_inputs = {resolve(variable) for node in needed for variable in node.inputs}
    used_inputs.update(resolved_outputs)
    for variable, name in zip(op.inputs, inputs, strict=True):
        if variable in used_inputs and input_loads and name in input_loads:
            local = f"{_prefix}{name}"
            statements.append(f"const {local} = {input_loads[name]};")
            names[variable] = local
    uses = Counter(resolve(variable) for node in needed for variable in node.inputs)
    uses.update(resolve(variable) for variable in op.outputs)
    for index, node in enumerate(nodes):
        if node not in needed:
            continue
        if len(node.outputs) != 1:
            raise NotImplementedError("JS Composite scalar Ops with multiple outputs")
        name = f"{_prefix}q{index}"
        arguments = [operand(arg) for arg in node.inputs]
        base = softplus_bases.get(node)
        if base is not None and counts[base] > 1:
            if base not in shared_softplus:
                common = f"{_prefix}sp{len(shared_softplus)}"
                shared_softplus[base] = common
                statements.append(
                    f"const {common} = Math.log1p(Math.exp(-Math.abs({operand(base)})));"
                )
            expression = f"({shared_softplus[base]} + Math.max({arguments[0]}, 0))"
        elif isinstance(node.op, Switch) and node.inputs[0].dtype == "bool":
            expression = f"({arguments[0]} ? {arguments[1]} : {arguments[2]})"
        elif isinstance(node.op, OR) and node.outputs[0].dtype == "bool":
            expression = f"({arguments[0]} || {arguments[1]})"
        elif isinstance(node.op, Invert) and node.outputs[0].dtype == "bool":
            expression = f"(!{arguments[0]})"
        else:
            expression = js_scalar(node.op, arguments)
        if uses[node.outputs[0]] == 1:
            names[node.outputs[0]] = expression
        else:
            statements.append(f"const {name} = {expression};")
            names[node.outputs[0]] = name
    return statements, [operand(output) for output in op.outputs]
