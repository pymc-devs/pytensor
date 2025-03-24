from functools import singledispatch

import numpy as np
from egglog import Expr, convert

import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor.graph import Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.sandbox.scrambled.basic import Int, IntTuple, ScalarOp, Tensor, egraph
from pytensor.sandbox.scrambled.op import (
    Elemwise,
    ExpandDims,
    Reduce,
    ScalarAdd,
    ScalarMul,
    ScalarSub,
    Squeeze,
)
from pytensor.sandbox.scrambled.rewrites.tensorify import CACHE_VARS


@singledispatch
def eggify(typ, *args, **kwargs) -> Expr:
    raise NotImplementedError(f"No egg dispatch for type {typ}")


@eggify.register
def _(op: ps.ScalarOp, *args, **kwargs):
    egraph.constant(str(op), ScalarOp)


@eggify.register
def _(op: ps.Add):
    return ScalarAdd


@eggify.register
def _(op: ps.Mul):
    return ScalarMul


@eggify.register
def _(op: ps.Sub):
    return ScalarSub


@eggify.register
def _(op: pt.elemwise.CAReduce, node, x):
    scalar_op = eggify(op.scalar_op)
    return Reduce(scalar_op, axis=tuple(op.axis))(x)


@eggify.register
def _(op: pt.elemwise.Elemwise, node, *inputs):
    scalar_op = eggify(op.scalar_op)
    return Elemwise(scalar_op)(inputs)


@eggify.register
def _(op: pt.elemwise.DimShuffle, node, x):
    if op.transposition:
        raise NotImplementedError("Dimshuffle with transposition not implemented")
    if op.drop and op.augment:
        raise NotImplementedError(
            "Dimshuffle with both augmentation and drop not implemented"
        )
    if op.augment:
        return ExpandDims(axis=tuple(op.augment))(x)
    if op.drop:
        return Squeeze(axis=tuple(op.dro))(x)


def eggify_constant(var):
    data = np.asarray(var.data)
    if np.asarray(data).dtype != "int64":
        raise NotImplementedError("Only int64 constants allowed")
    first_item = data.item(0)
    if data.ndim > 0:
        if not np.allequal(first_item, data):
            raise NotImplementedError("Only homogenous constants allowed")

    shape = convert(var.type.shape, IntTuple)
    return Tensor.constant(first_item, shape=shape)


def eggify_variable(var):
    if var.type.dtype != "int64":
        raise NotImplementedError(
            f"Only int64 variables are allowed: {var}, {var.type}"
        )

    if var.name:
        var_name = var.auto_name.replace("auto", var.name)
    else:
        var_name = var.auto_name.replace("auto_", "var")

    shape = convert(
        tuple(
            Int.var(f"{var_name}_dim_{i}") if dim is None else Int(dim)
            for i, dim in enumerate(var.type.shape)
        ),
        IntTuple,
    )

    CACHE_VARS[(var_name, var.type.shape)] = var
    return Tensor(var_name, shape)


def eggify_fg(fg: FunctionGraph) -> Expr:
    mapping = {var: eggify_variable(var) for var in fg.inputs}
    for node in fg.toposort():
        node_inputs = []
        for var in node.inputs:
            inp = mapping.get(var, None)
            if inp is None:
                assert isinstance(var, Constant), var
                inp = eggify_constant(var)
                mapping[var] = inp
            node_inputs.append(inp)
        [out] = node.outputs
        mapping[out] = eggify(node.op, node, *node_inputs)
    return [mapping[out] for out in fg.outputs]
