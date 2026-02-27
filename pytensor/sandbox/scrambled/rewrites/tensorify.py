from typing import Any

import numpy as np
from egglog import PyObject, String, eq, i64, py_eval_fn, rule, set_
from frozendict import frozendict

import pytensor.tensor as pt
from pytensor.graph import Apply, Variable
from pytensor.sandbox.scrambled.basic import (
    Int,
    IntTuple,
    Tensor,
    TensorTuple,
    egraph,
    tensorify_ruleset,
)
from pytensor.sandbox.scrambled.op import (
    Alloc,
    Elemwise,
    ExpandDims,
    Reduce,
    ScalarAdd,
    ScalarMul,
    ScalarSub,
    Shape,
    Shape_i,
    Squeeze,
)
from pytensor.tensor.shape import shape_i


CACHE_VARS: dict[Any, Variable] = {}
CACHE_APPLY_NODES: dict[Any, Apply] = {}


def cached_var(name, shape):
    hash = (name, shape)
    var = CACHE_VARS.get(hash, None)
    if var is None:
        var = pt.tensor(name, shape=shape)
        CACHE_VARS[hash] = var
    return var


def cached_consts(value, shape):
    hash = (value, shape)
    var = CACHE_VARS.get(hash, None)
    if var is None:
        var = pt.constant(np.broadcast_to(np.array(value, dtype="int64"), shape=shape))
        CACHE_VARS[hash] = var
    return var


def cached_apply(op, *args, **kwargs):
    node = op(*args, **kwargs).owner
    hash = (op, args, frozendict(kwargs))
    out = CACHE_APPLY_NODES.get(hash, None)
    if out is None:
        out = node.outputs
        if len(out) == 1:
            [out] = out
        CACHE_APPLY_NODES[hash] = out
    return out


@egraph.register
def tensorify_int_tuples(
    i: Int,
    i64_: i64,
    name: String,
    int_tuple: IntTuple,
    int_tuple1: IntTuple,
    int_tuple2: IntTuple,
):
    yield rule(
        eq(i).to(Int(i64_)),
        ruleset=tensorify_ruleset,
    ).then(set_(i.tensorify).to(PyObject.from_int(i64_)))

    # TODO: Convert variable Int to None only if in the shape of a Tensor!
    yield rule(
        eq(i).to(Int.var(name)),
        ruleset=tensorify_ruleset,
    ).then(set_(i.tensorify).to(PyObject(None)))

    yield rule(
        eq(int_tuple).to(IntTuple.empty()),
        ruleset=tensorify_ruleset,
    ).then(set_(int_tuple.tensorify).to(PyObject(tuple())))

    yield rule(
        eq(int_tuple).to(IntTuple(i)),
        i.tensorify,
        ruleset=tensorify_ruleset,
    ).then(set_(int_tuple.tensorify).to(py_eval_fn(lambda x: (x,))(i.tensorify)))

    yield rule(
        eq(int_tuple).to(int_tuple1 + int_tuple2),
        int_tuple1.tensorify,
        int_tuple2.tensorify,
        ruleset=tensorify_ruleset,
    ).then(
        set_(int_tuple.tensorify).to(
            py_eval_fn(lambda x, y: x + y)(int_tuple1.tensorify, int_tuple2.tensorify)
        )
    )


@egraph.register
def tensorify_tensor_tuples(
    tensor_tuple: TensorTuple,
    tensor_tuple1: TensorTuple,
    tensor_tuple2: TensorTuple,
    x: Tensor,
):
    yield rule(
        eq(tensor_tuple).to(TensorTuple(x)),
        x.tensorify,
        ruleset=tensorify_ruleset,
    ).then(set_(tensor_tuple.tensorify).to(py_eval_fn(lambda x: (x,))(x.tensorify)))

    yield rule(
        eq(tensor_tuple).to(tensor_tuple1 + tensor_tuple2),
        tensor_tuple1.tensorify,
        tensor_tuple2.tensorify,
        ruleset=tensorify_ruleset,
    ).then(
        set_(tensor_tuple.tensorify).to(
            py_eval_fn(lambda x, y: x + y)(
                tensor_tuple1.tensorify, tensor_tuple2.tensorify
            )
        )
    )


@egraph.register
def tensorify_tensor(
    x: Tensor,
    name: String,
    static_sh: IntTuple,
    i: Int,
):
    yield rule(
        eq(x).to(Tensor(name, static_sh)),
        # Todo: Shape.replace(Int.var("x"), Int.None())
        static_sh.tensorify,
        ruleset=tensorify_ruleset,
    ).then(
        set_(x.tensorify).to(
            py_eval_fn(cached_var)(PyObject.from_string(name), static_sh.tensorify)
        )
    )

    yield rule(
        eq(x).to(Tensor.constant(i, static_sh)),
        i.tensorify,
        static_sh.tensorify,
        ruleset=tensorify_ruleset,
    ).then(
        set_(x.tensorify).to(
            py_eval_fn(cached_consts)(i.tensorify, static_sh.tensorify)
        )
    )


@egraph.register
def tensorify_ops(
    out: Tensor,
    x: Tensor,
    axis: IntTuple,
    sh: TensorTuple,
    inputs: TensorTuple,
    i: Int,
):
    yield rule(
        eq(out).to(ExpandDims(axis)(x)),
        axis.tensorify,
        x.tensorify,
        ruleset=tensorify_ruleset,
    ).then(
        set_(out.tensorify).to(
            py_eval_fn(lambda axis, x: cached_apply(pt.expand_dims, x, axis=axis))(
                axis.tensorify, x.tensorify
            )
        )
    )

    yield rule(
        eq(out).to(Squeeze(axis)(x)),
        axis.tensorify,
        x.tensorify,
        ruleset=tensorify_ruleset,
    ).then(
        set_(out.tensorify).to(
            py_eval_fn(lambda axis, x: cached_apply(pt.squeeze, x, axis=axis))(
                axis.tensorify, x.tensorify
            )
        )
    )

    yield rule(
        eq(out).to(Alloc(x, sh)),
        x.tensorify,
        sh.tensorify,
        ruleset=tensorify_ruleset,
    ).then(
        set_(out.tensorify).to(
            py_eval_fn(lambda x, shape: cached_apply(pt.alloc, x, *shape))(
                x.tensorify, sh.tensorify
            )
        )
    )

    for scalar_op, reduce_pytensor_fn in (
        (ScalarAdd, pt.sum),
        (ScalarMul, pt.prod),
    ):
        yield rule(
            eq(out).to(Reduce(scalar_op, axis)(x)),
            axis.tensorify,
            x.tensorify,
            ruleset=tensorify_ruleset,
        ).then(
            set_(out.tensorify).to(
                py_eval_fn(
                    lambda axis, x, fn=reduce_pytensor_fn: cached_apply(
                        fn, x, axis=axis
                    )
                )(axis.tensorify, x.tensorify)
            )
        )

    for scalar_op, elemwise_pytensor_fn in (
        (ScalarAdd, pt.add),
        (ScalarSub, pt.sub),
        (ScalarMul, pt.mul),
    ):
        yield rule(
            eq(out).to(Elemwise(scalar_op)(inputs)),
            inputs.tensorify,
            ruleset=tensorify_ruleset,
        ).then(
            set_(out.tensorify).to(
                py_eval_fn(
                    lambda inputs, fn=elemwise_pytensor_fn: cached_apply(fn, *inputs)
                )(inputs.tensorify)
            )
        )

    # yield rule(
    #     eq(out_tuple).to(Shape(x)),
    #     x.tensorify,
    #     ruleset=tensorify_ruleset,
    # ).then(
    #     set_(out_tuple.tensorify).to(
    #         py_eval_fn(lambda x: cached_apply(pt.shape(x)))(x.tensorify)
    #     )
    # )

    yield rule(
        eq(out).to(Shape_i(x, i)),
        x.tensorify,
        i.tensorify,
        ruleset=tensorify_ruleset,
    ).then(
        set_(out.tensorify).to(
            py_eval_fn(lambda x, i: cached_apply(shape_i, x, i))(
                x.tensorify, i.tensorify
            )
        )
    )
