from copy import deepcopy
from hashlib import sha256
from textwrap import dedent

import numba
import numpy as np

from pytensor.compile.builders import OpFromGraph
from pytensor.compile.ops import DeepCopyOp, TypeCastingOp
from pytensor.ifelse import IfElse, LazyIfElse, _output_may_alias
from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    numba_funcify_and_cache_key,
    register_funcify_and_cache_key,
    register_funcify_default_op_cache_key,
)
from pytensor.raise_op import CheckAndRaise


def numba_deepcopy(x):
    return deepcopy(x)


@numba.extending.overload(numba_deepcopy)
def numba_deepcopy_tensor(x):
    if isinstance(x, numba.types.Number | numba.types.Boolean):

        def number_deepcopy(x):
            return x

        return number_deepcopy

    if isinstance(x, numba.types.Array):

        def array_deepcopy(x):
            return np.copy(x)

        return array_deepcopy

    if isinstance(x, numba.types.UnicodeType):

        def string_deepcopy(x):
            return x

        return string_deepcopy


@register_funcify_and_cache_key(OpFromGraph)
def numba_funcify_OpFromGraph(
    op,
    node=None,
    ofg_memo=None,
    **kwargs,
):
    _ = kwargs.pop("storage_map", None)

    if ofg_memo is not None and op in ofg_memo:
        return ofg_memo[op]

    # The inner graph is already optimized and inplace/deepcopy-baked by the
    # ``ofg_inner_graph`` rewrite (numba contract in ``pytensor.tensor.rewriting``),
    # so we funcify ``op.fgraph`` directly.
    fgraph_fn, fgraph_cache_key = numba_funcify_and_cache_key(
        op.fgraph,
        squeeze_output=True,
        fgraph_name="numba_ofg",
        ofg_memo=ofg_memo,
        **kwargs,
    )

    if fgraph_cache_key is None:
        # Can't cache the inner graph
        ofg_cache_key = None
    else:
        ofg_cache_key = sha256(
            str(
                (
                    type(op),
                    fgraph_cache_key,
                )
            ).encode()
        ).hexdigest()

    if ofg_memo is not None:
        ofg_memo[op] = (fgraph_fn, ofg_cache_key)

    return fgraph_fn, ofg_cache_key


@register_funcify_default_op_cache_key(TypeCastingOp)
def numba_funcify_type_casting(op, **kwargs):
    @numba_basic.numba_njit
    def identity(x):
        return x

    return identity


@register_funcify_default_op_cache_key(DeepCopyOp)
def numba_funcify_DeepCopyOp(op, node, **kwargs):
    @numba_basic.numba_njit
    def deepcopy(x):
        return numba_deepcopy(x)

    return deepcopy, 1


@register_funcify_default_op_cache_key(IfElse)
def numba_funcify_IfElse(op, **kwargs):
    n_outs = op.n_outs
    as_view = op.as_view

    true_names = [f"t{i}" for i in range(n_outs)]
    false_names = [f"f{i}" for i in range(n_outs)]
    arg_list = ", ".join((*true_names, *false_names))

    if as_view:
        true_returns = ", ".join(true_names)
    else:
        true_returns = ", ".join(f"{name}.copy()" for name in true_names)
    # We only ever view (alias) variables from the true branch. False branch variables must always be copied.
    false_returns = ", ".join(f"{name}.copy()" for name in false_names)

    func_src = dedent(
        f"""
            def ifelse(cond, {arg_list}):
                if cond:
                    return {true_returns}
                else:
                    return {false_returns}
        """
    )

    ifelse_func = numba_basic.numba_njit(
        compile_numba_function_src(func_src, "ifelse", globals())
    )

    cache_version = 1
    return ifelse_func, cache_version


@register_funcify_and_cache_key(LazyIfElse)
def numba_funcify_LazyIfElse(op, node=None, **kwargs):
    _ = kwargs.pop("storage_map", None)

    then_fn, then_key = numba_funcify_and_cache_key(
        op.then_fgraph, fgraph_name="numba_lazy_ifelse_then", **kwargs
    )
    else_fn, else_key = numba_funcify_and_cache_key(
        op.else_fgraph, fgraph_name="numba_lazy_ifelse_else", **kwargs
    )

    n_outs = len(op.output_types)
    then_args = [f"t{i}" for i in range(len(op.then_fgraph.inputs))]
    else_args = [f"f{i}" for i in range(len(op.else_fgraph.inputs))]
    signature = ", ".join(("cond", *then_args, *else_args))

    def branch_returns(fgraph):
        # A branch result that aliases a branch input, a constant, or another
        # output must be copied; fresh arrays are returned as they are.
        results = (
            f"numba_deepcopy(res[{i}])"
            if _output_may_alias(out, fgraph)
            else f"res[{i}]"
            for i, out in enumerate(fgraph.outputs)
        )
        return ", ".join(results)

    func_src = dedent(
        f"""
            def lazy_ifelse({signature}):
                if cond:
                    res = then_fn({", ".join(then_args)})
                    return {branch_returns(op.then_fgraph)}
                else:
                    res = else_fn({", ".join(else_args)})
                    return {branch_returns(op.else_fgraph)}
        """
    )

    lazy_ifelse_fn = numba_basic.numba_njit(
        compile_numba_function_src(
            func_src,
            "lazy_ifelse",
            globals() | {"then_fn": then_fn, "else_fn": else_fn},
        )
    )

    if then_key is None or else_key is None:
        cache_key = None
    else:
        cache_key = sha256(
            str((type(op), n_outs, then_key, else_key, func_src)).encode()
        ).hexdigest()

    return lazy_ifelse_fn, cache_key


@register_funcify_and_cache_key(CheckAndRaise)
def numba_funcify_CheckAndRaise(op, node, **kwargs):
    error = op.exc_type
    msg = op.msg

    @numba_basic.numba_njit
    def check_and_raise(x, *conditions):
        for cond in conditions:
            if not cond:
                raise error(msg)
        return x

    cache_key = sha256(str((type(op), error, msg)).encode()).hexdigest()
    return check_and_raise, cache_key
