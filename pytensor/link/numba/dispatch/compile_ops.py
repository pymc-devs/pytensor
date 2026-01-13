from copy import deepcopy
from hashlib import sha256
from textwrap import dedent

import numba
import numpy as np

from pytensor.compile.builders import OpFromGraph
from pytensor.compile.function.types import add_supervisor_to_fgraph, insert_deepcopy
from pytensor.compile.io import In, Out
from pytensor.compile.mode import NUMBA
from pytensor.compile.ops import DeepCopyOp, TypeCastingOp
from pytensor.ifelse import IfElse
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
def numba_funcify_OpFromGraph(op, node=None, **kwargs):
    _ = kwargs.pop("storage_map", None)

    # Apply inner rewrites
    # TODO: Not sure this is the right place to do this, should we have a rewrite that
    #  explicitly triggers the optimization of the inner graphs of OpFromGraph?
    #  The C-code defers it to the make_thunk phase
    fgraph = op.fgraph
    input_specs = [In(x, borrow=True, mutable=False) for x in fgraph.inputs]
    add_supervisor_to_fgraph(
        fgraph=fgraph,
        input_specs=input_specs,
        accept_inplace=True,
    )
    NUMBA.optimizer(fgraph)
    output_specs = [Out(o, borrow=False) for o in fgraph.outputs]
    insert_deepcopy(fgraph, wrapped_inputs=input_specs, wrapped_outputs=output_specs)
    fgraph_fn, fgraph_cache_key = numba_funcify_and_cache_key(
        fgraph, squeeze_output=True, fgraph_name="numba_ofg", **kwargs
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
