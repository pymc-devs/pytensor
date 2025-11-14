from hashlib import sha256

import numpy as np

from pytensor.compile.builders import OpFromGraph
from pytensor.compile.function.types import add_supervisor_to_fgraph
from pytensor.compile.io import In
from pytensor.compile.mode import NUMBA
from pytensor.compile.ops import DeepCopyOp, TypeCastingOp
from pytensor.ifelse import IfElse
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    numba_funcify_and_cache_key,
    register_funcify_and_cache_key,
    register_funcify_default_op_cache_key,
)
from pytensor.raise_op import CheckAndRaise
from pytensor.tensor.type import TensorType


@register_funcify_and_cache_key(OpFromGraph)
def numba_funcify_OpFromGraph(op, node=None, **kwargs):
    _ = kwargs.pop("storage_map", None)

    # Apply inner rewrites
    # TODO: Not sure this is the right place to do this, should we have a rewrite that
    #  explicitly triggers the optimization of the inner graphs of OpFromGraph?
    #  The C-code defers it to the make_thunk phase
    fgraph = op.fgraph
    add_supervisor_to_fgraph(
        fgraph=fgraph,
        input_specs=[In(x, borrow=True, mutable=False) for x in fgraph.inputs],
        accept_inplace=True,
    )
    NUMBA.optimizer(fgraph)
    fgraph_fn, fgraph_cache_key = numba_funcify_and_cache_key(
        op.fgraph, squeeze_output=True, **kwargs
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
    if isinstance(node.inputs[0].type, TensorType):

        @numba_basic.numba_njit
        def deepcopy(x):
            return np.copy(x)

    else:

        @numba_basic.numba_njit
        def deepcopy(x):
            return x

    return deepcopy


@register_funcify_default_op_cache_key(IfElse)
def numba_funcify_IfElse(op, **kwargs):
    n_outs = op.n_outs

    if n_outs > 1:

        @numba_basic.numba_njit
        def ifelse(cond, *args):
            if cond:
                res = args[:n_outs]
            else:
                res = args[n_outs:]

            return res

    else:

        @numba_basic.numba_njit
        def ifelse(cond, *args):
            if cond:
                res = args[:n_outs]
            else:
                res = args[n_outs:]

            return res[0]

    return ifelse


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
