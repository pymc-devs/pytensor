import numpy as np

from pytensor.compile.builders import OpFromGraph
from pytensor.compile.function.types import add_supervisor_to_fgraph
from pytensor.compile.io import In
from pytensor.compile.mode import NUMBA
from pytensor.compile.ops import DeepCopyOp, TypeCastingOp
from pytensor.ifelse import IfElse
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    numba_funcify,
    numba_njit,
)
from pytensor.raise_op import CheckAndRaise
from pytensor.tensor.type import TensorType


@numba_funcify.register(OpFromGraph)
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
    return numba_funcify(op.fgraph, squeeze_output=True, **kwargs)


@numba_funcify.register(TypeCastingOp)
def numba_funcify_type_casting(op, **kwargs):
    @numba_basic.numba_njit
    def identity(x):
        return x

    return identity


@numba_funcify.register(DeepCopyOp)
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


@numba_funcify.register(IfElse)
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


@numba_funcify.register(CheckAndRaise)
def numba_funcify_CheckAndRaise(op, node, **kwargs):
    error = op.exc_type
    msg = op.msg

    @numba_basic.numba_njit
    def check_and_raise(x, *conditions):
        for cond in conditions:
            if not cond:
                raise error(msg)
        return x

    return check_and_raise
