import importlib

import torch

from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.scalar.basic import (
    Cast,
    Clip,
    Invert,
    ScalarOp,
)
from pytensor.scalar.loop import ScalarLoop
from pytensor.scalar.math import Softplus


@pytorch_funcify.register(Invert)
def pytorch_funcify_invert(op, node, **kwargs):
    return torch.bitwise_not


@pytorch_funcify.register(ScalarOp)
def pytorch_funcify_ScalarOp(op, node, **kwargs):
    """Return pytorch function that implements the same computation as the Scalar Op.

    This dispatch is expected to return a pytorch function that works on Array inputs as Elemwise does,
    even though it's dispatched on the Scalar Op.
    """

    nfunc_spec = getattr(op, "nfunc_spec", None)
    if nfunc_spec is None:
        raise NotImplementedError(f"Dispatch not implemented for Scalar Op {op}")

    func_name = nfunc_spec[0].replace("scipy.", "")

    if "." in func_name:
        loc = func_name.split(".")
        mod = importlib.import_module(".".join(["torch", *loc[:-1]]))
        pytorch_func = getattr(mod, loc[-1])
    else:
        pytorch_func = getattr(torch, func_name)

    if len(node.inputs) > op.nfunc_spec[1]:
        # Some Scalar Ops accept multiple number of inputs, behaving as a variadic function,
        # even though the base Op from `func_name` is specified as a binary Op.
        # This happens with `Add`, which can work as a `Sum` for multiple scalars.
        pytorch_variadic_func = getattr(torch, op.nfunc_variadic, None)
        if not pytorch_variadic_func:
            raise NotImplementedError(
                f"Dispatch not implemented for Scalar Op {op} with {len(node.inputs)} inputs"
            )

        def pytorch_func(*args):
            return pytorch_variadic_func(
                torch.stack(torch.broadcast_tensors(*args), axis=0), axis=0
            )

    return pytorch_func


@pytorch_funcify.register(Cast)
def pytorch_funcify_Cast(op: Cast, node, **kwargs):
    dtype = getattr(torch, op.o_type.dtype)

    def cast(x):
        return x.to(dtype=dtype)

    return cast


@pytorch_funcify.register(Softplus)
def pytorch_funcify_Softplus(op, node, **kwargs):
    return torch.nn.Softplus()


@pytorch_funcify.register(Clip)
def pytorch_funcify_Clip(op, node, **kwargs):
    def clip(x, min_val, max_val):
        return torch.where(x < min_val, min_val, torch.where(x > max_val, max_val, x))

    return clip


@pytorch_funcify.register(ScalarLoop)
def pytorch_funicify_ScalarLoop(op, node, **kwargs):
    update = pytorch_funcify(op.fgraph, **kwargs)
    state_length = op.nout
    if op.is_while:

        def scalar_loop(steps, *start_and_constants):
            carry, constants = (
                start_and_constants[:state_length],
                start_and_constants[state_length:],
            )
            done = True
            for _ in range(steps):
                *carry, done = update(*carry, *constants)
                if torch.any(done):
                    break
            return *carry, done
    else:

        def scalar_loop(steps, *start_and_constants):
            carry, constants = (
                start_and_constants[:state_length],
                start_and_constants[state_length:],
            )
            for _ in range(steps):
                carry = update(*carry, *constants)
            if len(node.outputs) == 1:
                return carry[0]
            else:
                return carry

    return scalar_loop
