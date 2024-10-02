import torch
import torch.compiler

from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.scalar.basic import ScalarOp
from pytensor.scalar.loop import ScalarLoop


@pytorch_funcify.register(ScalarOp)
def pytorch_funcify_ScalarOp(op, node, **kwargs):
    """Return pytorch function that implements the same computation as the Scalar Op.

    This dispatch is expected to return a pytorch function that works on Array inputs as Elemwise does,
    even though it's dispatched on the Scalar Op.
    """

    nfunc_spec = getattr(op, "nfunc_spec", None)
    if nfunc_spec is None:
        raise NotImplementedError(f"Dispatch not implemented for Scalar Op {op}")

    func_name = nfunc_spec[0]

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


@pytorch_funcify.register(ScalarLoop)
def pytorch_funicify_ScalarLoop(op, node, **kwargs):
    update = pytorch_funcify(op.fgraph)
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
            if len(node.outputs) == 2:
                return carry[0], done
            else:
                return carry, done
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

    return torch.compiler.disable(scalar_loop, recursive=False)
