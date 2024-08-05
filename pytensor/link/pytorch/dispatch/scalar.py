import torch

from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.scalar.basic import (
    ScalarOp,
)


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
