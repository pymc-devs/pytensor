import warnings
from functools import singledispatch

# import jax
# import jax.numpy as jnp
import torch
import numpy as np

from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.ifelse import IfElse
from pytensor.link.utils import fgraph_to_python
from pytensor.raise_op import Assert, CheckAndRaise


# if config.floatX == "float64":
#     jax.config.update("jax_enable_x64", True)
# else:
#     jax.config.update("jax_enable_x64", False)


@singledispatch
def pytorch_typify(data, dtype=None, **kwargs):
    r"""Convert instances of PyTensor `Type`\s to PyTorch types."""
    if dtype is None:
        return data
    else:
        return torch.tensor(data, dtype=dtype)


@pytorch_typify.register(np.ndarray)
def pytorch_typify_ndarray(data, dtype=None, **kwargs):
    if len(data.shape) == 0:
        return data.item()
    return torch.tensor(data, dtype=dtype)


@singledispatch
def pytorch_funcify(op, node=None, storage_map=None, **kwargs):
    """Create a PyTorch compatible function from an PyTensor `Op`."""
    raise NotImplementedError(f"No PyTorch conversion for the given `Op`: {op}")


@pytorch_funcify.register(FunctionGraph)
def pytorch_funcify_FunctionGraph(
    fgraph,
    node=None,
    fgraph_name="pytorch_funcified_fgraph",
    **kwargs,
):
    return fgraph_to_python(
        fgraph,
        pytorch_funcify,
        type_conversion_fn=pytorch_typify,
        fgraph_name=fgraph_name,
        **kwargs,
    )


@pytorch_funcify.register(IfElse)
def pytorch_funcify_IfElse(op, **kwargs):
    n_outs = op.n_outs

    def ifelse(cond, *args, n_outs=n_outs):
        res = torch.where(
            cond, lambda _: args[:n_outs], lambda _: args[n_outs:], operand=None
        )
        return res if n_outs > 1 else res[0]

    return ifelse


@pytorch_funcify.register(Assert)
@pytorch_funcify.register(CheckAndRaise)
def pytorch_funcify_CheckAndRaise(op, **kwargs):
    warnings.warn(
        f"""Skipping `CheckAndRaise` Op (assertion: {op.msg}) as JAX tracing would remove it.""",
        stacklevel=2,
    )

    def assert_fn(x, *inputs):
        return x

    return assert_fn


def pytorch_safe_copy(x):
    try:
        res = x.clone().detach()
    except NotImplementedError:
        # warnings.warn(
        #     "`jnp.copy` is not implemented yet. Using the object's `copy` method."
        # )
        if hasattr(x, "copy"):
            res = torch.tensor(x.copy())
        else:
            warnings.warn(f"Object has no `copy` method: {x}")
            res = x

    return res


@pytorch_funcify.register(DeepCopyOp)
def pytorch_funcify_DeepCopyOp(op, **kwargs):
    def deepcopyop(x):
        return pytorch_safe_copy(x)

    return deepcopyop


@pytorch_funcify.register(ViewOp)
def pytorch_funcify_ViewOp(op, **kwargs):
    def viewop(x):
        return x

    return viewop
