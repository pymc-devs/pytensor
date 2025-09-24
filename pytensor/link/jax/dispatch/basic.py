import warnings
from collections.abc import Callable
from functools import singledispatch

import jax
import jax.numpy as jnp
import numpy as np

from pytensor.compile import JAX
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.ops import DeepCopyOp, TypeCastingOp
from pytensor.configdefaults import config
from pytensor.graph import Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.ifelse import IfElse
from pytensor.link.jax.ops import JAXOp
from pytensor.link.utils import fgraph_to_python
from pytensor.raise_op import CheckAndRaise


if config.floatX == "float64":
    jax.config.update("jax_enable_x64", True)
else:
    jax.config.update("jax_enable_x64", False)


@singledispatch
def jax_typify(data, dtype=None, **kwargs):
    r"""Convert instances of PyTensor `Type`\s to JAX types."""
    if dtype is None:
        return data
    else:
        return jnp.array(data, dtype=dtype)


@jax_typify.register(np.ndarray)
def jax_typify_ndarray(data, dtype=None, **kwargs):
    if len(data.shape) == 0:
        return data.item()
    return jnp.array(data, dtype=dtype)


@singledispatch
def jax_funcify(op, node=None, storage_map=None, **kwargs):
    """Create a JAX compatible function from an PyTensor `Op`."""
    raise NotImplementedError(f"No JAX conversion for the given `Op`: {op}")


@jax_funcify.register(FunctionGraph)
def jax_funcify_FunctionGraph(
    fgraph,
    node=None,
    fgraph_name="jax_funcified_fgraph",
    **kwargs,
):
    return fgraph_to_python(
        fgraph,
        jax_funcify,
        type_conversion_fn=jax_typify,
        fgraph_name=fgraph_name,
        **kwargs,
    )


@jax_funcify.register(IfElse)
def jax_funcify_IfElse(op, **kwargs):
    n_outs = op.n_outs

    def ifelse(cond, *args, n_outs=n_outs):
        res = jax.lax.cond(
            cond, lambda _: args[:n_outs], lambda _: args[n_outs:], operand=None
        )
        return res if n_outs > 1 else res[0]

    return ifelse


@jax_funcify.register(CheckAndRaise)
def jax_funcify_CheckAndRaise(op, node, **kwargs):
    conds = node.inputs[1:]
    if any(isinstance(cond, Constant) and not bool(cond.data) for cond in conds):
        raise op.exc_type(op.msg)

    warnings.warn(
        f"""Skipping {op} Op (assertion: {op.msg}) as JAX tracing would remove it.""",
        stacklevel=2,
    )

    def assert_fn(x, *inputs):
        return x

    return assert_fn


def jnp_safe_copy(x):
    try:
        res = jnp.copy(x)
    except NotImplementedError:
        warnings.warn(
            "`jnp.copy` is not implemented yet. Using the object's `copy` method."
        )
        if hasattr(x, "copy"):
            res = jnp.array(x.copy())
        else:
            warnings.warn(f"Object has no `copy` method: {x}")
            res = x

    return res


@jax_funcify.register(DeepCopyOp)
def jax_funcify_DeepCopyOp(op, **kwargs):
    def deepcopyop(x):
        return jnp_safe_copy(x)

    return deepcopyop


@jax_funcify.register(TypeCastingOp)
def jax_funcify_TypeCastingOp(op, **kwargs):
    def type_cast(x):
        return x

    return type_cast


@jax_funcify.register(OpFromGraph)
def jax_funcify_OpFromGraph(ofg: OpFromGraph, node=None, **kwargs) -> Callable:
    _ = kwargs.pop("storage_map", None)

    # Apply inner rewrites
    JAX.optimizer(ofg.fgraph)
    fgraph_fn = jax_funcify(ofg.fgraph, **kwargs)

    if len(ofg.fgraph.outputs) == 1:

        def opfromgraph(*inputs):
            return fgraph_fn(*inputs)[0]

    else:

        def opfromgraph(*inputs):
            return fgraph_fn(*inputs)

    return opfromgraph


@jax_funcify.register(JAXOp)
def jax_op_funcify(op, **kwargs):
    return op.perform_jax
