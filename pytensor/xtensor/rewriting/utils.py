import typing
from collections.abc import Sequence

from pytensor.compile import optdb
from pytensor.graph.rewriting.basic import NodeRewriter, dfs_rewriter
from pytensor.graph.rewriting.db import EquilibriumDB, RewriteDatabase
from pytensor.tensor.rewriting.ofg import inline_ofg_expansion
from pytensor.tensor.variable import TensorVariable
from pytensor.xtensor.type import XTensorVariable


lower_xtensor_db = EquilibriumDB(ignore_newtrees=False)

optdb.register(
    "lower_xtensor",
    lower_xtensor_db,
    "fast_run",
    "fast_compile",
    "minimum_compile",
    position=0.1,
)

# Register OFG inline again after lowering xtensor
optdb.register(
    "inline_ofg_expansion_xtensor",
    dfs_rewriter(inline_ofg_expansion),
    "fast_run",
    "fast_compile",
    position=0.11,
)


def register_lower_xtensor(
    node_rewriter: RewriteDatabase | NodeRewriter | str, *tags: str, **kwargs
):
    if isinstance(node_rewriter, str):

        def register(inner_rewriter: RewriteDatabase | NodeRewriter):
            return register_lower_xtensor(
                inner_rewriter, node_rewriter, *tags, **kwargs
            )

        return register

    else:
        name = kwargs.pop("name", None) or node_rewriter.__name__  # type: ignore
        lower_xtensor_db.register(
            name,
            node_rewriter,
            "fast_run",
            "fast_compile",
            "minimum_compile",
            *tags,
            **kwargs,
        )
        return node_rewriter


def lower_aligned(x: XTensorVariable, out_dims: Sequence[str]) -> TensorVariable:
    """Lower an XTensorVariable to a TensorVariable so that it's dimensions are aligned with "out_dims"."""
    inp_dims = {d: i for i, d in enumerate(x.type.dims)}
    ds_order = tuple(inp_dims.get(dim, "x") for dim in out_dims)
    return typing.cast(TensorVariable, x.values.dimshuffle(ds_order))
