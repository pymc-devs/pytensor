from typing import Union

from pytensor.compile import optdb
from pytensor.graph.rewriting.basic import NodeRewriter
from pytensor.graph.rewriting.db import RewriteDatabase, EquilibriumDB


optdb.register(
    "xcanonicalize",
    EquilibriumDB(ignore_newtrees=False),
    "fast_run",
    "fast_compile",
    "xtensor",
    position=0,
)


def register_xcanonicalize(
    node_rewriter: Union[RewriteDatabase, NodeRewriter, str], *tags: str, **kwargs
):
    if isinstance(node_rewriter, str):

        def register(inner_rewriter: Union[RewriteDatabase, NodeRewriter]):
            return register_xcanonicalize(inner_rewriter, node_rewriter, *tags, **kwargs)

        return register

    else:
        name = kwargs.pop("name", None) or node_rewriter.__name__
        optdb["xtensor"].register(
            name, node_rewriter, "fast_run", "fast_compile", *tags, **kwargs
        )
        return node_rewriter
