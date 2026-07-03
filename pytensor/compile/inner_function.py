"""Shared machinery for `Op`\\s that execute a lazily linked inner function."""

from typing import TYPE_CHECKING

from pytensor.compile.io import In, Out
from pytensor.compile.mode import Mode
from pytensor.graph.fg import FrozenFunctionGraph, FunctionGraph
from pytensor.graph.op import HasInnerGraph
from pytensor.graph.rewriting.basic import SequentialGraphRewriter
from pytensor.link.basic import Linker


if TYPE_CHECKING:
    from pytensor.compile.function.types import Function


def link_only_mode(linker: str | Linker) -> Mode:
    """A `Mode` that links a graph without rewriting it at all.

    The bare rewriter also bypasses the ``minimum_compile`` pass the linker
    would otherwise force onto a database query.
    """
    return Mode(linker, SequentialGraphRewriter())


class HasInnerFunction(HasInnerGraph):
    """`HasInnerGraph` op whose ``perform`` runs a lazily linked inner function.

    The frozen inner graph was already baked for the backend by the
    ``compile_inner_graph`` rewrites during the outer compile, so linking it
    needs no further rewrites (see `link_only_mode`).

    The linker never comes from the config default mode: ``perform`` only runs
    under the py/c backend family -- JIT backends (numba/jax/...) funcify
    ``op.fgraph`` directly and never call ``perform`` -- so a JIT default must
    not win, and the JIT inner-graph rewrites were never applied to this graph.
    """

    _fn = None
    fgraph: FrozenFunctionGraph

    def link_mode(self, impl: str | None) -> Mode:
        """The `Mode` to link the inner function with, given a thunk ``impl``."""
        return link_only_mode("cvm" if impl == "c" else "vm")

    def link_fgraph(self, fgraph: FunctionGraph, mode: Mode) -> "Function":
        """Link an already-baked inner ``fgraph`` under ``mode``, no rewrites."""
        fn = mode.function_maker(
            [In(inp) for inp in fgraph.inputs],
            [Out(out) for out in fgraph.outputs],
            mode,
            fgraph=fgraph,
            accept_inplace=True,
            on_unused_input="ignore",
        ).create()
        fn.trust_input = True
        return fn

    def compile_fn(self, mode: Mode) -> "Function":
        """Build the inner function under ``mode`` (override to massage the graph)."""
        return self.link_fgraph(self.fgraph.unfreeze(), mode)

    @property
    def fn(self) -> "Function":
        if self._fn is None:
            self._fn = self.compile_fn(self.link_mode(None))
        return self._fn

    def make_thunk(self, node, storage_map, compute_map, no_recycling, impl=None):
        if self._fn is None:
            self._fn = self.compile_fn(self.link_mode(impl))
        return super().make_thunk(
            node, storage_map, compute_map, no_recycling, impl=impl
        )

    @property
    def inner_inputs(self):
        # Read-only views of the immutable inner graph, as lists so callers
        # that concatenate inputs/outputs keep list semantics.
        return list(self.fgraph.inputs)

    @property
    def inner_outputs(self):
        return list(self.fgraph.outputs)
