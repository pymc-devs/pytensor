"""Shared machinery for `Op`\\s that execute a lazily linked inner function."""

from copy import copy
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
    # Backend whose python fallback runs ``perform`` (e.g. numba objmode), set on
    # a private copy by that backend so the inner function is linked with it.
    _fallback_impl: str | None = None
    fgraph: FrozenFunctionGraph

    def link_mode(self, impl: str | None) -> Mode:
        """The `Mode` to link the inner function with, given a thunk ``impl``."""
        if impl in (None, "py"):
            return link_only_mode("vm")
        if impl == "c":
            return link_only_mode("cvm")
        # A JIT backend running ``perform`` through its python fallback
        # (e.g. numba objmode): link the inner function with that backend.
        return link_only_mode(impl)

    def _reset_inner_fn(self) -> None:
        """Drop cached inner functions. Subclasses with extra caches extend this."""
        self._fn = None

    def with_inner_fn_for(self, impl: str) -> "HasInnerFunction":
        """A copy of this op whose inner function will be linked for ``impl``.

        Nothing is compiled here: the inner function is still linked lazily on
        first use. Works on a copy so a shared op instance never carries a
        backend-specific inner function into other compiles.
        """
        new_op = copy(self)
        new_op._reset_inner_fn()
        new_op._fallback_impl = impl
        return new_op

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
            self._fn = self.compile_fn(self.link_mode(self._fallback_impl))
        return self._fn

    def make_thunk(self, node, storage_map, compute_map, no_recycling, impl=None):
        if self._fn is None:
            self._fn = self.compile_fn(self.link_mode(impl))
        return super().make_thunk(
            node, storage_map, compute_map, no_recycling, impl=impl
        )
