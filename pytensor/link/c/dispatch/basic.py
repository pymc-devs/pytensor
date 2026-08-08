import warnings
from collections.abc import Collection
from functools import singledispatch
from typing import NoReturn

from pytensor.graph.basic import Apply, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import ComputeMapType, Op, StorageMapType, ThunkType
from pytensor.graph.utils import MethodNotDefined
from pytensor.link.c.interface import CLinkerOp
from pytensor.link.c.op import (
    COp,
    CThunkWrapperType,
    is_cthunk_wrapper_type,
)


@singledispatch
def c_funcify(op: Op, node: Apply | None = None, **kwargs) -> CLinkerOp:
    """Return the C implementation of `op` at `node`.

    By default an op implementing `CLinkerOp` (every `COp`) is its own
    implementation; otherwise raise `NotImplementedError` and let the caller fall
    back to the Python thunk.
    """
    if isinstance(op, CLinkerOp):
        return op
    raise NotImplementedError(f"No C implementation registered for {type(op).__name__}")


def _hashable_aliasing_map(aliasing_map: dict[int, list[int]]) -> tuple:
    return tuple(sorted((idx, tuple(vals)) for idx, vals in aliasing_map.items()))


class CImpl(CLinkerOp):
    """A C implementation of an `Op`, detached from the op.

    Returned by `c_funcify`; never a graph op. Subclasses that add configuration
    must extend `_impl_props`, which backs equality, hashing, and the cache key.
    """

    # `Apply.clone_with_new_inputs` reads this off the node's op when the
    # single-node graph is cloned for compilation; impl outputs never depend on
    # input values (the graph op already fixed the output types).
    _output_type_depends_on_input_value = False

    def __init__(
        self,
        op: Op,
        *,
        destroy_map: dict[int, list[int]] | None = None,
        view_map: dict[int, list[int]] | None = None,
    ):
        self.op = op
        if destroy_map is None:
            destroy_map = getattr(op, "destroy_map", {})
        if view_map is None:
            view_map = getattr(op, "view_map", {})
        self.destroy_map = destroy_map
        self.view_map = view_map

    def _impl_props(self) -> tuple:
        return (
            self.op,
            _hashable_aliasing_map(self.destroy_map),
            _hashable_aliasing_map(self.view_map),
        )

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self._impl_props() == other._impl_props()

    def __hash__(self) -> int:
        return hash((type(self), *self._impl_props()))

    def __str__(self) -> str:
        return f"{type(self).__name__}{{{self.op}}}"

    def make_node(self, *inputs) -> NoReturn:
        raise RuntimeError(
            f"{type(self).__name__} is a C implementation, not a graph op."
        )

    def prepare_node(
        self,
        node: Apply,
        storage_map: StorageMapType,
        compute_map: ComputeMapType | None,
        impl: str | None,
    ) -> None:
        """No-op: C preparation happens when `c_funcify` constructs the impl."""


def c_thunk_from_dispatch(
    node: Apply,
    storage_map: StorageMapType,
    compute_map: ComputeMapType | None,
    no_recycling: Collection[Variable],
) -> CThunkWrapperType:
    """Compile a C thunk for `node`, taking its implementation from `c_funcify`.

    Raises
    ------
    NotImplementedError
        If `node.op` has no C implementation, or has float16 inputs/outputs.
    MethodNotDefined
        If the implementation declines this node (e.g. an unsupported dtype).

    Callers fall back to a Python thunk on either.
    """
    # Imported here to avoid an import cycle.
    import pytensor.link.c.basic

    # Resolve eagerly so an unimplemented op raises before prepare_node runs and
    # before any compilation work; CLinker re-resolves (memoized) during codegen.
    c_funcify(node.op, node=node)

    node.op.prepare_node(
        node, storage_map=storage_map, compute_map=compute_map, impl="c"
    )

    node_input_storage = [storage_map[r] for r in node.inputs]
    node_output_storage = [storage_map[r] for r in node.outputs]

    fgraph = FunctionGraph(node.inputs, node.outputs)
    fgraph_no_recycling = [
        new_o
        for (new_o, old_o) in zip(fgraph.outputs, node.outputs, strict=True)
        if old_o in no_recycling
    ]
    cl = pytensor.link.c.basic.CLinker().accept(
        fgraph, no_recycling=fgraph_no_recycling
    )

    # float16 gets special treatment since running unprepared C code will get bad
    # results.
    if not getattr(node.op, "_f16_ok", False):

        def is_f16(t):
            return getattr(t, "dtype", "") == "float16"

        if any(is_f16(i.type) for i in node.inputs) or any(
            is_f16(o.type) for o in node.outputs
        ):
            # get_dynamic_module just tries to build the C code; it raises for
            # impls without C code, in which case we don't want to warn.
            cl.get_dynamic_module()
            warnings.warn(f"Disabling C code for {node.op} due to unsupported float16")
            raise NotImplementedError("float16")

    outputs = cl.make_thunk(
        input_storage=node_input_storage, output_storage=node_output_storage
    )
    thunk, _node_input_filters, _node_output_filters = outputs

    if compute_map is None:
        rval = is_cthunk_wrapper_type(thunk)
    else:
        cm_entries = [compute_map[o] for o in node.outputs]

        @is_cthunk_wrapper_type
        def rval(thunk=thunk, cm_entries=cm_entries):
            thunk()
            for entry in cm_entries:
                entry[0] = True

    rval.thunk = thunk
    rval.cthunk = thunk.cthunk
    rval.inputs = node_input_storage
    rval.outputs = node_output_storage
    rval.lazy = False
    return rval


# Ops whose `make_thunk` is one of these run through the dispatch; anything else
# overrode `make_thunk` (e.g. `IfElse`, `Scan`) and keeps its custom path.
_DEFAULT_MAKE_THUNKS = (Op.make_thunk, COp.make_thunk)


def make_node_thunk_with_c_dispatch(
    node: Apply,
    storage_map: StorageMapType,
    compute_map: ComputeMapType | None,
    no_recycling: Collection[Variable],
    *,
    try_c: bool,
    fallback_impl: str | None = None,
) -> ThunkType:
    """Make a thunk for `node`, trying the C dispatch first when `try_c`.

    When the C attempt fails (no implementation, or the implementation declines
    the node) the fallback passes ``impl="py"`` so `COp.make_thunk` does not
    retry the C path.
    """
    if try_c and type(node.op).make_thunk in _DEFAULT_MAKE_THUNKS:
        try:
            return c_thunk_from_dispatch(node, storage_map, compute_map, no_recycling)
        except (NotImplementedError, MethodNotDefined):
            fallback_impl = "py"
    # Op.make_thunk is untyped upstream; pin the result to its real type.
    thunk: ThunkType = node.op.make_thunk(
        node, storage_map, compute_map, no_recycling, impl=fallback_impl
    )
    return thunk
