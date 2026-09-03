from functools import singledispatch

from pytensor.graph.fg import AbstractFunctionGraph
from pytensor.link.utils import fgraph_to_python


@singledispatch
def python_funcify(op, node=None, **kwargs):
    """Return a fast pure-Python implementation of ``op`` as a callable.

    The callable takes the node's inputs positionally and returns its output (a
    single value, or a tuple for multi-output nodes). Register a specialization
    to override an `Op`'s ``perform`` with a faster numpy/scipy path on the
    Python backend.

    Unregistered ops raise `NotImplementedError`. The ``py`` (VM) linker catches
    this and falls back to ``Op.make_thunk(impl="py")``; the whole-graph ``pyjit``
    linker catches it and falls back to a ``perform`` wrapper.
    """
    raise NotImplementedError(
        f"No python_funcify implementation registered for {type(op).__name__}"
    )


def _perform_wrapper(op, node):
    """Wrap an `Op`'s ``perform`` into a `python_funcify`-style callable."""
    n_outputs = len(node.outputs)
    single_output = n_outputs == 1

    def perform(*inputs):
        output_storage = [[None] for _ in range(n_outputs)]
        op.perform(node, list(inputs), output_storage)
        if single_output:
            return output_storage[0][0]
        return tuple(storage[0] for storage in output_storage)

    return perform


def _funcify_or_perform(op, node=None, **kwargs):
    try:
        return python_funcify(op, node=node)
    except NotImplementedError:
        return _perform_wrapper(op, node)


@python_funcify.register(AbstractFunctionGraph)
def python_funcify_FunctionGraph(
    fgraph, node=None, fgraph_name="python_funcified_fgraph", **kwargs
):
    return fgraph_to_python(
        fgraph,
        op_conversion_fn=_funcify_or_perform,
        fgraph_name=fgraph_name,
        **kwargs,
    )


def make_node_thunk_with_python_dispatch(
    node, storage_map, compute_map, *, fallback, implementation
):
    """Build a per-node thunk, preferring a registered `python_funcify` implementation.

    When `python_funcify` has a specialization for ``node.op``, its callable is
    wrapped into a thunk that reads inputs from and writes outputs to
    ``storage_map``. Otherwise ``fallback`` (``Op.make_thunk``) is used, which
    covers ``perform`` ops and lazy ops like ``IfElse`` unchanged.
    """
    try:
        fn = python_funcify(node.op, node=node)
    except NotImplementedError:
        return fallback(node, storage_map, compute_map, implementation)

    return _wrap_callable_as_thunk(fn, node, storage_map, compute_map)


def _wrap_callable_as_thunk(fn, node, storage_map, compute_map):
    input_storage = [storage_map[variable] for variable in node.inputs]
    output_compute = [compute_map[variable] for variable in node.outputs]

    if len(node.outputs) == 1:
        [output] = (storage_map[variable] for variable in node.outputs)

        def thunk(fn=fn, inputs=input_storage, output=output, compute=output_compute):
            output[0] = fn(*(inp[0] for inp in inputs))
            compute[0][0] = True
    else:
        output_storage = [storage_map[variable] for variable in node.outputs]

        def thunk(
            fn=fn, inputs=input_storage, outputs=output_storage, compute=output_compute
        ):
            for output, value in zip(outputs, fn(*(inp[0] for inp in inputs))):
                output[0] = value
            for entry in compute:
                entry[0] = True

    thunk.lazy = False
    return thunk
