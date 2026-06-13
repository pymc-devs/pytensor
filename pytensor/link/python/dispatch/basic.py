from functools import singledispatch


@singledispatch
def python_funcify(op, node=None, **kwargs):
    """Return a fast pure-Python implementation of ``op`` as a callable.

    The callable takes the node's inputs positionally and returns its output (a
    single value, or a tuple for multi-output nodes). Register a specialization
    to override an `Op`'s ``perform`` with a faster numpy/scipy path on the
    Python backend.

    Unregistered ops raise `NotImplementedError`, signalling the linker to fall
    back to ``perform`` (via ``Op.make_thunk(impl="py")``).
    """
    raise NotImplementedError(
        f"No python_funcify implementation registered for {type(op).__name__}"
    )


def make_node_thunk_with_python_dispatch(
    node, storage_map, compute_map, *, fallback, impl
):
    """Build a per-node thunk, preferring a registered `python_funcify` impl.

    When `python_funcify` has a specialization for ``node.op``, its callable is
    wrapped into a thunk that reads inputs from and writes outputs to
    ``storage_map``. Otherwise ``fallback`` (``Op.make_thunk``) is used, which
    covers ``perform`` ops and lazy ops like ``IfElse`` unchanged.
    """
    try:
        fn = python_funcify(node.op, node=node)
    except NotImplementedError:
        return fallback(node, storage_map, compute_map, impl)

    return _wrap_callable_as_thunk(fn, node, storage_map, compute_map)


def _wrap_callable_as_thunk(fn, node, storage_map, compute_map):
    input_storage = [storage_map[v] for v in node.inputs]
    output_compute = [compute_map[v] for v in node.outputs]

    if len(node.outputs) == 1:
        [out_storage] = (storage_map[v] for v in node.outputs)

        def thunk(fn=fn, inputs=input_storage, out=out_storage, cm=output_compute):
            out[0] = fn(*(inp[0] for inp in inputs))
            cm[0][0] = True
    else:
        output_storage = [storage_map[v] for v in node.outputs]

        def thunk(fn=fn, inputs=input_storage, outs=output_storage, cm=output_compute):
            for storage, value in zip(outs, fn(*(inp[0] for inp in inputs))):
                storage[0] = value
            for entry in cm:
                entry[0] = True

    thunk.lazy = False
    return thunk
