from pytensor.link.numba.dispatch.basic import generate_fallback_impl, numba_funcify
from pytensor.tensor.optimize import ScipyWrapperOp


@numba_funcify.register(ScipyWrapperOp)
def numba_funcify_ScipyWrapperOp(op, node=None, storage_map=None, **kwargs):
    # No numba implementation: ``perform`` runs in objmode, but the inner
    # function can still be linked with numba (see #2668). Work on a private
    # copy so a shared op instance never carries a numba-linked ``_fn`` into
    # other compiles.
    numba_op = op.clone_with_inner_graph(op.fgraph)
    numba_op._fn = numba_op.compile_fn(numba_op.link_mode("numba"))
    return generate_fallback_impl(numba_op, node, storage_map, **kwargs)
