"""Build an ordinary VJP, infer its nonzeros, and batch cotangent directions."""

import pytensor.tensor as pt
from pytensor.gradient import pullback
from pytensor.graph.basic import Variable
from pytensor.graph.replace import vectorize_graph
from pytensor.tensor.type import DenseTensorType, float_dtypes
from research.vjp_sparsity.rewriting import infer_nonzero


def jacobian_sparsity_vjp(outputs, wrt, *, diagnostics=None):
    """Infer a dense boolean Jacobian pattern from the existing symbolic VJP."""
    outputs = [outputs] if isinstance(outputs, Variable) else list(outputs)
    wrt = [wrt] if isinstance(wrt, Variable) else list(wrt)
    if any(
        not isinstance(variable.type, DenseTensorType)
        or variable.dtype not in float_dtypes
        for variable in [*outputs, *wrt]
    ):
        raise NotImplementedError("The prototype requires real, dense endpoints")
    n_rows = sum((output.size for output in outputs), pt.constant(0, dtype="int64"))
    n_columns = sum((variable.size for variable in wrt), pt.constant(0, dtype="int64"))
    if not outputs or not wrt:
        return pt.zeros((n_rows, n_columns), dtype="bool")

    cotangents = [
        output.type(name=f"cotangent_{i}") for i, output in enumerate(outputs)
    ]
    unique = {}
    for output, cotangent in zip(outputs, cotangents, strict=True):
        unique[output] = unique[output] + cotangent if output in unique else cotangent
    vjp = pullback(
        list(unique), wrt, list(unique.values()), disconnected_inputs="ignore"
    )

    lifted = infer_nonzero(vjp, cotangents, wrt=wrt)

    basis = {}
    offset = pt.constant(0, dtype="int64")
    for output, cotangent in zip(outputs, cotangents, strict=True):
        seed = pt.eq(
            pt.arange(n_rows)[:, None],
            offset + pt.arange(output.size)[None, :],
        ).reshape((n_rows, *output.shape))
        basis[cotangent] = pt.specify_shape(seed, (None, *output.type.shape))
        offset += output.size

    batched = vectorize_graph(lifted, replace=basis)
    blocks = []
    for mask, variable in zip(batched, wrt, strict=True):
        if mask.ndim == variable.ndim:
            mask = pt.broadcast_to(mask, (n_rows, *variable.shape))
        blocks.append(mask.reshape((n_rows, variable.size)))
    if diagnostics is not None:
        diagnostics.update(vjp=vjp, lifted_vjp=lifted)
    return pt.concatenate(blocks, axis=1)
