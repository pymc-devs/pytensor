"""Differentiation of graphs containing XTensorVariables.

xtensor Ops carry no gradient of their own, so :func:`pytensor.grad` raises on a graph
that still holds them. These functions lower the graph first, on a clone, and hand the
result to the ordinary tensor rules.

Labels are preserved throughout: a gradient with respect to an ``XTensorVariable`` is
an ``XTensorVariable`` with the same ``dims``, and cotangents and tangents are matched
to what they seed by dim name rather than by position. ``wrt`` and ``consider_constant``
may be intermediate results, not only inputs.

.. warning::

    This module is experimental and its API may change without a deprecation cycle.
"""

from collections.abc import Sequence

from pytensor.gradient import grad as tensor_grad
from pytensor.gradient import pullback as tensor_pullback
from pytensor.graph.basic import Variable
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.xtensor.type import XTensorType


def lower(outputs: Variable | Sequence[Variable]):
    """Rewrite the xtensor Ops in ``outputs`` into the tensor Ops they stand for.

    Runs the ``lower_xtensor`` rewrite that compilation applies anyway, on a clone.
    """
    return rewrite_graph(outputs, include=("lower_xtensor",), clone=True)


def _lower_for(outputs, referenced):
    """Lower ``outputs`` without rewriting away any of the ``referenced`` variables.

    ``wrt`` and ``consider_constant`` act by identity, so they have to come out of
    lowering as the same variables that went in. Lowering rewrites every xtensor Op it
    can reach, which would replace any of them that is interior to the graph.

    Detaching a variable from its owner makes it an input of the graph being rewritten,
    and no rewrite reaches past an input -- the same trick `rewrite_subgraph` uses.
    Unlike that helper this one clones, so the caller's graph is left alone, and the
    owners are put back afterwards: the region below each frontier variable is simply
    left for the ordinary pipeline to lower later.
    """
    detached = [
        (var, var.owner, var.index) for var in referenced if var.owner is not None
    ]
    for var, _, _ in detached:
        var.owner = None
    try:
        return lower(outputs)
    finally:
        for var, owner, index in detached:
            var.owner = owner
            var.index = index


def _referenced(wrt, consider_constant):
    named = [wrt] if isinstance(wrt, Variable) else list(wrt)
    return named + list(consider_constant or ())


def grad(cost, wrt, consider_constant=None, **kwargs):
    """Gradient of a scalar ``cost`` with respect to ``wrt``.

    The xtensor counterpart of :func:`pytensor.grad`, which takes the remaining
    keyword arguments. Each gradient carries the type of its variable, so an
    xtensor one keeps its ``dims``.
    """
    lowered = _lower_for(cost, _referenced(wrt, consider_constant))
    return tensor_grad(lowered, wrt, consider_constant=consider_constant, **kwargs)


def pullback(outputs, wrt, cotangents, consider_constant=None, **kwargs):
    """Vector-Jacobian product: ``vjp[j] = sum_i cotangents[i] d outputs[i] / d wrt[j]``.

    Unlike `grad` the outputs need not be scalar, and each is seeded with a
    cotangent of its own type -- a labelled one may give its ``dims`` in any order.
    Remaining keyword arguments go to :func:`pytensor.gradient.pullback`.
    """
    outs = [outputs] if isinstance(outputs, Variable) else list(outputs)
    cots = [cotangents] if isinstance(cotangents, Variable) else list(cotangents)
    lowered = _lower_for(outs, _referenced(wrt, consider_constant))

    # Lowering makes dims positional, and the tensor rule pairs a cotangent with its
    # output by position from there on. A labelled cotangent has to be put in the
    # output's dim order first, or one written in a different order -- which xtensor
    # semantics say means the same thing -- would be transposed into the gradient
    # without anything noticing, whenever the shape happens to allow it.
    aligned = []
    for out, cot in zip(lowered, cots, strict=True):
        labelled = isinstance(out.type, XTensorType)
        if labelled != isinstance(cot.type, XTensorType):
            raise TypeError(
                f"Cotangent {cot} and the output it belongs to, {out}, must both be "
                "labelled or both be plain tensors; there is no way to tell which "
                "dims a plain cotangent carries."
            )
        if labelled and cot.type.dims != out.type.dims:
            cot = cot.transpose(*out.type.dims)
        aligned.append(cot)

    return tensor_pullback(
        lowered, wrt, aligned, consider_constant=consider_constant, **kwargs
    )


def pushforward(outputs, wrt, tangents, **kwargs):
    """Jacobian-vector product: ``jvp[i] = sum_j d outputs[i] / d wrt[j] tangents[j]``.

    Each tangent has the type of its ``wrt``, and each result that of its output.
    A pullback is linear in its cotangents, so this is built from two `pullback`
    calls rather than needing a forward rule of its own.
    """
    single = isinstance(outputs, Variable)
    outs = [outputs] if single else list(outputs)

    # The two pullbacks are taken here rather than handed to the tensor rule whole,
    # because the first can leave fresh xtensor Ops behind -- accumulating cotangents of
    # an xtensor variable, say -- which the second has to see lowered. The placeholders
    # are free variables so that lowering, which rewrites expressions but never inputs,
    # cannot cost us the handle.
    placeholders = [out.type() for out in outs]
    jvps = pullback(pullback(outs, wrt, placeholders), placeholders, tangents, **kwargs)
    return jvps[0] if single else jvps
