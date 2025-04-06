from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
    indices_from_subtensor,
)
from pytensor.tensor.type_other import MakeSlice


BOOLEAN_MASK_ERROR = """JAX does not support resizing arrays with boolean
masks. In some cases, however, it is possible to re-express your model
in a form that JAX can compile:

>>> import pytensor.tensor as pt
>>> x_pt = pt.vector('x')
>>> y_pt = x_pt[x_pt > 0].sum()

can be re-expressed as:

>>> import pytensor.tensor as pt
>>> x_pt = pt.vector('x')
>>> y_pt = pt.where(x_pt > 0, x_pt, 0).sum()
"""

DYNAMIC_SLICE_LENGTH_ERROR = """JAX does not support slicing arrays with a dynamic
slice length.
"""


@jax_funcify.register(Subtensor)
@jax_funcify.register(AdvancedSubtensor)
@jax_funcify.register(AdvancedSubtensor1)
def jax_funcify_Subtensor(op, node, **kwargs):
    idx_list = getattr(op, "idx_list", None)

    def subtensor(x, *ilists):
        indices = indices_from_subtensor(ilists, idx_list)
        if len(indices) == 1:
            indices = indices[0]

        return x.__getitem__(indices)

    return subtensor


@jax_funcify.register(IncSubtensor)
@jax_funcify.register(AdvancedIncSubtensor1)
def jax_funcify_IncSubtensor(op, node, **kwargs):
    idx_list = getattr(op, "idx_list", None)

    if getattr(op, "set_instead_of_inc", False):

        def jax_fn(x, indices, y):
            return x.at[indices].set(y)

    else:

        def jax_fn(x, indices, y):
            return x.at[indices].add(y)

    def incsubtensor(x, y, *ilist, jax_fn=jax_fn, idx_list=idx_list):
        indices = indices_from_subtensor(ilist, idx_list)
        if len(indices) == 1:
            indices = indices[0]

        if isinstance(op, AdvancedIncSubtensor1):
            op._check_runtime_broadcasting(node, x, y, indices)

        return jax_fn(x, indices, y)

    return incsubtensor


@jax_funcify.register(AdvancedIncSubtensor)
def jax_funcify_AdvancedIncSubtensor(op, node, **kwargs):
    if getattr(op, "set_instead_of_inc", False):

        def jax_fn(x, indices, y):
            return x.at[indices].set(y)

    else:

        def jax_fn(x, indices, y):
            return x.at[indices].add(y)

    def advancedincsubtensor(x, y, *ilist, jax_fn=jax_fn):
        return jax_fn(x, ilist, y)

    return advancedincsubtensor


@jax_funcify.register(MakeSlice)
def jax_funcify_MakeSlice(op, **kwargs):
    def makeslice(*x):
        return slice(*x)

    return makeslice
