import jax
import jax.numpy as jnp
from jax import lax

from pytensor.graph.basic import Constant
from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
    get_idx_list,
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


def _get_static_length(start_var, stop_var):
    """
    Analyzes the PyTensor graph to prove that (stop_var - start_var) is a constant.
    This bypasses PyTensor's (None,) static shape inference.
    """

    def extract_offset(var):
        # Unwrap ScalarFromTensor to get to the actual math operations
        if (
            hasattr(var, "owner")
            and var.owner
            and var.owner.op.__class__.__name__ == "ScalarFromTensor"
        ):
            var = var.owner.inputs[0]

        # Check if the variable is a pure constant
        if isinstance(var, Constant):
            try:
                return None, int(var.data.item())
            except Exception:
                pass

        # Check if the variable is an operation  like (base + offset) or (base - offset)
        if hasattr(var, "owner") and var.owner and isinstance(var.owner.op, Elemwise):
            scalar_op = getattr(var.owner.op, "scalar_op", None)
            if scalar_op:
                op_name = getattr(scalar_op, "name", "")

                if op_name == "add":
                    c_in = [i for i in var.owner.inputs if isinstance(i, Constant)]
                    v_in = [i for i in var.owner.inputs if not isinstance(i, Constant)]
                    if len(c_in) == 1 and len(v_in) == 1:
                        return v_in[0], int(c_in[0].data.item())

                elif op_name == "sub":
                    if isinstance(var.owner.inputs[1], Constant) and not isinstance(
                        var.owner.inputs[0], Constant
                    ):
                        return var.owner.inputs[0], -int(
                            var.owner.inputs[1].data.item()
                        )

        return var, 0

    if start_var is None or stop_var is None:
        return None

    start_base, start_off = extract_offset(start_var)
    stop_base, stop_off = extract_offset(stop_var)

    # If both variables  share the same dynamic base  , the size is static
    if start_base is not None and stop_base is not None and start_base == stop_base:
        return stop_off - start_off

    return None


@jax_funcify.register(Subtensor)
@jax_funcify.register(AdvancedSubtensor)
@jax_funcify.register(AdvancedSubtensor1)
def jax_funcify_Subtensor(op, node, **kwargs):
    idx_list = getattr(op, "idx_list", None)
    out_shape = list(node.outputs[0].type.shape)
    is_basic_subtensor = isinstance(op, Subtensor)

    # Extract original PyTensor symbolic variables to deduce static slice lengths
    pt_idx_list = list(get_idx_list(node.inputs, idx_list))

    def subtensor(x, *ilists):
        indices = indices_from_subtensor(ilists, idx_list)
        idx_iter = indices if isinstance(indices, tuple) else (indices,)

        has_tracer = False
        for idx in idx_iter:
            if isinstance(idx, jax.core.Tracer):
                has_tracer = True
            elif isinstance(idx, slice):
                if isinstance(idx.start, jax.core.Tracer) or isinstance(
                    idx.stop, jax.core.Tracer
                ):
                    has_tracer = True

        if has_tracer and is_basic_subtensor:
            try:
                start_indices = []
                slice_sizes = []
                squeeze_dims = []

                out_dim_idx = 0
                for i, (idx, pt_idx) in enumerate(zip(idx_iter, pt_idx_list)):
                    if isinstance(idx, slice):
                        if idx.step not in (None, 1):
                            raise ValueError(
                                "Dynamic slicing with step != 1 is not supported by JAX."
                            )

                        start = 0 if idx.start is None else idx.start

                        # Determine slice size
                        size = out_shape[out_dim_idx]
                        if size is None:
                            # Mathematical Prover
                            size = _get_static_length(pt_idx.start, pt_idx.stop)
                            if size is None:
                                raise ValueError(
                                    "Could not prove static slice size for JAX lowering."
                                )

                        start_indices.append(start)
                        slice_sizes.append(size)
                        out_dim_idx += 1
                    else:
                        start_indices.append(idx)
                        slice_sizes.append(1)
                        squeeze_dims.append(i)

                for i in range(len(start_indices), x.ndim):
                    start_indices.append(0)
                    size = out_shape[out_dim_idx]
                    if size is None:
                        # unlikely to hit but unless the trailing dimension is genuinely dynamic
                        size = x.shape[i]
                    slice_sizes.append(size)
                    out_dim_idx += 1

                sliced = lax.dynamic_slice(x, start_indices, slice_sizes)

                if squeeze_dims:
                    sliced = jnp.squeeze(sliced, axis=tuple(squeeze_dims))

                return sliced
            except Exception:
                # If prover fails or assumptions break, fall back to standard indexing crash
                pass

        if len(indices) == 1 and isinstance(indices, tuple):
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
