from itertools import chain

from pytensor.compile import optdb
from pytensor.configdefaults import config
from pytensor.graph import ancestors
from pytensor.graph.op import compute_test_value
from pytensor.graph.rewriting.basic import copy_stack_trace, in2out, node_rewriter
from pytensor.scalar import integer_types
from pytensor.tensor import NoneConst
from pytensor.tensor.basic import constant, get_vector_length
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.extra_ops import broadcast_to
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.utils import broadcast_params
from pytensor.tensor.shape import Shape, Shape_i, shape_padleft
from pytensor.tensor.subtensor import (
    AdvancedSubtensor,
    AdvancedSubtensor1,
    Subtensor,
    as_index_variable,
    get_idx_list,
)
from pytensor.tensor.type_other import SliceType


def is_rv_used_in_graph(base_rv, node, fgraph):
    """Determine whether or not `base_rv` is used by a node other than `node` in `fgraph`.

    If a node uses `Shape` or `Shape_i` on the `base_rv`, we ignore it, because
    those `Op`s don't rely on the actual sample values of `base_rv`.

    TODO: We should apply all the shape rewrites before these rewrites, since
    that would properly remove the unnecessary dependencies on `base_rv` (when
    possible).

    """

    def _node_check(n, i):
        if n == "output":
            n = fgraph.outputs[i].owner
        return n == node or isinstance(n.op, (Shape, Shape_i))

    return not all(_node_check(n, i) for n, i in fgraph.clients.get(base_rv, ()))


@node_rewriter([RandomVariable], inplace=True)
def random_make_inplace(fgraph, node):
    op = node.op

    if isinstance(op, RandomVariable) and not op.inplace:
        props = op._props_dict()
        props["inplace"] = True
        new_op = type(op)(**props)
        return new_op.make_node(*node.inputs).outputs

    return False


optdb.register(
    "random_make_inplace",
    in2out(random_make_inplace, ignore_newtrees=True),
    "fast_run",
    "inplace",
    position=99,
)


@node_rewriter(tracks=None)
def local_rv_size_lift(fgraph, node):
    """Lift the ``size`` parameter in a ``RandomVariable``.

    In other words, this will broadcast the distribution parameters by adding
    the extra dimensions implied by the ``size`` parameter, and remove the
    ``size`` parameter in the process.

    For example, ``normal(0, 1, size=(1, 2))`` becomes
    ``normal([[0, 0]], [[1, 1]], size=())``.

    """

    if not isinstance(node.op, RandomVariable):
        return

    rng, size, dtype, *dist_params = node.inputs

    dist_params = broadcast_params(dist_params, node.op.ndims_params)

    if get_vector_length(size) > 0:
        dist_params = [
            broadcast_to(
                p,
                (
                    tuple(size)
                    + (
                        tuple(p.shape)[-node.op.ndims_params[i] :]
                        if node.op.ndims_params[i] > 0
                        else ()
                    )
                )
                if node.op.ndim_supp > 0
                else size,
            )
            for i, p in enumerate(dist_params)
        ]
    else:
        return

    new_node = node.op.make_node(rng, None, dtype, *dist_params)

    if config.compute_test_value != "off":
        compute_test_value(new_node)

    return new_node.outputs


@node_rewriter([DimShuffle])
def local_dimshuffle_rv_lift(fgraph, node):
    """Lift a ``DimShuffle`` through ``RandomVariable`` inputs.

    For example, ``normal(mu, std).T == normal(mu.T, std.T)``.

    This rewrite is only applicable when the Dimshuffle operation does
    not affect support dimensions.

    TODO: Support dimension dropping
    """

    ds_op = node.op

    if not isinstance(ds_op, DimShuffle):
        return False

    base_rv = node.inputs[0]
    rv_node = base_rv.owner

    if not (rv_node and isinstance(rv_node.op, RandomVariable)):
        return False

    # Dimshuffle which drop dimensions not supported yet
    if ds_op.drop:
        return False

    rv_op = rv_node.op
    rng, size, dtype, *dist_params = rv_node.inputs
    rv = rv_node.default_output()

    # Check that Dimshuffle does not affect support dims
    supp_dims = set(range(rv.ndim - rv_op.ndim_supp, rv.ndim))
    shuffled_dims = {dim for i, dim in enumerate(ds_op.shuffle) if dim != i}
    augmented_dims = {d - rv_op.ndim_supp for d in ds_op.augment}
    if (shuffled_dims | augmented_dims) & supp_dims:
        return False

    # If no one else is using the underlying RandomVariable, then we can
    # do this; otherwise, the graph would be internally inconsistent.
    if is_rv_used_in_graph(base_rv, node, fgraph):
        return False

    batched_dims = rv.ndim - rv_op.ndim_supp
    batched_dims_ds_order = tuple(o for o in ds_op.new_order if o not in supp_dims)

    # Make size explicit
    missing_size_dims = batched_dims - get_vector_length(size)
    if missing_size_dims > 0:
        full_size = tuple(broadcast_params(dist_params, rv_op.ndims_params)[0].shape)
        size = full_size[:missing_size_dims] + tuple(size)

    # Update the size to reflect the DimShuffled dimensions
    new_size = [
        constant(1, dtype="int64") if o == "x" else size[o]
        for o in batched_dims_ds_order
    ]

    # Updates the params to reflect the Dimshuffled dimensions
    new_dist_params = []
    for param, param_ndim_supp in zip(dist_params, rv_op.ndims_params):
        # Add broadcastable dimensions to the parameters that would have been expanded by the size
        padleft = batched_dims - (param.ndim - param_ndim_supp)
        if padleft > 0:
            param = shape_padleft(param, padleft)

        # Add the parameter support dimension indexes to the batched dimensions Dimshuffle
        param_new_order = batched_dims_ds_order + tuple(
            range(batched_dims, batched_dims + param_ndim_supp)
        )
        new_dist_params.append(param.dimshuffle(param_new_order))

    new_node = rv_op.make_node(rng, new_size, dtype, *new_dist_params)

    if config.compute_test_value != "off":
        compute_test_value(new_node)

    out = new_node.outputs[1]
    if base_rv.name:
        out.name = f"{base_rv.name}_lifted"
    return [out]


@node_rewriter([Subtensor, AdvancedSubtensor1, AdvancedSubtensor])
def local_subtensor_rv_lift(fgraph, node):
    """Lift a ``*Subtensor`` through ``RandomVariable`` inputs.

    For example, ``normal(mu, std)[0] == normal(mu[0], std[0])``.

    This rewrite also applies to multivariate distributions as long
    as indexing does not happen within core dimensions, such as in
    ``mvnormal(mu, cov, size=(2,))[0, 0]``.
    """

    def is_nd_advanced_idx(idx, dtype):
        if isinstance(dtype, str):
            return (getattr(idx.type, "dtype", None) == dtype) and (idx.type.ndim >= 1)
        else:
            return (getattr(idx.type, "dtype", None) in dtype) and (idx.type.ndim >= 1)

    subtensor_op = node.op

    old_subtensor = node.outputs[0]
    rv = node.inputs[0]
    rv_node = rv.owner

    if not (rv_node and isinstance(rv_node.op, RandomVariable)):
        return False

    shape_feature = getattr(fgraph, "shape_feature", None)
    if not shape_feature:
        return None

    # Use shape_feature to facilitate inferring final shape.
    # Check that neither the RV nor the old Subtensor are in the shape graph.
    output_shape = fgraph.shape_feature.shape_of.get(old_subtensor, None)
    if output_shape is None or {old_subtensor, rv} & set(ancestors(output_shape)):
        return None

    rv_op = rv_node.op
    rng, size, dtype, *dist_params = rv_node.inputs

    # Parse indices
    idx_list = getattr(subtensor_op, "idx_list", None)
    if idx_list:
        idx_vars = get_idx_list(node.inputs, idx_list)
    else:
        idx_vars = node.inputs[1:]
    indices = tuple(as_index_variable(idx) for idx in idx_vars)

    # The rewrite doesn't apply if advanced indexing could broadcast the samples (leading to duplicates)
    # Note: For simplicity this also excludes subtensor-related expand_dims (np.newaxis).
    #  If we wanted to support that we could rewrite it as subtensor + dimshuffle
    #  and make use of the dimshuffle lift rewrite
    integer_dtypes = {type.dtype for type in integer_types}
    if any(
        is_nd_advanced_idx(idx, integer_dtypes) or NoneConst.equals(idx)
        for idx in indices
    ):
        return False

    # Check that indexing does not act on support dims
    batch_ndims = rv.ndim - rv_op.ndim_supp
    # We decompose the boolean indexes, which makes it clear whether they act on support dims or not
    non_bool_indices = tuple(
        chain.from_iterable(
            idx.nonzero() if is_nd_advanced_idx(idx, "bool") else (idx,)
            for idx in indices
        )
    )
    if len(non_bool_indices) > batch_ndims:
        # If the last indexes are just dummy `slice(None)` we discard them instead of quitting
        non_bool_indices, supp_indices = (
            non_bool_indices[:batch_ndims],
            non_bool_indices[batch_ndims:],
        )
        for idx in supp_indices:
            if not (
                isinstance(idx.type, SliceType)
                and all(NoneConst.equals(i) for i in idx.owner.inputs)
            ):
                return False
        n_discarded_idxs = len(supp_indices)
        indices = indices[:-n_discarded_idxs]

    # If no one else is using the underlying `RandomVariable`, then we can
    # do this; otherwise, the graph would be internally inconsistent.
    if is_rv_used_in_graph(rv, node, fgraph):
        return False

    # Update the size to reflect the indexed dimensions
    new_size = output_shape[: len(output_shape) - rv_op.ndim_supp]

    # Propagate indexing to the parameters' batch dims.
    # We try to avoid broadcasting the parameters together (and with size), by only indexing
    # non-broadcastable (non-degenerate) parameter dims. These parameters and the new size
    # should still correctly broadcast any degenerate parameter dims.
    new_dist_params = []
    for param, param_ndim_supp in zip(dist_params, rv_op.ndims_params):
        # We first expand any missing parameter dims (and later index them away or keep them with none-slicing)
        batch_param_dims_missing = batch_ndims - (param.ndim - param_ndim_supp)
        batch_param = (
            shape_padleft(param, batch_param_dims_missing)
            if batch_param_dims_missing
            else param
        )
        # Check which dims are actually broadcasted
        bcast_batch_param_dims = tuple(
            dim
            for dim, (param_dim, output_dim) in enumerate(
                zip(batch_param.type.shape, rv.type.shape)
            )
            if (param_dim == 1) and (output_dim != 1)
        )
        batch_indices = []
        curr_dim = 0
        for idx in indices:
            # Advanced boolean indexing
            if is_nd_advanced_idx(idx, "bool"):
                # Check if any broadcasted dim overlaps with advanced boolean indexing.
                # If not, we use that directly, instead of the more inefficient `nonzero` form
                bool_dims = range(curr_dim, curr_dim + idx.type.ndim)
                # There's an overlap, we have to decompose the boolean mask as a `nonzero`
                if set(bool_dims) & set(bcast_batch_param_dims):
                    int_indices = list(idx.nonzero())
                    # Indexing by 0 drops the degenerate dims
                    for bool_dim in bool_dims:
                        if bool_dim in bcast_batch_param_dims:
                            int_indices[bool_dim - curr_dim] = 0
                    batch_indices.extend(int_indices)
                # No overlap, use index as is
                else:
                    batch_indices.append(idx)
                curr_dim += len(bool_dims)
            # Basic-indexing (slice or integer)
            else:
                # Broadcasted dim
                if curr_dim in bcast_batch_param_dims:
                    # Slice indexing, keep degenerate dim by none-slicing
                    if isinstance(idx.type, SliceType):
                        batch_indices.append(slice(None))
                    # Integer indexing, drop degenerate dim by 0-indexing
                    else:
                        batch_indices.append(0)
                # Non-broadcasted dim
                else:
                    # Use index as is
                    batch_indices.append(idx)
                curr_dim += 1

        new_dist_params.append(batch_param[tuple(batch_indices)])

    # Create new RV
    new_node = rv_op.make_node(rng, new_size, dtype, *new_dist_params)
    new_rv = new_node.default_output()

    copy_stack_trace(rv, new_rv)

    return [new_rv]
