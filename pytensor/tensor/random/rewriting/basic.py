from itertools import chain

from pytensor.compile import optdb
from pytensor.configdefaults import config
from pytensor.graph import ancestors
from pytensor.graph.op import compute_test_value
from pytensor.graph.rewriting.basic import (
    copy_stack_trace,
    dfs_rewriter,
    node_rewriter,
)
from pytensor.tensor import TensorVariable
from pytensor.tensor.basic import constant
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.extra_ops import broadcast_to
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.utils import broadcast_params
from pytensor.tensor.shape import Shape, Shape_i
from pytensor.tensor.subtensor import (
    AdvancedSubtensor,
    AdvancedSubtensor1,
    Subtensor,
    indices_from_subtensor,
)
from pytensor.tensor.type import integer_dtypes
from pytensor.tensor.type_other import NoneTypeT, SliceType


def is_rv_used_in_graph(base_rv, node, fgraph):
    """Determine whether or not `base_rv` is used by a node other than `node` in `fgraph`.

    If a node uses `Shape` or `Shape_i` on the `base_rv`, we ignore it, because
    those `Op`s don't rely on the actual sample values of `base_rv`.

    TODO: We should apply all the shape rewrites before these rewrites, since
    that would properly remove the unnecessary dependencies on `base_rv` (when
    possible).
    """
    return any(
        n
        for n, i in fgraph.clients.get(base_rv, ())
        if not (n is node or isinstance(n.op, Shape | Shape_i))
    )


@node_rewriter([RandomVariable], inplace=True)
def random_make_inplace(fgraph, node):
    op = node.op

    if isinstance(op, RandomVariable) and not op.inplace:
        props = op._props_dict()
        props["inplace"] = True
        new_op = type(op)(**props)
        new_outputs = new_op.make_node(*node.inputs).outputs
        for old_out, new_out in zip(node.outputs, new_outputs, strict=True):
            copy_stack_trace(old_out, new_out)
        return new_outputs

    return False


optdb.register(
    "random_make_inplace",
    dfs_rewriter(random_make_inplace, ignore_newtrees=True),
    "fast_run",
    "inplace",
    position=50.9,
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

    rng, size, *dist_params = node.inputs

    if isinstance(size.type, NoneTypeT):
        return

    dist_params = broadcast_params(dist_params, node.op.ndims_params)

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

    new_node = node.op.make_node(rng, None, *dist_params)

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

    # Dimshuffle which drop dimensions not supported yet
    if ds_op.drop:
        return False

    [ds_rv] = node.outputs
    rv_node = node.inputs[0].owner

    if not (rv_node and isinstance(rv_node.op, RandomVariable)):
        return False

    rv_op = rv_node.op
    rng, size, *dist_params = rv_node.inputs
    next_rng, rv = rv_node.outputs

    # If no one else is using the underlying `RandomVariable`, then we can
    # do this; otherwise, the graph would be internally inconsistent.
    if is_rv_used_in_graph(rv, node, fgraph):
        return False

    # Check that Dimshuffle does not affect support dims
    supp_dims = set(range(rv.ndim - rv_op.ndim_supp, rv.ndim))
    shuffled_dims = {dim for i, dim in enumerate(ds_op.shuffle) if dim != i}
    augmented_dims = {d - rv_op.ndim_supp for d in ds_op.augment}
    if (shuffled_dims | augmented_dims) & supp_dims:
        return False

    # If no one else is using the underlying RandomVariable, then we can
    # do this; otherwise, the graph would be internally inconsistent.
    if is_rv_used_in_graph(rv, node, fgraph):
        return False

    batched_dims = rv.ndim - rv_op.ndim_supp
    batched_dims_ds_order = tuple(o for o in ds_op.new_order if o not in supp_dims)

    if isinstance(size.type, NoneTypeT):
        new_size = size
    else:
        # Update the size to reflect the DimShuffled dimensions
        new_size = [
            constant(1, dtype="int64") if o == "x" else size[o]
            for o in batched_dims_ds_order
        ]

    # Updates the params to reflect the Dimshuffled dimensions
    new_dist_params = []
    for param, param_ndim_supp in zip(dist_params, rv_op.ndims_params, strict=True):
        # Add the parameter support dimension indexes to the batched dimensions Dimshuffle
        param_new_order = batched_dims_ds_order + tuple(
            range(batched_dims, batched_dims + param_ndim_supp)
        )
        new_dist_params.append(param.dimshuffle(param_new_order))

    new_node = rv_op.make_node(rng, new_size, *new_dist_params)

    if config.compute_test_value != "off":
        compute_test_value(new_node)

    new_next_rng, new_rv = new_node.outputs

    if rv.name:
        new_rv.name = f"{rv.name}_lifted"

    # We replace uses of the dimshuffled RV by the new RV
    # And uses of the old RNG update by the new RNG update
    return {
        ds_rv: new_rv,
        next_rng: new_next_rng,
    }


@node_rewriter([Subtensor, AdvancedSubtensor1, AdvancedSubtensor])
def local_subtensor_rv_lift(fgraph, node):
    """Lift a ``*Subtensor`` through ``RandomVariable`` inputs.

    For example, ``normal(mu, std)[0] == normal(mu[0], std[0])``.

    This rewrite also applies to multivariate distributions as long
    as indexing does not happen within core dimensions, such as in
    ``mvnormal(mu, cov, size=(2,))[0, 0]``.
    """

    def is_nd_advanced_idx(idx, dtype) -> bool:
        if not isinstance(idx, TensorVariable):
            return False
        if isinstance(dtype, str):
            return (getattr(idx.type, "dtype", None) == dtype) and (idx.type.ndim >= 1)
        else:
            return (getattr(idx.type, "dtype", None) in dtype) and (idx.type.ndim >= 1)

    subtensor_op = node.op

    [indexed_rv] = node.outputs
    rv_node = node.inputs[0].owner

    if not (rv_node and isinstance(rv_node.op, RandomVariable)):
        return False

    rv_op = rv_node.op
    rng, size, *dist_params = rv_node.inputs
    next_rng, rv = rv_node.outputs

    # If no one else is using the underlying `RandomVariable`, then we can
    # do this; otherwise, the graph would be internally inconsistent.
    if is_rv_used_in_graph(rv, node, fgraph):
        return False

    # Parse indices
    if isinstance(subtensor_op, Subtensor):
        indices = indices_from_subtensor(node.inputs[1:], subtensor_op.idx_list)
    else:
        indices = node.inputs[1:]
        # The rewrite doesn't apply if advanced indexing could broadcast the samples (leading to duplicates)
        # Note: For simplicity this also excludes subtensor-related expand_dims (np.newaxis).
        #  If we wanted to support that we could rewrite it as subtensor + dimshuffle
        #  and make use of the dimshuffle lift rewrite
        # TODO: This rewrite is aborting with dummy indexing dimensions which aren't a problem
        if any(
            is_nd_advanced_idx(idx, integer_dtypes) or isinstance(idx.type, NoneTypeT)
            for idx in indices
        ):
            return False

    # Check that indexing does not act on support dims
    batch_ndims = rv_op.batch_ndim(rv_node)
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
                and all(isinstance(i.type, NoneTypeT) for i in idx.owner.inputs)
            ):
                return False
        n_discarded_idxs = len(supp_indices)
        indices = indices[:-n_discarded_idxs]

    # Update the size to reflect the indexed dimensions
    if isinstance(size.type, NoneTypeT):
        new_size = size
    else:
        shape_feature = getattr(fgraph, "shape_feature", None)
        if not shape_feature:
            return None

        # Use shape_feature to facilitate inferring final shape.
        # Check that neither the RV nor the old Subtensor are in the shape graph.
        output_shape = fgraph.shape_feature.shape_of.get(indexed_rv, None)
        if output_shape is None or {indexed_rv, rv} & set(ancestors(output_shape)):
            return None

        new_size = output_shape[: len(output_shape) - rv_op.ndim_supp]

    # Propagate indexing to the parameters' batch dims.
    # We try to avoid broadcasting the parameters together (and with size), by only indexing
    # non-broadcastable (non-degenerate) parameter dims. These parameters and the new size
    # should still correctly broadcast any degenerate parameter dims.
    new_dist_params = []
    for param, param_ndim_supp in zip(dist_params, rv_op.ndims_params, strict=True):
        # Check which dims are broadcasted by either size or other parameters
        bcast_param_dims = tuple(
            dim
            for dim, (param_dim_bcast, output_dim_bcast) in enumerate(
                zip(param.type.broadcastable, rv.type.broadcastable, strict=False)
            )
            if param_dim_bcast and not output_dim_bcast
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
                if set(bool_dims) & set(bcast_param_dims):
                    int_indices = list(idx.nonzero())
                    # Indexing by 0 drops the degenerate dims
                    for bool_dim in bool_dims:
                        if bool_dim in bcast_param_dims:
                            int_indices[bool_dim - curr_dim] = 0
                    batch_indices.extend(int_indices)
                # No overlap, use boolean index as is
                else:
                    batch_indices.append(idx)
                curr_dim += len(bool_dims)
            # Basic-indexing (slice or integer)
            else:
                # Broadcasted dim
                if curr_dim in bcast_param_dims:
                    # Slice indexing, keep degenerate dim by none-slicing
                    if isinstance(idx, slice) or isinstance(idx.type, SliceType):
                        batch_indices.append(slice(None))
                    # Integer indexing, drop degenerate dim by 0-indexing
                    else:
                        batch_indices.append(0)
                # Non-broadcasted dim
                else:
                    # Use index as is
                    batch_indices.append(idx)
                curr_dim += 1

        new_dist_params.append(param[tuple(batch_indices)])

    # Create new RV
    new_node = rv_op.make_node(rng, new_size, *new_dist_params)
    new_next_rng, new_rv = new_node.outputs

    copy_stack_trace(rv, new_rv)

    # We replace uses of the indexed RV by the new RV
    # And uses of the old RNG update by the new RNG update
    return {
        indexed_rv: new_rv,
        next_rng: new_next_rng,
    }
