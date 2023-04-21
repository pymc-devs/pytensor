from itertools import zip_longest

from pytensor.compile import optdb
from pytensor.configdefaults import config
from pytensor.graph.op import compute_test_value
from pytensor.graph.rewriting.basic import in2out, node_rewriter
from pytensor.tensor import NoneConst
from pytensor.tensor.basic import constant, get_vector_length
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.extra_ops import broadcast_to
from pytensor.tensor.math import sum as at_sum
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.utils import broadcast_params
from pytensor.tensor.shape import Shape, Shape_i, shape_padleft
from pytensor.tensor.subtensor import (
    AdvancedSubtensor,
    AdvancedSubtensor1,
    Subtensor,
    as_index_variable,
    get_idx_list,
    indexed_result_shape,
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

    st_op = node.op

    if not isinstance(st_op, (AdvancedSubtensor, AdvancedSubtensor1, Subtensor)):
        return False

    rv = node.inputs[0]
    rv_node = rv.owner

    if not (rv_node and isinstance(rv_node.op, RandomVariable)):
        return False

    rv_op = rv_node.op
    rng, size, dtype, *dist_params = rv_node.inputs

    # Parse indices
    idx_list = getattr(st_op, "idx_list", None)
    if idx_list:
        cdata = get_idx_list(node.inputs, idx_list)
    else:
        cdata = node.inputs[1:]
    st_indices, st_is_bool = zip(
        *tuple(
            (as_index_variable(i), getattr(i, "dtype", None) == "bool") for i in cdata
        )
    )

    # Check that indexing does not act on support dims
    batched_ndims = rv.ndim - rv_op.ndim_supp
    if len(st_indices) > batched_ndims:
        # If the last indexes are just dummy `slice(None)` we discard them
        st_is_bool = st_is_bool[:batched_ndims]
        st_indices, supp_indices = (
            st_indices[:batched_ndims],
            st_indices[batched_ndims:],
        )
        for index in supp_indices:
            if not (
                isinstance(index.type, SliceType)
                and all(NoneConst.equals(i) for i in index.owner.inputs)
            ):
                return False

    # If no one else is using the underlying `RandomVariable`, then we can
    # do this; otherwise, the graph would be internally inconsistent.
    if is_rv_used_in_graph(rv, node, fgraph):
        return False

    # Update the size to reflect the indexed dimensions
    # TODO: Could use `ShapeFeature` info.  We would need to be sure that
    # `node` isn't in the results, though.
    # if hasattr(fgraph, "shape_feature"):
    #     output_shape = fgraph.shape_feature.shape_of(node.outputs[0])
    # else:
    output_shape_ignoring_bool = indexed_result_shape(rv.shape, st_indices)
    new_size_ignoring_boolean = (
        output_shape_ignoring_bool
        if rv_op.ndim_supp == 0
        else output_shape_ignoring_bool[: -rv_op.ndim_supp]
    )

    # Boolean indices can actually change the `size` value (compared to just *which* dimensions of `size` are used).
    # The `indexed_result_shape` helper does not consider this
    if any(st_is_bool):
        new_size = tuple(
            at_sum(idx) if is_bool else s
            for s, is_bool, idx in zip_longest(
                new_size_ignoring_boolean, st_is_bool, st_indices, fillvalue=False
            )
        )
    else:
        new_size = new_size_ignoring_boolean

    # Update the parameters to reflect the indexed dimensions
    new_dist_params = []
    for param, param_ndim_supp in zip(dist_params, rv_op.ndims_params):
        # Apply indexing on the batched dimensions of the parameter
        batched_param_dims_missing = batched_ndims - (param.ndim - param_ndim_supp)
        batched_param = shape_padleft(param, batched_param_dims_missing)
        batched_st_indices = []
        for st_index, batched_param_shape in zip(st_indices, batched_param.type.shape):
            # If we have a degenerate dimension indexing it should always do the job
            if batched_param_shape == 1:
                batched_st_indices.append(0)
            else:
                batched_st_indices.append(st_index)
        new_dist_params.append(batched_param[tuple(batched_st_indices)])

    # Create new RV
    new_node = rv_op.make_node(rng, new_size, dtype, *new_dist_params)
    new_rv = new_node.default_output()

    if config.compute_test_value != "off":
        compute_test_value(new_node)

    return [new_rv]
