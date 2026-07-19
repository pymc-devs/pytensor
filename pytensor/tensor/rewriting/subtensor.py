import sys
import warnings
from itertools import pairwise, zip_longest

import numpy as np

from pytensor import compile
from pytensor.assumptions.core import UNIQUE_INDICES, check_assumption
from pytensor.compile import optdb
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import (
    WalkingGraphRewriter,
    copy_stack_trace,
    dfs_rewriter,
    in2out,
    node_rewriter,
)
from pytensor.raise_op import Assert
from pytensor.scalar import Add, ScalarConstant, ScalarMinimum, Sub
from pytensor.scalar import constant as scalar_constant
from pytensor.tensor.basic import (
    Alloc,
    ARange,
    Join,
    Nonzero,
    ScalarFromTensor,
    TensorFromScalar,
    alloc,
    arange,
    cast,
    concatenate,
    expand_dims,
    get_scalar_constant_value,
    get_underlying_scalar_constant_value,
    moveaxis,
    register_infer_shape,
    switch,
)
from pytensor.tensor.basic import constant as tensor_constant
from pytensor.tensor.blockwise import _squeeze_left
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.extra_ops import broadcast_to, squeeze
from pytensor.tensor.math import (
    add,
    and_,
    eq,
    ge,
    gt,
    le,
    lt,
    maximum,
    minimum,
    or_,
    variadic_add,
)
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.rewriting.blockwise import blockwise_of
from pytensor.tensor.shape import (
    Reshape,
    Shape,
    Shape_i,
    shape_padleft,
    shape_padright,
)
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedSubtensor,
    IncSubtensor,
    Subtensor,
    _is_provably_non_negative,
    _is_provably_positive,
    _non_consecutive_adv_indexing,
    as_index_literal,
    flatten_index_variables,
    get_canonical_form_slice,
    get_constant_idx,
    get_slice_elements,
    indices_from_subtensor,
    unflatten_index_variables,
)
from pytensor.tensor.type import TensorType
from pytensor.tensor.variable import TensorConstant, TensorVariable


def register_useless(lopt, *tags, **kwargs):
    if isinstance(lopt, str):

        def register(inner_lopt):
            return register_useless(inner_lopt, lopt, *tags, **kwargs)

        return register
    else:
        name = kwargs.pop("name", None) or lopt.__name__

        compile.mode.local_useless.register(
            name, lopt, "fast_run", *tags, position="last", **kwargs
        )
        return lopt


def is_full_slice(x):
    warnings.warn(
        "The function is deprecated, use x==slice(None) instead.",
        DeprecationWarning,
    )
    return x == slice(None)


def _constant_has_unique_indices(idx) -> bool:
    """Check whether a constant index has no duplicate entries.

    Boolean indices, scalars, and single-element arrays are trivially unique.
    For larger integer arrays, indices that mix positive and negative values
    may alias, so those are treated as potentially duplicated.  The result
    is cached on ``idx.tag``.
    """
    if not isinstance(idx, Constant):
        return False
    cached = getattr(idx.tag, "unique_indices", None)
    if cached is not None:
        return bool(cached)
    idx_val = np.asarray(idx.data)
    if idx_val.dtype == bool:
        result = True
    elif idx_val.size <= 1:
        result = True
    else:
        has_pos = (idx_val >= 0).any()
        has_neg = (idx_val < 0).any()
        result = not (has_pos and has_neg) and np.unique(idx_val).size == idx_val.size
    idx.tag.unique_indices = result
    return result


def _arange_provably_unique(start, stop, step) -> bool:
    """Whether ``arange(start, stop, step)`` selects each position at most once.

    Its entries are always distinct values; they map to distinct positions as
    long as they don't wrap around zero, i.e. they all share a sign (``arange(-2,
    2)`` aliases on a size-2 axis, but ``arange(2, 6)`` and ``arange(-6, -2)`` do
    not). This is proved from whichever bounds are statically known.
    """
    constants: list[int | None] = []
    for v in (start, stop, step):
        try:
            constants.append(int(get_scalar_constant_value(v)))
        except NotScalarConstantError:
            constants.append(None)
    start_c, stop_c, step_c = constants

    # Both endpoints non-negative -> every entry lies between them -> all >= 0,
    # whatever the step direction (covers symbolic ``arange(x.shape[0])``).
    if _is_provably_non_negative(start) and _is_provably_non_negative(stop):
        return True

    # With a known direction only one endpoint binds each sign. Ascending entries
    # span ``[start, stop)`` (min is ``start``, max ``< stop``); descending entries
    # span ``(stop, start]`` (max is ``start``, min ``> stop``).
    if _is_provably_positive(step):  # ascending
        if _is_provably_non_negative(start):  # all >= 0
            return True
        if stop_c is not None and stop_c <= 0:  # all <= -1
            return True
    elif step_c is not None and step_c < 0:  # descending
        if _is_provably_non_negative(stop):  # all >= 0  (e.g. arange(k, 5, -1))
            return True
        if start_c is not None and start_c < 0:  # all <= -1  (e.g. arange(-1, k, -1))
            return True

    # Fully constant: compute the exact entry range without materializing it. The
    # checks above only bound the last entry as past ``stop``, so they need ``stop``
    # on the right side of zero. This catches the rest -- ranges whose last entry
    # overshoots ``stop`` across zero yet stays single-signed, e.g.
    # ``arange(6, -2, -2)`` is ``[6, 4, 2, 0]`` and ``arange(-5, 1, 3)`` is ``[-5, -2]``.
    if (
        start_c is not None
        and stop_c is not None
        and step_c is not None
        and step_c != 0
    ):
        n = max(0, -(-(stop_c - start_c) // step_c))  # length, via ceil division
        if n <= 1:
            return True
        last_c = start_c + (n - 1) * step_c
        return min(start_c, last_c) >= 0 or max(start_c, last_c) < 0
    return False


def _arange_shifted_bounds(idx):
    """``arange(a, b, s) +/- k`` as the bounds of the ``arange`` it is equal to.

    A scalar shift slides the whole range, so whether the entries straddle zero — the
    only thing that makes distinct values alias onto one position — is still decided by
    the bounds, once the shift is folded into them. ``arange(-5, 5) + 5`` is
    ``arange(0, 10)``, unique, even though the unshifted range is not.
    """
    if not (
        isinstance(idx.owner_op, Elemwise)
        and isinstance(idx.owner.op.scalar_op, Add | Sub)
    ):
        return None
    x, y = idx.owner.inputs
    if isinstance(x.owner_op, ARange) and all(y.type.broadcastable):
        arange_var, shift_var = x, y
    elif (
        isinstance(idx.owner.op.scalar_op, Add)
        and isinstance(y.owner_op, ARange)
        and all(x.type.broadcastable)
    ):
        arange_var, shift_var = y, x
    else:
        return None
    try:
        shift = int(get_underlying_scalar_constant_value(shift_var))
    except NotScalarConstantError:
        # A symbolic shift moves the range by an unknown amount, so where it sits
        # relative to zero -- the only thing that matters here -- stays unknown.
        return None
    if isinstance(idx.owner.op.scalar_op, Sub):
        shift = -shift

    def shifted(bound):
        # Fold eagerly: the sign checks below read the bound itself, not a value
        # they could recover from an unevaluated Add. The shifted value can leave the
        # original bound's dtype (``arange(5)`` carries int8 bounds), and only its
        # sign and magnitude are ever read, so let the constant pick its own.
        try:
            value = int(get_scalar_constant_value(bound)) + shift
        except NotScalarConstantError:
            return bound + shift
        return tensor_constant(np.asarray(value))

    start, stop, step = arange_var.owner.inputs
    return shifted(start), shifted(stop), step


def _index_provably_unique(idx, fgraph: FunctionGraph | None) -> bool:
    """Whether a single index selects each position on its own axis at most once.

    This is the duplicate-free reasoning shared by accumulation gating (where
    repeated positions would scatter-add) and by ``_index_provably_not_larger``
    (a duplicate-free index can't enlarge its axis). It excludes only the
    statically-smaller fallback, which bounds the size without ruling out
    repeated positions.

    ``fgraph`` is the handle to the ``AssumptionFeature``: it lets a user-declared
    ``unique_indices`` assumption prove uniqueness when the static value, ``arange``
    shape, or view structure can't. Pass ``None`` to skip that leg (no assumptions).
    """
    if isinstance(idx, slice) or idx.ndim == 0:
        return True
    if all(idx.type.broadcastable):
        return True
    if idx.type.dtype == "bool":
        return True
    if _constant_has_unique_indices(idx):
        return True
    if check_assumption(fgraph, idx, UNIQUE_INDICES):
        return True
    if isinstance(idx.owner_op, ARange):
        return _arange_provably_unique(*idx.owner.inputs)
    if (shifted_bounds := _arange_shifted_bounds(idx)) is not None:
        return _arange_provably_unique(*shifted_bounds)
    if isinstance(idx.owner_op, Reshape | DimShuffle):
        # Views that only reorder or insert size-1 dims keep the value multiset.
        return _index_provably_unique(idx.owner.inputs[0], fgraph)
    return False


def _constant_indices_jointly_unique(consts) -> bool:
    """Whether stacked constant indices have no duplicate coordinate tuples.

    The stacked ``np.unique`` can be expensive on large indices, so the result
    is cached on the first constant's tag. Uniqueness is a property of the whole
    group, and a constant may belong to several groups (constants are shared
    across the graph), so the cache is keyed by the group's identities rather
    than a single flag.
    """
    key = tuple(id(c) for c in consts)
    cache = getattr(consts[0].tag, "jointly_unique_indices", None)
    if cache is None:
        cache = consts[0].tag.jointly_unique_indices = {}
    if key not in cache:
        datas = [np.asarray(c.data) for c in consts]
        # A coordinate axis that mixes positive and negative values may alias
        # (``0`` and ``-dim`` are the same position), so distinctness of the raw
        # values no longer proves distinctness of the coordinates.
        if any((data >= 0).any() and (data < 0).any() for data in datas):
            cache[key] = False
        else:
            coords = np.broadcast_arrays(*datas)
            stacked = np.stack([coord.ravel() for coord in coords])
            cache[key] = bool(np.unique(stacked, axis=1).shape[1] == stacked.shape[1])
    return bool(cache[key])


def _indices_jointly_unique(idxs, fgraph: FunctionGraph | None) -> bool:
    """Whether advanced indices produce no duplicate joint coordinate tuples.

    For accumulation (``inc``), and for bounding a gather by the indexed axes'
    size, what matters is that the broadcast coordinate tuples
    ``(idx0[k], idx1[k], ...)`` are all distinct. Sufficient conditions, in
    increasing generality:

    - every index is duplicate-free on its own axis, so the tuples are distinct
      regardless of the others (sound under broadcasting, and the path basic
      slice/scalar indexing trivially takes);
    - the indices are all the coordinates of a single ``Nonzero``, distinct by
      construction (e.g. symbolic ``tril_indices``);
    - the indices are all constants whose stacked coordinate tuples have no
      duplicates (catches cases where no single axis is unique on its own).

    ``fgraph`` is forwarded to ``_index_provably_unique`` so a user-declared
    ``unique_indices`` assumption can satisfy the per-axis leg; pass ``None`` to
    skip assumptions.
    """
    if all(_index_provably_unique(idx, fgraph) for idx in idxs):
        return True
    if len(idxs) > 1:
        owners = {idx.owner for idx in idxs}
        if (
            len(owners) == 1
            and (owner := next(iter(owners))) is not None
            and isinstance(owner.op, Nonzero)
            and set(idxs) == set(owner.outputs)
        ):
            return True
        if all(isinstance(idx, Constant) for idx in idxs):
            return _constant_indices_jointly_unique(idxs)
    return False


def _advanced_indices_jointly_unique(indices, fgraph: FunctionGraph | None) -> bool:
    """Whether the advanced (``ndim > 0`` tensor) indices in a reconstructed index
    tuple have distinct joint coordinate tuples.

    Slice and scalar (basic) indices are ignored: basic indexing is trivially
    duplicate-free, so only the advanced array indices are weighed. ``fgraph`` is
    forwarded to ``_indices_jointly_unique`` for the ``unique_indices`` assumption
    lookup (see there); pass ``None`` to skip assumptions.
    """
    adv_idxs = [
        idx for idx in indices if isinstance(idx, TensorVariable) and idx.type.ndim > 0
    ]
    return _indices_jointly_unique(adv_idxs, fgraph)


def _constant_is_arange(idx) -> tuple[int, int, int] | None:
    """Match ``idx`` to ``np.arange(offset, offset + d * step, step)``
    and return ``(d, offset, step)``, else ``None``.

    Single-element constants return ``(1, value, 1)``.  The result is cached
    on ``idx.tag.is_arange`` (``False`` sentinels a no-match).
    """
    if not isinstance(idx, Constant):
        return None
    cached = getattr(idx.tag, "is_arange", None)
    if cached is not None:
        return cached if cached is not False else None
    idx_val = np.asarray(idx.data)
    if idx_val.ndim != 1 or idx_val.size == 0 or idx_val.dtype.kind not in "iu":
        result: tuple[int, int, int] | None = None
    elif idx_val.size == 1:
        result = (1, int(idx_val[0]), 1)
    else:
        diffs = np.diff(idx_val)
        step = int(diffs[0])
        if step != 0 and np.all(diffs == step):
            result = (int(idx_val.size), int(idx_val[0]), step)
        else:
            result = None
    idx.tag.is_arange = result if result is not None else False
    return result


def _match_arange_0_to_d_plus_offset(idx):
    """Match ``arange(0, d, 1) + offset`` and return ``(arange_node, offset)``
    where ``arange_node`` is the ``arange(0, d, 1)`` output and ``offset`` is
    a scalar ``TensorVariable`` (``constant(0)`` for bare aranges), else
    ``None``.
    """
    if not isinstance(idx, TensorVariable):
        return None
    if isinstance(idx.owner_op, ARange):
        start, _stop, step = idx.owner.inputs
        if not (isinstance(start, TensorConstant) and int(start.data) == 0):
            return None
        if not (isinstance(step, TensorConstant) and int(step.data) == 1):
            return None
        return idx, tensor_constant(np.zeros((), dtype=idx.type.dtype))
    if not (
        isinstance(idx.owner_op, Elemwise) and isinstance(idx.owner.op.scalar_op, Add)
    ):
        return None
    arange_node = None
    offset_terms = []
    for inp in idx.owner.inputs:
        if (
            arange_node is None
            and isinstance(inp.owner_op, ARange)
            and isinstance(inp.owner.inputs[0], TensorConstant)
            and int(inp.owner.inputs[0].data) == 0
            and isinstance(inp.owner.inputs[2], TensorConstant)
            and int(inp.owner.inputs[2].data) == 1
        ):
            arange_node = inp
        elif inp.type.shape == (1,):
            offset_terms.append(inp)
        else:
            return None
    if arange_node is None:
        return None
    return arange_node, variadic_add(*offset_terms)


def eager_add_zero(x, y):
    """``x + y``, but return ``x`` when ``y`` is provably zero."""
    if isinstance(y, int | np.integer):
        return x if y == 0 else x + y
    try:
        if int(get_underlying_scalar_constant_value(y)) == 0:
            return x
    except NotScalarConstantError:
        pass
    return x + y


def _eager_scalar(x):
    """Reduce a 0d or ``(1,)``-shaped tensor to the simplest scalar form."""
    if isinstance(x, TensorConstant):
        # ``.item()`` covers the ``(1,)`` case the 0d-only ``int()`` rejects.
        return int(x.data.item())
    if isinstance(x.owner_op, DimShuffle) and x.owner.op.input_ndim == 0:
        inner = x.owner.inputs[0]
        return int(inner.data) if isinstance(inner, TensorConstant) else inner
    if isinstance(x.owner_op, TensorFromScalar):
        return x.owner.inputs[0]
    if x.type.ndim > 0:
        return squeeze(x)
    return x


def _idx_to_int_array(idx):
    """Materialize a 1-D integer index as a numpy array.

    Handles ``TensorConstant`` (read ``.data``) and symbolic
    ``arange(n) + offset`` with constant ``n`` and ``offset``
    (eager constant-fold). Boolean masks are converted to integer positions.
    Return ``None`` when the index can't be materialized at rewrite time.
    """
    if isinstance(idx, TensorConstant):
        arr = np.asarray(idx.data)
        if arr.dtype == bool:
            return np.flatnonzero(arr)
        return arr
    sym = _match_arange_0_to_d_plus_offset(idx)
    if sym is None:
        return None
    arange_node, offset = sym
    _, stop, _ = arange_node.owner.inputs
    if not isinstance(stop, TensorConstant):
        return None
    n = int(stop.data)
    if n < 0:
        return None
    try:
        off_val = int(get_underlying_scalar_constant_value(offset))
    except NotScalarConstantError:
        return None
    return np.arange(n, dtype=np.int64) + off_val


def _is_shape_of_x_at(var, x, axis):
    """``True`` when ``var`` is statically equivalent to ``x.shape[axis]``."""
    if isinstance(var, TensorConstant):
        s = x.type.shape[axis]
        return s is not None and int(var.data) == s
    op = var.owner_op
    if isinstance(op, Shape_i):
        return op.i == axis and var.owner.inputs[0] is x
    if isinstance(op, Subtensor):
        shape_node = var.owner.inputs[0]
        if not isinstance(shape_node.owner_op, Shape):
            return False
        if shape_node.owner.inputs[0] is not x:
            return False
        try:
            idx_val = get_constant_idx(
                var.owner.op.idx_list, var.owner.inputs, allow_partial=False
            )
        except NotScalarConstantError:
            return False
        return len(idx_val) == 1 and idx_val[0] == axis
    if isinstance(op, DimShuffle) and op.new_order == ():
        # ``Squeeze(Shape(x))`` collapses Shape(x) (length-1 vector) to scalar:
        # only valid when ``x`` is 1-D, in which case it's ``x.shape[0]``.
        shape_node = var.owner.inputs[0]
        if not isinstance(shape_node.owner_op, Shape):
            return False
        return axis == 0 and shape_node.owner.inputs[0] is x
    return False


@register_infer_shape
@register_useless
@register_canonicalize
@register_specialize
@register_stabilize
@node_rewriter([Subtensor, IncSubtensor, AdvancedSubtensor, AdvancedIncSubtensor])
def local_useless_slice(fgraph, node):
    """Remove useless slices and canonicalize redundant slice bounds to ``None``.

    Applies to all Subtensor Ops with slices (basic and advanced, get and set).

    - ``X[0, :]`` -> ``X[0]`` (trailing full slices dropped)
    - ``X[:]`` -> ``X``
    - ``X[0:7:1]`` -> ``X[:]`` when ``X.shape[0] <= 7``
    - ``X[-1:-8:-1]`` -> ``X[::-1]`` when ``X.shape[0] <= 7``
    """
    op = node.op
    idx_list = op.idx_list
    if not idx_list:
        if isinstance(op, Subtensor | AdvancedSubtensor):
            return [node.inputs[0]]
        else:
            # We let local_useless_inc_subtensor handle these
            return None

    if is_inc_subtensor := isinstance(op, IncSubtensor | AdvancedIncSubtensor):
        x, y, *idx_vars = node.inputs
    else:
        x, *idx_vars = node.inputs

    new_idxs = list(indices_from_subtensor(idx_vars, idx_list))
    change_flag = False
    last_useful_idx = -1
    for dim, s in enumerate(new_idxs):
        if not isinstance(s, slice):
            last_useful_idx = dim
            continue

        if s == slice(None):
            continue

        step = s.step

        if step is None:
            positive_step = True
        elif isinstance(step, Constant):
            step_value = step.data
            positive_step = step.data > 0
            if step_value == 1:
                change_flag = True
                step = None
        else:
            # We can only canonicalize start and stop if we know the sign of step
            last_useful_idx = dim
            continue

        start = s.start
        stop = s.stop

        dim_length = x.type.shape[dim] if dim < x.type.ndim else None
        if start is not None and isinstance(start, Constant):
            start_val = start.data
            if positive_step:
                if (
                    start_val == 0
                    # Negative start that wraps to or before index 0
                    or (dim_length is not None and -start_val >= dim_length)
                ):
                    change_flag = True
                    start = None
            else:
                if (
                    start_val == -1
                    # Positive start at or beyond the last index
                    or (dim_length is not None and start_val >= dim_length - 1)
                ):
                    change_flag = True
                    start = None

        if dim_length is not None and stop is not None and isinstance(stop, Constant):
            stop_val = stop.data
            if positive_step:
                # Positive stop at or beyond the length
                if stop_val >= dim_length:
                    change_flag = True
                    stop = None
            else:
                # Negative stop that wraps to or before index 0
                if -stop_val > dim_length:
                    change_flag = True
                    stop = None

        # Drop a redundant stop that equals ``x.shape[dim]`` or is wrapped in
        # ``min(..., x.shape[dim])``: ``Subtensor`` already clips at runtime,
        # so such stops are just noise. Peek through ``ScalarFromTensor``.
        if positive_step and stop is not None:
            tensor_stop = (
                stop.owner.inputs[0]
                if isinstance(stop.owner_op, ScalarFromTensor)
                else stop
            )
            if _is_shape_of_x_at(tensor_stop, x, dim):
                change_flag = True
                stop = None
            elif isinstance(tensor_stop.owner_op, Elemwise) and isinstance(
                tensor_stop.owner.op.scalar_op, ScalarMinimum
            ):
                a, b = tensor_stop.owner.inputs
                kept = (
                    a
                    if _is_shape_of_x_at(b, x, dim)
                    else b
                    if _is_shape_of_x_at(a, x, dim)
                    else None
                )
                if kept is not None:
                    change_flag = True
                    stop = kept

        if start is not None or stop is not None or step is not None:
            last_useful_idx = dim

        new_idxs[dim] = slice(start, stop, step)

    if change_flag or (last_useful_idx + 1) < len(idx_list):
        new_idxs = new_idxs[: last_useful_idx + 1]
        new_idx_list, new_flat_vars = flatten_index_variables(new_idxs)
        props = op._props_dict() | {"idx_list": new_idx_list}
        if is_inc_subtensor:
            new_node = type(op)(**props)(x, y, *new_flat_vars).owner
            if not new_idx_list:
                ret = local_useless_inc_subtensor.fn(fgraph, new_node)
                if ret:
                    copy_stack_trace(node.outputs, ret)
                    return ret
            out = new_node.outputs[0]
        else:
            out = type(op)(**props)(x, *new_flat_vars) if new_idx_list else x
        copy_stack_trace(node.outputs, out)
        return [out]


def _merge_slice_into_slice_no_shape_ref(slice1, slice2):
    """Merge ``slice1`` then ``slice2`` (both pure slices on the same dim)
    when the result is computable from the bounds alone -- no shape required.

    Returns the merged slice, or ``None`` if the merge would need to know the
    array length (and thus would emit a switch / min / max tree).

    Cases handled (steps in ``{None, -1}``):

    * Both forward (step ``None``): combine starts and stops by addition,
      with sign-aware checks. Examples: ``x[1:-1][1:-1]`` -> ``x[2:-2]``.
    * ``x[a:b][::-1]`` -> single negative-step slice over the same range.
    * ``x[::-1][a:b]`` -> negative-step slice via index reflection.
    * ``x[::-1][a:b:-1]`` -> forward slice via double reflection
      (subsumes ``x[::-1][::-1]`` -> ``x[:]``).
    * ``x[a:b:-1][::-1]`` -> forward slice, restricted to non-negative
      ``a`` / ``b`` (or ``None``).

    Anything else returns ``None``.
    """

    def _const_int_or_none(v):
        if v is None:
            return None
        if isinstance(v, Constant):
            return int(v.data)
        return "unknown"

    s1_step = _const_int_or_none(slice1.step)
    s2_step = _const_int_or_none(slice2.step)

    if s1_step not in (None, -1) or s2_step not in (None, -1):
        # Unknown or non unit: don't bother
        # Wait for canonicalize: step = 1 -> None
        return None

    a1, b1 = _const_int_or_none(slice1.start), _const_int_or_none(slice1.stop)
    a2, b2 = _const_int_or_none(slice2.start), _const_int_or_none(slice2.stop)

    if "unknown" in (a1, a2, b1, b2):
        return None  # TODO: Handle symbolic cases with known sign

    if s2_step is None:
        if s1_step is None:
            # [a1:b1][a2:b2]

            # Ignore here as it will be canonicalized
            # a1 = None, b1 = None -> useless slice

            if a2 is None or a2 >= 0:
                # [±a1:±b1][a2:±b2]
                if a2 is None:
                    a2 = 0

                if a1 is None or a1 >= 0:
                    # [a1:±b1][a2:±b2]
                    if a1 is None:
                        a1 = 0
                    if b1 is None:
                        # [a1:][a2:±b2]
                        if b2 is None:
                            # [a1:][a2:]
                            return slice(a1 + a2, None)
                        elif b2 > 0:
                            # [a1:][a2:b2]
                            return slice(a1 + a2, a1 + b2)
                        else:  # b2 <= 0
                            # [a1:][a2:-b2]
                            return slice(a1 + a2, b2)
                    else:
                        # [a1:±b1][a2:±b2]
                        if b2 is None:
                            # [a1:±b1][a2:]
                            return slice(a1 + a2, b1)
                        elif b2 < 0:
                            if b1 < 0:
                                # [a1:-b1][a2:-b2]
                                return slice(a1 + a2, b1 + b2)
                            else:
                                # [a1:b1][a2:-b2] -- b1 + b2 would flip sign, needs shape
                                return None
                        elif b1 > 0:
                            # [a1:b1][a2:b2]
                            return slice(a1 + a2, min(b1, a1 + b2))
                        else:
                            # [a1:-b1][a2:b2] -- needs shape
                            return None
                else:  # a1 < 0
                    # [-a1:±b1][a2:±b2]
                    # Only sound if a2 == 0; otherwise the start mapping
                    # depends on whether len(x) >= |a1|.
                    if a2 != 0:
                        return None
                    if b1 is None:
                        # [-a1:][:b2]
                        if b2 is None:
                            return slice(a1, None)
                        elif b2 < 0:
                            return slice(a1, b2)
                        else:
                            # [-a1:][:b2] with b2 > 0 -- needs shape
                            return None
                    elif b1 < 0:
                        # [-a1:-b1][:b2]
                        if b2 is None:
                            return slice(a1, b1)
                        elif b2 < 0:
                            return slice(a1, b1 + b2)
                        else:
                            return None
                    else:
                        # [-a1:b1] with b1 >= 0 -- needs shape
                        return None

            else:  # a2 < 0
                # [±a1:±b1][-a2:±b2]
                if (
                    (a1 is not None and a1 < 0)
                    and b1 is None
                    and (b2 is None or b2 < 0)
                ):
                    # [-a1:][-a2:-b2]
                    return slice(max(a1, a2), b2)
                else:
                    return None  # complex (or trivially useless)

    if s1_step is None and s2_step == -1:
        # [a1:b1][a2:b2:-1] -- only handle the full-reverse outer.
        if a2 is None and b2 is None:
            # [a1:b1][::-1] -> single negative-step slice over the same range
            if b1 == 0:
                # [a1:0][::-1] -- always empty
                return slice(0, 0, -1)
            new_start = None if b1 is None else b1 - 1
            new_stop = None if (a1 is None or a1 == 0) else a1 - 1
            return slice(new_start, new_stop, -1)
        return None

    if s1_step == -1 and s2_step is None:
        # [a1:b1:-1][a2:b2] -- only handle the full-reverse inner.
        if a1 is None and b1 is None:
            # [::-1][a2:b2] -> negative-step slice via index reflection
            new_start = None if a2 is None else -a2 - 1
            new_stop = None if b2 is None else -b2 - 1
            return slice(new_start, new_stop, -1)
        return None

    if s1_step == -1 and s2_step == -1:
        # [a1:b1:-1][a2:b2:-1]
        if a1 is None and b1 is None:
            # [::-1][a2:b2:-1] -> forward slice via double reflection
            if a2 is None and b2 is None:
                # [::-1][::-1] -> [:]
                return slice(None)
            new_start = None if a2 is None else -a2 - 1
            new_stop = None if b2 is None else -b2 - 1
            return slice(new_start, new_stop, None)
        if a2 is None and b2 is None:
            # [a1:b1:-1][::-1] -> forward slice
            # Sound only when a1 in {None, >=0} and b1 in {None, >=0}.
            # a1 == -1 or b1 == -1 are danger cases: a1+1 == 0 / b1+1 == 0
            # flip between "before idx 0" and "L" semantics.
            if (a1 is None or a1 >= 0) and (b1 is None or b1 >= 0):
                new_start = None if b1 is None else b1 + 1
                new_stop = None if a1 is None else a1 + 1
                return slice(new_start, new_stop, None)
        return None

    return None


def _merge_scalar_into_slice_unsafe(inner_slice, scalar_index, dim, xshape):
    """Merge ``x[slice][scalar]`` into a single scalar index, or return None.

    Returns None when the step is symbolic or has magnitude != 1.

    Each sign of idx uses exactly one endpoint: positive idx counts from
    start, negative idx counts from stop. We clamp the used endpoint
    when it overflows (e.g. stop > n for step=1, start > n-1 for step=-1).
    We don't clamp the unused endpoint, so invalid indices may silently
    produce a wrong result. Valid indices always remain valid after the
    conversion. This is part of the shape_unsafe contract.
    """

    def _eager_lt_0(x):
        """Return ``True``/``False`` (Python bool) when the sign of *x* is
        known, otherwise return the ``lt(x, 0)`` graph node."""
        if _is_provably_non_negative(x):
            return False
        if isinstance(x, Constant):
            return int(x.data) < 0
        return lt(x, 0)

    def _eager_switch(cond, a, b):
        if cond is True:
            return a
        if cond is False:
            return b
        if a is b:
            return a
        return switch(cond, a, b)

    def _eager_minimum(a, b):
        if a is b:
            return a
        if _eager_lt_0(a) is True and _is_provably_non_negative(b):
            return a
        if _eager_lt_0(b) is True and _is_provably_non_negative(a):
            return b
        return minimum(a, b)

    step = inner_slice.step
    step_val = (
        1 if step is None else int(step.data) if isinstance(step, Constant) else None
    )

    start = inner_slice.start
    stop = inner_slice.stop

    if step_val == 1:
        # x[±a:±b][±idx]
        # Positive idx counts from effective start in [0, ∞).
        # Negative idx counts from effective stop in (-∞, n].
        if start is None:
            pos_idx_result = scalar_index
        else:
            a_eff = _eager_switch(
                _eager_lt_0(start),
                maximum(start + xshape[dim], 0),
                start,
            )
            pos_idx_result = a_eff + scalar_index

        if stop is None:
            neg_idx_result = scalar_index
        else:
            neg_idx_result = _eager_minimum(stop, xshape[dim]) + scalar_index

        return _eager_switch(_eager_lt_0(scalar_index), neg_idx_result, pos_idx_result)

    if step_val == -1:
        # x[±a:±b:-1][±idx]
        # Positive idx counts from effective start in (-∞, n-1].
        # Negative idx counts from effective stop in [-1, ∞).
        # When both are None (x[::-1]), both branches give -1 - idx
        # and _eager_switch deduplicates via identity check.
        default = -1 - scalar_index

        if start is None:
            pos_idx_result = default
        else:
            a_eff = _eager_switch(
                _eager_lt_0(start),
                maximum(start + xshape[dim], 0),
                _eager_minimum(start, xshape[dim] - 1),
            )
            pos_idx_result = a_eff - scalar_index

        if stop is None:
            neg_idx_result = default
        else:
            b_eff = _eager_switch(
                _eager_lt_0(stop),
                maximum(stop + xshape[dim], -1),
                stop,
            )
            neg_idx_result = b_eff - scalar_index

        return _eager_switch(_eager_lt_0(scalar_index), neg_idx_result, pos_idx_result)

    return None


def _local_subtensor_merge_rewrite(fgraph, node, *, merge_integer_index):
    """Merge ``Subtensor(Subtensor(x))`` into fewer operations.

    Both modes try shape-free slice+slice merges and constant-only
    ``merge_two_slices``.  Pairs that can't be merged are kept as a
    residual outer ``Subtensor``.

    When *merge_integer_index* is ``True`` (``local_subtensor_merge_integer``),
    scalar-into-slice merges are additionally attempted even with symbolic
    bounds (tagged ``shape_unsafe`` because it assumes indices are in-bounds).
    """

    u, *outer_index_vars = node.inputs
    match u.owner_op_and_inputs:
        case (Subtensor(idx_list=inner_idx_list), x, *inner_index_vars):
            pass
        case _:
            return None

    indices_inner = unflatten_index_variables(inner_index_vars, inner_idx_list)
    indices_outer = unflatten_index_variables(outer_index_vars, node.op.idx_list)

    try:
        xshape = fgraph.shape_feature.shape_tuple(x)
    except AttributeError:
        xshape = tuple(x.shape)

    try:
        ushape = fgraph.shape_feature.shape_tuple(u)
    except AttributeError:
        ushape = tuple(u.shape)

    merged_inner = []
    unmerged_outer = []
    pos_outer = 0
    any_merged = False

    for pos_inner, idx_inner in enumerate(indices_inner):
        if pos_outer >= len(indices_outer):
            # No more outer indices to pair; keep remaining inner as-is
            merged_inner.extend(indices_inner[pos_inner:])
            break

        if not isinstance(idx_inner, slice):
            # Integer index consumes input dim without producing output dim
            merged_inner.append(idx_inner)
            continue

        idx_outer = indices_outer[pos_outer]
        pos_outer += 1

        if isinstance(idx_outer, slice) and idx_outer == slice(None):
            # Useless outer slice, nothing to merge
            merged_inner.append(idx_inner)
            unmerged_outer.append(slice(None))
            continue

        merged = None

        if isinstance(idx_outer, slice):
            merged = _merge_slice_into_slice_no_shape_ref(idx_inner, idx_outer)
        elif merge_integer_index:
            merged = _merge_scalar_into_slice_unsafe(
                idx_inner, idx_outer, pos_inner, xshape
            )

        if merged is None:
            merged = merge_two_slices(
                fgraph,
                idx_inner,
                xshape[pos_inner],
                idx_outer,
                ushape[pos_outer - 1],
                allow_symbolic_refs=False,
            )

        if merged is not None:
            any_merged = True
            merged_inner.append(merged)
            if isinstance(merged, slice):
                # Placeholder to keep unmerged_outer aligned; stripped at the end
                unmerged_outer.append(slice(None))
        else:
            merged_inner.append(idx_inner)
            unmerged_outer.append(idx_outer)
    else:  # no-break
        # Outer had more indices not paired to an inner index
        if indices_outer[pos_outer:]:
            any_merged = True
            merged_inner.extend(indices_outer[pos_outer:])

    if not any_merged:
        return None

    # Strip trailing slice(None) from unmerged outer
    while unmerged_outer and unmerged_outer[-1] == slice(None):
        unmerged_outer.pop()

    out = x[tuple(merged_inner)]
    if unmerged_outer:
        out = out[tuple(unmerged_outer)]

    copy_stack_trace([node.outputs[0], u], out)
    return [out]


@register_canonicalize
@register_specialize
@node_rewriter([Subtensor])
def local_subtensor_merge_slice(fgraph, node):
    return _local_subtensor_merge_rewrite(fgraph, node, merge_integer_index=False)


@register_canonicalize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([Subtensor])
def local_subtensor_merge_integer(fgraph, node):
    return _local_subtensor_merge_rewrite(fgraph, node, merge_integer_index=True)


@register_specialize
@register_canonicalize
@node_rewriter([Subtensor])
def local_subtensor_remove_broadcastable_index(fgraph, node):
    """
    Remove broadcastable dimension with index 0 or -1
    a[:,:,:,0] -> a.dimshuffle(0,1,2), when
        a.broadcastable = (False, False, False, True)
    a[0,:,-1,:] -> a.dimshuffle(1,3), when
        a.broadcastable = (True, False, True, False)

    """
    idx = node.op.idx_list

    remove_dim = []
    node_inputs_idx = 1
    for dim, elem in enumerate(idx):
        if isinstance(elem, int):
            # The idx is a integer position.
            dim_index = node.inputs[node_inputs_idx]
            if isinstance(dim_index, ScalarConstant):
                dim_index = dim_index.value
            if dim_index in (0, -1) and node.inputs[0].broadcastable[dim]:
                remove_dim.append(dim)
                node_inputs_idx += 1
            else:
                return
        elif isinstance(elem, slice):
            if elem != slice(None):
                return
        else:
            raise TypeError("case not expected")

    if len(remove_dim) == 0:
        return
    else:
        all_dim = range(node.inputs[0].ndim)
        remain_dim = [x for x in all_dim if x not in remove_dim]
        return [node.inputs[0].dimshuffle(tuple(remain_dim))]


@register_canonicalize
@register_specialize
@node_rewriter([IncSubtensor, AdvancedIncSubtensor])
def local_useless_inc_subtensor(fgraph, node):
    r"""Remove redundant `IncSubtensor`\s.

    - ``x[full_slices].set(y)`` -> ``y``  (broadcast/cast to x's shape)
    - ``zeros[full_slices].inc(y)`` -> ``y``  (broadcast/cast to x's shape)
    - ``x[full_slices].inc(y)`` -> ``x + y``
    """

    x, y, *index_vars = node.inputs

    indices = indices_from_subtensor(index_vars, node.op.idx_list)

    # Check that all indices are full slices or full reversals
    if not all(
        isinstance(e, slice)
        and e.start is None
        and e.stop is None
        and (
            e.step is None
            or get_scalar_constant_value(
                e.step, only_process_constants=True, raise_not_constant=False
            )
            == -1
        )
        for e in indices
    ):
        return

    is_inc = not node.op.set_instead_of_inc
    x_is_zero = False
    if is_inc:
        try:
            x_is_zero = get_underlying_scalar_constant_value(x) == 0
        except NotScalarConstantError:
            pass

    # IncSubtensor casts y to x's dtype and broadcasts y onto x's shape
    out_dtype = node.outputs[0].type.dtype

    static_same = x.type.shape == y.type.shape and all(
        s is not None for s in x.type.shape
    )
    if not static_same:
        if hasattr(fgraph, "shape_feature") and fgraph.shape_feature.same_shape(x, y):
            static_same = True

    if y.type.dtype != out_dtype:
        y = cast(y, out_dtype)

    if not static_same:
        y = alloc(y, *x.shape)
        copy_stack_trace(node.outputs[0], y)

    if not all(e.step is None for e in node.op.idx_list):
        y = Subtensor(node.op.idx_list)(y, *index_vars)

    if not is_inc or x_is_zero:
        return [y]

    r = add(x, y)
    copy_stack_trace(node.outputs[0], r)
    return [r]


@register_canonicalize
@register_specialize
@node_rewriter([AdvancedIncSubtensor])
def local_set_to_inc_subtensor(fgraph, node):
    r"""Set of a read at the same indices: ``x[idx].set(x[idx] + other)``.

    .. code::

        x[idx].set(x[idx] + other) -> x[idx].inc(other)

    Only valid when ``idx`` is duplicate-free: a dense set is last-wins while an
    inc accumulates, so duplicate indices would over-count.

    TODO FIXME: Why doesn't this apply to all `*IncSubtensor*` `Op`\s?  If it
    did this wouldn't need to also be included in the "specialize" pass.

    """
    if not (
        node.op.set_instead_of_inc
        and node.inputs[1].owner
        and isinstance(node.inputs[1].owner.op, Elemwise)
        and isinstance(node.inputs[1].owner.op.scalar_op, Add)
    ):
        return
    addn = node.inputs[1].owner
    subn = None
    other = None

    if addn.inputs[0].owner and isinstance(addn.inputs[0].owner.op, AdvancedSubtensor):
        subn = addn.inputs[0].owner
        other = addn.inputs[1]
    elif addn.inputs[1].owner and isinstance(
        addn.inputs[1].owner.op, AdvancedSubtensor
    ):
        subn = addn.inputs[1].owner
        other = addn.inputs[0]
    else:
        return
    # The read has to gather exactly the positions the write stores to: same base,
    # same axes, same indices. `idx_list` must be compared too -- it is what pins
    # the indices to their axes, so without it a read of ``x[:, i]`` would match a
    # write to ``x[i]``.
    if (
        subn.inputs[0] != node.inputs[0]
        or subn.op.idx_list != node.op.idx_list
        or subn.inputs[1:] != node.inputs[2:]
    ):
        return
    # set->inc is only valid when the written positions are duplicate-free:
    # ``set(x[idx] + other)`` is last-wins at repeated positions, while
    # ``inc(other)`` would accumulate the contributions of every occurrence and
    # over-count them.
    indices = indices_from_subtensor(node.inputs[2:], node.op.idx_list)
    if not _advanced_indices_jointly_unique(indices, fgraph):
        return
    new_op = type(node.op)(
        idx_list=node.op.idx_list,
        inplace=node.op.inplace,
        set_instead_of_inc=False,
        ignore_duplicates=node.op.ignore_duplicates,
    )
    ret = new_op(node.inputs[0], other, *node.inputs[2:])

    copy_stack_trace(node.outputs, ret)

    return [ret]


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([add])
def local_add_of_sparse_write(fgraph, node):
    """Absorb a sparse write into a surrounding add: ``x + zeros[idx].set(v) -> x[idx].inc(v)``.

    A set into a zero-filled base is just the dense form of a sparse update.
    Adding it to another tensor is equivalent to incrementing in place, which
    avoids materialising the dense sparse representation.

    The ``zeros[idx].inc(v)`` form is rewritten unconditionally: inc applies the
    same per-position delta whether the base is zeros (then added to ``x``) or
    ``x`` itself, so duplicate indices accumulate identically on both sides. Only
    the ``zeros[idx].set(v)`` form needs duplicate-free indices, since a dense set
    is last-wins and collapsing it to an inc would over-count repeats.
    """
    for i, sparse_candidate in enumerate(node.inputs):
        if not (
            sparse_candidate.owner
            and isinstance(
                sparse_candidate.owner.op,
                IncSubtensor | AdvancedIncSubtensor,
            )
        ):
            continue

        inner_op = sparse_candidate.owner.op
        base, v, *idx_vars = sparse_candidate.owner.inputs

        if (
            get_underlying_scalar_constant_value(
                base, elemwise=False, raise_not_constant=False
            )
            != 0
        ):
            continue

        # Only the set->inc conversion needs duplicate-free indices. An inc into
        # zeros and the resulting inc into ``other`` apply the same per-position
        # delta (accumulating any duplicates identically), so ``x + zeros[idx].inc(v)
        # -> x[idx].inc(v)`` holds for any indices. A dense set, by contrast, is
        # last-wins, so collapsing it to an inc would over-count repeated
        # positions. Basic (slice/scalar) IncSubtensor is always unique; advanced
        # integer-array set indices must be jointly duplicate-free, weighing only
        # the advanced indices and not the flattened slice bounds.
        if inner_op.set_instead_of_inc and not isinstance(inner_op, IncSubtensor):
            indices = indices_from_subtensor(idx_vars, inner_op.idx_list)
            if not _advanced_indices_jointly_unique(indices, fgraph):
                continue

        others = [node.inputs[j] for j in range(len(node.inputs)) if j != i]
        other = variadic_add(*others)

        if inner_op.set_instead_of_inc:
            new_op = type(inner_op)(
                **(inner_op._props_dict() | {"set_instead_of_inc": False})
            )
        else:
            new_op = inner_op
        r = new_op(other, v, *idx_vars)
        copy_stack_trace([node.outputs[0], sparse_candidate], r)
        return [r]

    return None


@register_canonicalize
@register_specialize
@node_rewriter([Subtensor])
def local_useless_subtensor(fgraph, node):
    """Remove `Subtensor` if it takes the full input."""

    if not node.op.idx_list:
        return [node.inputs[0]]

    # The more elaborate optimization needs ShapeOpt and fgraph.shape_feature
    if not hasattr(fgraph, "shape_feature"):
        return

    shape_feature = fgraph.shape_feature

    cdata = get_constant_idx(
        node.op.idx_list,
        node.inputs,
        allow_partial=True,
        only_process_constants=True,
    )
    for pos, idx in enumerate(cdata):
        if not isinstance(idx, slice):
            # If idx is not a slice, this means we remove this dimension
            # from the output, so the subtensor is not useless
            return False
        if idx.start is not None and idx.start != 0:
            # If the start of the slice is different from 0, or is a
            # variable, then we assume the subtensor is not useless
            return False
        if idx.step is not None and idx.step != 1:
            # If we are going backwards, or skipping elements, then this
            # is not a useless subtensor
            return False

        length_pos = shape_feature.get_shape(node.inputs[0], pos)

        if isinstance(idx.stop, int | np.integer):
            length_pos_data = sys.maxsize
            try:
                length_pos_data = get_scalar_constant_value(
                    length_pos, only_process_constants=True
                )
            except NotScalarConstantError:
                pass

            if idx.stop < length_pos_data:
                return False
        elif isinstance(idx.stop, Variable):
            length_pos_shape_i = idx.stop
            # length_pos is a tensor variable, but length_pos_shape_i
            # is a scalar variable. We try to see if they represent
            # the same underlying variable.
            if length_pos_shape_i.owner and isinstance(
                length_pos_shape_i.owner.op, ScalarFromTensor
            ):
                length_pos_shape_i = length_pos_shape_i.owner.inputs[0]
            elif length_pos.owner and isinstance(length_pos.owner.op, TensorFromScalar):
                length_pos = length_pos.owner.inputs[0]
            else:
                # We did not find underlying variables of the same type
                return False

            # The type can be different: int32 vs int64. length_pos
            # should always be int64 as that is what the shape
            # tracker keep. Subtensor accept any scalar int{8,16,32,64}
            # as index type.
            assert str(length_pos.type.dtype) == "int64"
            assert str(length_pos_shape_i.type.dtype) in [
                "int8",
                "int16",
                "int32",
                "int64",
            ]

            # length_pos_shape_i cannot be None
            if length_pos_shape_i != length_pos:
                return False
        elif idx.stop is None:
            continue
        else:
            return False

    return [node.inputs[0]]


@register_canonicalize
@node_rewriter([Subtensor])
def local_convert_negative_indices(fgraph, node):
    """Convert negative indices in `Subtensor` with static length to positive indices."""
    x, *raw_idxs = node.inputs
    idxs = indices_from_subtensor(raw_idxs, node.op.idx_list)

    new_idxs = None
    for i, (dim_length, idx) in enumerate(zip(x.type.shape, idxs)):
        if (
            dim_length is None
            or isinstance(idx, slice)
            or not isinstance(idx, Constant)
        ):
            continue

        val = idx.data
        if val >= 0:
            continue

        new_val = val + dim_length
        if new_val < 0:
            # This is an invalid index, keep original to not confuse the user
            return None

        if new_idxs is None:
            new_idxs = list(idxs)
        new_idxs[i] = new_val

    if new_idxs is None:
        # No negative indices to convert
        return None

    new_subtensor = x[tuple(new_idxs)]
    copy_stack_trace(node.outputs, new_subtensor)
    return [new_subtensor]


def _arange_index_to_slice(idx):
    """Return the Python ``slice`` equivalent to ``idx`` if it is a constant or
    symbolic arange with non-negative offset, else ``None``.

    Negative offsets are not currently supported — some could be mapped to
    negative-start slices, but mixed-sign indices (where some wrap and some
    don't) cannot.  Symbolic matching covers
    step=1 only and additionally requires the arange ``stop`` to be provably
    non-negative — for negative ``stop``, ``arange`` returns empty whereas
    the slice ``[0:stop]`` wraps, and the two cannot be safely interchanged.
    """
    if not isinstance(idx, TensorVariable) or idx.type.ndim != 1:
        return None

    const_match = _constant_is_arange(idx)
    if const_match is not None:
        d, offset, step = const_match
        if offset < 0 or offset + (d - 1) * step < 0:
            return None
        stop_int = offset + d * step
        if step > 0:
            return (
                slice(offset, stop_int, step) if step != 1 else slice(offset, stop_int)
            )
        # Negative step: a negative ``stop`` would wrap, so walk through 0 with None.
        stop = stop_int if stop_int >= 0 else None
        return slice(offset, stop, step)

    sym_match = _match_arange_0_to_d_plus_offset(idx)
    if sym_match is None:
        return None
    arange_node, offset = sym_match
    _, arange_stop, _ = arange_node.owner.inputs
    arange_stop = _eager_scalar(arange_stop)
    if isinstance(arange_stop, TensorVariable) and arange_stop.type.dtype != "int64":
        arange_stop = arange_stop.astype("int64")
    offset = _eager_scalar(offset)
    if not _is_provably_non_negative(offset):
        return None
    if not _is_provably_non_negative(arange_stop):
        return None
    stop = eager_add_zero(arange_stop, offset)
    return slice(offset, stop)


@register_canonicalize
@register_specialize
@node_rewriter([AdvancedSubtensor])
def local_adv_idx_to_diagonal(fgraph, node):
    """Rewrite paired-arange advanced indices to ``base.diagonal(...)``.

    Recognizes ``base[arange(d)+r, arange(d)+c]`` on consecutive axes
    ``(a1, a2)`` and rewrites it to ``base.diagonal(offset=c-r, axis1=a1,
    axis2=a2)``. The const arm requires ``d == min(dim_a - r, dim_b - c)``
    (full diagonal coverage at this offset), making the rewrite shape-safe.
    """
    var, *idx_inputs = node.inputs
    indices = list(indices_from_subtensor(idx_inputs, node.op.idx_list))
    indices += [slice(None)] * (var.type.ndim - len(indices))

    adv_axes = [axis for axis, idx in enumerate(indices) if not isinstance(idx, slice)]
    if len(adv_axes) != 2 or adv_axes[1] != adv_axes[0] + 1:
        return None
    if any(isinstance(idx, slice) and idx != slice(None) for idx in indices):
        return None

    a1, a2 = adv_axes
    idx1, idx2 = indices[a1], indices[a2]

    # Match both indices as arange(d) + offset (const or symbolic).
    # Both must be the same kind (both const or both symbolic).
    def _match_arange(idx):
        const = _constant_is_arange(idx)
        if const is not None and const[2] == 1:
            return "const", const[0], const[1]
        sym = _match_arange_0_to_d_plus_offset(idx)
        if sym is not None:
            arange_node, offset = sym
            try:
                off_val = int(get_underlying_scalar_constant_value(offset))
            except NotScalarConstantError:
                return None
            return "sym", arange_node, off_val
        return None

    m1 = _match_arange(idx1)
    m2 = _match_arange(idx2)
    if m1 is None or m2 is None or m1[0] != m2[0]:
        return None
    kind, const_d_or_arange_to_d, row_off = m1
    _, const_d_or_arange_to_d2, col_off = m2

    # diagonal(offset=k) maps to [arange(d), arange(d)+k] (k>=0) or
    # [arange(d)+|k|, arange(d)] (k<0). Both offsets nonzero means this
    # is a sub-range gather that diagonal() can't express.
    if row_off != 0 and col_off != 0:
        return None

    # Both indices must cover the same number of elements.
    if const_d_or_arange_to_d != const_d_or_arange_to_d2:
        return None
    if kind == "const":
        stop = tensor_constant(np.int64(const_d_or_arange_to_d))
    else:
        stop = const_d_or_arange_to_d.owner.inputs[1]

    # Verify that the arange spans the full diagonal at this offset:
    # d == min(shape[a1] - row_off, shape[a2] - col_off).
    def _is_diag_length_term(term, axis, offset):
        if offset == 0:
            return _is_shape_of_x_at(term, var, axis)
        if not isinstance(term.owner_op, Elemwise):
            return False
        if not isinstance(term.owner.op.scalar_op, Sub):
            return False
        sub_a, sub_b = term.owner.inputs
        if not _is_shape_of_x_at(sub_a, var, axis):
            return False
        try:
            return int(get_underlying_scalar_constant_value(sub_b)) == offset
        except NotScalarConstantError:
            return False

    try:
        stop_val = int(get_underlying_scalar_constant_value(stop))
    except NotScalarConstantError:
        stop_val = None
    if stop_val is not None:
        dim_a, dim_b = var.type.shape[a1], var.type.shape[a2]
        if dim_a is None or dim_b is None:
            return None
        if stop_val != min(dim_a - row_off, dim_b - col_off):
            return None
    else:
        if not isinstance(stop.owner_op, Elemwise):
            return None
        if not isinstance(stop.owner.op.scalar_op, ScalarMinimum):
            return None
        term_a, term_b = stop.owner.inputs
        if not (
            (
                _is_diag_length_term(term_a, a1, row_off)
                and _is_diag_length_term(term_b, a2, col_off)
            )
            or (
                _is_diag_length_term(term_a, a2, col_off)
                and _is_diag_length_term(term_b, a1, row_off)
            )
        ):
            return None

    offset = col_off - row_off
    out = var.diagonal(offset=offset, axis1=a1, axis2=a2)
    # ``diagonal`` appends the diagonal as the last axis, but the original
    # ``base[..., arange, arange, ...]`` keeps the broadcast group at ``a1``
    # (consecutive advanced axes preserve their position in numpy).
    if a1 != out.type.ndim - 1:
        out = moveaxis(out, -1, a1)
    copy_stack_trace(node.outputs[0], out)
    return [out]


@register_canonicalize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([AdvancedSubtensor])
def local_adv_idx_to_slice(fgraph, node):
    """Rewrite a single arange-shaped advanced index to a basic slice.

    ``base[..., arange(d)+offset, ...]`` (all other axes full slices) becomes
    ``base[..., offset:offset+d, ...]``. Tagged ``shape_unsafe``: the slice
    silently truncates when the gather would be out-of-bounds, masking the
    ``IndexError`` the original ``AdvancedSubtensor`` would raise.
    """
    var, *idx_inputs = node.inputs
    indices = list(indices_from_subtensor(idx_inputs, node.op.idx_list))
    indices += [slice(None)] * (var.type.ndim - len(indices))

    adv_axes = [axis for axis, idx in enumerate(indices) if not isinstance(idx, slice)]
    if len(adv_axes) != 1:
        return None
    if any(isinstance(idx, slice) and idx != slice(None) for idx in indices):
        return None

    [axis] = adv_axes
    sl = _arange_index_to_slice(indices[axis])
    if sl is None:
        return None
    new_indices = list(indices)
    new_indices[axis] = sl
    out = var[tuple(new_indices)]
    copy_stack_trace(node.outputs[0], out)
    return [out]


def merge_two_slices(fgraph, slice1, len1, slice2, len2, allow_symbolic_refs=True):
    """Merge two consecutive slices into a single indexing operation.

    ``slice1`` must be a ``slice``; ``slice2`` can be a ``slice`` or a
    scalar index.  Both must have been applied consecutively on the same
    tensor.  ``len1`` / ``len2`` are the dimension lengths *before* and
    *after* applying ``slice1``.

    When *allow_symbolic_refs* is ``False``, the merge is only attempted
    when all components (slice bounds, steps, lengths) are constants.
    This avoids the symbolic ``switch / min / max`` trees that
    ``get_canonical_form_slice`` would otherwise produce.  Returns
    ``None`` when a symbolic component is detected and the flag is off.
    """

    if not isinstance(slice1, slice):
        raise ValueError("slice1 should be of type `slice`")

    # Simple case where one of the slices is useless
    if slice1 == slice(None):
        return slice2
    elif slice2 == slice(None):
        return slice1

    if not allow_symbolic_refs:
        vals = [len1, len2, slice1.start, slice1.stop, slice1.step]
        if isinstance(slice2, slice):
            vals.extend([slice2.start, slice2.stop, slice2.step])
        else:
            vals.append(slice2)
        if not all(v is None or isinstance(v, Constant) for v in vals):
            return None

    sl1, reverse1 = get_canonical_form_slice(slice1, len1)
    sl2, reverse2 = get_canonical_form_slice(slice2, len2)

    if not isinstance(sl2, slice):
        if reverse1 is None:
            # The first slice is not in reverse, which makes things a lot
            # more clear.
            # In this case we need to take care only of the special cases:
            # len2 <=0    -> throw index error regardless of sl2
            # sl2 > len2  -> throw index error
            # sl2 < -len2 -> throw index error
            # To get a index error we simply use len1+1 to indicate we are
            # out of bounds, because passing this index through the formula
            # of getting the mixed slice is not guaranteed to result in an
            # index error. The **issue though** if that the error will
            # complain about accessing element len1+1 which is probably not
            # too intuitive for the user
            val = sl1.start + sl2 * sl1.step
            val = switch(le(len2, 0), len1 + 1, val)
            val = switch(ge(sl2, len2), len1 + 1, val)
            val = switch(lt(sl2, 0), -len1 - 1, val)
            if sl1.step:
                val = switch(eq(sl1.step, 0), len1 + 1, val)
            return val
        else:
            # We are in the more complex case when we do not actually know
            # if the first slice was in reverse or not.
            # in case it was not in reverse:
            p_val = sl1.start + sl2 * sl1.step
            # case it was in reverse we need to realize that we do not want
            # the k-th element from sl.start but the k-th element from
            # sl.stop backwards
            n_val = sl1.stop - 1 - sl2 * sl1.step
            # we need to pick either n_val or p_val and then follow same
            # steps as above for covering the index error cases
            val = switch(lt(reverse1, 0), n_val, p_val)
            val = switch(le(len2, 0), len1 + 1, val)
            val = switch(ge(sl2, len2), len1 + 1, val)
            val = switch(lt(sl2, 0), -len1 - 1, val)
            if sl1.step is not None:
                val = switch(eq(sl1.step, 0), len1 + 1, val)
            return val
    else:
        # We are deleaing with two slices that need to be put together
        # according to the two steps we have 4 different combinations of
        # positive/negative. I will denote the case I'm looking at by
        # suffixes to the variables (nn,np,pn,pp):
        flen = sl2.stop - sl2.start
        p_step = sl1.step * sl2.step
        n_step = sl1.step * sl2.step * -1

        pp_start = minimum(sl1.start + sl2.start * sl1.step, sl1.stop)
        pp_stop = minimum(sl1.start + sl2.stop * sl1.step, sl1.stop)

        pn_stop = sl1.start + (sl2.start - 1) * sl1.step
        pn_stop = switch(
            and_(lt(pn_stop, 0), gt(flen, 0)),
            -len1 - 1,
            minimum(pn_stop, sl1.stop),
        )
        pn_start = sl1.start + (sl2.stop - 1) * sl1.step
        pn_start = minimum(pn_start, sl1.stop)
        pn_start = maximum(pn_start, 0)

        np_stop = sl1.stop - sl2.stop * sl1.step - 1
        np_stop = switch(
            and_(lt(np_stop, 0), gt(flen, 0)),
            -len1 - 1,
            maximum(sl1.start - 1, np_stop),
        )
        np_start = maximum(sl1.start, sl1.stop - sl2.start * sl1.step - 1)

        nn_start = maximum(sl1.start, (sl1.stop - 1) - (sl2.stop - 1) * sl1.step)
        nn_stop = maximum(sl1.start, sl1.stop - sl2.start * sl1.step)

        start = switch(
            lt(reverse2 * reverse1, 0),
            switch(lt(reverse1, 0), np_start, pn_start),
            switch(lt(reverse1, 0), nn_start, pp_start),
        )

        stop = switch(
            lt(reverse2 * reverse1, 0),
            switch(lt(reverse1, 0), np_stop, pn_stop),
            switch(lt(reverse1, 0), nn_stop, pp_stop),
        )

        step = switch(lt(reverse2 * reverse1, 0), n_step, p_step)
        start = switch(le(flen, 0), 0, start)
        stop = switch(le(flen, 0), 0, stop)

        return slice(start, stop, step)


@register_canonicalize
@node_rewriter([add])
def local_IncSubtensor_serialize(fgraph, node):
    """
    When using Subtensor, gradient graphs can be ugly.

    If we ask for grad(f(a[0]), a), we are going to get something like

        IncSubtensor(Elemwise{second}(a, 0), g(f(a[0])), [0])

    This might be ugly, but at least it's as fast as you could want.
    If we ask for grad(f(a[0], a[1], a[2]), a), it's much worse...

        Elemwise{Add}
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[0])), [0])
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[1])), [1])
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[2])), [2])

    This is much worse because this time we have to produce 3 matrices
    the size of 'a', just so we can add them together.

    This Op rearranges IncSubtensor's that all work on the same
    initial argument (here, Elemwise{second}(a,0)) into a chain.  The
    advantage of the chain structure is that each one can be optimized
    later in the pipeline to operate inplace.

    Ideally, the op will do something like this:

    #
    #  add(x, incsubtensor(b, c), incsubtensor(b, d))
    #  -> incsubtensor(incsubtensor(add(x,b,b), c), d)

    """

    def movable(i):
        # Return True iff this is a incsubtensor that we can move
        return (
            i.owner
            and isinstance(
                i.owner.op,
                IncSubtensor | AdvancedIncSubtensor,
            )
            and i.type.is_super(o_type)
            and len(fgraph.clients[i]) == 1
            and not i.owner.op.set_instead_of_inc
        )

    o_type = node.outputs[0].type

    movable_inputs = [i for i in node.inputs if movable(i)]

    if movable_inputs:
        new_inputs = [i for i in node.inputs if not movable(i)] + [
            mi.owner.inputs[0] for mi in movable_inputs
        ]
        new_add = variadic_add(*new_inputs)
        # Copy over stacktrace from original output, as an error
        # (e.g. an index error) in this add operation should
        # correspond to an error in the original add operation.
        copy_stack_trace(node.outputs[0], new_add)

        # stack up the new incsubtensors
        tip = new_add
        for mi in movable_inputs:
            assert o_type.is_super(tip.type)
            tip = mi.owner.op(tip, *mi.owner.inputs[1:])
            # Copy over stacktrace from outputs of the original
            # "movable" operation to the new operation.
            copy_stack_trace(node.outputs + mi.owner.outputs, tip)

        return [tip]


# We register it in a WalkingGraphRewriter inside the canonizer EQ optimizer.
# Otherwise in some cases it was making the EQ optimizer use 45. In
# the WalkingGraphRewriter, the EQ only use 5 passes.
compile.optdb.register(
    "pre_local_IncSubtensor_serialize",
    in2out(local_IncSubtensor_serialize),
    "fast_run",
    # Just before canonizer
    position=0.99,
)


# after priority 50 Destructive inplace operations
# gemm is the first one now, at priority 70


@node_rewriter([IncSubtensor], inplace=True)
def local_inplace_setsubtensor(fgraph, node):
    if node.op.inplace:
        return False
    dta = node.op.destroyhandler_tolerate_aliased
    new_op = node.op.__class__(
        node.op.idx_list,
        inplace=True,
        set_instead_of_inc=node.op.set_instead_of_inc,
        destroyhandler_tolerate_aliased=dta,
    )
    new_node = new_op(*node.inputs)
    val = getattr(node.outputs[0].tag, "nan_guard_mode_check", True)
    new_node.tag.nan_guard_mode_check = val

    # Copy stacktrace from original outputs to new outputs.
    # This is sensible, because the new operation is the
    # same as the old one, but now with different attributes.
    copy_stack_trace(node.outputs, new_node)
    return [new_node]


compile.optdb.register(
    "local_inplace_setsubtensor",
    WalkingGraphRewriter(
        local_inplace_setsubtensor, failure_callback=WalkingGraphRewriter.warn_inplace
    ),
    "fast_run",
    "inplace",
    position=50.1,
)


@node_rewriter([AdvancedIncSubtensor], inplace=True)
def local_inplace_AdvancedIncSubtensor(fgraph, node):
    if node.op.inplace:
        return False
    x, y, *idxs = node.inputs
    if fgraph.has_destroyers([x]):
        # In this case we can't operate inplace, but if x is just an alloc of zeros
        # We're better off duplicating it and then acting on it inplace.
        if (
            x.owner is not None
            and isinstance(x.owner.op, Alloc)
            and x.owner.op.value_is_scalar_zero(x.owner.inputs[0])
        ):
            x = x.owner.clone().outputs[0]
        else:
            return None  # Inplace isn't valid
    new_op = type(node.op)(
        node.op.idx_list,
        inplace=True,
        set_instead_of_inc=node.op.set_instead_of_inc,
        ignore_duplicates=node.op.ignore_duplicates,
    )
    new_node = new_op(x, y, *idxs)
    copy_stack_trace(node.outputs, new_node)
    return [new_node]


compile.optdb.register(
    "local_inplace_AdvancedIncSubtensor",
    WalkingGraphRewriter(
        local_inplace_AdvancedIncSubtensor,
        failure_callback=WalkingGraphRewriter.warn_inplace,
    ),
    "fast_run",
    "inplace",
    position=70.6,
)


# Register old name
@register_canonicalize("local_incsubtensor_of_allocs")
@register_stabilize("local_incsubtensor_of_allocs")
@node_rewriter([IncSubtensor, AdvancedIncSubtensor])
def local_incsubtensor_of_zeros(fgraph, node):
    """
    IncSubtensor(x, zeros, idx) -> x

    """
    if (
        isinstance(node.op, IncSubtensor | AdvancedIncSubtensor)
        and not node.op.set_instead_of_inc
    ):
        x = node.inputs[0]
        y = node.inputs[1]
        try:
            # Don't use only_process_constants=True. We need to
            # investigate Alloc of 0s but with non constant shape.
            if get_underlying_scalar_constant_value(y, elemwise=False) == 0:
                # No need to copy over the stacktrace,
                # because x should already have a stacktrace
                return [x]
        except NotScalarConstantError:
            return


@register_canonicalize
@register_specialize
@node_rewriter([IncSubtensor])
def local_incsubtensor_of_zeros_to_setsubtensor(fgraph, node):
    """
    IncSubtensor(zeros, x, ...) -> SetSubtensor(zeros, x, ...)
    """
    if node.op.set_instead_of_inc:
        return
    x = node.inputs[0]

    if isinstance(x, Constant) and not np.any(x.data):
        return [
            IncSubtensor(
                node.op.idx_list,
                node.op.inplace,
                set_instead_of_inc=True,
                destroyhandler_tolerate_aliased=node.op.destroyhandler_tolerate_aliased,
            )(*node.inputs)
        ]


@register_canonicalize("local_setsubtensor_of_allocs")
@register_stabilize("local_setsubtensor_of_allocs")
@node_rewriter([IncSubtensor])
def local_setsubtensor_of_constants(fgraph, node):
    """
    SetSubtensor(x, x[idx], idx) -> x

    when x is constant or alloc.

    """
    if not node.op.set_instead_of_inc:
        return
    x = node.inputs[0]
    y = node.inputs[1]

    # Don't use only_process_constants=True. We need to
    # investigate Alloc of 0s but with non constant shape.
    try:
        replace_x = get_underlying_scalar_constant_value(x, elemwise=False)
    except NotScalarConstantError:
        return

    try:
        replace_y = get_underlying_scalar_constant_value(y, elemwise=False)
    except NotScalarConstantError:
        return

    if replace_x == replace_y:
        # No need to copy over the stacktrace,
        # because x should already have a stacktrace
        return [x]
    else:
        return False


@register_canonicalize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([Subtensor, AdvancedSubtensor])
def local_read_of_write_same_indices(fgraph, node):
    """Read of a write at the same indices: ``x[idx].set/inc(v)[idx]``.

    .. code::

        x[idx].set(v)[idx] -> v
        x[idx].inc(v)[idx] -> x[idx] + v   (idx must be duplicate-free)

    Applies when the outer read and inner write share identical index
    variables (``is`` check) and the same ``idx_list``.  The inc case
    additionally requires duplicate-free indices: slices and scalar indices
    are trivially unique, while advanced integer-array indices must be jointly
    duplicate-free (see ``_indices_jointly_unique``).

    Companion rewrites:

    - ``local_advanced_read_of_write_constant_indices`` handles the multi-axis
      case when read and write indices differ but are both constant.
    - ``local_write_of_write_same_indices`` collapses nested write chains.
    """
    if isinstance(node.op, Subtensor):
        write_type = (IncSubtensor,)
    else:
        write_type = (AdvancedIncSubtensor,)

    inner = node.inputs[0]
    if not (inner.owner and isinstance(inner.owner.op, write_type)):
        return None

    if node.op.idx_list != inner.owner.op.idx_list:
        return None

    x, v, *inner_idx_vars = inner.owner.inputs
    outer_idx_vars = node.inputs[1:]

    if not all(o is i for o, i in zip(outer_idx_vars, inner_idx_vars, strict=True)):
        return None

    out = node.outputs[0]

    if inner.owner.op.set_instead_of_inc:
        r = cast(v, out.dtype)
        if not r.type.is_super(out.type):
            r = alloc(r, *out.shape)
        copy_stack_trace(out, r)
        return [r]
    else:
        # Inc case: advanced integer-array indices must be jointly duplicate-free;
        # slices and scalar indices are trivially unique.
        indices = indices_from_subtensor(outer_idx_vars, node.op.idx_list)
        if not _advanced_indices_jointly_unique(indices, fgraph):
            return None

        x_at_idx = x[tuple(indices)]
        copy_stack_trace(out, x_at_idx)
        r = x_at_idx + v
        copy_stack_trace(out, r)
        return [r]


def _slice_to_arange(sl, dim_length):
    """Convert ``sl`` to the equivalent ``arange``-shaped index, or ``None``.

    - constant ``slice(start, stop, step)`` with all ``>= 0`` and ``step > 0``
      → ``tensor_constant(np.arange(start, stop, step))``.
    - symbolic ``slice(0|None, stop, 1|None)`` with provably non-negative
      ``stop`` → ``arange(minimum(stop, dim_length))``.
    - ``slice(None, None, None)`` → ``arange(dim_length)``.
    """
    try:
        start = 0 if sl.start is None else int(as_index_literal(sl.start))
        stop = int(as_index_literal(sl.stop))
        step = 1 if sl.step is None else int(as_index_literal(sl.step))
        if start >= 0 and stop >= 0 and step > 0:
            return tensor_constant(np.arange(start, stop, step))
        return None
    except (TypeError, NotScalarConstantError):
        pass
    if sl.start is not None:
        try:
            if int(as_index_literal(sl.start)) != 0:
                return None
        except (TypeError, NotScalarConstantError):
            return None
    if sl.step is not None:
        try:
            if int(as_index_literal(sl.step)) != 1:
                return None
        except (TypeError, NotScalarConstantError):
            return None
    if sl.stop is None:
        return arange(dim_length)
    if not _is_provably_non_negative(sl.stop):
        return None
    return arange(minimum(sl.stop, dim_length))


@register_canonicalize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([Subtensor])
def local_slice_read_of_write(fgraph, node):
    """Simplify ``x[write_idx].set/inc(v)[slices]`` when we slice the written axes.

    Converts slice indices to ``arange`` on axes where the write uses advanced
    indexing, then delegates to ``local_advanced_read_of_write_constant_indices``.
    """
    read_node = node

    write_node = node.inputs[0].owner
    if not (write_node is not None and isinstance(write_node.op, AdvancedIncSubtensor)):
        return None

    read_idx_list = read_node.op.idx_list
    write_idx_list = write_node.op.idx_list

    if len(read_idx_list) > len(write_idx_list) or read_idx_list == write_idx_list:
        return None

    read_indices = unflatten_index_variables(read_node.inputs[1:], read_idx_list)
    write_indices = unflatten_index_variables(write_node.inputs[2:], write_idx_list)

    buffer_shape = tuple(write_node.inputs[0].shape)
    new_indices: list = []
    for axis, (read_idx, write_idx) in enumerate(
        zip_longest(read_indices, write_indices, fillvalue=slice(None))
    ):
        read_is_slice = isinstance(read_idx, slice)
        write_is_slice = isinstance(write_idx, slice)
        if read_is_slice and not write_is_slice:
            arange_index = _slice_to_arange(read_idx, buffer_shape[axis])
            if arange_index is None:
                return None
            else:
                new_indices.append(arange_index)
                continue
        elif read_is_slice != write_is_slice:
            return None
        else:
            new_indices.append(read_idx)

    new_read = write_node.out[tuple(new_indices)].owner
    return local_advanced_read_of_write_constant_indices.fn(fgraph, new_read)


@register_canonicalize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([AdvancedSubtensor])
def local_advanced_read_of_write_constant_indices(fgraph, node):
    """Read of a write at possibly-different constant indices.

    .. code::

        x[w_idx].set(v)[r_idx] ->
            v[lookup]                     (full coverage — base irrelevant)
            x[r_idx]                      (no coverage — set irrelevant)
            x[r_idx].set(v[k])[...]       (partial — mix base and v)

        x[w_idx].inc(v)[r_idx] ->
            x[r_idx] + v[lookup]          (full coverage, unique writes)
            x[r_idx]                      (no coverage)
            x[r_idx].inc(v[k])[...]       (partial, unique writes)

    Fires when each advanced index is materializable: a 1-D integer
    ``TensorConstant`` (boolean masks accepted) or a symbolic
    ``arange(n_const) + offset`` whose stop is a ``TensorConstant``.  The
    ``idx_list`` must match between read and write.  The inc case
    additionally requires unique joint write coords so each read coord
    matches at most one write.

    Companion rewrites:

    - ``local_read_of_write_same_indices`` handles the identity-check case
      (symbolic indices allowed) for basic and advanced subtensors.
    - ``local_write_of_write_same_indices`` collapses nested write chains.
    """
    inner = node.inputs[0]
    if not (inner.owner and isinstance(inner.owner.op, AdvancedIncSubtensor)):
        return None

    inner_op = inner.owner.op
    is_set = inner_op.set_instead_of_inc

    # Both must have the same idx_list structure (same axes advanced-indexed).
    if node.op.idx_list != inner_op.idx_list:
        return None

    base, v, *write_idx_inputs = inner.owner.inputs
    read_idx_inputs = node.inputs[1:]

    write_indices = indices_from_subtensor(write_idx_inputs, inner_op.idx_list)
    read_indices = indices_from_subtensor(read_idx_inputs, node.op.idx_list)

    # Collect advanced (integer) axes; other axes must be identical slices.
    # Cross-sign indices are rejected since negatives can alias positives
    # (a normalisation rewrite handles those separately).
    write_arrs = []
    read_arrs = []
    for w, r in zip(write_indices, read_indices, strict=True):
        if isinstance(w, TensorVariable) and isinstance(r, TensorVariable):
            if w.type.broadcastable != (False,) or r.type.broadcastable != (False,):
                return None
            w_arr = _idx_to_int_array(w)
            r_arr = _idx_to_int_array(r)
            if w_arr is None or r_arr is None:
                return None
            # Reject only cross-sign within an axis — negatives can alias
            # positives on the same axis, but uniformly negative (or
            # uniformly non-negative) indices compare correctly as raw values.
            # Short-circuit so the common all-non-negative case skips most checks.
            if ((w_arr < 0).any() or (r_arr < 0).any()) and (
                (w_arr >= 0).any() or (r_arr >= 0).any()
            ):
                return None
            write_arrs.append(w_arr)
            read_arrs.append(r_arr)
        elif isinstance(w, slice) and isinstance(r, slice):
            if w != r:
                return None
        else:
            return None

    if not write_arrs:
        return None

    n_write = len(write_arrs[0])

    # Extend indices with implicit trailing slices so axis bookkeeping is
    # uniform regardless of whether the subtensor indexed all base dims.
    n_trailing = base.type.ndim - len(write_indices)
    full_write = list(write_indices) + [slice(None)] * n_trailing

    # Compute where the advanced axis lands in the result of x[indices], per
    # numpy semantics: hoisted to position 0 if the adv indices are split by
    # slices, otherwise kept in place at the position of the first adv axis
    # (counting only slice axes that come before it, since collapsed adv
    # axes share one output dim).
    adv_axes = [
        i for i, idx in enumerate(full_write) if isinstance(idx, TensorVariable)
    ]
    if _non_consecutive_adv_indexing(full_write):
        adv_pos = 0
        slice_shapes = [
            base.shape[i] for i in range(base.type.ndim) if i not in set(adv_axes)
        ]
    else:
        first_adv = min(adv_axes)
        last_adv = max(adv_axes)
        pre = [
            base.shape[i] for i in range(first_adv) if isinstance(full_write[i], slice)
        ]
        post = [
            base.shape[i]
            for i in range(last_adv + 1, base.type.ndim)
            if isinstance(full_write[i], slice)
        ]
        adv_pos = len(pre)
        slice_shapes = pre + post

    # Bring v to its full natural shape so we can index the adv axis directly.
    natural_shape_v = [*slice_shapes[:adv_pos], n_write, *slice_shapes[adv_pos:]]
    v = alloc(v, *natural_shape_v)
    # set_subtensor/inc_subtensor cast v to the buffer dtype internally; we need
    # to do it explicitly so v[lookup] (and subsequent ops) match the output dtype.
    out_dtype = node.outputs[0].type.dtype
    if v.type.dtype != out_dtype:
        v = cast(v, out_dtype)

    write_coords = np.column_stack(write_arrs)  # (n_write, n_axes)
    read_coords = np.column_stack(read_arrs)  # (n_read, n_axes)

    if is_set:
        # Set: last-write-wins; uncovered positions need the base.
        write_dict: dict[tuple, int] = {}
        for k in range(len(write_coords)):
            write_dict[tuple(write_coords[k])] = k
    else:
        # Inc: require unique write coords so each read matches at most one
        # write.  With duplicates we'd need a scatter-add at write positions,
        # which generally isn't simpler than the original inc.
        write_dict = {}
        for k in range(len(write_coords)):
            coord = tuple(write_coords[k])
            if coord in write_dict:
                return None
            write_dict[coord] = k

    lookup = np.array(
        [write_dict.get(tuple(rc), -1) for rc in read_coords], dtype=np.int64
    )
    covered = lookup >= 0

    def constant_idx(idx, merge_feature=getattr(fgraph, "merge_feature", None)):
        # Build (or reuse) the TensorConstant that indexes the adv axis.
        # Read-write graphs on the same idx require structural identity
        # To not be at the mercy of MergeOptimizer firing in time,
        # we eagerly reuse index variables if they already exist in the graph
        # (which is the case in which those rewrites would need)
        idx = tensor_constant(idx)
        if merge_feature is None:
            return idx
        else:
            return merge_feature.atomic_sig_inv.get(idx.signature(), idx)

    def index_adv(t, positions):
        # Index axis `adv_pos` of t with `positions`. Skip if identity.
        if len(positions) == n_write and np.array_equal(positions, np.arange(n_write)):
            return t
        indexer = [slice(None)] * t.type.ndim
        indexer[adv_pos] = constant_idx(positions)
        return t[tuple(indexer)]

    if is_set:
        if covered.all():
            # Every read position is overwritten; base is irrelevant.
            out = index_adv(v, lookup)
        elif not covered.any():
            # No read position is overwritten; the set is irrelevant.
            out = base[tuple(read_indices)]
        else:
            # Mix: read base, then overwrite covered positions with v values.
            base_part = base[tuple(read_indices)]
            covered_read = np.flatnonzero(covered)
            covered_write = lookup[covered]
            indexer = [slice(None)] * base_part.type.ndim
            indexer[adv_pos] = constant_idx(covered_read)
            out = base_part[tuple(indexer)].set(index_adv(v, covered_write))
    else:
        # Inc case (write coords are unique by construction above).
        base_part = base[tuple(read_indices)]
        copy_stack_trace(node.outputs[0], base_part)
        if not covered.any():
            return [base_part]

        if covered.all():
            out = base_part + index_adv(v, lookup)
        else:
            covered_read = np.flatnonzero(covered)
            covered_write = lookup[covered]
            indexer = [slice(None)] * base_part.type.ndim
            indexer[adv_pos] = constant_idx(covered_read)
            out = base_part[tuple(indexer)].inc(index_adv(v, covered_write))

    copy_stack_trace(node.outputs[0], out)
    return [out]


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([IncSubtensor, AdvancedIncSubtensor])
def local_write_of_write_same_indices(fgraph, node):
    """Collapse nested write ops that share the same indices.

    .. code::

        x[idx].set/inc(a)[idx].set(b) -> x[idx].set(b)
        x[idx].inc(a)[idx].inc(b)     -> x[idx].inc(a + b)
        x[idx].set(a)[idx].inc(b)     -> x[idx].set(a + b)   [unique idx]

    Outer-set always applies (it shadows the inner write).  Inc+inc is safe
    because addition is associative.  Inc-of-set requires duplicate-free
    indices: slices are trivially unique, advanced indices are checked
    per-axis (conservative — joint-tuple uniqueness would be exact).

    If the inc-of-set base is zero-filled the result is emitted as an
    ``inc`` so downstream zero-aware rewrites can still fire.

    Typically arises from gradient accumulation or user code that writes
    then updates the same slice (e.g. Scan updates).

    Companion rewrites:

    - ``local_read_of_write_same_indices`` simplifies a read following a
      write at the same indices.
    - ``local_advanced_read_of_write_constant_indices`` handles multi-axis
      reads with differing constant indices.
    """
    # AdvancedIncSubtensor.ignore_duplicates is not a concern here:
    # the outer-set and inc+inc cases are valid regardless of duplicates,
    # and the inc-of-set case requires verified-unique indices so there
    # are no duplicates for the flag to affect.
    inner_x, b, *outer_idx_vars = node.inputs
    if not (inner_x.owner and isinstance(inner_x.owner.op, type(node.op))):
        return

    base, a, *inner_idx_vars = inner_x.owner.inputs

    # Same indices: idx_list (slice specs) must match and all index
    # variables must be identical.
    if node.op.idx_list != inner_x.owner.op.idx_list:
        return
    if not all(o is i for o, i in zip(outer_idx_vars, inner_idx_vars, strict=True)):
        return

    outer_is_set = node.op.set_instead_of_inc
    inner_is_set = inner_x.owner.op.set_instead_of_inc

    if outer_is_set:
        # Outer set shadows inner completely.
        new_val = b
        use_set = True
    elif inner_is_set:
        # x[idx].set(a)[idx].inc(b) — needs unique indices. Basic indexing
        # (slices/scalars) is always duplicate-free; advanced indices must have
        # duplicate-free joint coordinate tuples.
        if not isinstance(node.op, IncSubtensor):
            indices = indices_from_subtensor(outer_idx_vars, node.op.idx_list)
            if not _advanced_indices_jointly_unique(indices, fgraph):
                return
        new_val = a + b
        if (
            get_underlying_scalar_constant_value(
                base, elemwise=False, raise_not_constant=False
            )
            == 0
        ):
            use_set = False
        else:
            use_set = True
    else:
        # x[idx].inc(a)[idx].inc(b) — always safe (addition is associative).
        new_val = a + b
        use_set = False

    # ignore_duplicates is deliberately not propagated: the merged op
    # should use the safe np.add.at path (the default).
    new_op = type(node.op)(idx_list=node.op.idx_list, set_instead_of_inc=use_set)
    r = new_op(base, new_val, *outer_idx_vars)
    copy_stack_trace(node.outputs[0], r)
    return [r]


@register_specialize
@register_stabilize
@register_canonicalize
@register_useless
@node_rewriter([IncSubtensor, AdvancedIncSubtensor])
def local_useless_inc_subtensor_alloc(fgraph, node):
    """
    Replaces an [Advanced]IncSubtensor[1], whose increment is an `alloc` of
    a fully or partially broadcastable variable, by one that skips the
    intermediate `alloc` where possible.

    """
    if isinstance(node.op, IncSubtensor | AdvancedIncSubtensor):
        x, y, *index_variables = node.inputs

        if y.owner is not None and isinstance(y.owner.op, Alloc):
            # `z` is the input of the Alloc op, i.e. at.alloc(z, <shape>)
            z = y.owner.inputs[0]

            try:
                shape_feature = fgraph.shape_feature
            except AttributeError:
                # The shape feature may not be available in some mode, but we
                # need it for this optimization, so don't continue.
                return False

            same_shape = shape_feature.same_shape

            # Get the subtensor of `x` indexed by `i` in order to compare
            # shapes later.
            if isinstance(node.op, IncSubtensor):
                xi = Subtensor(node.op.idx_list)(x, *index_variables)
            elif isinstance(node.op, AdvancedIncSubtensor):
                xi = AdvancedSubtensor(node.op.idx_list)(x, *index_variables)
            else:
                raise Exception("Should never happen!")

            # `xi` may have more dimensions than `y` since the subtensor ops
            # do automatic broadcasting of the increment internally. Thus, we
            # need to make the leading implicitly broadcasted dimensions
            # explicit for shape comparison later.
            if xi.ndim > y.ndim:
                y = shape_padleft(y, xi.ndim - y.ndim)

            # Build `z_broad` explicitly to include extra implicit dimensions.
            z_broad = (True,) * (xi.ndim - z.ndim) + z.broadcastable

            cond = [
                # The shapes of `y` and `xi` must either agree or `y` may
                # also have shape equal to 1 which may be treated as a
                # broadcastable dimension by the subtensor op.
                or_(eq(y.shape[k], 1), eq(y.shape[k], xi.shape[k]))
                # Loop over all dimensions.
                for k in range(xi.ndim)
                # We need to check the above shapes, if
                # * the pre-alloc increment `z` is broadcastable in
                # dimension `k` (if it isn't, then the shapes of `z` and
                # `y` are the same by the definition of the `Alloc` op in
                # this dimension and replacing `y` by `z` will not hide a
                # shape error), and
                # * `xi` and `y` do not have the same shape in dimension
                # `k` or we cannot infer the shape statically (if the
                # shapes of `xi` and `y` are not the same, then replacing
                # `y` by `z` will hide the shape error of `y`), and
                # * the shape of `y` is not equal to 1 or we cannot infer
                # the shape statically (if the shape of `y` is equal to
                # 1, then `y` is broadcasted by the inc_subtensor op
                # internally, so the shapes of `xi` and `y` do not need
                # to match in dimension `k`; else we need to check at
                # runtime that the shape of `y` is either 1 or the same
                # as `xi` or otherwise replacing `y` by `z` will hide a
                # shape error).
                if (
                    z_broad[k]
                    and not same_shape(xi, y, dim_x=k, dim_y=k)
                    and shape_feature.get_shape(y, k) != 1
                )
            ]

            if len(cond) > 0:
                msg = "`x[i]` and `y` do not have the same shape."
                z = Assert(msg)(z, *cond)

            r = node.op(x, z, *index_variables)
            # Copy over stacktrace from previous output, since
            # we don't expect problems when removing the intermediate
            # alloc operation and so we still want to point at the line
            # of the inc_subtensor operation.
            copy_stack_trace(node.outputs, r)

            return [r]


@register_specialize
@node_rewriter([Join])
def local_join_subtensors(fgraph, node):
    r"""Simplify contiguous :class:`Subtensor`\s inside a :class:`Join`.

    `join((x[:3], x[3:5]), axis=0) -> x[:5]`
    """
    # TODO: Generalize to AdvancedSubtensors

    tensors = node.inputs
    axis = node.op.axis

    for subtensor1_idx, (subtensor1, subtensor2) in enumerate(pairwise(tensors)):
        # Check that two consecutive Subtensors are operating on the same base tensor
        if not (
            (
                subtensor1.owner is not None
                and isinstance(subtensor1.owner.op, Subtensor)
            )
            and (
                subtensor2.owner is not None
                and isinstance(subtensor2.owner.op, Subtensor)
            )
            and (subtensor1.owner.inputs[0] is subtensor2.owner.inputs[0])
        ):
            continue

        # Check that subtensors have consecutive indexes across the join axis
        idxs_subtensor1 = indices_from_subtensor(
            subtensor1.owner.inputs[1:], subtensor1.owner.op.idx_list
        )
        idxs_subtensor2 = indices_from_subtensor(
            subtensor2.owner.inputs[1:], subtensor2.owner.op.idx_list
        )
        try:
            idxs_axis_subtensor1 = idxs_subtensor1[axis]
            idxs_axis_subtensor2 = idxs_subtensor2[axis]
        except IndexError:
            continue
        if not (
            isinstance(idxs_axis_subtensor1, slice)
            and isinstance(idxs_axis_subtensor2, slice)
        ):
            continue
        start_subtensor1, stop_subtensor1, step_subtensor1 = (
            idxs_axis_subtensor1.start,
            idxs_axis_subtensor1.stop,
            idxs_axis_subtensor1.step,
        )
        start_subtensor2, stop_subtensor2, step_subtensor2 = (
            idxs_axis_subtensor2.start,
            idxs_axis_subtensor2.stop,
            idxs_axis_subtensor2.step,
        )
        if not (
            (stop_subtensor1 is not None and start_subtensor2 is not None)
            and (stop_subtensor1 == start_subtensor2)
        ):
            continue

        # Check that step is None or 1
        # For non-unit steps (perhaps except for -1) we would need to know the
        # exact values of start and stop to know if they can be merged
        for step in (step_subtensor1, step_subtensor2):
            if step is None:
                continue
            try:
                if get_scalar_constant_value(step, only_process_constants=True) != 1:
                    return None
            except NotScalarConstantError:
                return None

        # Check that all other idxs of subtensor are the same
        if all(
            idxs_nonaxis_subtensor1 == idxs_nonaxis_subtensor2
            for i, (idxs_nonaxis_subtensor1, idxs_nonaxis_subtensor2) in enumerate(
                zip(idxs_subtensor1, idxs_subtensor2, strict=True)
            )
            if i != axis
        ):
            base_tensor = subtensor1.owner.inputs[0]
            new_idxs = list(idxs_subtensor1)
            new_idxs[axis] = slice(start_subtensor1, stop_subtensor2, step_subtensor1)
            merged_subtensors = base_tensor[new_idxs]

            new_joined_tensors = [
                *tensors[:subtensor1_idx],
                merged_subtensors,
                *tensors[subtensor1_idx + 2 :],
            ]
            if len(new_joined_tensors) > 1:
                return [concatenate(new_joined_tensors, axis=axis)]
            else:
                return [merged_subtensors]


@node_rewriter(
    [
        Subtensor,
        AdvancedSubtensor,
        IncSubtensor,
        AdvancedIncSubtensor,
    ]
)
def local_uint_constant_indices(fgraph, node):
    """Convert constant indices to unsigned dtypes."""

    op = node.op
    if isinstance(op, IncSubtensor | AdvancedIncSubtensor):
        x, y, *indices = node.inputs
    else:
        x, *indices = node.inputs
        y = None

    new_indices = list(indices_from_subtensor(indices, node.op.idx_list))
    has_new_index = False

    for i, index in enumerate(new_indices):
        if not isinstance(index, Constant):
            continue

        index_val = index.data

        if index_val is None or isinstance(index_val, slice):
            # TODO: If slice index dtypes matter, we can consider converting
            # those, as well.
            continue

        assert isinstance(index_val, np.generic | np.ndarray)

        if index_val.size == 0:
            continue

        if index_val.dtype == bool:
            continue

        if np.ndim(index_val) > 0:
            minval = index_val.min()
        else:
            minval = index_val

        if minval >= 0:
            maxval = index_val.max()
            dtype = np.min_scalar_type(maxval)
        else:
            # If we can't convert to unsigned, then don't attempt to minimize
            # the type size either--at least not for now.
            # dtype = np.min_scalar_type(-max(-minval, maxval))
            continue

        if dtype == index_val.dtype:
            continue

        if isinstance(index.type, TensorType):
            new_index = tensor_constant(index_val.astype(dtype), dtype=dtype)
        else:
            new_index = scalar_constant(index_val.astype(dtype), dtype=dtype)

        new_indices[i] = new_index
        has_new_index = True

    if not has_new_index:
        return False

    new_indices = get_slice_elements(new_indices)
    new_args = (x, *new_indices) if y is None else (x, y, *new_indices)
    new_out = op(*new_args)
    copy_stack_trace(node.outputs[0], new_out)
    return [new_out]


compile.optdb.register(
    local_uint_constant_indices.__name__,
    dfs_rewriter(local_uint_constant_indices),
    # We don't include in the Python / C because those always cast indices to int64 internally.
    "numba",
    "jax",
    # After specialization and uncanonicalization
    # Other rewrites don't worry about the dtype of the indices
    # And can cause unnecessary passes of this optimization
    # Such as x.shape[np.int(0)] -> x.shape[np.uint(0)]
    position=4,
)


@register_stabilize
@register_specialize
@node_rewriter([blockwise_of(Subtensor)])
def local_blockwise_of_subtensor(fgraph, node):
    """Rewrite Blockwise of Subtensor, where the only batch input is the indexed tensor.

    Blockwise(Subtensor{a: b})(x, a, b) -> x[:, a:b] when x has one batch dimension, and a/b none

    TODO: Handle batched indices like we do with blockwise of inc_subtensor
    TODO: Extend to AdvanceSubtensor
    """
    x, *idxs = node.inputs
    if not all(all(idx.type.broadcastable) for idx in idxs):
        return

    core_idxs = indices_from_subtensor(
        [idx.squeeze() for idx in idxs], node.op.core_op.idx_list
    )
    # Add empty slices for the batch dims
    none_slices = (slice(None),) * node.op.batch_ndim(node)
    return [x[(*none_slices, *core_idxs)]]


@register_canonicalize("shape_unsafe")
@register_stabilize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([blockwise_of(IncSubtensor | AdvancedIncSubtensor)])
def local_blockwise_inc_subtensor(fgraph, node):
    """Rewrite blockwised inc_subtensors.

    Note: The reason we don't apply this rewrite eagerly in the `_vectorize_node` dispatch
    Is that we often have batch dimensions from alloc of shapes/reshape that can be removed by rewrites

    such as x[:vectorized(w.shape[0])].set(y), that will later be rewritten as x[:w.shape[1]].set(y),
    and can be safely rewritten without Blockwise.
    """
    core_op = node.op.core_op
    x, y, *idxs = node.inputs
    [out] = node.outputs
    advanced = isinstance(core_op, AdvancedIncSubtensor)

    if advanced and any(idx.type.dtype == "bool" for idx in idxs):
        # Get out if we have boolean indices as they cross dimension boundaries
        # / can't be safely broadcasted depending on their runtime content
        return None

    batch_ndim = node.op.batch_ndim(node)
    idxs_core_ndim = [len(inp_sig) for inp_sig in node.op.inputs_sig[2:]]
    max_idx_core_ndim = max(idxs_core_ndim, default=0)

    # Broadcast buffer to batch_shape. The output batch shape and broadcast
    # pattern are derived from the inputs, never from `out.type`, which can be
    # stale after an upstream rewrite swaps an input.
    batch_shape = [1] * batch_ndim
    out_batch_bcast = [True] * batch_ndim
    for inp in node.inputs:
        for i, (broadcastable, batch_dim) in enumerate(
            zip(inp.type.broadcastable[:batch_ndim], tuple(inp.shape)[:batch_ndim])
        ):
            if broadcastable:
                # This dimension is broadcastable, it doesn't provide shape information
                continue
            out_batch_bcast[i] = False
            if batch_shape[i] != 1:
                # We already found a source of shape for this batch dimension
                continue
            batch_shape[i] = batch_dim

    if list(x.type.broadcastable[:batch_ndim]) != out_batch_bcast:
        x = broadcast_to(x, (*batch_shape, *x.shape[batch_ndim:]))

    # Massage indices so they respect blockwise semantics while using regular indexing
    core_idxs = []
    for idx_entry in core_op.idx_list:
        if isinstance(idx_entry, slice):
            # Squeeze away dummy dimensions so we can convert to slice
            new_entries = [None, None, None]
            for i, slice_idx_entry in enumerate(
                (idx_entry.start, idx_entry.stop, idx_entry.step)
            ):
                if slice_idx_entry is None:
                    continue
                else:
                    new_entries[i] = new_entry = idxs[slice_idx_entry].squeeze()
                    if new_entry.ndim > 0:
                        # If the slice entry has dimensions after the squeeze we can't convert it to a slice
                        # We could try to convert to equivalent integer indices, but nothing guarantees
                        # that the slice is "square".
                        return None
            squeezed_index = slice(*new_entries)
        else:
            if advanced:
                # For AdvancedIncSubtensor we have tensor integer indices,
                # We need to expand batch indexes on the right, so they don't interact with core index dimensions
                # We still squeeze on the left in case that allows us to use simpler indices
                squeezed_index = _squeeze_left(
                    shape_padright(
                        idxs[idx_entry], max_idx_core_ndim - idxs_core_ndim[idx_entry]
                    ),
                    stop_at_dim=batch_ndim,
                )
            else:
                # For basic IncSubtensor integers indices can be used as is, but we try to squeeze away dummy
                # batch dimensions in case we can end up with a basic IncSubtensor again
                squeezed_index = _squeeze_left(idxs[idx_entry])

        core_idxs.append(squeezed_index)

    # Create new indices for the batch dimensions
    has_batched_indices = not all(
        all(idx.type.broadcastable[:batch_ndim])
        for idx in idxs
        if not isinstance(idx, slice)
    )
    if has_batched_indices:
        # If indices have batch dimensions, we need to align them element-wise with the respective batch dimensions of x
        # We achieve this by creating `arange` indices and adding expand_dims for correct broadcasting.
        # Example:
        # x = pt.zeros(5); idx = [0, 1, 0]; out = x[idx].set(y)
        # batch_x = pt.zeros((2, 5)); batch_idx = [[0, 1, 0], [1, 1, 2]]
        # batch_out = batch_x[[0, 1][:, None], batch_idx].set(y)
        # If instead batch_x = pt.zeros((2, 2, 5))
        # batch_out = batch_x[[0, 1][:, None, None], [0, 1][None, 1, None], batch_idx]

        # Note: For simplicity we use arange for all batch dimensions of x,
        # even if not all may have corresponding batch index dimensions
        batch_slices = [
            shape_padright(arange(x_batch_shape, dtype="int64"), n)
            for (x_batch_shape, n) in zip(
                tuple(x.shape)[:batch_ndim],
                reversed(range(max_idx_core_ndim, max_idx_core_ndim + batch_ndim)),
            )
        ]
    else:
        # In the case we don't have batch indices,
        # we can use slice(None) to broadcast the core indices to each new batch dimension of x / y
        batch_slices = [slice(None)] * batch_ndim

    new_idxs = (*batch_slices, *core_idxs)
    x_view = x[new_idxs]

    # Introduce any implicit expand_dims on core dimension of y
    missing_y_core_ndim = x_view.type.ndim - y.type.ndim
    implicit_axes = tuple(range(batch_ndim, batch_ndim + missing_y_core_ndim))
    y = expand_dims(y, implicit_axes)

    # Transpose y if needed
    if has_batched_indices:
        # By introducing arange slices we may caused a transposition of the advanced group to the front
        # If this was not already happening in the core graph, we'll need to transpose y to align it correctly
        if max_idx_core_ndim and not (
            advanced and _non_consecutive_adv_indexing(core_idxs)
        ):
            integer_pos = [
                i for i, entry in enumerate(core_op.idx_list) if isinstance(entry, int)
            ]
            slice_pos = [
                i
                for i, entry in enumerate(core_op.idx_list)
                if isinstance(entry, slice)
            ]
            if slice_pos and integer_pos and (slice_pos[0] < integer_pos[-1]):
                y = moveaxis(
                    y,
                    [batch_ndim + integer_pos[0] + i for i in range(max_idx_core_ndim)],
                    [batch_ndim + i for i in range(max_idx_core_ndim)],
                )
    else:
        # Conversely if we tried to use `slice(None)` for the batch dimensions but there was already transposition
        # in the core case, we'll need to move the batch slices of y to after the advanced indexing group
        if advanced and _non_consecutive_adv_indexing(core_idxs):
            y = moveaxis(
                y,
                [i for i in range(batch_ndim)],  # noqa: C416
                [max_idx_core_ndim + i for i in range(batch_ndim)],
            )

    # Remove useless left-batch dimensions of y (if any)
    y = _squeeze_left(y, stop_at_dim=batch_ndim)

    if core_op.set_instead_of_inc:
        new_out = x[new_idxs].set(y)
    else:
        new_out = x[new_idxs].inc(y)

    copy_stack_trace(out, new_out)
    return [new_out]


@node_rewriter(tracks=[AdvancedSubtensor, AdvancedIncSubtensor])
def bool_idx_to_nonzero(fgraph, node):
    """Convert boolean indexing into equivalent vector boolean index, supported by our dispatch

    x[1:, eye(3, dtype=bool), 1:] -> x[1:, *eye(3).nonzero()]
    """
    if isinstance(node.op, AdvancedSubtensor):
        x, *idxs = node.inputs
    else:
        x, y, *idxs = node.inputs

    idxs = indices_from_subtensor(idxs, node.op.idx_list)

    bool_pos = {
        i
        for i, idx in enumerate(idxs)
        if isinstance(idx, TensorVariable) and idx.dtype == "bool"
    }

    if not bool_pos:
        return None

    new_idxs = []
    for i, idx in enumerate(idxs):
        if i in bool_pos:
            new_idxs.extend(idx.nonzero())
        else:
            new_idxs.append(idx)

    if isinstance(node.op, AdvancedSubtensor):
        new_out = x[tuple(new_idxs)]
    else:
        new_out = (
            x[tuple(new_idxs)].set(y)
            if node.op.set_instead_of_inc
            else x[tuple(new_idxs)].inc(y)
        )

    return [copy_stack_trace(node.outputs[0], new_out)]


optdb["specialize"].register(
    bool_idx_to_nonzero.__name__,
    bool_idx_to_nonzero,
    "numba",
    "shape_unsafe",  # It can mask invalid mask sizes
    use_db_name_as_tag=False,  # Not included if only "specialize" is requested
)
