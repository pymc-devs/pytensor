from collections import deque
from warnings import warn

import numpy as np

import pytensor
from pytensor.configdefaults import config
from pytensor.graph.basic import Constant, Variable, equal_computations
from pytensor.graph.features import AlreadyThere, Feature
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import (
    GraphRewriter,
    copy_stack_trace,
    node_rewriter,
)
from pytensor.graph.type import HasShape
from pytensor.tensor.basic import (
    Alloc,
    MakeVector,
    alloc,
    as_tensor_variable,
    cast,
    constant,
    expand_dims,
    get_scalar_constant_value,
    register_infer_shape,
    stack,
)
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.exceptions import ShapeError
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
    register_useless,
)
from pytensor.tensor.shape import (
    Reshape,
    Shape,
    Shape_i,
    SpecifyShape,
    specify_shape,
)
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    IncSubtensor,
    Subtensor,
)
from pytensor.tensor.type import TensorType, integer_dtypes
from pytensor.tensor.type_other import NoneTypeT
from pytensor.tensor.variable import TensorVariable


class _ShapeOfProxy:
    """Dict-like proxy so ``shape_feature.shape_of[var]`` keeps working."""

    def __init__(self, feature):
        self._feature = feature

    def __getitem__(self, var):
        result = self._feature.shape_tuple(var)
        if result is None:
            raise KeyError(var)
        return result

    def __contains__(self, var):
        return isinstance(var.type, HasShape)


class ShapeFeature(Feature):
    r"""Lazy `Feature` that provides shape information for the graph it tracks.

    Shapes are derived on demand by calling each ``Op.infer_shape`` on the
    current (live) node inputs, recursing toward the graph inputs. Use:

    - ``get_shape(var, i)`` / ``shape_tuple(var)`` for the shape expressed in
      terms of the graph inputs (recursing through the ancestors of ``var``);
    - ``get_non_recursive_shape(var, i)`` for the shape expressed in terms of
      ``var.owner``'s direct inputs only;
    - ``same_shape(x, y)`` to statically compare two shapes.

    Inferred shapes are cached per node, but the cache is invalidated whenever
    an ancestor input changes (``on_change_input``), so the expressions handed
    out always reference live graph variables.
    """

    def __init__(self):
        self.fgraph: FunctionGraph | None = None
        # node -> tuple of (tuple of shape vars) per output, recursive shapes only,
        # lazily populated. Non-recursive shapes read only the node's own inputs, so
        # they are cheap and recomputed on demand rather than cached.
        self._cache: dict = {}
        # Nodes whose input changed since the last query. The recursive shape of a node
        # embeds the shapes of its whole ancestor cone, so a change anywhere above a
        # cached node invalidates it. We defer that to the next query and walk
        # ``fgraph.clients`` downstream from these nodes then — the live graph already
        # encodes the dependencies, so no reverse-dependency map is kept.
        self._stale: set = set()
        # var -> {i: Shape_i(i)(v)}, ensures Apply identity for leaves
        self._shape_i_cache: dict = {}
        self.lscalar_one = constant(1, dtype="int64")
        # Compat: scheduled replacements for local_track_shape_i
        self.scheduled: dict = {}

    def _shape_i_var(self, v, i):
        per_dim = self._shape_i_cache.get(v)
        if per_dim is not None:
            cached = per_dim.get(i)
            if cached is not None:
                return cached
        else:
            per_dim = {}
            self._shape_i_cache[v] = per_dim
        if isinstance(v.type, HasShape) and v.type.shape[i] is not None:
            res = constant(v.type.shape[i], dtype="int64")
        else:
            res = Shape_i(i)(v)
        per_dim[i] = res
        return res

    @staticmethod
    def _fresh_shape_i(v, i):
        """Like ``_shape_i_var``, but never reusing a cached variable.

        The cached variable may live in the graph with its own constraints
        (e.g. its buffer destroyed by an inplace Op); a fresh read carries no
        constraints beyond reading ``v``, which non-recursive consumers rely on.
        """
        if isinstance(v.type, HasShape) and v.type.shape[i] is not None:
            return constant(v.type.shape[i], dtype="int64")
        return Shape_i(i)(v)

    def _coerce_shape_element(self, element, node):
        """Validate and normalize a single shape element from infer_shape."""
        if isinstance(element, np.ndarray):
            if element.ndim != 0:
                raise TypeError(
                    f"infer_shape for {node.op} returned a non-scalar "
                    f"ndarray for shape element: {element!r}"
                )
            element = element.item()
        if isinstance(element, Variable):
            if element.type.dtype not in integer_dtypes:
                raise TypeError(
                    f"infer_shape for {node.op} returned a non-integer "
                    f"Variable for shape element: {element!r}"
                )
            if getattr(element.type, "ndim", 0):
                raise TypeError(
                    f"infer_shape for {node.op} returned a non-scalar "
                    f"Variable for shape element: {element!r}"
                )
            if element.type.dtype != "int64":
                if isinstance(element, Constant):
                    return constant(int(element.data), dtype="int64")
                return cast(element, "int64")
            return element
        if isinstance(element, int | np.integer):
            if int(element) < 0:
                raise ValueError(
                    f"infer_shape for {node.op} returned a negative shape: {int(element)}"
                )
            return constant(int(element), dtype="int64")
        raise TypeError(
            f"infer_shape for {node.op} returned an unsupported "
            f"shape element of type {type(element).__name__}: {element!r}"
        )

    def _get_node_shapes(self, node, recursive=True):
        """Return validated per-output shape tuples for ``node``.

        With ``recursive=False``, input shapes are fresh constant/``Shape_i``
        reads of the node's own inputs instead of inferred expressions, so the
        result references no variables beyond those inputs (and nothing is
        cached).
        """
        if self._stale:
            self._flush_stale()

        if not recursive:
            return self._compute_node_shapes(node, recursive=False)

        cache = self._cache
        cached = cache.get(node)
        if cached is not None:
            return cached

        # Fill the cache for ``node`` and its ancestor cone bottom-up with an
        # explicit stack. Recursing through ``get_shape`` would overflow the
        # Python stack on deep graphs; here a node is computed only once every
        # input-producing node it needs is cached, so ``_compute_node_shapes``
        # reads its input shapes straight from the cache instead of recursing.
        stack = [node]
        while stack:
            top = stack[-1]
            if top in cache:
                stack.pop()
                continue
            ready = True
            for inp in top.inputs:
                inp_node = inp.owner
                if (
                    inp_node is not None
                    and inp_node not in cache
                    and isinstance(inp.type, HasShape)
                    # A fully-static input shape is read without touching its
                    # owner (see ``get_shape``), so don't bother caching it.
                    and any(s is None for s in inp.type.shape)
                ):
                    stack.append(inp_node)
                    ready = False
            if ready:
                stack.pop()
                cache[top] = self._compute_node_shapes(top, recursive=True)

        return cache[node]

    def _compute_node_shapes(self, node, recursive):
        """Call ``infer_shape`` for ``node`` and validate the result.

        Input shapes are taken from ``get_shape`` (recursive) or from fresh
        ``Shape_i`` reads of the node's own inputs (non-recursive). In the
        recursive case the caller must have cached the input-producing nodes
        already, so ``get_shape`` resolves them from the cache without recursing.
        """
        input_shape_of = self.get_shape if recursive else self._fresh_shape_i
        shape_i_var = self._shape_i_var if recursive else self._fresh_shape_i

        input_shapes = []
        for inp in node.inputs:
            if isinstance(inp.type, HasShape):
                input_shapes.append(
                    tuple(input_shape_of(inp, j) for j in range(inp.type.ndim))
                )
            else:
                input_shapes.append(None)

        output_shapes = None
        shape_infer = getattr(node.op, "infer_shape", None)
        if shape_infer is not None:
            try:
                output_shapes = shape_infer(node, input_shapes)
            except ShapeError:
                pass
            except NotImplementedError:
                pass
            except Exception as exc:
                if config.on_shape_error == "raise":
                    raise
                warn(
                    f"Failed to infer_shape from Op {node.op}: "
                    f"{type(exc).__name__}: {exc}"
                )

        result = []
        for k, out in enumerate(node.outputs):
            if not isinstance(out.type, HasShape):
                result.append(None)
                continue
            sh = None
            if output_shapes is not None and k < len(output_shapes):
                sh = output_shapes[k]
            if sh is None or not isinstance(sh, list | tuple):
                result.append(tuple(shape_i_var(out, j) for j in range(out.type.ndim)))
                continue
            coerced = []
            for j, s in enumerate(sh):
                coerced.append(self._coerce_shape_element(s, node))
            result.append(tuple(coerced))

        return tuple(result)

    def get_shape(self, var, idx):
        """Return a symbolic expression for ``var.shape[idx]``."""
        if isinstance(var.type, HasShape) and var.type.shape[idx] is not None:
            return constant(var.type.shape[idx], dtype="int64")

        node = var.owner
        if node is None:
            return self._shape_i_var(var, idx)

        node_shapes = self._get_node_shapes(node)
        out_idx = node.outputs.index(var)
        sh = node_shapes[out_idx]
        if sh is not None:
            return sh[idx]
        return self._shape_i_var(var, idx)

    def get_non_recursive_shape(self, var, idx):
        """Return an expression for ``var.shape[idx]`` reading only ``var.owner``'s inputs.

        Unlike ``get_shape``, input shapes are not recursively expanded: the
        expression references the direct inputs of ``var.owner`` (through
        constants and fresh ``Shape_i`` reads) and nothing else. It can
        therefore be introduced next to any consumer of those inputs no matter
        what the destroy maps in the surrounding graph are, whereas the
        recursion of ``get_shape`` may surface variables that an inplace Op
        destroys.

        Works on an unattached feature.
        """
        if isinstance(var.type, HasShape) and var.type.shape[idx] is not None:
            return constant(var.type.shape[idx], dtype="int64")

        node = var.owner
        if node is None:
            return self._fresh_shape_i(var, idx)

        node_shapes = self._get_node_shapes(node, recursive=False)
        out_idx = node.outputs.index(var)
        sh = node_shapes[out_idx]
        if sh is not None:
            return sh[idx]
        return self._fresh_shape_i(var, idx)

    def shape_tuple(self, var):
        if not isinstance(var.type, HasShape):
            return None
        return tuple(self.get_shape(var, i) for i in range(var.type.ndim))

    @property
    def shape_of(self):
        """Deprecated back-compat shim. Use ``shape_tuple(var)`` instead."""
        warn(
            "ShapeFeature.shape_of is deprecated; use shape_tuple(var) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _ShapeOfProxy(self)

    def on_attach(self, fgraph):
        if hasattr(fgraph, "shape_feature"):
            raise AlreadyThere("This FunctionGraph already has a ShapeFeature")
        if self.fgraph is not None and self.fgraph is not fgraph:
            raise Exception("This ShapeFeature is already attached to a graph")
        self.fgraph = fgraph
        fgraph.shape_feature = self

    def on_detach(self, fgraph):
        self._cache.clear()
        self._stale.clear()
        self._shape_i_cache.clear()
        self.scheduled.clear()
        self.fgraph = None
        if hasattr(fgraph, "shape_feature"):
            del fgraph.shape_feature

    def _flush_stale(self):
        """Drop cached shapes for the changed nodes and everything downstream of them.

        Walks ``fgraph.clients`` from each stale node, so a change above a cached node
        evicts it no matter how many levels down it sits.
        """
        cache = self._cache
        clients = self.fgraph.clients
        queue = deque(self._stale)
        seen = set(self._stale)
        self._stale = set()
        while queue:
            node = queue.popleft()
            cache.pop(node, None)
            for out in node.outputs:
                for client_node, _ in clients.get(out, ()):
                    if client_node not in seen:
                        seen.add(client_node)
                        queue.append(client_node)

    def on_prune(self, fgraph, node, reason):
        self._cache.pop(node, None)
        self._stale.discard(node)
        for out in node.outputs:
            self._shape_i_cache.pop(out, None)

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        if r is new_r:
            return
        # Defer invalidation: the next query flushes this node and its downstream cone.
        self._stale.add(node)

        # Schedule Shape_i(r) replacements for local_track_shape_i
        if isinstance(r.type, HasShape):
            for shpnode, _idx in fgraph.clients.get(r, []):
                if isinstance(getattr(shpnode, "op", None), Shape_i):
                    self.scheduled[shpnode] = new_r

    def same_shape(
        self,
        x: Variable,
        y: Variable,
        dim_x: int | None = None,
        dim_y: int | None = None,
    ) -> bool:
        """Return True if we can statically prove x and y have the same shape."""
        sx = self.shape_tuple(x)
        sy = self.shape_tuple(y)
        if sx is None or sy is None:
            return False
        if dim_x is not None:
            sx = (sx[dim_x],)
        if dim_y is not None:
            sy = (sy[dim_y],)
        if len(sx) != len(sy):
            return False
        for dx, dy in zip(sx, sy, strict=True):
            if dx is dy:
                continue
            if isinstance(dx, Constant) and isinstance(dy, Constant):
                if dx.data == dy.data:
                    continue
                return False
            if equal_computations([dx], [dy]):
                continue
            return False
        return True

    def clone(self):
        return type(self)()


class ShapeOptimizer(GraphRewriter):
    """Rewriter that adds `ShapeFeature` as a feature."""

    def apply(self, fgraph):
        fgraph.attach_feature(ShapeFeature())


class UnShapeOptimizer(GraphRewriter):
    """Rewriter that removes `ShapeFeature` as a feature."""

    def apply(self, fgraph):
        for feature in fgraph._features:
            if isinstance(feature, ShapeFeature):
                fgraph.remove_feature(feature)


# Register it after merge1 optimization at 0. We don't want to track
# the shape of merged node.
pytensor.compile.mode.optdb.register(
    "ShapeOpt", ShapeOptimizer(), "fast_run", "fast_compile", position=0.1
)
# Not enabled by default for now. Some crossentropy opt use the
# shape_feature.  They are at step 2.01. uncanonicalize is at step
# 3. After it goes to 48.5 that move to the gpu. So 10 seems reasonable.
pytensor.compile.mode.optdb.register("UnShapeOpt", UnShapeOptimizer(), position=10)


@register_useless
@register_canonicalize
@node_rewriter([Reshape])
def local_useless_expand_dims_in_reshape(fgraph, node):
    """
    Removes useless expand_dims `DimShuffle` operations inside Reshape:
      reshape(expand_dims(vector, axis=0), shp) => reshape(vector, shp)
      reshape(expand_dims(matrix, axis=(0, 2), shp) => reshape(matrix, shp)

    Implicit (and useless) squeezes are kept in the graph, as they are
    part of the canonical form of the graph.
    """
    expanded_x, new_shape = node.inputs

    if not (
        expanded_x.owner is not None
        and isinstance(expanded_x.owner.op, DimShuffle)
        and expanded_x.owner.op.augment
    ):
        return False

    [x] = expanded_x.owner.inputs

    new_order = tuple(o for o in expanded_x.owner.op.new_order if o != "x")
    if new_order != tuple(range(x.type.ndim)):
        x = x.dimshuffle(new_order)

    new_reshaped_x = x.reshape(new_shape)
    copy_stack_trace(node.outputs[0], new_reshaped_x)
    return [new_reshaped_x]


@register_canonicalize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([Reshape])
def local_reshape_chain(fgraph, node):
    """
    Reshape(Reshape(x, shape1),shape2) -> Reshape(x, shape2)

    """
    inner_reshape, final_shape = node.inputs

    if not (inner_reshape.owner and isinstance(inner_reshape.owner.op, Reshape)):
        return None

    x, _ = inner_reshape.owner.inputs
    new_reshape = node.op(x, final_shape)

    copy_stack_trace(node.outputs, new_reshape)
    return [new_reshape]


def _is_shape_i_of_x(
    var: TensorVariable,
    x: TensorVariable,
    i: int,
    shape_feature: ShapeFeature | None = None,
) -> bool:
    if var.type.ndim != 0:
        return False

    constant_var = get_scalar_constant_value(
        var,
        only_process_constants=False,
        # Don't go through Elemwise to keep things fast
        elemwise=False,
        raise_not_constant=False,
    )

    # Check var is a constant expression with the same value as x.type.shape[i]
    if constant_var == x.type.shape[i]:
        return True

    # Match shape_of[x][i] or its constant equivalent
    if shape_feature is not None:
        i_shape_of_x = shape_feature.get_shape(x, i)
        if i_shape_of_x == var or (
            isinstance(i_shape_of_x, Constant) and (i_shape_of_x.data == constant_var)
        ):
            return True

    if var.owner is None:
        # No more constant possibilities
        return False

    # Match Shape_i{i}(x)
    if isinstance(var.owner.op, Shape_i):
        return (var.owner.op.i == i) and (var.owner.inputs[0] == x)  # type: ignore

    # Match Subtensor((int,))(Shape(input), i) - single integer index into shape
    if isinstance(var.owner.op, Subtensor):
        idx_entry = (
            var.owner.op.idx_list[0] if len(var.owner.op.idx_list) == 1 else None
        )
        return (
            # Check we have integer indexing operation
            # (and not slice or multiple indexing)
            len(var.owner.op.idx_list) == 1
            and isinstance(idx_entry, int)
            # Check we are indexing on the shape of x
            and var.owner.inputs[0].owner is not None
            and isinstance(var.owner.inputs[0].owner.op, Shape)
            and var.owner.inputs[0].owner.inputs[0] == x
            # Check that index == i
            and (
                get_scalar_constant_value(var.owner.inputs[1], raise_not_constant=False)
                == i
            )
        )

    return False


def _unpack_shape_vector(shape: TensorVariable) -> tuple[TensorVariable, ...]:
    """Return the elements of a symbolic vector representing a shape.

    Handles the most common constant vector or make_vector cases.

    Returns tuple(shape) as fallback.
    """
    if isinstance(shape, Constant):
        return tuple(as_tensor_variable(dim, ndim=0) for dim in shape.data)
    elif shape.owner and isinstance(shape.owner.op, MakeVector):
        return tuple(shape.owner.inputs)
    else:
        return tuple(shape)


@register_useless("shape_unsafe")
@register_canonicalize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([Reshape])
def local_useless_reshape(fgraph, node):
    """Remove two kinds of useless `Reshape`.

    - Remove `Reshape` when both the input and output have a single dimension.
    - Remove `Reshape` when reshaping to the shape of the input.

    """
    inp, output_shape = node.inputs
    [output] = node.outputs

    if inp.type.ndim != output.type.ndim:
        return False

    # Simple case: both input and output have a single dimension.
    if (
        inp.type.ndim == 1
        and output.type.ndim == 1
        and inp.type.broadcastable == output.type.broadcastable
    ):
        return [inp]

    # Second case: all the shapes match the input shape
    # Match Reshape(x, x.shape)
    if output_shape.owner and isinstance(output_shape.owner.op, Shape):
        shape_input = output_shape.owner.inputs[0]
        if shape_input == inp:
            return [inp]

    shape_feature = getattr(fgraph, "shape_feature", None)

    # Match case where at least (n-1) entries correspond to the original shape:
    # Reshape(x, [x.shape[0], ..., x.shape[-1]]), or Reshape(x, [x.shape[0], y, x.shape[2], ... x.shape[-1]])
    # Where y can be -1 or anything with an unknown value, since the only valid reshape is still a no reshape.
    output_shape_is = _unpack_shape_vector(output_shape)
    nb_m1 = 0
    shape_match = [False] * inp.type.ndim
    for dim in range(inp.type.ndim):
        outshp_i = output_shape_is[dim]
        if _is_shape_i_of_x(outshp_i, inp, dim, shape_feature=shape_feature):
            shape_match[dim] = True
        elif isinstance(outshp_i, Constant) and outshp_i.data == -1:
            shape_match[dim] = True
            nb_m1 += 1

    if nb_m1 <= 1 and all(shape_match):
        return [inp]  # This is provably correct

    # There is one missing match, but all other dimensions match
    # Such as x.type.shape == (3, 5, None) and output_shape == (3, 5, y)
    if (nb_m1 == 0) and (shape_match.count(False) == 1):
        return [inp]  # This could mask a shape error

    return False


@register_canonicalize("shape_unsafe")
@node_rewriter([Reshape])
def local_reshape_to_dimshuffle(fgraph, node):
    r"""Remove `Reshape` operations over length-1 (broadcastable) dimensions.

    It's always valid to squeeze an input before doing the same reshape operation.
    Equivalently, it's always valid to remove `1` entries from the reshape shape
    and replace them by an expand_dims after the rewritten reshape operation.

    We chose to canonicalize the graph in this way as it allows isolating
    operations that are unique to the reshaping operation (mixing dimensions)
    from those that can be more legibly encoded by DimShuffle (squeeze and expand_dims).
    This can allow further simplifications by other rewrites that target
    DimShuffle but not Reshape, as well as facilitate the removal of useless reshape operations.

    For example:
        - reshape(col, (m, n)) -> reshape(squeeze(col, axis=1), (m, n))
        - reshape(col, (1, m, n)) -> expand_dims(reshape(squeeze(col, axis=1), (m, n)), axis=0)
        - reshape(x, (1, m, 1, n, 1, 1)) -> expand_dims(reshape(x, (m, n)), axis=(0, 2, 4, 5))

    """
    inp, shape = node.inputs
    [output] = node.outputs

    new_output_shape = []
    expand_axes = []
    # We look at both output.type.broadcastable and shape
    # The first may encode understanding about -1, but may miss knowledge about
    # constant 1 shape that only simplified later
    for i, (static_one, dim_length) in enumerate(
        zip(output.type.broadcastable, _unpack_shape_vector(shape))
    ):
        # -1 can be an implicit expand_dims, but it's tricky to prove
        # Example: np.zeros((2, 2, 2)).reshape((2, -1, 4))
        # We rely on the output static shape which will already have figured it out (sometimes)
        if static_one or (isinstance(dim_length, Constant) and dim_length.data == 1):
            expand_axes.append(i)
        else:
            new_output_shape.append(dim_length)

    if all(inp.type.broadcastable) or not new_output_shape:
        # Trivial case we have provably size 1 as input or output, reshape can't be doing anything useful
        new_out = inp.dimshuffle(["x"] * output.type.ndim)
        copy_stack_trace(output, new_out)
        return [new_out]

    squeeze_axes = [i for i, b in enumerate(inp.type.broadcastable) if b]

    if not squeeze_axes and not expand_axes:
        return None

    new_out = inp.squeeze(squeeze_axes)
    new_out = new_out.reshape(new_output_shape)
    new_out = expand_dims(new_out, expand_axes)
    copy_stack_trace(output, new_out)
    return [new_out]


@register_specialize
@node_rewriter([Reshape])
def local_fuse_squeeze_reshape(fgraph, node):
    r"""If there is a squeeze right before a reshape, merge them.

    This undoes the effect of `local_reshape_to_dimshuffle` that is applied during canonicalization.
    """
    x, new_shape = node.inputs

    if (
        x.owner is not None
        and isinstance(x.owner.op, DimShuffle)
        and x.owner.op.is_squeeze
    ):
        # A reshape can always subsume a squeeze.
        x = x.owner.inputs[0]
        return [x.reshape(new_shape)]


@register_specialize
@node_rewriter([DimShuffle])
def local_fuse_expand_dims_reshape(fgraph, node):
    r"""If there is an expand_dims right after a reshape, merge them.

    This undoes the effect of `local_reshape_to_dimshuffle` that is applied during canonicalization.
    """
    if not node.op.is_expand_dims:
        return None

    reshaped_x = node.inputs[0]

    if not (reshaped_x.owner and isinstance(reshaped_x.owner.op, Reshape)):
        return None

    if len(fgraph.clients[reshaped_x]) > 1:
        # The reshape is used elsewhere, don't fuse as it can sometimes require a copy.
        # Example: `x = pt.matrix(); y = x.T.reshape(-1); out = y[: None] * y[None, :]`
        return None

    x, new_shape = reshaped_x.owner.inputs

    # Add expand_dims to shape
    new_shape = list(_unpack_shape_vector(new_shape))
    for i in node.op.augment:
        new_shape.insert(i, 1)

    new_reshaped_x = x.reshape(new_shape)
    copy_stack_trace(node.outputs[0], new_reshaped_x)
    return [new_reshaped_x]


@register_canonicalize
@register_specialize
@node_rewriter([Reshape])
def local_reshape_lift(fgraph, node):
    """
        Reshape(UnaryElemwise(x)) -> UnaryElemwise(Reshape(x))

    Notes
    -----
    This rewrite is needed by `log1msigm_to_softplus` in order to get applied
    when there is a reshape.

    """
    if not (
        node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Elemwise)
        and len(node.inputs[0].owner.inputs) == 1
    ):
        return None
    r = node.op(node.inputs[0].owner.inputs[0], node.inputs[1])
    # Copy stacktrace from previous Reshape op, as an error in new
    # Reshape op could only have been caused by old one.
    copy_stack_trace(node.outputs, r)

    e = node.inputs[0].owner.op(r)
    # Copy stacktrace from both previous Reshape and UnaryElemwise op
    # because an error in new cg could have been caused by either ops.
    copy_stack_trace(node.outputs + node.inputs, e)
    return [e]


@register_useless
@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([SpecifyShape])
def local_useless_specify_shape(fgraph, node):
    """Remove SpecifyShape when the asserted shapes are already encoded in the static type of the input."""
    x, *shape = node.inputs
    for static_dim, specified_dim in zip(x.type.shape, shape, strict=True):
        if isinstance(specified_dim.type, NoneTypeT):
            continue
        if static_dim is None:
            # There is an unknown static dimension that is being specified
            return None
        if not (
            isinstance(specified_dim, Constant) and specified_dim.data == static_dim
        ):
            # The specified dim is either:
            # 1. Not constant or
            # 2. Constant that does not match the static dim
            # Either way, we must keep the SpecifyShape
            return None

    # If we arrived here, it means SpecifyShape was already encoded in the static shape
    # We don't need it
    copy_stack_trace(node.outputs[0], x)
    return [x]


@register_infer_shape
@register_useless
@register_canonicalize
@node_rewriter([SpecifyShape])
def local_merge_consecutive_specify_shape(fgraph, node):
    """Replace ``specify_shape(specify_shape(x, s1), s2)`` with ``specify_shape(x, s3)``,
    where s3 is the union of specified dimensions in s1 and s2, with preference given to s2.
    """

    obj = node.inputs[0]
    if not (obj.owner and isinstance(obj.owner.op, SpecifyShape)):
        return False

    inner_obj, *shape = obj.owner.inputs
    for dim, sh in enumerate(node.inputs[1:]):
        if not isinstance(sh.type, NoneTypeT):
            shape[dim] = sh

    # TODO: We could make sure that the overlapping shapes of the two `SpecifyShape`s are
    # the same.

    return [specify_shape(inner_obj, shape)]


_empty_shape = constant([], dtype="int64")


@register_infer_shape
@node_rewriter([Shape])
def local_shape_ground(fgraph, node):
    """Rewrite shape(x) -> make_vector(x.type.shape) when this is constant."""
    [x] = node.inputs
    static_shape = x.type.shape
    if len(static_shape) == 0:
        return [_empty_shape]
    if not any(dim is None for dim in static_shape):
        return [stack([constant(dim, dtype="int64") for dim in static_shape])]


@register_infer_shape
@register_useless
@register_canonicalize
@node_rewriter([Shape])
def local_Shape_of_SpecifyShape(fgraph, node):
    """Replace ``specify_shape(x, s).shape`` with ``s``."""

    specified_shape = node.inputs[0]

    if not (
        specified_shape.owner is not None
        and isinstance(specified_shape.owner.op, SpecifyShape)
    ):
        return False

    x, *shape = specified_shape.owner.inputs

    # Replace `NoneConst` by `shape_i`
    for i, sh in enumerate(shape):
        if isinstance(sh.type, NoneTypeT):
            shape[i] = x.shape[i]

    return [stack(shape).astype(np.int64)]


@register_infer_shape
@register_canonicalize
@register_specialize
@node_rewriter([SpecifyShape])
def local_lift_specify_shape_elemwise(fgraph, node):
    """Lift SpecifyShape of Elemwise towards the inputs."""
    inp, *shape = node.inputs
    if inp.owner and isinstance(inp.owner.op, Elemwise):
        if len(inp.owner.outputs) != 1:
            return None

        elem_inps = inp.owner.inputs
        if len(elem_inps) == 1:
            new_elem_inps = [specify_shape(elem_inps[0], shape)]
        else:
            # Rewrite does not support case where specify_shape provides new broadcastable information,
            # As that may require a specify_shape for each input
            out_broadcastable = node.outputs[0].type.broadcastable
            if out_broadcastable != inp.type.broadcastable:
                return None

            # All non-broadcastable dimensions of inputs must match the non-broadcastbale specify_shape dims
            # We look for a sufficient input to assign all the specify_shape dims
            # We could consider distributing the SpecifyShape across multiple inputs, when none is sufficient

            nonbcast_dims = {
                i
                for i, (dim, bcast) in enumerate(
                    zip(shape, out_broadcastable, strict=True)
                )
                if (not bcast and not isinstance(dim.type, NoneTypeT))
            }
            new_elem_inps = elem_inps.copy()
            for i, elem_inp in enumerate(elem_inps):
                if all(
                    bcast_dim is False
                    for dim, bcast_dim in enumerate(elem_inp.type.broadcastable)
                    if dim in nonbcast_dims
                ):
                    new_elem_inps[i] = specify_shape(elem_inp, shape)
                    break
            else:  # no-break, no sufficient candidate found
                return None

        new_out = inp.owner.op.make_node(*new_elem_inps).outputs
        copy_stack_trace(node.outputs, new_out)
        return new_out


@register_canonicalize
@register_specialize
@node_rewriter([SpecifyShape])
def local_lift_specify_shape_inc_subtensor(fgraph, node):
    """specify_shape(x[idx].inc(y)) -> specify_shape(x)[idx].inc(y).

    IncSubtensor always preserves the shape of the buffer
    """
    inc_x, *specified_shape = node.inputs
    if isinstance(
        (inc_op := inc_x.owner_op),
        IncSubtensor | AdvancedIncSubtensor,
    ):
        x, y, *idx_vars = inc_x.owner.inputs
        new_x = specify_shape(x, specified_shape)
        new_out = inc_op(new_x, y, *idx_vars)
        copy_stack_trace(node.outputs[0], new_out)
        return [new_out]


@register_canonicalize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([SpecifyShape])
def local_specify_shape_alloc(fgraph, node):
    """Replace specify_shape(alloc(x, *shape), *specified) -> alloc(x, *new_shape).

    Each new_shape dim is the specified dim if given, otherwise the original alloc dim.
    """
    alloc_out, *specified = node.inputs
    if not isinstance(alloc_out.owner_op, Alloc):
        return None

    value, *alloc_shape = alloc_out.owner.inputs

    new_shape = list(alloc_shape)
    changed = False
    for i, s in enumerate(specified):
        if isinstance(s.type, NoneTypeT):
            continue
        new_shape[i] = s
        changed = True

    if not changed:
        return None

    new_out = alloc(value, *new_shape)
    copy_stack_trace(node.outputs[0], new_out)
    return [new_out]


@register_infer_shape
@register_useless
@register_canonicalize
@node_rewriter([Shape_i])
def local_Shape_i_ground(fgraph, node):
    """Replace ``shape_i(x, i)`` with ``s`` when ``x.type.shape[i] == s``."""

    shape_arg = node.inputs[0]

    if not isinstance(shape_arg.type, TensorType):
        return False

    s_val = shape_arg.type.shape[node.op.i]
    if s_val is not None:
        return [as_tensor_variable(s_val, dtype=np.int64)]


@register_infer_shape
@register_specialize
@register_canonicalize
@node_rewriter([Shape])
def local_shape_to_shape_i(fgraph, node):
    if not hasattr(fgraph, "shape_feature"):
        return
    shape_feature = fgraph.shape_feature
    ret = as_tensor_variable(shape_feature.shape_tuple(node.inputs[0]), dtype="int64")

    # We need to copy over stack trace from input to output
    copy_stack_trace(node.outputs[0], ret)
    return [ret]


@register_infer_shape
@register_specialize
@register_canonicalize
@node_rewriter([Shape_i])
def local_track_shape_i(fgraph, node):
    """Replace ``Shape_i(v, i)`` with the inferred shape expression.

    When ``v.owner.op`` has ``infer_shape``, ``get_shape(v, i)`` returns
    a non-``Shape_i`` expression. Rewriting the literal ``Shape_i(v, i)``
    with that expression lets downstream rewrites see the inferred form
    and typically lets the original producer of ``v`` be pruned when only
    its shape is consumed.
    """
    shape_feature = getattr(fgraph, "shape_feature", None)
    if shape_feature is None:
        return False

    # Handle scheduled replacements from on_change_input
    replacement = shape_feature.scheduled.pop(node, None)
    if replacement is not None:
        return [shape_feature.get_shape(replacement, node.op.i)]

    [v] = node.inputs
    if v.owner is None:
        return False

    i = node.op.i
    new_shape = shape_feature.get_shape(v, i)
    if new_shape is None:
        return False

    # Avoid replacing Shape_i(v, i) with itself
    if new_shape.owner is node or (
        isinstance(new_shape, Variable)
        and new_shape.owner is not None
        and isinstance(new_shape.owner.op, Shape_i)
        and new_shape.owner.op.i == i
        and new_shape.owner.inputs[0] is v
    ):
        return False

    return [new_shape]
