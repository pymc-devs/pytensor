from io import StringIO
from warnings import warn

import numpy as np

import pytensor
from pytensor.configdefaults import config
from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.features import AlreadyThere, Feature
from pytensor.graph.fg import FrozenFunctionGraph, FunctionGraph
from pytensor.graph.rewriting.basic import (
    GraphRewriter,
    copy_stack_trace,
    node_rewriter,
)
from pytensor.graph.traversal import ancestors
from pytensor.graph.utils import get_variable_trace_string
from pytensor.tensor.basic import (
    MakeVector,
    as_tensor_variable,
    cast,
    constant,
    expand_dims,
    get_scalar_constant_value,
    register_infer_shape,
    stack,
)
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.exceptions import NotScalarConstantError, ShapeError
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
    register_useless,
)
from pytensor.tensor.rewriting.elemwise import apply_local_dimshuffle_lift
from pytensor.tensor.shape import (
    Reshape,
    Shape,
    Shape_i,
    SpecifyShape,
    specify_shape,
)
from pytensor.tensor.subtensor import Subtensor, get_idx_list
from pytensor.tensor.type import TensorType, discrete_dtypes, integer_dtypes, lscalar
from pytensor.tensor.type_other import NoneTypeT
from pytensor.tensor.variable import TensorVariable


def _materialize_frozen(root, replacements):
    """Walk a frozen kernel expression from ``root`` and rebuild it using live
    ``make_node`` calls and ``replacements`` for leaves.

    We cannot use ``graph_replace`` here: its cloning path goes through
    ``Apply.clone_with_new_inputs``, which falls into a branch that mutates
    the node's ``.inputs`` in place when input types match.  Against a
    ``FrozenApply`` — which is globally interned and shared across kernel
    builds — that mutation leaks live variables from one materialization into
    another.  Walking and rebuilding with ``make_node`` side-steps the issue.
    """
    memo: dict = dict(replacements)

    def _walk(v):
        if v in memo:
            return memo[v]
        if v.owner is None:
            memo[v] = v
            return v
        node = v.owner
        new_inputs = [_walk(inp) for inp in node.inputs]
        new_node = node.op.make_node(*new_inputs)
        for old, new in zip(node.outputs, new_node.outputs, strict=True):
            memo.setdefault(old, new)
        return memo[v]

    return _walk(root)


def _strip_shape_noise(v):
    """Strip same-dtype casts from a shape expression.

    Asserts are NOT stripped: they carry real semantic information
    (broadcasting checks), so ``Assert(y.shape[0], ...)`` and
    ``y.shape[0]`` must not be treated as equal for ``same_shape``.
    """
    from pytensor.scalar import Cast as _ScalarCast
    from pytensor.tensor.elemwise import Elemwise as _Elemwise

    seen = 0
    while v.owner is not None and seen < 8:
        op = v.owner.op
        if (
            isinstance(op, _Elemwise)
            and isinstance(op.scalar_op, _ScalarCast)
            and len(v.owner.inputs) == 1
            and v.owner.inputs[0].type.dtype == v.type.dtype
        ):
            v = v.owner.inputs[0]
            seen += 1
            continue
        break
    return v


class _LazyShapeTuple:
    """TEMPORARY back-compat shim for ``shape_of[v][i]``.

    .. deprecated::
        ``shape_of`` is scheduled for removal. New code must use
        ``ShapeFeature.get_shape(v, i)`` or ``ShapeFeature.shape_tuple(v)``
        directly.  This class — and ``_ShapeOfProxy`` — only exist to keep
        the legacy surface working while we migrate internal callers
        (``pytensor.compile.builders``, ``pytensor.scan.rewriting``,
        ``pytensor.tensor.rewriting.subtensor`` / ``basic``,
        ``pytensor.tensor.random.rewriting.basic``, and a handful of tests).
        Once those are migrated, delete both shims along with
        ``ShapeFeature.shape_of``.

    Lazy materialization of the shape expression per dim, cached
    per instance so within a single access repeats are idempotent.
    Different accesses of ``shape_of[v]`` produce different
    ``_LazyShapeTuple`` instances; cross-instance identity is NOT
    guaranteed (materialization happens at request time).
    """

    __slots__ = ("_cache", "_feature", "_ndim", "_v")

    def __init__(self, feature, v):
        self._feature = feature
        self._v = v
        self._ndim = v.type.ndim
        self._cache = [None] * self._ndim

    def __len__(self):
        return self._ndim

    def __iter__(self):
        for i in range(self._ndim):
            yield self[i]

    def __getitem__(self, i):
        if isinstance(i, slice):
            return tuple(self)[i]
        if i < 0:
            i += self._ndim
        cached = self._cache[i]
        if cached is None:
            cached = self._feature.get_shape(self._v, i)
            self._cache[i] = cached
        return cached

    def __add__(self, other):
        return tuple(self) + tuple(other)

    def __radd__(self, other):
        return tuple(other) + tuple(self)

    def __eq__(self, other):
        if other is None:
            return False
        try:
            return tuple(self) == tuple(other)
        except TypeError:
            return NotImplemented

    def __ne__(self, other):
        res = self.__eq__(other)
        if res is NotImplemented:
            return res
        return not res

    def __hash__(self):
        return hash((id(self._feature), id(self._v)))


class _ShapeOfProxy:
    """TEMPORARY back-compat shim for ``fgraph.shape_feature.shape_of``.

    .. deprecated::
        Remove together with ``_LazyShapeTuple`` once internal callers are
        migrated to ``ShapeFeature.get_shape`` / ``ShapeFeature.shape_tuple``.
        See ``_LazyShapeTuple`` for the caller inventory.
    """

    __slots__ = ("_feature",)

    def __init__(self, feature):
        self._feature = feature

    def __getitem__(self, v):
        if v is None or not hasattr(v.type, "ndim"):
            return None
        return _LazyShapeTuple(self._feature, v)

    def get(self, v, default=None):
        if v is None or not hasattr(v.type, "ndim"):
            return default
        return _LazyShapeTuple(self._feature, v)

    def __contains__(self, v):
        # A var is "in shape_of" if its owner has a kernel cached (i.e. was
        # imported), or if it's a graph input of the attached fgraph, or if
        # an override was set for it.
        feature = self._feature
        if v in feature._overrides:
            return True
        if v.owner is not None:
            return v.owner in feature._cache
        fg = feature.fgraph
        if fg is not None and v in fg.inputs:
            return True
        return False

    def __iter__(self):
        feature = self._feature
        seen = set()
        for v in feature._overrides:
            if hasattr(v.type, "ndim") and v not in seen:
                seen.add(v)
                yield v
        fg = feature.fgraph
        if fg is not None:
            for v in fg.variables:
                if hasattr(v.type, "ndim") and v not in seen:
                    seen.add(v)
                    yield v

    def values(self):
        for v in self:
            yield _LazyShapeTuple(self._feature, v)

    def keys(self):
        return iter(self)

    def items(self):
        for v in self:
            yield v, _LazyShapeTuple(self._feature, v)


class ShapeFeature(Feature):
    r"""Kernel-based `Feature` that tracks shape information in a graph.

    For each `Apply`, a `FrozenFunctionGraph` "kernel" is built once and
    stored in ``self._cache[node]``. The kernel is rooted in *dummy*
    variables — never the live outer variables — so it can't go stale as
    the fgraph mutates. Shape requests instantiate the kernel via
    ``graph_replace`` against today's ``node.inputs`` (and recursive shape
    lookups), so materialized expressions are always rooted in live
    variables.

    Preferred API for new callers:

    - ``get_shape(v, i)`` — materialize ``v.shape[i]``.
    - ``shape_tuple(v)`` — materialize ``tuple(v.shape)``.
    - ``same_shape(x, y, dim_x=None, dim_y=None)`` — via content-addressed
      ``shape_key`` and a union-find for externally-registered equalities.

    Temporary legacy surface (SHIM — scheduled for removal):

    - ``shape_of[v]`` / ``shape_of[v][i]`` / ``shape_of.get`` / iteration
      — see ``_LazyShapeTuple`` / ``_ShapeOfProxy``. Internal callers
      should migrate to ``get_shape`` / ``shape_tuple``; then delete both
      shims and ``self.shape_of``.
    - ``set_shape(r, s)`` / ``init_r(r)`` / ``on_import(fgraph, node, ...)``
      for ``pytensor.compile.builders.infer_shape``.
    - ``scheduled`` — kept as an (unused) empty dict so the old
      ``local_track_shape_i`` rewrite stays a no-op.
    """

    def __init__(self):
        self._cache: dict = {}
        self._overrides: dict = {}
        # Memoizes ``Shape_i(i)(v)`` for leaves/fallbacks so callers that
        # cross-reference ``shape_of[v][i]`` with Shape_i nodes in the graph
        # observe Apply identity (the graph's MergeFeature would otherwise
        # merge structurally equal copies, but by then compare-by-identity
        # rewrites may have already bailed out).
        self._shape_i_cache: dict = {}
        # TEMPORARY back-compat surface.  Remove once all in-tree callers
        # move to ``get_shape`` / ``shape_tuple`` — see ``_LazyShapeTuple``
        # for the inventory.
        self.shape_of = _ShapeOfProxy(self)
        self._uf_parent: dict = {}
        self.scheduled: dict = {}
        self.fgraph: FunctionGraph | None = None
        self.lscalar_one = constant(1, dtype="int64")

    def _shape_i_var(self, v, i):
        key = (id(v), i)
        cached = self._shape_i_cache.get(key)
        if cached is not None:
            return cached
        res = Shape_i(i)(v)
        self._shape_i_cache[key] = res
        return res

    # ---- Feature callbacks ----

    def on_attach(self, fgraph):
        if hasattr(fgraph, "shape_feature"):
            raise AlreadyThere("This FunctionGraph already has a ShapeFeature")
        if self.fgraph is not None and self.fgraph is not fgraph:
            raise Exception("This ShapeFeature is already attached to a graph")
        self.fgraph = fgraph
        fgraph.shape_feature = self

    def on_detach(self, fgraph):
        self._cache.clear()
        self._overrides.clear()
        self._uf_parent.clear()
        self.scheduled.clear()
        self._shape_i_cache.clear()
        self.fgraph = None
        if hasattr(fgraph, "shape_feature"):
            del fgraph.shape_feature

    def on_import(self, fgraph, node, reason):
        # Kernel is built lazily in ``get_shape`` — don't front-load the
        # cost for nodes whose shapes never get queried.
        return

    def on_prune(self, fgraph, node, reason):
        self._cache.pop(node, None)
        # Drop cached Shape_i variables whose owner is being pruned — without
        # this the memo grows monotonically over a long canonicalize pass.
        for out in node.outputs:
            oid = id(out)
            for j in range(getattr(out.type, "ndim", 0) or 0):
                self._shape_i_cache.pop((oid, j), None)

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        # Carry r's shape forward as an override when new_r's Op has no
        # ``infer_shape`` (mirrors legacy ``update_shape``'s merge).
        # Optional: per-dim salvage for partial fallbacks (e.g. Scan).
        if r is new_r or not hasattr(new_r.type, "ndim"):
            return
        if new_r in self._overrides:
            return
        if new_r.owner is None:
            return  # graph inputs have their own Shape_i fallback; nothing better to install
        if getattr(new_r.owner.op, "infer_shape", None) is not None:
            return  # new_r's own kernel will produce a real shape
        if not hasattr(r.type, "ndim") or r.type.ndim != new_r.type.ndim:
            return
        # r may or may not have a cached kernel — ``get_shape`` builds it
        # lazily.  r.owner (if any) is still intact at this point: the
        # client edges are being rewired but r itself is not yet pruned.
        self._overrides[new_r] = tuple(self.get_shape(r, k) for k in range(r.type.ndim))

    # ---- kernel construction ----

    def _build_kernel(self, node):
        # When the same live input appears at multiple positions (e.g.
        # ``Elemwise.add(x, x)``), share the dummy clone AND the dummy
        # input-shape lscalars between those positions.  Ops like Elemwise
        # call ``broadcast_shape(*i_shapes)``, which only drops the runtime
        # ``Assert`` guard when the incoming shape expressions are
        # identical — so identity here is what lets ``x + x`` infer a
        # clean shape instead of ``Assert(x.shape[0], ...)``.
        input_slot: dict[int, int] = {}
        unique_dummies: list[Variable] = []
        unique_shape_tuples: list[tuple | None] = []

        dummy_inputs: list[Variable] = []
        dummy_input_shapes: list[tuple | None] = []
        for inp in node.inputs:
            key = id(inp)
            slot = input_slot.get(key)
            if slot is None:
                slot = len(unique_dummies)
                input_slot[key] = slot
                d = inp.clone()
                unique_dummies.append(d)
                if hasattr(inp.type, "ndim"):
                    static_shape = getattr(inp.type, "shape", (None,) * inp.type.ndim)
                    shp_tuple = tuple(
                        constant(s, dtype="int64") if s is not None else lscalar()
                        for s in static_shape
                    )
                else:
                    shp_tuple = None
                unique_shape_tuples.append(shp_tuple)
            dummy_inputs.append(unique_dummies[slot])
            dummy_input_shapes.append(unique_shape_tuples[slot])

        dummy_outputs = [out.clone() for out in node.outputs]
        dummy_node = Apply(node.op, dummy_inputs, dummy_outputs)

        output_shapes = None
        shape_infer = getattr(node.op, "infer_shape", None)
        if shape_infer is not None:
            try:
                output_shapes = shape_infer(None, dummy_node, dummy_input_shapes)
            except ShapeError:
                output_shapes = None
            except NotImplementedError:
                output_shapes = None
            except Exception as exc:
                if config.on_shape_error == "raise":
                    raise
                warn(
                    f"Failed to infer_shape from Op {node.op}: "
                    f"{type(exc).__name__}: {exc}"
                )
                output_shapes = None

        if output_shapes is None:
            output_shapes = [None] * len(dummy_outputs)

        # Fallback: Shape_i(i)(dummy_output) where the op couldn't provide
        # an infer_shape for a given output. Reuse dummy_outputs — no extra
        # placeholders.
        fallback_out_dummies = [None] * len(dummy_outputs)
        coerced_output_shapes = []
        for k, dummy_out in enumerate(dummy_outputs):
            sh = output_shapes[k] if k < len(output_shapes) else None
            if not hasattr(dummy_out.type, "ndim"):
                coerced_output_shapes.append(None)
                continue
            if sh is None:
                fallback_out_dummies[k] = dummy_out
                coerced_output_shapes.append(
                    tuple(Shape_i(i)(dummy_out) for i in range(dummy_out.type.ndim))
                )
                continue
            if not isinstance(sh, list | tuple):
                # Malformed return — fall back to Shape_i
                fallback_out_dummies[k] = dummy_out
                coerced_output_shapes.append(
                    tuple(Shape_i(i)(dummy_out) for i in range(dummy_out.type.ndim))
                )
                continue
            coerced = []
            for i, s in enumerate(sh):
                if (
                    hasattr(dummy_out.type, "shape")
                    and dummy_out.type.shape[i] is not None
                ):
                    coerced.append(constant(dummy_out.type.shape[i], dtype="int64"))
                    continue
                coerced.append(self._coerce_shape_el(s, dummy_out))
            coerced_output_shapes.append(tuple(coerced))

        flat_out = []
        layout = []
        for sh in coerced_output_shapes:
            if sh is None:
                layout.append(None)
                continue
            layout.append(len(sh))
            flat_out.extend(sh)

        # ``meta`` carries only what ``get_shape`` / ``shape_key`` need to
        # re-wire the frozen kernel against live ``node.inputs``.
        meta = {"output_layout": tuple(layout)}
        if not flat_out:
            return (None, meta)

        # Build kernel_inputs with unique dummies only.  Shape slots are
        # attached by unique-slot index so duplicate live inputs share the
        # same set of kernel-input positions.  Each kernel_input needs a
        # role that maps back to the live graph at materialization time.
        kernel_inputs: list[Variable] = []
        roles: list[tuple] = []
        for slot, dummy in enumerate(unique_dummies):
            kernel_inputs.append(dummy)
            roles.append(("input_slot", slot))
        for slot, shape_tuple in enumerate(unique_shape_tuples):
            if shape_tuple is None:
                continue
            for j, s in enumerate(shape_tuple):
                kernel_inputs.append(s)
                roles.append(("input_shape_slot", slot, j))
        for k, ph in enumerate(fallback_out_dummies):
            if ph is not None:
                kernel_inputs.append(ph)
                roles.append(("fallback_out", k))

        # Sanity: every free Variable in flat_out should be in kernel_inputs.
        # An orphan indicates a buggy ``infer_shape`` that leaked a variable
        # outside of ``node.inputs`` / their shape scalars. In development
        # mode (config.on_shape_error == "raise") we surface this eagerly
        # instead of silently falling back to ``Shape_i``.
        kernel_input_set = set(kernel_inputs)
        for anc in ancestors(flat_out):
            if anc.owner is None:
                if isinstance(anc, Constant):
                    continue
                if anc not in kernel_input_set:
                    msg = (
                        f"Op {node.op}.infer_shape leaked an orphan variable "
                        f"{anc!r} that is not one of node.inputs or their "
                        f"shape scalars; falling back to Shape_i."
                    )
                    if config.on_shape_error == "raise":
                        raise ShapeError(msg)
                    return (None, dict(meta, kernel_build_error=msg))

        # Find any live input index that maps to this slot, so materialization
        # can look up ``node.inputs[<any representative>]``.
        slot_to_input_idx: list[int] = [-1] * len(unique_dummies)
        for inp_idx, inp in enumerate(node.inputs):
            s = input_slot[id(inp)]
            if slot_to_input_idx[s] == -1:
                slot_to_input_idx[s] = inp_idx

        try:
            kernel = FrozenFunctionGraph(kernel_inputs, flat_out)
        except Exception as exc:
            return (None, dict(meta, kernel_build_error=str(exc)))

        meta["roles"] = tuple(roles)
        meta["slot_to_input_idx"] = tuple(slot_to_input_idx)
        return (kernel, meta)

    def _coerce_shape_el(self, s, var):
        """Coerce one infer_shape output element into an int64 scalar Variable."""
        if isinstance(s, Variable):
            if s.type.dtype != "int64":
                if getattr(s.type, "ndim", 0) == 0 and (
                    s.type.dtype in discrete_dtypes
                ):
                    if isinstance(s, Constant):
                        return constant(int(s.data), dtype="int64")
                    return cast(s, "int64")
            return s
        if isinstance(s, (np.integer, int)) or (
            isinstance(s, np.ndarray) and s.ndim == 0
        ):
            if int(s) < 0:
                raise AssertionError(
                    "There is a negative shape in the graph!"
                    + get_variable_trace_string(var)
                )
            return constant(int(s), dtype="int64")
        if isinstance(s, float) and int(s) == s:
            return constant(int(s), dtype="int64")
        # Fallback: try as_tensor_variable + cast
        s = as_tensor_variable(s)
        if s.type.dtype != "int64":
            s = cast(s, "int64")
        return s

    # ---- materialization ----

    def get_shape(self, v, i):
        if hasattr(v.type, "shape") and v.type.shape[i] is not None:
            return constant(v.type.shape[i], dtype="int64")
        if v in self._overrides:
            ov = self._overrides[v]
            if ov is not None:
                return ov[i]
        if v.owner is None:
            return self._shape_i_var(v, i)

        node = v.owner
        entry = self._cache.get(node)
        if entry is None:
            entry = self._build_kernel(node)
            self._cache[node] = entry
        kernel, meta = entry
        if kernel is None:
            return self._shape_i_var(v, i)

        out_idx = node.outputs.index(v)
        layout = meta["output_layout"]
        if layout[out_idx] is None:
            return self._shape_i_var(v, i)
        slot = sum((layout[k] or 0) for k in range(out_idx)) + i

        slot_to_input_idx = meta["slot_to_input_idx"]
        replacements = {}
        for k_input, role in zip(kernel.inputs, meta["roles"], strict=True):
            tag = role[0]
            if tag == "input_slot":
                slot_idx = role[1]
                replacements[k_input] = node.inputs[slot_to_input_idx[slot_idx]]
            elif tag == "input_shape_slot":
                slot_idx, j = role[1], role[2]
                replacements[k_input] = self.get_shape(
                    node.inputs[slot_to_input_idx[slot_idx]], j
                )
            elif tag == "fallback_out":
                k = role[1]
                replacements[k_input] = node.outputs[k]

        return _materialize_frozen(kernel.outputs[slot], replacements)

    # ---- back-compat API used by pytensor.compile.builders and tests ----

    def get_node_infer_shape(self, node):
        """Back-compat: synthesize a per-output shape list for ``node``
        using the kernel-based materialization.
        """
        res: list[tuple | None] = []
        for out in node.outputs:
            if not hasattr(out.type, "ndim"):
                res.append(None)
                continue
            res.append(tuple(self.get_shape(out, i) for i in range(out.type.ndim)))
        return res

    # ---- helpers kept from the legacy implementation ----

    def shape_ir(self, i, r):
        """Return symbolic r.shape[i] for tensor variable r, int i."""
        return self.get_shape(r, i)

    def shape_tuple(self, r):
        """Return a tuple of symbolic shape vars for tensor variable r."""
        if not hasattr(r.type, "ndim"):
            return None
        return tuple(self.get_shape(r, i) for i in range(r.type.ndim))

    def unpack(self, s_i, var):
        """Return a symbolic integer scalar for the shape element s_i."""
        assert s_i is not None

        if s_i == 1:
            return self.lscalar_one
        if isinstance(s_i, float) and int(s_i) == s_i:
            s_i = int(s_i)
        if isinstance(s_i, np.integer | int) or (
            isinstance(s_i, np.ndarray) and s_i.ndim == 0
        ):
            if s_i < 0:
                msg = "There is a negative shape in the graph!"
                msg += get_variable_trace_string(var)
                raise AssertionError(msg)
            return constant(s_i, dtype="int64")
        if isinstance(s_i, tuple | list):
            raise NotImplementedError(s_i)

        if (
            s_i.owner
            and isinstance(s_i.owner.op, Subtensor)
            and s_i.owner.inputs[0].owner
            and isinstance(s_i.owner.inputs[0].owner.op, Shape)
        ):
            assert s_i.type.ndim == 0
            assert len(s_i.owner.op.idx_list) == 1
            idx = get_idx_list(s_i.owner.inputs, s_i.owner.op.idx_list)
            assert len(idx) == 1
            idx = idx[0]
            try:
                i = get_scalar_constant_value(idx)
            except NotScalarConstantError:
                pass
            else:
                x = s_i.owner.inputs[0].owner.inputs[0]
                s_i = self.get_shape(x, int(i))

        if s_i.type.dtype in integer_dtypes:
            if getattr(s_i.type, "ndim", 0):
                raise TypeError("Shape element must be scalar", s_i)
            if s_i.type.dtype != "int64":
                if isinstance(s_i, Constant):
                    s_i = constant(int(s_i.data), dtype="int64")
                else:
                    s_i = cast(s_i, "int64")
            return s_i
        else:
            raise TypeError(
                "Unsupported shape element", s_i, type(s_i), getattr(s_i, "type", None)
            )

    def set_shape(self, r, s, override=False):
        """Record a shape override for ``r``.

        With the kernel-based feature, per-variable shape is materialized
        lazily via ``get_shape``. ``set_shape`` installs an override that
        ``get_shape`` consults first. This is how
        ``pytensor.compile.builders.infer_shape`` supplies input shapes
        for an otherwise empty ``FunctionGraph``.
        """
        if s is None:
            self._overrides[r] = None
            return
        if not isinstance(s, tuple | list | _LazyShapeTuple):
            raise TypeError("shapes must be tuple/list", (r, s))

        if r.type.ndim != len(s):
            sio = StringIO()
            pytensor.printing.debugprint(r, file=sio, print_type=True)
            raise AssertionError(
                f"Something inferred a shape with {len(s)} dimensions "
                f"for a variable with {int(r.type.ndim)} dimensions"
                f" for the variable:\n{sio.getvalue()}"
            )

        shape_vars = []
        for i in range(r.type.ndim):
            if hasattr(r.type, "shape") and r.type.shape[i] is not None:
                shape_vars.append(constant(r.type.shape[i], dtype="int64"))
            else:
                shape_vars.append(self.unpack(s[i], r))
        self._overrides[r] = tuple(shape_vars)

    def init_r(self, r):
        """Back-compat no-op: kernel feature materializes lazily."""
        return

    def update_shape(self, r, other_r):
        """Register that ``r`` and ``other_r`` have the same shape.

        Feeds the equivalence into the union-find used by ``same_shape``.
        """
        if other_r is None or not hasattr(other_r.type, "ndim"):
            return
        if not hasattr(r.type, "ndim"):
            return
        if r.type.ndim != other_r.type.ndim:
            return
        for i in range(r.type.ndim):
            try:
                self.register_equal(r, i, other_r, i)
            except Exception:
                pass

    def make_vector_shape(self, r):
        """Return an int64 vector of ``r``'s shape, built lazily from kernels."""
        elems = [self.get_shape(r, i) for i in range(r.type.ndim)]
        return as_tensor_variable(elems, ndim=1, dtype="int64")

    # ---- same_shape via content-addressed shape_key + union-find ----

    def shape_key(self, v, i):
        """Canonical key for ``v.shape[i]`` (structural, with UF closure)."""
        return self._uf_find(self._raw_shape_key(v, i))

    def _raw_shape_key(self, v, i):
        if hasattr(v.type, "shape") and v.type.shape[i] is not None:
            return ("const", int(v.type.shape[i]))
        if v in self._overrides:
            ov = self._overrides[v]
            if ov is not None:
                return self._expr_key(ov[i])
        if v.owner is None:
            return ("leaf", id(v), i)
        node = v.owner
        entry = self._cache.get(node)
        if entry is None:
            entry = self._build_kernel(node)
            self._cache[node] = entry
        kernel, meta = entry
        if kernel is None:
            return ("leaf", id(v), i)
        out_idx = node.outputs.index(v)
        layout = meta["output_layout"]
        if layout[out_idx] is None:
            return ("leaf", id(v), i)
        slot = sum((layout[k] or 0) for k in range(out_idx)) + i
        sv = kernel.outputs[slot]
        return self._hash_sv(sv, kernel, meta, node)

    def _hash_sv(self, sv, kernel, meta, node):
        # See through Assert / CheckAndRaise wrappers and through no-op casts
        # so `same_shape` ignores runtime checks and dtype coercions.
        sv = _strip_shape_noise(sv)
        if sv.owner is None:
            if isinstance(sv, Constant):
                try:
                    return ("const", int(sv.data))
                except Exception:
                    return ("const", id(sv))
            try:
                k_idx = kernel.inputs.index(sv)
            except ValueError:
                return ("opaque", id(sv))
            role = meta["roles"][k_idx]
            tag = role[0]
            slot_to_input_idx = meta["slot_to_input_idx"]
            if tag == "input_slot":
                slot_idx = role[1]
                return ("input", id(node.inputs[slot_to_input_idx[slot_idx]]))
            if tag == "input_shape_slot":
                slot_idx, j = role[1], role[2]
                return self.shape_key(node.inputs[slot_to_input_idx[slot_idx]], j)
            if tag == "fallback_out":
                k = role[1]
                return ("fallback", id(node.outputs[k]), getattr(sv.owner, "op", None))
            return ("opaque", id(sv))

        op_props = getattr(sv.owner.op, "_props", None)
        op_key = op_props() if callable(op_props) else None
        child_keys = tuple(
            self._hash_sv(inp, kernel, meta, node) for inp in sv.owner.inputs
        )
        return (type(sv.owner.op).__name__, op_key, child_keys)

    def _expr_key(self, v):
        """Canonical key for a shape expression rooted in live variables."""
        v = _strip_shape_noise(v)
        if isinstance(v, Constant):
            try:
                return ("const", int(v.data))
            except Exception:
                return ("const", id(v))
        if v.owner is None:
            return ("leaf", id(v), 0)
        op_props = getattr(v.owner.op, "_props", None)
        op_key = op_props() if callable(op_props) else None
        child_keys = tuple(self._expr_key(inp) for inp in v.owner.inputs)
        return (type(v.owner.op).__name__, op_key, child_keys)

    def same_shape(
        self,
        x: Variable,
        y: Variable,
        dim_x: int | None = None,
        dim_y: int | None = None,
    ) -> bool:
        """Return ``True`` if ``x`` and ``y`` have the same shape (along
        ``dim_x`` / ``dim_y`` if given, else all dims).
        """
        if dim_x is None and dim_y is None:
            if x.type.ndim != y.type.ndim:
                return False
            for i in range(x.type.ndim):
                if not self.same_shape(x, y, i, i):
                    return False
            return True
        if dim_x is None:
            dim_x = dim_y
        if dim_y is None:
            dim_y = dim_x
        # Force IndexError semantics matching the legacy impl.
        x.type.shape[dim_x]
        y.type.shape[dim_y]
        return self.shape_key(x, dim_x) == self.shape_key(y, dim_y)

    # ---- union-find over shape_keys ----

    def register_equal(self, v1, i1, v2, i2):
        k1 = self.shape_key(v1, i1)
        k2 = self.shape_key(v2, i2)
        self._uf_union(k1, k2)

    def _uf_find(self, key):
        parent = self._uf_parent
        root = key
        while parent.get(root, root) != root:
            root = parent[root]
        while key != root:
            nxt = parent.get(key, key)
            parent[key] = root
            key = nxt
        return root

    def _uf_union(self, k1, k2):
        r1 = self._uf_find(k1)
        r2 = self._uf_find(k2)
        if r1 == r2:
            return
        if r1[0] == "const" and r2[0] != "const":
            self._uf_parent[r2] = r1
        elif r2[0] == "const" and r1[0] != "const":
            self._uf_parent[r1] = r2
        else:
            self._uf_parent[r2] = r1

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
    inp, output_shape = node.inputs
    [output] = node.outputs

    # Trivial case, all dimensions of input/output are known to be broadcastable:
    # there's nothing to reshape
    if all(inp.type.broadcastable) or all(output.type.broadcastable):
        squeeze_axes = tuple(range(inp.type.ndim))
        new_output_shape = []
        expand_axes = tuple(range(output.type.ndim))

    else:
        squeeze_axes = [i for i, bcast in enumerate(inp.type.broadcastable) if bcast]
        unpacked_shape = _unpack_shape_vector(output_shape)
        new_output_shape = []
        expand_axes = []
        for i, dim_length in enumerate(unpacked_shape):
            if isinstance(dim_length, Constant) and (
                dim_length.data == 1
                # -1 can be an implicit expand_dims, but it's tricky to prove
                # as we would need to check whether all other dimensions
                # already explain the full size of the array.
                # Example: np.zeros((2, 2, 2)).reshape((8, -1))
                # We rely on the output static shape which will already have figured
                # it out for some (but not all) cases
                or (dim_length.data == -1 and output.type.shape[i] == 1)
            ):
                expand_axes.append(i)
            else:
                new_output_shape.append(dim_length)

    if squeeze_axes or expand_axes:
        new_out = inp.squeeze(squeeze_axes)

        if new_output_shape:
            new_out = new_out.reshape(new_output_shape)
            copy_stack_trace(output, new_out)

        new_out = expand_dims(new_out, expand_axes)

        if not new_output_shape:
            # Eagerly merge consecutive squeeze and expand_dims
            new_out = apply_local_dimshuffle_lift(fgraph, new_out)

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
    if (
        isinstance(node.op, Reshape)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Elemwise)
        and len(node.inputs[0].owner.inputs) == 1
    ):
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

    if not isinstance(node.op, SpecifyShape):
        return False

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

    if not isinstance(node.op, Shape):
        return False

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
def local_specify_shape_lift(fgraph, node):
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


@register_infer_shape
@register_useless
@register_canonicalize
@node_rewriter([Shape_i])
def local_Shape_i_ground(fgraph, node):
    """Replace ``shape_i(x, i)`` with ``s`` when ``x.type.shape[i] == s``."""

    if not isinstance(node.op, Shape_i):
        return False

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
    if isinstance(node.op, Shape):
        if not hasattr(fgraph, "shape_feature"):
            return
        shape_feature = fgraph.shape_feature
        ret = shape_feature.make_vector_shape(node.inputs[0])

        # We need to copy over stack trace from input to output
        copy_stack_trace(node.outputs[0], ret)
        return [ret]


@register_infer_shape
@register_specialize
@register_canonicalize
@node_rewriter([Shape_i])
def local_track_shape_i(fgraph, node):
    """Rewrite ``Shape_i(v, i)`` to the kernel-inferred shape expression.

    With the kernel-based `ShapeFeature`, per-node shape kernels are
    always rooted in live inputs.  Whenever ``v`` has an ``infer_shape``
    available, the kernel yields a non-``Shape_i`` expression for
    ``v.shape[i]``.  Rewriting the literal ``Shape_i(v, i)`` with the
    kernel expression lets rewrites downstream see the inferred form and
    typically lets the original producer node of ``v`` be pruned when
    only its shape is consumed.
    """
    shape_feature = getattr(fgraph, "shape_feature", None)
    if shape_feature is None:
        return False

    [v] = node.inputs
    if v.owner is None:
        return False

    i = node.op.i
    new_shape = shape_feature.get_shape(v, i)
    if new_shape is None:
        return False

    # Avoid rewriting Shape_i(v, i) to itself.
    if new_shape.owner is node or (
        isinstance(new_shape, Variable)
        and new_shape.owner is not None
        and isinstance(new_shape.owner.op, Shape_i)
        and new_shape.owner.op.i == i
        and new_shape.owner.inputs[0] is v
    ):
        return False

    return [new_shape]
