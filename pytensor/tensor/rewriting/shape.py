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
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    IncSubtensor,
    Subtensor,
)
from pytensor.tensor.type import TensorType, integer_dtypes, lscalar
from pytensor.tensor.type_other import NoneTypeT
from pytensor.tensor.variable import TensorVariable


class ShapeFeature(Feature):
    r"""Kernel-based `Feature` that tracks shape information in a graph.

    For each `Apply`, a `FrozenFunctionGraph` "kernel" is built once and
    stored in ``_shape_kernel_cache[node]``. The kernel is rooted in *dummy*
    variables — never the live outer variables — so it can't go stale
    as the fgraph mutates. Shape requests materialize the kernel
    against today's ``node.inputs`` (and recursive shape lookups), so
    returned expressions are always rooted in live variables.

    Public API:

    - ``get_shape(v, i)`` — materialize ``v.shape[i]``.
    - ``shape_tuple(v)`` — materialize ``tuple(v.shape)``.
    - ``unaliased_shape_tuple(v, dims=None)`` — like ``shape_tuple`` but
      breaks aliasing-induced cycles so the result is safe to import
      into the attached fgraph alongside its inplace destroyers.
    - ``same_shape(x, y, dim_x=None, dim_y=None)`` — via content-addressed ``shape_key``.
    """

    _scalar_shape = constant(np.array([], dtype="int64"))

    def __init__(self):
        # node -> (kernel, meta) from _build_kernel, lazily populated
        self._shape_kernel_cache: dict = {}
        # node -> {slot: (dim_kernel, used_roles) | None}, per-dim views of _shape_kernel_cache
        self._dim_kernel_cache: dict = {}
        # var -> ndim-tuple of (dim_kernel, role_bindings) | None,
        # installed by on_change_input when new_r's Op has no infer_shape
        self._overrides: dict = {}
        # (id(v), i) -> Shape_i(i)(v), ensures Apply identity for leaves
        self._shape_i_cache: dict = {}
        # node -> {(out_idx, i): result}, canonicalized get_shape results
        self._materialized_dim_cache: dict = {}
        self.fgraph: FunctionGraph | None = None

    def _shape_i_var(self, v, i):
        key = (id(v), i)
        cached = self._shape_i_cache.get(key)
        if cached is not None:
            return cached
        res = Shape_i(i)(v)
        self._shape_i_cache[key] = res
        return res

    def _canonicalize_live_shape(self, s, memo=None):
        """Rewrite ``Shape(x)`` / ``Subtensor(Shape(x), const_i)`` patterns
        into ``MakeVector(Shape_i_0, …)`` / ``Shape_i(const_i)(x)``.

        Why: some ``infer_shape`` impls (e.g. ``Alloc``: ``return [node.inputs[1:]]``)
        pipe live shape inputs through verbatim. Those live inputs were
        often built by user code as ``v.shape[axis]`` — i.e.
        ``Subtensor(Shape(v), axis)`` Applies. If those reach the
        materialized output of ``get_shape`` unchanged, EquilibriumGraphRewriter
        keeps re-firing ``local_shape_to_shape_i`` on each fresh ``Shape(v)``
        we re-emit, never reaching a fixed point. Pre-canonicalizing here
        means the materialized shape never contains ``Shape(...)`` Apply
        nodes — only ``Shape_i`` leaves the optimizer leaves alone.
        """
        if memo is None:
            memo = {}
        cached = memo.get(s)
        if cached is not None:
            return cached
        if not isinstance(s, Variable) or s.owner is None:
            memo[s] = s
            return s

        node = s.owner
        op = node.op

        if isinstance(op, Subtensor) and op.idx_list == (0,):
            base, idx = node.inputs
            if isinstance(base.owner_op, Shape):
                x = base.owner.inputs[0]
                try:
                    idx_const = int(get_scalar_constant_value(idx))
                    memo[s] = result = self.get_shape(x, idx_const)
                    return result
                except (NotScalarConstantError, IndexError):
                    pass

        if isinstance(op, Shape):
            x = node.inputs[0]
            dims = [self.get_shape(x, j) for j in range(x.type.ndim)]
            if dims:
                result = stack(dims)
            else:
                result = self._scalar_shape
            memo[s] = result
            return result

        new_inputs = [self._canonicalize_live_shape(inp, memo) for inp in node.inputs]
        if all(ni is oi for ni, oi in zip(new_inputs, node.inputs)):
            memo[s] = s
            return s
        new_node = op.make_node(*new_inputs)
        new_out = new_node.outputs[node.outputs.index(s)]
        memo[s] = new_out
        return new_out

    def on_attach(self, fgraph):
        if hasattr(fgraph, "shape_feature"):
            raise AlreadyThere("This FunctionGraph already has a ShapeFeature")
        if self.fgraph is not None and self.fgraph is not fgraph:
            raise Exception("This ShapeFeature is already attached to a graph")
        self.fgraph = fgraph
        fgraph.shape_feature = self

    def on_detach(self, fgraph):
        self._shape_kernel_cache.clear()
        self._overrides.clear()
        self._shape_i_cache.clear()
        self._materialized_dim_cache.clear()
        self._dim_kernel_cache.clear()
        self.fgraph = None
        if hasattr(fgraph, "shape_feature"):
            del fgraph.shape_feature

    def on_prune(self, fgraph, node, reason):
        self._shape_kernel_cache.pop(node, None)
        self._dim_kernel_cache.pop(node, None)
        self._materialized_dim_cache.pop(node, None)
        for out in node.outputs:
            oid = id(out)
            for j in range(getattr(out.type, "ndim", 0)):
                self._shape_i_cache.pop((oid, j), None)
            self._overrides.pop(out, None)

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        # Carry r's shape forward as a *kernel-borrow* override when
        # ``new_r``'s Op has no ``infer_shape``. Per-dim, we rederive r's
        # shape kernel against ``new_r.owner.inputs`` by matching
        # kernel-input bindings via ``shape_key``; if every binding finds
        # a structurally-equal counterpart we store the dim_kernel plus
        # the ``(input_idx, dim)`` positions to look up in
        # ``new_r.owner.inputs`` at access time. No live Variables are
        # pinned. Per-dim ``None`` means "couldn't reroute; fall back to
        # ``Shape_i``".
        if r is new_r or not hasattr(new_r.type, "ndim"):
            return
        self._materialized_dim_cache.pop(node, None)
        if new_r in self._overrides:
            return
        if new_r.owner is None:
            return  # graph inputs have their own Shape_i fallback
        if getattr(new_r.owner.op, "infer_shape", None) is not None:
            return  # new_r's own kernel will produce a real shape
        if not hasattr(r.type, "ndim") or r.type.ndim != new_r.type.ndim:
            return
        new_owner_inputs = new_r.owner.inputs
        entries = []
        any_set = False
        for k in range(r.type.ndim):
            e = self._reroute_dim(r, k, new_owner_inputs)
            if e is not None:
                any_set = True
            entries.append(e)
        if any_set:
            self._overrides[new_r] = tuple(entries)

    def _reroute_dim(self, r, k, new_r_owner_inputs):
        """Try to rederive ``r.shape[k]`` against ``new_r_owner_inputs``.

        Returns ``(dim_kernel, role_bindings)`` on success, where
        ``role_bindings`` is a tuple of ``(input_idx, dim)`` aligned with
        ``dim_kernel.inputs`` — indices into ``new_r_owner_inputs`` whose
        ``shape_key`` matches the corresponding live binding under r's
        owner.

        Returns ``None`` when (a) the kernel uses any role other than
        ``input_shape_slot`` (``input_slot`` would need value-equality;
        ``self_out`` references r's own outputs and can't reroute),
        or (b) some role's binding has no structurally-equal
        counterpart in ``new_r_owner_inputs``.
        """
        if r.owner is None:
            return None
        if (entry := self._shape_kernel_cache.get(r.owner)) is None:
            entry = self._build_kernel(r.owner)
            self._shape_kernel_cache[r.owner] = entry
        kernel, meta = entry
        if kernel is None:
            return None
        out_idx = r.owner.outputs.index(r)
        layout = meta["output_layout"]
        if layout[out_idx] is None:
            return None
        slot = sum((layout[k_] or 0) for k_ in range(out_idx)) + k
        dk = self._dim_kernel(r.owner, slot)
        if dk is None:
            return None
        dim_kernel, used_roles = dk

        if any(role[0] != "input_shape_slot" for role in used_roles):
            return None

        slot_to_input_idx = meta["slot_to_input_idx"]
        role_bindings = []
        for role in used_roles:
            s, j = role[1], role[2]
            r_inp = r.owner.inputs[slot_to_input_idx[s]]
            target_key = self.shape_key(r_inp, j)
            match = None
            for idx, inp in enumerate(new_r_owner_inputs):
                if not hasattr(inp.type, "ndim"):
                    continue
                for d in range(inp.type.ndim):
                    if self.shape_key(inp, d) == target_key:
                        match = (idx, d)
                        break
                if match is not None:
                    break
            if match is None:
                return None
            role_bindings.append(match)
        return (dim_kernel, tuple(role_bindings))

    def _override_shape_key(self, v, i, entry):
        """Content-addressed key for an override entry; see ``shape_key``."""
        if entry is None:
            return ("leaf", id(v), i)
        dim_kernel, role_bindings = entry
        new_owner_inputs = v.owner.inputs
        sv = dim_kernel.outputs[0]
        if sv.owner is None:
            # Passthrough: kernel output is one of its inputs (no computation).
            # Collapse to the source input's shape_key directly so equality
            # is transitive across chains of passthrough overrides.
            if isinstance(sv, Constant):
                return ("const", int(sv.data))
            try:
                k_idx = dim_kernel.inputs.index(sv)
            except ValueError:
                return ("opaque", id(sv))
            idx, dim = role_bindings[k_idx]
            return self.shape_key(new_owner_inputs[idx], dim)
        bindings = tuple(
            self.shape_key(new_owner_inputs[idx], dim) for idx, dim in role_bindings
        )
        return (dim_kernel, bindings)

    def _build_kernel(self, node):
        # Phase 1: Deduplicate inputs.
        # When the same live input appears at multiple positions (e.g.
        # ``add(x, x)``), share the dummy clone and shape scalars so
        # ``broadcast_shape`` sees identity-equal shapes and elides
        # the runtime Assert.
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

        # Phase 2: Call infer_shape with dummy node.
        dummy_outputs = [out.clone() for out in node.outputs]
        dummy_node = Apply(node.op, dummy_inputs, dummy_outputs)

        output_shapes = None
        shape_infer = getattr(node.op, "infer_shape", None)
        if shape_infer is not None:
            try:
                output_shapes = shape_infer(dummy_node, dummy_input_shapes)
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

        # Phase 3: Coerce and validate each shape element returned by
        # infer_shape. Static type shape overrides infer_shape when known,
        # ensuring the canonical constant form.
        def coerce_shape_el(s, dummy_out):
            if isinstance(s, np.ndarray):
                if s.ndim != 0:
                    raise TypeError(
                        f"infer_shape for {node.op} returned a non-scalar "
                        f"ndarray for shape element: {s!r}"
                    )
                s = s.item()
            if isinstance(s, Variable):
                if s.type.dtype not in integer_dtypes:
                    raise TypeError(
                        f"infer_shape for {node.op} returned a non-integer "
                        f"Variable for shape element: {s!r}"
                    )
                if getattr(s.type, "ndim", 0):
                    raise TypeError(
                        f"infer_shape for {node.op} returned a non-scalar "
                        f"Variable for shape element: {s!r}"
                    )
                return s
            if isinstance(s, int | np.integer):
                if int(s) < 0:
                    raise ValueError(
                        f"infer_shape for {node.op} returned a negative "
                        f"shape: {int(s)}" + get_variable_trace_string(dummy_out)
                    )
                return constant(int(s), dtype="int64")
            raise TypeError(
                f"infer_shape for {node.op} returned an unsupported "
                f"shape element of type {type(s).__name__}: {s!r}"
            )

        # Outputs with missing/malformed infer_shape get None, which
        # propagates to output_layout[k] = None. get_shape / shape_key
        # short-circuit to _shape_i_var(v, i) for those.
        coerced_output_shapes = []
        for k, dummy_out in enumerate(dummy_outputs):
            sh = output_shapes[k] if k < len(output_shapes) else None
            if not hasattr(dummy_out.type, "ndim"):
                coerced_output_shapes.append(None)
                continue
            if sh is None or not isinstance(sh, list | tuple):
                coerced_output_shapes.append(None)
                continue
            coerced = []
            for i, s in enumerate(sh):
                if (
                    hasattr(dummy_out.type, "shape")
                    and dummy_out.type.shape[i] is not None
                ):
                    coerced.append(constant(dummy_out.type.shape[i], dtype="int64"))
                    continue
                coerced.append(coerce_shape_el(s, dummy_out))
            coerced_output_shapes.append(tuple(coerced))

        # Phase 4: Flatten per-output shape tuples into a single list.
        # layout[k] records how many dims output k contributed (or None).
        flat_out = []
        layout = []
        for sh in coerced_output_shapes:
            if sh is None:
                layout.append(None)
                continue
            layout.append(len(sh))
            flat_out.extend(sh)

        meta = {"output_layout": tuple(layout)}
        if not flat_out:
            return (None, meta)

        # Phase 5: Build kernel inputs with roles.
        # Three role types: input_slot (the dummy tensor itself),
        # input_shape_slot (a shape scalar of a dummy), self_out (a dummy
        # output referenced by infer_shape, e.g. Scan).
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

        # Some infer_shape impls (e.g. Scan) reference dummy_node.outputs
        # in the returned expression. Register those as self_out inputs so
        # materialization can substitute live outputs.
        anc_set = set(ancestors(flat_out))
        for k, dummy_out in enumerate(dummy_outputs):
            if dummy_out in anc_set and dummy_out not in kernel_inputs:
                kernel_inputs.append(dummy_out)
                roles.append(("self_out", k))

        # Phase 6: Map unique slots back to live input indices.
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

    def get_shape(self, v, i):
        if hasattr(v.type, "shape") and v.type.shape[i] is not None:
            return constant(v.type.shape[i], dtype="int64")

        node = v.owner
        if node is None:
            return self._shape_i_var(v, i)

        node_cache = self._materialized_dim_cache.get(node)
        if node_cache is not None:
            cached = node_cache.get((v, i))
            if cached is not None:
                return cached
        else:
            node_cache = {}
            self._materialized_dim_cache[node] = node_cache

        def _walk(dim_kernel, memo):
            for fa in dim_kernel.toposort():
                new_inputs = [memo.get(inp, inp) for inp in fa.inputs]
                new_node = fa.op.make_node(*new_inputs)
                memo.update(zip(fa.outputs, new_node.outputs, strict=True))
            raw = memo.get(dim_kernel.outputs[0], dim_kernel.outputs[0])
            return self._canonicalize_live_shape(raw)

        def _fallback():
            node_cache[(v, i)] = result = self._shape_i_var(v, i)
            return result

        if (ov := self._overrides.get(v)) is not None:
            entry = ov[i]
            if entry is None:
                return _fallback()
            dim_kernel, role_bindings = entry
            memo = {
                k_input: self.get_shape(node.inputs[idx], dim)
                for k_input, (idx, dim) in zip(
                    dim_kernel.inputs, role_bindings, strict=True
                )
            }
            node_cache[(v, i)] = result = _walk(dim_kernel, memo)
            return result

        if (entry := self._shape_kernel_cache.get(node)) is None:
            self._shape_kernel_cache[node] = entry = self._build_kernel(node)

        kernel, meta = entry
        if kernel is None:
            return _fallback()

        out_idx = node.outputs.index(v)
        layout = meta["output_layout"]
        if layout[out_idx] is None:
            return _fallback()
        slot = sum((layout[k] or 0) for k in range(out_idx)) + i
        dk = self._dim_kernel(node, slot)
        if dk is None:
            return _fallback()
        dim_kernel, used_roles = dk

        slot_to_input_idx = meta["slot_to_input_idx"]
        memo = {}
        for k_input, role in zip(dim_kernel.inputs, used_roles, strict=True):
            tag = role[0]
            if tag == "input_slot":
                memo[k_input] = node.inputs[slot_to_input_idx[role[1]]]
            elif tag == "input_shape_slot":
                memo[k_input] = self.get_shape(
                    node.inputs[slot_to_input_idx[role[1]]], role[2]
                )
            else:
                memo[k_input] = node.outputs[role[1]]
        node_cache[(v, i)] = result = _walk(dim_kernel, memo)
        return result

    def unaliased_shape_tuple(self, v, dims=None):
        """Like :meth:`shape_tuple`, but free of aliasing-induced cycles
        so the result can be imported into ``self.fgraph`` alongside its
        inplace destroyers.

        ``shape_tuple`` returns dim expressions that may share live Apply
        nodes with the rest of the fgraph. If one of those Applies reads
        a destroyed scalar ``x`` directly *and* (via another input)
        depends on its destroyer's output, importing the shape into the
        fgraph would trip the destroy handler — that Apply would have to
        run both before and after the destroyer. This wrapper materializes
        the requested dims and runs the per-Apply cycle break in one pass
        via :func:`pytensor.graph.replace.break_aliasing_cycles`. No-op
        when the fgraph has no ``DestroyHandler`` or no destroyers; in
        that case it's equivalent to ``shape_tuple``.

        Parameters
        ----------
        v
            Variable whose shape we want.
        dims
            Optional iterable of dim indices to materialize (defaults to
            all dims of ``v``). Negative indices follow Python convention.
            Pass an explicit subset to avoid materializing dims you don't
            need — both the kernel call and the cycle-break walk are
            scoped to the dims actually requested.

        Returns ``None`` if ``v`` has no ``ndim``.
        """
        if not hasattr(v.type, "ndim"):
            return None
        if dims is None:
            dims = range(v.type.ndim)
        shape = [self.get_shape(v, i) for i in dims]
        fgraph = self.fgraph
        dh = getattr(fgraph, "destroy_handler", None) if fgraph is not None else None
        if dh is None or not dh.destroyers:
            return tuple(shape)

        from pytensor.graph.replace import break_aliasing_cycles

        return tuple(break_aliasing_cycles(shape, fgraph.destroyers))

    def shape_tuple(self, r):
        """Return a tuple of symbolic shape vars for tensor variable r."""
        if not hasattr(r.type, "ndim"):
            return None
        return tuple(self.get_shape(r, i) for i in range(r.type.ndim))

    def _dim_kernel(self, node, slot):
        """Lazily-built per-dim ``FrozenFunctionGraph`` view of the
        per-node kernel for ``kernel.outputs[slot]``.

        Returns ``(dim_kernel, used_roles)`` or ``None`` if no kernel.
        ``dim_kernel`` is a single-output ``FrozenFunctionGraph`` whose
        inputs are the subset of ``kernel.inputs`` reachable from the
        slot, in their original kernel order. Two structurally identical
        slot DAGs produce ``__eq__`` ``FrozenFunctionGraph`` objects (via
        global ``FrozenApply``/``NominalVariable`` interning), letting
        ``shape_key`` collapse the structural comparison to one hash and
        only descend into inputs that are themselves shape lookups.
        """
        node_cache = self._dim_kernel_cache.get(node)
        if node_cache is None:
            node_cache = {}
            self._dim_kernel_cache[node] = node_cache
        if slot in node_cache:
            return node_cache[slot]
        if (entry := self._shape_kernel_cache.get(node)) is None:
            entry = self._build_kernel(node)
            self._shape_kernel_cache[node] = entry
        kernel, meta = entry
        if kernel is None:
            node_cache[slot] = None
            return None
        sv = kernel.outputs[slot]
        kernel_input_set = set(kernel.inputs)
        used = {anc for anc in ancestors([sv]) if anc in kernel_input_set}
        used_inputs = tuple(inp for inp in kernel.inputs if inp in used)
        roles = meta["roles"]
        used_roles = tuple(
            roles[i] for i, inp in enumerate(kernel.inputs) if inp in used
        )
        try:
            dim_kernel = FrozenFunctionGraph(used_inputs, [sv])
        except Exception:
            node_cache[slot] = None
            return None
        result = (dim_kernel, used_roles)
        node_cache[slot] = result
        return result

    def shape_key(self, v, i):
        """Hashable key for ``v.shape[i]`` such that two keys compare equal
        iff this feature can prove the two shapes are the same.

        The key is shaped ``(dim_kernel, bindings)``:

        - ``dim_kernel`` is the per-dim ``FrozenFunctionGraph`` view from
          ``_dim_kernel``. ``FrozenApply`` and ``NominalVariable`` are
          globally interned, so two structurally identical shape
          expressions produce ``__eq__`` kernels — content-addressed
          structural equality with no manual op-tree walk on this side.
        - ``bindings`` records what's bound at each kernel-input
          position. An ``id`` for the live var at ``input_slot`` /
          ``self_out`` leaves, and a recursive ``shape_key`` call for
          ``input_shape_slot`` leaves — whose binding is itself a
          sub-shape (``node.inputs[k]``'s dim j), which can in turn
          hit any of these branches again. The recursion is bounded
          by graph depth.

        Special cases handled before the kernel path:

        - **static dim** → ``("const", value)``.
        - **override** → routed through ``_override_shape_key`` against
          the borrowed ``(dim_kernel, role_bindings)`` tuple. Same
          structure as the kernel path below: passthrough slots collapse
          to the underlying live var's key, otherwise
          ``(dim_kernel, recursive_shape_keys)``. No identity-only
          fallback — a rerouted override compares equal to any
          structurally-equal kernel shape.
        - **untracked leaf** (no owner, kernel build failed, or this
          output isn't laid out) → ``("leaf", id(v), i)``.
        - **passthrough slot** (kernel output is a kernel input
          directly) → return the underlying live var's binding so the
          key matches that var's own ``shape_key``.

        Known limitation: shape sub-expressions baked into a kernel via
        ``Op(input).shape`` (e.g. an ``infer_shape`` impl that takes
        ``foo(node.inputs[0]).shape[0]``) are compared *structurally* as
        part of the parent kernel — ``same_shape`` will not equate two
        such kernels even when the inner ops have equivalent shape
        kernels. Cross-kernel shape-equivalence is only detected through
        ``input_shape_slot`` bindings, which are the explicit seams
        ``_build_kernel`` creates. A follow-up could inline sub-kernels
        at build time (analogous to how ``_canonicalize_live_shape``
        resolves ``Subtensor(Shape(...))`` at materialization) to close
        this gap.
        """
        if hasattr(v.type, "shape") and v.type.shape[i] is not None:
            return ("const", int(v.type.shape[i]))
        if (ov := self._overrides.get(v)) is not None:
            return self._override_shape_key(v, i, ov[i])
        node = v.owner
        if node is None:
            return ("leaf", id(v), i)
        if (entry := self._shape_kernel_cache.get(node)) is None:
            entry = self._build_kernel(node)
            self._shape_kernel_cache[node] = entry
        kernel, meta = entry
        if kernel is None:
            return ("leaf", id(v), i)
        out_idx = node.outputs.index(v)
        layout = meta["output_layout"]
        if layout[out_idx] is None:
            return ("leaf", id(v), i)
        slot = sum((layout[k] or 0) for k in range(out_idx)) + i
        sv = kernel.outputs[slot]
        slot_to_input_idx = meta["slot_to_input_idx"]

        # Bind one kernel-input role to a live key. Only ``input_shape_slot``
        # needs to recurse (its leaf is a sub-shape, not a live var); every
        # other role bottoms out at a live ``node.inputs``/``node.outputs``,
        # whose ``id`` already discriminates by identity. Heterogeneous
        # element types (int id vs recursive tuple) don't collide.
        def bind(role):
            if role[0] == "input_shape_slot":
                return self.shape_key(node.inputs[slot_to_input_idx[role[1]]], role[2])
            if role[0] == "input_slot":
                return id(node.inputs[slot_to_input_idx[role[1]]])
            return id(node.outputs[role[1]])  # self_out

        # Passthrough slot: sv is a kernel input (or Constant) directly,
        # no shape function around it. Skip the dim-kernel wrapper so the
        # key matches the underlying live var's own shape_key.
        if sv.owner is None:
            if isinstance(sv, Constant):
                return ("const", int(sv.data))
            try:
                k_idx = kernel.inputs.index(sv)
            except ValueError:
                return ("opaque", id(sv))
            return bind(meta["roles"][k_idx])
        dk = self._dim_kernel(node, slot)
        if dk is None:
            return ("leaf", id(v), i)
        dim_kernel, used_roles = dk
        return (dim_kernel, tuple(bind(role) for role in used_roles))

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
        return bool(self.shape_key(x, dim_x) == self.shape_key(y, dim_y))

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
        IncSubtensor | AdvancedIncSubtensor1 | AdvancedIncSubtensor,
    ):
        x, y, *idx_vars = inc_x.owner.inputs
        new_x = specify_shape(x, specified_shape)
        new_out = inc_op(new_x, y, *idx_vars)
        copy_stack_trace(node.outputs[0], new_out)
        return [new_out]


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
        r = node.inputs[0]
        elems = [shape_feature.get_shape(r, i) for i in range(r.type.ndim)]
        ret = as_tensor_variable(elems, ndim=1, dtype="int64")

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
