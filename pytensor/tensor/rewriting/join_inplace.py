"""Ops and rewrites for eliminating Join buffer copies.

Instead of Join allocating a fresh buffer and copying each stream into it,
we pre-allocate the output buffer and have each stream write directly into
its slice. This eliminates intermediate allocations and concat copies.

Key ops:
- WriteSplit: Splits a buffer into non-overlapping slices WITHOUT
  declaring view_map. This lets the DestroyHandler treat each slice as
  independent, so multiple ops can destroy their own slices without conflict.
- WriteJoin: Returns the buffer after all inplace writes are complete.
  destroy_map ensures correct ordering and prevents stale reads.
"""

import pytensor.tensor as pt
from pytensor.compile import optdb
from pytensor.graph.basic import Apply
from pytensor.graph.rewriting.basic import GraphRewriter
from pytensor.link.c.op import COp
from pytensor.scalar.basic import Composite, ScalarType
from pytensor.tensor.basic import Join, as_tensor_variable
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.shape import Reshape
from pytensor.tensor.type import TensorType


class WriteSplit(COp):
    """Split a buffer into contiguous slices along a given axis.

    Unlike Subtensor, this op does NOT declare view_map. The DestroyHandler
    treats each output as an independent variable, allowing multiple
    downstream ops to destroy different outputs without conflict.

    At the memory level, each output IS a view of the input buffer.
    """

    __props__ = ("n_splits", "axis")

    def __init__(self, n_splits, axis=0):
        self.n_splits = n_splits
        self.axis = axis

    def make_node(self, buffer, *split_sizes):
        buffer = as_tensor_variable(buffer)
        split_sizes = [as_tensor_variable(s) for s in split_sizes]

        if len(split_sizes) != self.n_splits:
            raise ValueError(
                f"Expected {self.n_splits} split sizes, got {len(split_sizes)}"
            )

        outputs = []
        base_shape = list(buffer.type.shape)
        for i in range(self.n_splits):
            size_var = split_sizes[i]
            static_size = None
            if hasattr(size_var, "data"):
                static_size = int(size_var.data)
            out_shape = list(base_shape)
            out_shape[self.axis] = static_size
            out_type = TensorType(
                dtype=buffer.type.dtype,
                shape=tuple(out_shape),
            )
            outputs.append(out_type())

        return Apply(self, [buffer, *split_sizes], outputs)

    def perform(self, node, inputs, output_storage):
        buffer = inputs[0]
        split_sizes = inputs[1:]
        axis = self.axis

        offset = 0
        for i, size in enumerate(split_sizes):
            size = int(size)
            slices = [slice(None)] * buffer.ndim
            slices[axis] = slice(offset, offset + size)
            output_storage[i][0] = buffer[tuple(slices)]
            offset += size

    def infer_shape(self, fgraph, node, input_shapes):
        buffer_shape = list(input_shapes[0])
        out_shapes = []
        for i in range(self.n_splits):
            split_size = node.inputs[1 + i]
            shape = list(buffer_shape)
            shape[self.axis] = split_size
            out_shapes.append(tuple(shape))
        return out_shapes

    def c_code(self, node, name, inputs, outputs, sub):
        buf = inputs[0]
        size_vars = inputs[1:]
        fail = sub["fail"]
        n = self.n_splits
        ndim = node.inputs[0].type.ndim
        axis = self.axis

        outputs_pointers = "&" + ", &".join(outputs)

        size_lines = []
        for i, sv in enumerate(size_vars):
            dtype = node.inputs[1 + i].type.dtype_specs()[1]
            size_lines.append(
                f"npy_intp sz_{i} = (npy_intp)(({dtype}*)PyArray_DATA({sv}))[0];"
            )
        sizes_src = "\n".join(size_lines)

        return f"""
        {{
            PyArrayObject** outs[] = {{{outputs_pointers}}};
            int ndim = {ndim};
            int axis = {axis};
            npy_intp split_dims[{ndim}];
            npy_intp offset = 0;

            {sizes_src}
            npy_intp sizes[] = {{{", ".join(f"sz_{i}" for i in range(n))}}};

            // Bounds check: sum of sizes must equal buffer dim along axis
            npy_intp total = 0;
            for (int i = 0; i < {n}; ++i) total += sizes[i];
            if (total != PyArray_DIM({buf}, axis)) {{
                PyErr_Format(PyExc_ValueError,
                    "WriteSplit: sum of split sizes (%lld) != buffer dim %d (%lld)",
                    (long long)total, axis, (long long)PyArray_DIM({buf}, axis));
                {fail}
            }}

            memcpy(split_dims, PyArray_DIMS({buf}), ndim * sizeof(npy_intp));

            for (int i = 0; i < {n}; ++i) {{
                Py_XDECREF(*outs[i]);

                npy_intp data_offset = PyArray_STRIDE({buf}, axis) * offset;
                split_dims[axis] = sizes[i];

                PyArray_Descr *descr = PyArray_DESCR({buf});
                Py_INCREF(descr);
                *outs[i] = (PyArrayObject*)PyArray_NewFromDescr(
                    &PyArray_Type, descr, ndim, split_dims,
                    PyArray_STRIDES({buf}),
                    PyArray_BYTES({buf}) + data_offset,
                    PyArray_FLAGS({buf}) & ~NPY_ARRAY_OWNDATA,
                    NULL);

                if (*outs[i] == NULL) {{
                    PyErr_SetString(PyExc_RuntimeError,
                        "WriteSplit: unable to create view");
                    {fail}
                }}

                Py_INCREF((PyObject*){buf});
                PyArray_SetBaseObject(*outs[i], (PyObject*){buf});

                offset += sizes[i];
            }}
        }}
        """

    def c_code_cache_version(self):
        return (2,)


class WriteJoin(COp):
    """Return the buffer after all dependent inplace writes complete.

    Takes (buffer, *deps) where deps are outputs of inplace operations
    that wrote into the buffer. The deps create data dependencies ensuring
    all writes finish before the buffer is returned.

    destroy_map = {0: [0]} prevents downstream ops from reading stale
    buffer data and enables proper ordering via the DestroyHandler.
    """

    __props__ = ()
    destroy_map = {0: [0]}

    def make_node(self, buffer, *deps):
        buffer = as_tensor_variable(buffer)
        deps = [as_tensor_variable(d) for d in deps]
        out = buffer.type()
        return Apply(self, [buffer, *deps], [out])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = inputs[0]

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]]

    def c_code(self, node, name, inputs, outputs, sub):
        buf = inputs[0]
        out = outputs[0]

        return f"""
        Py_XDECREF({out});
        {out} = {buf};
        Py_INCREF({out});
        """

    def c_code_cache_version(self):
        return (1,)


class JoinBufferElimination(GraphRewriter):
    """Replace Join with pre-allocated buffer + inplace writes.

    For each Join(axis=0, stream_0, stream_1, ...), allocate the output
    buffer upfront and have each Elemwise stream write directly into its
    slice via WriteSplit. Non-Elemwise streams fall back to set_subtensor.

    Requires >= 2 expandable (Elemwise) streams; with fewer streams the
    overhead of WriteSplit + WriteJoin exceeds the savings from eliminating
    the Join allocation and copy.
    """

    def apply(self, fgraph):
        if not hasattr(fgraph, "shape_feature"):
            return
        for node in list(fgraph.toposort()):
            if isinstance(node.op, Join):
                self._try_rewrite(fgraph, node)

    def _try_rewrite(self, fgraph, join_node):
        axis_var = join_node.inputs[0]
        if not hasattr(axis_var, "data") or int(axis_var.data) != 0:
            return

        streams = join_node.inputs[1:]
        if len(streams) < 2:
            return

        plans = []
        claimed_outputs = set()  # Track Elemwise outputs already claimed
        for s in streams:
            # Walk up through view ops (DimShuffle, Reshape) to find
            # the Elemwise producer. These view ops get absorbed by
            # inverse-transforming the buffer slice instead.
            chain = []
            current = s
            while current.owner and isinstance(current.owner.op, (DimShuffle, Reshape)):
                chain.append(current.owner)
                current = current.owner.inputs[0]

            if current.owner and isinstance(current.owner.op, Elemwise):
                elemwise_node = current.owner
                output_idx = elemwise_node.outputs.index(current)

                # Can't expand the same output twice — the second would
                # try to destroy a buffer slice that's already been claimed.
                if current in claimed_outputs:
                    plans.append(("fallback", None, None, []))
                    continue

                # Check this Elemwise output has no other clients besides
                # the view chain leading to this Join stream. If it does,
                # we can't redirect its output without duplicating computation.
                has_other_clients = False
                for v in [current, *[n.outputs[0] for n in chain]]:
                    clients = fgraph.clients.get(v, [])
                    n_real_clients = sum(1 for c, _ in clients if c != "output")
                    if n_real_clients > 1:
                        has_other_clients = True
                        break

                if not has_other_clients:
                    claimed_outputs.add(current)
                    plans.append(("expand", elemwise_node, output_idx, chain))
                else:
                    plans.append(("fallback", None, None, []))
            else:
                plans.append(("fallback", None, None, []))

        # Need at least 2 expandable streams; with fewer, the overhead of
        # AllocEmpty + WriteSplit + WriteJoin exceeds savings from removing
        # intermediate allocations and the Join copy.
        if sum(1 for k, *_ in plans if k == "expand") < 2:
            return

        join_out = join_node.outputs[0]
        dtype = join_out.type.dtype
        shape_feature = fgraph.shape_feature
        sizes = [shape_feature.get_shape(s, 0) for s in streams]
        buf_shape = [
            shape_feature.get_shape(join_out, i) for i in range(join_out.type.ndim)
        ]
        buf = pt.empty(buf_shape, dtype=dtype)
        split_outputs = WriteSplit(len(streams))(buf, *sizes)

        deps = []
        for i, ((kind, elemwise_node, output_idx, view_chain), s_i) in enumerate(
            zip(plans, split_outputs)
        ):
            if kind == "expand":
                # Inverse-transform buffer slice to match Elemwise output shape.
                # e.g. if stream was Elemwise(...).ravel(), reshape the buffer
                # slice to the pre-ravel shape so the Elemwise can write into it.
                target = s_i
                for view_node in view_chain:
                    target = target.reshape(view_node.inputs[0].type.shape)

                # Wrap the scalar_op in a Composite with an extra dummy input
                # for the output buffer. The Elemwise writes into this buffer
                # via inplace_pattern on the dummy input.
                scalar_op = elemwise_node.op.scalar_op
                scalar_inputs = [
                    ScalarType(dtype=inp.type.dtype)() for inp in elemwise_node.inputs
                ]
                dummy = ScalarType(dtype=target.type.dtype)()
                scalar_outputs = scalar_op.make_node(*scalar_inputs).outputs
                composite = Composite([*scalar_inputs, dummy], scalar_outputs)
                new_elemwise = Elemwise(
                    composite,
                    inplace_pattern={output_idx: len(scalar_inputs)},
                )
                deps.append(new_elemwise(*elemwise_node.inputs, target))
            else:
                # Fallback: copy stream output into buffer slice via set_subtensor.
                # The local_inplace_setsubtensor rewrite at position 50.1 will
                # convert this to inplace.
                deps.append(pt.set_subtensor(s_i[:], streams[i]))

        wb = WriteJoin()(buf, *deps)
        try:
            fgraph.replace_all_validate(
                [(join_out, wb)], reason="JoinBufferElimination"
            )
        except Exception:
            return


# Register for C and Numba backends but not pure Python.
# After add_destroy_handler (49.5), before per-op inplace rewrites (50.1+).
# This rewrite coordinates multiple inplace ops that the per-op rewrites
# can't handle individually (multiple destroyers of the same buffer).
# Separate instances needed because the DB excludes by rewriter identity
# when any of its registrations carry an excluded tag.
for _name, _tags in [
    ("join_buffer_elimination", ("fast_run", "inplace", "cxx_only")),
    ("join_buffer_elimination_numba", ("numba", "inplace")),
]:
    optdb.register(_name, JoinBufferElimination(), *_tags, position=50.0)
