"""FusedElemwise: one loop for an Elemwise with indexed reads, indexed writes and
reductions (see pytensor.tensor.rewriting.fused_elemwise).

Outer inputs are `[elemwise inputs..., index arrays..., write targets...]`. Each Elemwise
input is plain (broadcast by its type) or read through a 1-d index array along one axis.
Each output is plain, a reduction over some loop axes, or written through an index into
a copy of its target (inc or set). Patterns outside this raise NotImplementedError and
the planner runs the inner graph node by node instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from pytensor.link.llvm.codegen.elemwise import identity_value
from pytensor.link.llvm.codegen.emit import (
    Array,
    ProvidedOutput,
    ScalarOutput,
    advance,
    cast,
    checked_index,
    constant,
    copy,
    for_each,
    index_bounds,
    is_contiguous,
    load,
    loop_order,
    loop_view,
    memory_type,
    size,
    store,
)
from pytensor.link.llvm.codegen.function import llvm_emit
from pytensor.link.llvm.codegen.scalar import emit_scalar
from pytensor.link.llvm.codegen.types import check_dtype
from pytensor.link.llvm.layout import Layout
from pytensor.scalar.basic import ScalarType, add
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.subtensor import AdvancedSubtensor


#: elements per chunk of a loop with register-carried increments (see `chunked`)
CHUNK = 256


@dataclass
class Read:
    index: int  # outer input position of the index array
    axis: int


@dataclass
class Write:
    index: int
    axis: int
    target: int  # outer input position of the target
    set: bool


@dataclass
class Reduce:
    scalar_op: object
    axes: tuple
    identity: object
    acc: str


def fused_spec(op, node):
    inner = op.fgraph
    elems = [n for n in inner.apply_nodes if isinstance(n.op, Elemwise)]
    if len(elems) != 1:
        raise NotImplementedError("FusedElemwise: expected one inner Elemwise")
    (E,) = elems
    n_idx = len(op.indexed_inputs)
    if len(op.indexed_outputs) not in (0, n_idx):
        raise NotImplementedError("FusedElemwise: inconsistent index specs")
    written = sorted(int(s) for e in op.indexed_outputs if e is not None for s in e[0])
    n_elem = len(node.inputs) - n_idx - len(written)
    if len(E.inputs) != n_elem or len(E.outputs) != len(node.outputs):
        raise NotImplementedError("FusedElemwise: unexpected inner structure")

    def check_index(p):
        t = node.inputs[p].type
        if t.ndim != 1 or not t.dtype.startswith(("int", "uint")):
            raise NotImplementedError(
                "FusedElemwise: only 1-d integer index arrays are fused"
            )

    reads = [None] * n_elem
    for k, entry in enumerate(op.indexed_inputs):
        if entry is None:
            continue
        sources, axis = entry
        check_index(n_elem + k)
        for s in sources:
            if reads[s] is not None:
                raise NotImplementedError(
                    "FusedElemwise: a source read through several indices"
                )
            reads[s] = Read(n_elem + k, int(axis))
    for j, e_in in enumerate(E.inputs):
        src = inner.inputs[j]
        if reads[j] is not None:
            if not (
                e_in.owner
                and isinstance(e_in.owner.op, AdvancedSubtensor)
                and e_in.owner.inputs[0] is src
            ):
                raise NotImplementedError("FusedElemwise: unexpected indexed read")
        elif e_in is not src and not (
            isinstance(node.inputs[j].type, ScalarType) and all(e_in.type.broadcastable)
        ):
            raise NotImplementedError("FusedElemwise: transformed Elemwise input")
    outs = [None] * len(node.outputs)
    for i, spec in enumerate(op.reduced_outputs):
        if spec is None:
            continue
        scalar_op, axes, identity, acc = spec
        if node.outputs[i].type.dtype != E.outputs[i].type.dtype:
            raise NotImplementedError("FusedElemwise: reduction changes the dtype")
        outs[i] = Reduce(
            scalar_op, tuple(int(a) for a in axes), identity, check_dtype(acc)
        )
    for k, entry in enumerate(op.indexed_outputs):
        if entry is None:
            continue
        sources, axis, mode = entry
        check_index(n_elem + k)
        for s in sources:
            if outs[s] is not None:
                raise NotImplementedError(
                    "FusedElemwise: an output written through several indices"
                )
            outs[s] = Write(
                n_elem + k,
                int(axis),
                n_elem + n_idx + written.index(int(s)),
                mode == "set",
            )
    for i, o in enumerate(node.outputs):
        if o.type.dtype != E.outputs[i].type.dtype:
            raise NotImplementedError(
                "FusedElemwise: output dtype differs from the loop's"
            )
    return E, n_elem, reads, outs


def _register():
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise

    @llvm_emit.register(FusedElemwise)
    def _(op, node, cg, inputs, outputs):
        emit_fused(op, node, cg, inputs, outputs)


def emit_fused(op, node, cg, inputs, outputs):
    b = cg.b
    E, n_elem, reads, outs = fused_spec(op, node)
    nd = E.outputs[0].type.ndim
    in_dtypes = [check_dtype(v.type.dtype) for v in E.inputs]
    out_dtypes = [check_dtype(v.type.dtype) for v in E.outputs]

    # ---- the loop shape, from everything that has lengths
    lengths = [[] for _ in range(nd)]  # candidates per axis: (value, what)
    static = list(E.outputs[0].type.shape)
    for j in range(n_elem):
        x = inputs[j]
        if not isinstance(x, Array):
            continue  # a scalar
        r = reads[j]
        etype = E.inputs[j].type
        for a in range(nd):
            if r is not None and a == r.axis:
                lengths[a].append(inputs[r.index].shape[0])
            elif etype.shape[a] != 1:
                lengths[a].append(x.shape[a])
    for i, spec in enumerate(outs):
        if isinstance(spec, Write):
            t = inputs[spec.target]
            for a in range(nd):
                lengths[a].append(
                    inputs[spec.index].shape[0] if a == spec.axis else t.shape[a]
                )
    shape = []
    for a in range(nd):
        cand = lengths[a]
        n = (
            b.i64(static[a])
            if static[a] is not None
            else (cand[0] if cand else b.i64(1))
        )
        for c in cand:
            if not (c.const_int is not None and c.const_int == n.const_int):
                cg.check(b.eq(c, n), f"FusedElemwise: shape mismatch on axis {a}")
        shape.append(n)

    # ---- outputs (ScalarType outputs are computed into a 0-d temporary)
    scalar_outs = {}
    outputs = list(outputs)
    for i, out in enumerate(outputs):
        if isinstance(out, ScalarOutput):
            tmp = cg.temporary(out.dtype, [])
            scalar_outs[i] = (out, tmp)
            outputs[i] = ProvidedOutput(tmp)
    out_arrays = []
    for i, (spec, out) in enumerate(zip(outs, outputs, strict=True)):
        if isinstance(spec, Reduce):
            oshape = [shape[a] for a in range(nd) if a not in spec.axes]
            res = out.allocate(oshape)
            acc = res if spec.acc == res.dtype else cg.temporary(spec.acc, oshape)
            init = constant(
                b, spec.acc, identity_value(spec.scalar_op, spec.identity, spec.acc)
            )
            for_each(
                b,
                oshape,
                [acc],
                lambda p, init=init, spec=spec: store(b, spec.acc, init, p[0]),
            )
            out_arrays.append((res, acc))
        elif isinstance(spec, Write):
            t = inputs[spec.target]
            res = out.allocate(t.shape)
            copy(cg, t, res)
            out_arrays.append((res, None))
        else:
            out_arrays.append((out.allocate(shape), None))

    # ---- operands with loop strides (and their layouts)
    plain_in = {}
    fixed = {}  # inputs that are one element for the whole loop: read once, before it
    for j in range(n_elem):
        x = inputs[j]
        if not isinstance(x, Array):
            continue
        if reads[j] is None and all(s == 1 for s in E.inputs[j].type.shape):
            fixed[j] = load(b, x.dtype, x.data)
            continue
        skip = reads[j].axis if reads[j] is not None else None
        # (an indexed axis is stepped through by the index, not by the loop)
        plain_in[j] = loop_view(b, x, shape, zero_axes=() if skip is None else (skip,))
    plain_out = {}
    for i, spec in enumerate(outs):
        res, acc = out_arrays[i]
        if isinstance(spec, Reduce):
            plain_out[i] = reduced_view(b, acc, shape, spec.axes)
        elif isinstance(spec, Write):
            plain_out[i] = loop_view(b, res, shape, zero_axes=(spec.axis,))
        else:
            plain_out[i] = Array(res.dtype, res.data, shape, res.strides)

    def index_value(k, i):
        """idx[i] for index array input k, wrapped and bounds-checked against `n`."""
        return inputs[k].load(b, i).astype("int64").value

    def checked(v, n, what, k):
        return checked_index(
            cg,
            v,
            n,
            f"FusedElemwise: index out of bounds ({what})",
            index_bounds(node.inputs[k]),
        )

    def body(ptr_in, ptr_out, loop_index, accs, runs=None, defer=None):
        """One element. `ptr_in[j]`/`ptr_out[i]`: element pointers of the plain parts;
        `accs`: register accumulators {output: value}; `runs`: {output: (index, value)}
        of the incremented element of an output kept in registers (see `inc_run`);
        `defer`: {output: pointer} where the value of an indexed write is stored
        instead of written (see `chunked`). Returns the new accumulators and runs."""
        runs = runs or {}
        defer = defer or {}
        new_runs = {}
        args = []
        for j, d in enumerate(in_dtypes):
            x = inputs[j]
            if not isinstance(x, Array):
                args.append((x, d))
                continue
            if j in fixed:
                args.append((cast(b, fixed[j], x.dtype, d), d))
                continue
            p = ptr_in[j]
            r = reads[j]
            if r is not None:
                n = x.shape[r.axis]
                ix = checked(
                    index_value(r.index, loop_index[r.axis]), n, "read", r.index
                )
                p = advance(b, p, ix, x.strides[r.axis], x.dtype)
            args.append((load(b, x.dtype, p), d))
        res = emit_scalar(E.op.scalar_op, b, args, out_dtypes)
        new_accs = {}
        for i, (spec, v) in enumerate(zip(outs, res, strict=True)):
            d = out_dtypes[i]
            if isinstance(spec, Reduce):
                cur = accs[i] if i in accs else load(b, spec.acc, ptr_out[i])
                new = emit_scalar(
                    spec.scalar_op,
                    b,
                    [(cur, spec.acc), (cast(b, v, d, spec.acc), spec.acc)],
                    [spec.acc],
                )[0]
                if i in accs:
                    new_accs[i] = new
                else:
                    store(b, spec.acc, new, ptr_out[i])
            elif isinstance(spec, Write):
                if i in defer:
                    store(b, d, v, defer[i])
                    continue
                res_arr = out_arrays[i][0]
                n = res_arr.shape[spec.axis]
                ix = checked(
                    index_value(spec.index, loop_index[spec.axis]),
                    n,
                    "write",
                    spec.index,
                )
                if i in runs:
                    new_runs[i] = inc_run(i, d, ptr_out[i], ix, v, *runs[i])
                    continue
                p = element(i, ptr_out[i], ix)
                if not spec.set:
                    v = emit_scalar(add, b, [(load(b, d, p), d), (v, d)], [d])[0]
                store(b, d, v, p)
            else:
                store(b, d, v, ptr_out[i])
        return new_accs, new_runs

    def element(i, base, ix):
        res_arr = out_arrays[i][0]
        return advance(b, base, ix, res_arr.strides[outs[i].axis], res_arr.dtype)

    def inc_run(i, d, base, ix, v, prev, cur):
        """`out[ix] += v` with the element being incremented kept in a register while
        the index repeats (`prev`, `cur`: its index, -1 before the first, and value):
        written back when the index changes, so consecutive increments of one element
        don't wait on each other through memory. The order of the additions is kept."""

        def same():
            return [prev, emit_scalar(add, b, [(cur, d), (v, d)], [d])[0]]

        def other():
            with b.if_(b.sge(prev, 0)):
                store(b, d, cur, element(i, base, prev))
            old = load(b, d, element(i, base, ix))
            return [ix, emit_scalar(add, b, [(old, d), (v, d)], [d])[0]]

        return tuple(b.if_else(b.eq(ix, prev), same, other))

    def flush_run(i, d, base, prev, cur):
        with b.if_(b.sge(prev, 0)):
            store(b, d, cur, element(i, base, prev))

    in_keys = sorted(plain_in)
    out_keys = sorted(plain_out)
    arrays = [plain_in[j] for j in in_keys] + [plain_out[i] for i in out_keys]
    any_indexed = any(r is not None for r in reads) or any(
        isinstance(s, Write) for s in outs
    )

    def split(ptrs):
        return dict(zip(in_keys, ptrs[: len(in_keys)], strict=True)), dict(
            zip(out_keys, ptrs[len(in_keys) :], strict=True)
        )

    # reductions that reduce the innermost axis are carried in registers there
    inner_red = [
        i
        for i, s in enumerate(outs)
        if isinstance(s, Reduce) and nd > 0 and (nd - 1) in s.axes
    ]
    # increments through an index along the innermost axis keep their run in registers
    inner_inc = [
        i
        for i, s in enumerate(outs)
        if isinstance(s, Write) and not s.set and nd > 0 and s.axis == nd - 1
    ]

    def chunked(n, ptrs, index, init, nr):
        """The innermost loop of `nest` with register-carried increments, in chunks of
        CHUNK elements: the values of a chunk go to stack buffers first, in a loop
        without the runs (so it can be vectorized), and are then added to their
        elements in a second loop. Same results as one loop, in the same order."""
        a = nd - 1
        bufs = {i: b.alloca(memory_type(out_dtypes[i]), CHUNK) for i in inner_inc}
        _, po = split(ptrs)

        def chunk(c, *carried):
            start = b.mul(c, CHUNK, nsw=True)
            m = b.smin(b.i64(CHUNK), b.sub(n, start))

            def compute(k, *accs):
                i = b.add(start, k, nsw=True)
                q = [
                    advance(b, p, i, arr.strides[a], arr.dtype)
                    for p, arr in zip(ptrs, arrays, strict=True)
                ]
                qi, qo = split(q)
                defer = {
                    j: b.gep(memory_type(out_dtypes[j]), bufs[j], k) for j in inner_inc
                }
                new, _ = body(
                    qi,
                    qo,
                    [*index, i],
                    dict(zip(inner_red, accs, strict=True)),
                    defer=defer,
                )
                return [new[j] for j in inner_red]

            accs = b.fold(m, list(carried[:nr]), compute)

            def increment(k, *runs):
                i = b.add(start, k, nsw=True)
                new = []
                for r, j in enumerate(inner_inc):
                    spec, d = outs[j], out_dtypes[j]
                    ix = checked(
                        index_value(spec.index, i),
                        out_arrays[j][0].shape[spec.axis],
                        "write",
                        spec.index,
                    )
                    v = load(b, d, b.gep(memory_type(d), bufs[j], k))
                    new += inc_run(j, d, po[j], ix, v, *runs[2 * r : 2 * r + 2])
                return new

            return [*accs, *b.fold(m, list(carried[nr:]), increment)]

        n_chunks = b.udiv(b.add(n, CHUNK - 1, nsw=True), CHUNK)
        return b.fold(n_chunks, init, chunk)

    def nest():
        def level(a, ptrs, index):
            if a == nd - 1 and nd > 0:
                pi, po = split(ptrs)
                init = [load(b, outs[i].acc, po[i]) for i in inner_red]
                nr = len(inner_red)
                for i in inner_inc:
                    d = out_dtypes[i]
                    init += [b.i64(-1), constant(b, d, 0)]

                def step(i, *carried):
                    q = [
                        advance(b, p, i, arr.strides[a], arr.dtype)
                        for p, arr in zip(ptrs, arrays, strict=True)
                    ]
                    qi, qo = split(q)
                    runs = {
                        k: carried[nr + 2 * n : nr + 2 * n + 2]
                        for n, k in enumerate(inner_inc)
                    }
                    new, new_runs = body(
                        qi,
                        qo,
                        [*index, i],
                        dict(zip(inner_red, carried[:nr], strict=True)),
                        runs,
                    )
                    return [new[k] for k in inner_red] + [
                        v for k in inner_inc for v in new_runs[k]
                    ]

                final = (
                    chunked(shape[a], ptrs, index, init, nr)
                    if inner_inc
                    else b.fold(shape[a], init, step)
                )
                for i, v in zip(inner_red, final[:nr], strict=True):
                    store(b, outs[i].acc, v, po[i])
                for n, i in enumerate(inner_inc):
                    flush_run(
                        i, out_dtypes[i], po[i], *final[nr + 2 * n : nr + 2 * n + 2]
                    )
                return
            if a == nd:
                pi, po = split(ptrs)
                body(pi, po, index, {})
                return
            with b.loop(shape[a]) as i:
                q = [
                    advance(b, p, i, arr.strides[a], arr.dtype)
                    for p, arr in zip(ptrs, arrays, strict=True)
                ]
                level(a + 1, q, [*index, i])

        level(0, [arr.data for arr in arrays], [])
        return []

    # one flat loop when nothing is indexed, every reduction reduces everything, and the
    # plain operands are contiguous (checked at run time unless known)
    flat_ok = (
        not any_indexed
        and nd > 0
        and all(len(s.axes) == nd for s in outs if isinstance(s, Reduce))
        and all(
            E.inputs[j].type.shape[a] != 1 or static[a] == 1
            for j in plain_in
            for a in range(nd)
        )
    )
    if not flat_ok:
        nest()
        _finish(cg, outs, out_arrays, scalar_outs)
        return

    # every operand packed in the same axis order (from the layouts, so only operands
    # whose layout leaves it open are checked at run time)
    flat_arrays = [plain_in[j] for j in in_keys] + [
        plain_out[i] for i in out_keys if not isinstance(outs[i], Reduce)
    ]
    order = loop_order(shape, flat_arrays)
    contig = b.const_bool(True)
    for arr in flat_arrays:
        contig = b.and_(contig, is_contiguous(b, shape, arr, order))

    def flat():
        red = [i for i in out_keys if isinstance(outs[i], Reduce)]
        init = [load(b, outs[i].acc, plain_out[i].data) for i in red]

        def step(i, *accs):
            pi = {
                j: b.gep(memory_type(plain_in[j].dtype), plain_in[j].data, i)
                for j in in_keys
            }
            po = {
                k: (
                    plain_out[k].data
                    if isinstance(outs[k], Reduce)
                    else b.gep(memory_type(plain_out[k].dtype), plain_out[k].data, i)
                )
                for k in out_keys
            }
            new, _ = body(pi, po, [None] * nd, dict(zip(red, accs, strict=True)))
            return [new[k] for k in red]

        final = b.fold(size(b, shape), init, step)
        for i, v in zip(red, final, strict=True):
            store(b, outs[i].acc, v, plain_out[i].data)
        return []

    c = contig.const_int
    if c == 1:
        flat()
    elif c == 0:
        nest()
    else:
        b.if_else(contig, flat, nest)
    _finish(cg, outs, out_arrays, scalar_outs)


def reduced_view(b, acc: Array, shape, axes) -> Array:
    """The accumulator of a reduction over `axes` iterated over the full loop `shape`:
    stride 0 on the reduced axes, the accumulator's own strides (and layout) elsewhere."""
    nd = len(shape)
    kept = [a for a in range(nd) if a not in axes]
    strides = [b.i64(0)] * nd
    kinds = [0] * nd
    for k, a in enumerate(kept):
        strides[a] = acc.strides[k]
        kinds[a] = acc.layout.strides[k]
    order = [a for a in range(nd) if a in axes] + [kept[k] for k in acc.layout.order]
    return Array(
        acc.dtype,
        acc.data,
        list(shape),
        strides,
        Layout.make([n.const_int for n in shape], order, kinds),
    )


def _finish(cg, outs, out_arrays, scalar_outs):
    """Convert reductions accumulated in another dtype; set scalar outputs."""
    for i, spec in enumerate(outs):
        if isinstance(spec, Reduce):
            res, acc = out_arrays[i]
            if acc is not res:
                copy(cg, acc, res)
    for out, tmp in scalar_outs.values():
        out.set(cg.b, load(cg.b, tmp.dtype, tmp.data))


_register()

__all__ = ["emit_fused", "fused_spec"]
