// Copyright 2026 Benjamin F. Maier / PyMC Labs. Apache-2.0.
// Adapted from pymc-labs/tymc b1c038979eb006b2d9ad1842a5b9ec2640688944:
// src/graph/{wasm_linalg,wasm_trsm,wasm_cholesky}.ts.
// Changes: isolate assemblers in closures; omit TyMC graph/record adapters.
// PyTensor's record adapters live in wasm_runtime.js. See special.LICENSE.
const ptWasmKernels = (() => {
/**
 * WebAssembly f64x2 kernels for `matmul` and `matvec`, reading and writing the compiled function's
 * buffers in place (D-196, after D-194).
 *
 * The compiled function's buffers and input records are views into one `WebAssembly.Memory`, the
 * arena, whenever the graph holds a product this file can run, so a kernel addresses its operands
 * with their offsets and strides as literals, the way the generated JavaScript kernels do, and
 * nothing is copied per call. A constant operand is copied into the arena once, when the step is
 * built, in whichever layout the kernel wants.
 *
 * The two lanes run across outputs, never along the summed index: each output starts at 0 and adds
 * its products in inner-index order, one multiply and one add per product, which is exactly what
 * `matmul_into`, `matvec_into` and their generated kernels do. The results are bit-identical to the
 * JavaScript kernels, and `compile({ wasm: false })` changes speed only.
 *
 * A tile of `rows × 2·vectors` outputs keeps one f64x2 accumulator per row and vector; per inner
 * index it loads `vectors` pairs of the right operand and one element of the left per row (BLAS
 * dgemm's register blocking, two wide). The lanes need the right operand's pairs adjacent in
 * memory. Where they are not, the transposed product `outᵀ = bᵀ aᵀ` is tried, which needs the left
 * operand's columns contiguous instead.
 *
 * The module is assembled here, a few hundred bytes per product, with no toolchain and no
 * dependency (the plan's "no WASM" non-goal is amended by D-194). Where WebAssembly SIMD is
 * unavailable, or a content-security policy forbids compiling it, `make_arena` returns null and the
 * JavaScript kernels run.
 *
 * Kernel-style file (loops, `let`).
 */
const leb = (n) => {
    const out = [];
    let rest = n;
    do {
        let b = rest & 0x7f;
        rest >>>= 7;
        if (rest !== 0)
            b |= 0x80;
        out.push(b);
    } while (rest !== 0);
    return out;
};
const sleb = (n) => {
    const out = [];
    let rest = n;
    for (;;) {
        const b = rest & 0x7f;
        rest >>= 7;
        if ((rest === 0 && (b & 0x40) === 0) || (rest === -1 && (b & 0x40) !== 0)) {
            out.push(b);
            return out;
        }
        out.push(b | 0x80);
    }
};
const section = (id, body) => [id, ...leb(body.length), ...body];
const vec = (items) => [...leb(items.length), ...items.flat()];
const text = (s) => [...leb(s.length), ...new TextEncoder().encode(s)];
/** The instructions the kernels are assembled from, as their byte encodings. */
const OP = {
    get: (i) => [0x20, ...leb(i)],
    set: (i) => [0x21, ...leb(i)],
    tee: (i) => [0x22, ...leb(i)],
    i32: (v) => [0x41, ...sleb(v)],
    i32add: [0x6a],
    i32sub: [0x6b],
    i32mul: [0x6c],
    i32ne: [0x47],
    i32eqz: [0x45],
    f64load: (offset) => [0x2b, 3, ...leb(offset)],
    f64store: (offset) => [0x39, 3, ...leb(offset)],
    f64zero: [0x44, 0, 0, 0, 0, 0, 0, 0, 0],
    f64add: [0xa0],
    f64sub: [0xa1],
    f64mul: [0xa2],
    f64div: [0xa3],
    block: [0x02, 0x40],
    loop: [0x03, 0x40],
    brif: (d) => [0x0d, ...leb(d)],
    end: [0x0b],
    drop: 0x1a,
    v128load: (offset) => [0xfd, 0, 3, ...leb(offset)],
    v128store: (offset) => [0xfd, 11, 3, ...leb(offset)],
    f64x2splat: [0xfd, 20],
    f64x2extract: (lane) => [0xfd, 33, lane],
    f64x2add: [0xfd, ...leb(0xf0)],
    f64x2sub: [0xfd, ...leb(0xf1)],
    f64x2mul: [0xfd, ...leb(0xf2)],
    f64x2div: [0xfd, ...leb(0xf3)],
};
// i32 locals: 0 row block's left byte base, 1 its output byte base, 2 column block's right byte
// base, 3 its output byte offset, 4 row blocks left, 5 column blocks left, 6 left pointer, 7 right
// pointer, 8 inner count; then the v128 accumulators, right vectors and the splatted left element;
// then the f64 left element, the odd column's accumulators and its right element
const IA = 0;
const IO = 1;
const JB = 2;
const JO = 3;
const IC = 4;
const JC = 5;
const PA = 6;
const PB = 7;
const CNT = 8;
const I32_LOCALS = 9;
/** Local indices for a tiling. */
function locals_of(t) {
    const acc = I32_LOCALS;
    const bv = acc + t.rows * t.vectors;
    const as = bv + t.vectors;
    const ar = as + 1;
    const sacc = ar + 1;
    const bs = sacc + t.rows;
    return { acc, bv, as, ar, sacc, bs, v128_count: as + 1 - acc, f64_count: bs + 1 - ar };
}
/** One inner index of a tile: the right operand's pairs, then each row's left element times them into the accumulators. */
function tile_inner(layout, t, shape) {
    const { rows, vectors, odd } = shape;
    const L = locals_of(t);
    const code = [];
    for (let v = 0; v < vectors; v++)
        code.push(...OP.get(PB), ...OP.v128load(16 * v), ...OP.set(L.bv + v));
    if (odd)
        code.push(...OP.get(PB), ...OP.f64load(16 * vectors), ...OP.set(L.bs));
    for (let r = 0; r < rows; r++) {
        code.push(...OP.get(PA), ...OP.f64load(8 * r * layout.a.row));
        if (odd)
            code.push(...OP.tee(L.ar));
        code.push(...(vectors > 0 ? [...OP.f64x2splat, ...OP.set(L.as)] : [OP.drop]));
        for (let v = 0; v < vectors; v++) {
            const acc = L.acc + r * t.vectors + v;
            code.push(...OP.get(acc), ...OP.get(L.as), ...OP.get(L.bv + v), ...OP.f64x2mul, ...OP.f64x2add, ...OP.set(acc));
        }
        if (odd)
            code.push(...OP.get(L.sacc + r), ...OP.get(L.ar), ...OP.get(L.bs), ...OP.f64mul, ...OP.f64add, ...OP.set(L.sacc + r));
    }
    return code;
}
/** A tile's outputs written from its accumulators, a pair at a time where the output's columns are contiguous. */
function tile_store(layout, t, shape) {
    const { out } = layout;
    const L = locals_of(t);
    const code = [...OP.get(IO), ...OP.get(JO), ...OP.i32add, ...OP.set(PA)];
    for (let r = 0; r < shape.rows; r++) {
        for (let v = 0; v < shape.vectors; v++) {
            const at = 8 * (r * out.row + 2 * v * out.col);
            const acc = L.acc + r * t.vectors + v;
            if (out.col === 1)
                code.push(...OP.get(PA), ...OP.get(acc), ...OP.v128store(at));
            else {
                code.push(...OP.get(PA), ...OP.get(acc), ...OP.f64x2extract(0), ...OP.f64store(at));
                code.push(...OP.get(PA), ...OP.get(acc), ...OP.f64x2extract(1), ...OP.f64store(at + 8 * out.col));
            }
        }
        if (shape.odd)
            code.push(...OP.get(PA), ...OP.get(L.sacc + r), ...OP.f64store(8 * (r * out.row + 2 * shape.vectors * out.col)));
    }
    return code;
}
/**
 * One tile at the row block in `IA`/`IO` and the column block in `JB`/`JO`: zero its accumulators,
 * run the inner index, store. `b_step` is the right operand's stride per inner index.
 */
function tile(layout, t, shape, b_step) {
    const L = locals_of(t);
    const code = [];
    for (let r = 0; r < shape.rows; r++) {
        for (let v = 0; v < shape.vectors; v++)
            code.push(...OP.f64zero, ...OP.f64x2splat, ...OP.set(L.acc + r * t.vectors + v));
        if (shape.odd)
            code.push(...OP.f64zero, ...OP.set(L.sacc + r));
    }
    code.push(...OP.get(IA), ...OP.set(PA), ...OP.get(JB), ...OP.set(PB), ...OP.i32(layout.k), ...OP.set(CNT), ...OP.loop);
    code.push(...tile_inner(layout, t, shape));
    code.push(...OP.get(PA), ...OP.i32(8 * layout.a.col), ...OP.i32add, ...OP.set(PA));
    code.push(...OP.get(PB), ...OP.i32(8 * b_step), ...OP.i32add, ...OP.set(PB));
    code.push(...OP.get(CNT), ...OP.i32(1), ...OP.i32sub, ...OP.tee(CNT), ...OP.brif(0), ...OP.end);
    return [...code, ...tile_store(layout, t, shape)];
}
/** Every column of the row block in `IA`/`IO`: full blocks of `2·vectors` columns in a loop, then the remainder. */
function column_pass(layout, t, rows) {
    const { n, k, b, out } = layout;
    const width = 2 * t.vectors;
    const blocks = Math.floor(n / width);
    const rest = n - blocks * width;
    const panel = layout.packed === true ? k * width : width;
    const code = [];
    if (blocks > 0) {
        code.push(...OP.i32(8 * b.at), ...OP.set(JB), ...OP.i32(0), ...OP.set(JO), ...OP.i32(blocks), ...OP.set(JC), ...OP.loop);
        code.push(...tile(layout, t, { rows, vectors: t.vectors, odd: false }, layout.packed === true ? width : b.row));
        code.push(...OP.get(JB), ...OP.i32(8 * panel), ...OP.i32add, ...OP.set(JB));
        code.push(...OP.get(JO), ...OP.i32(8 * width * out.col), ...OP.i32add, ...OP.set(JO));
        code.push(...OP.get(JC), ...OP.i32(1), ...OP.i32sub, ...OP.tee(JC), ...OP.brif(0), ...OP.end);
    }
    if (rest > 0) {
        code.push(...OP.i32(8 * (b.at + blocks * panel)), ...OP.set(JB), ...OP.i32(8 * blocks * width * out.col), ...OP.set(JO));
        code.push(...tile(layout, t, { rows, vectors: Math.floor(rest / 2), odd: rest % 2 === 1 }, layout.packed === true ? rest : b.row));
    }
    return code;
}
/** The kernel's body: full row blocks in a loop, then the remaining rows. */
function kernel_body(layout, t) {
    const { m, a, out } = layout;
    const blocks = Math.floor(m / t.rows);
    const code = [];
    if (blocks > 0) {
        code.push(...OP.i32(8 * a.at), ...OP.set(IA), ...OP.i32(8 * out.at), ...OP.set(IO), ...OP.i32(blocks), ...OP.set(IC), ...OP.loop);
        code.push(...column_pass(layout, t, t.rows));
        code.push(...OP.get(IA), ...OP.i32(8 * t.rows * a.row), ...OP.i32add, ...OP.set(IA));
        code.push(...OP.get(IO), ...OP.i32(8 * t.rows * out.row), ...OP.i32add, ...OP.set(IO));
        code.push(...OP.get(IC), ...OP.i32(1), ...OP.i32sub, ...OP.tee(IC), ...OP.brif(0), ...OP.end);
    }
    const rest = m - blocks * t.rows;
    if (rest > 0) {
        code.push(...OP.i32(8 * (a.at + blocks * t.rows * a.row)), ...OP.set(IA), ...OP.i32(8 * (out.at + blocks * t.rows * out.row)), ...OP.set(IO));
        code.push(...column_pass(layout, t, rest));
    }
    return code;
}
/** The module's bytes: one exported function `run` over the imported memory `m.mem` of at least `pages` pages. */
function matmul_module(layout, t, pages) {
    const L = locals_of(t);
    const locals = [
        [...leb(I32_LOCALS), 0x7f],
        [...leb(L.v128_count), 0x7b],
        [...leb(L.f64_count), 0x7c],
    ];
    return module_bytes(locals, kernel_body(layout, t), pages);
}
/** A module of one function `run` with these local declarations, body and result types, over the imported memory `m.mem` of `pages` pages. */
function module_bytes(locals, body, pages, results = []) {
    const code = [...vec(locals), ...body, ...OP.end];
    return new Uint8Array([
        0,
        0x61,
        0x73,
        0x6d,
        1,
        0,
        0,
        0,
        ...section(1, vec([[0x60, 0, ...leb(results.length), ...results]])),
        ...section(2, vec([[...text("m"), ...text("mem"), 2, 0, ...leb(pages)]])),
        ...section(3, vec([[0]])),
        ...section(7, vec([[...text("run"), 0, 0]])),
        ...section(10, vec([[...leb(code.length), ...code]])),
    ]);
}
/** The tiling for a product: four rows by two vectors, or one row by eight vectors for a single-row product. */
function tiling_for(layout) {
    if (layout.m === 1)
        return { rows: 1, vectors: Math.max(1, Math.min(8, Math.floor(layout.n / 2))) };
    return { rows: Math.min(4, layout.m), vectors: Math.max(1, Math.min(2, Math.floor(layout.n / 2))) };
}
/** Whether WebAssembly with SIMD can be compiled here (not under a CSP without `wasm-unsafe-eval`). */
function can_run_wasm_simd() {
    try {
        const layout = { m: 1, k: 1, n: 2, a: { at: 0, row: 0, col: 1 }, b: { at: 0, row: 2, col: 1 }, out: { at: 0, row: 0, col: 1 } };
        return typeof WebAssembly === "object" && WebAssembly.validate(matmul_module(layout, { rows: 1, vectors: 1 }, 1));
    }
    catch {
        return false;
    }
}
/**
 * An arena of `elements` float64 elements (each allocation rounded up to an even count), or null
 * where WebAssembly is unavailable.
 */
function make_arena(elements) {
    const pages = Math.max(1, Math.ceil((8 * elements) / 65536));
    let memory;
    try {
        memory = new WebAssembly.Memory({ initial: pages, maximum: pages });
    }
    catch {
        return null;
    }
    let next = 0;
    const alloc = (size) => {
        const view = new Float64Array(memory.buffer, 8 * next, size);
        next += size + (size % 2);
        if (next > elements)
            throw new Error(`You called ops.compile with a WebAssembly arena of ${elements} elements, but its buffers need ${next}. Pass wasm: false to ops.compile, and report the graph.`);
        return view;
    };
    return { memory, pages, alloc };
}
/** Elements an arena needs for these sizes, each rounded up to an even count as `alloc` does. */
function arena_elements(sizes) {
    let total = 0;
    for (const size of sizes)
        total += size + (size % 2);
    return total;
}
const trsm = (() => {
/**
 * WebAssembly f64x2 kernel for the triangular solves with a matrix right-hand side (D-211), on the
 * arena of `wasm_linalg.ts`.
 *
 * `out = T⁻¹ b` or `T⁻ᵀ b` for lower-triangular `T` (n × n) and `b` (n × k). The lanes run across
 * the right-hand side's columns, never along the substitution: each output column starts from its
 * right-hand side, subtracts the earlier solutions' products in the kernel's order (j ascending)
 * and divides by the diagonal last, which is exactly what `solve_lower_triangular_into` does, so
 * the results are bit-identical. A tile is one row of the solution by up to eight vectors (sixteen
 * columns), held in registers across the whole substitution of that row: the triangle entry is
 * loaded once per row of the earlier solutions and multiplied into every lane.
 *
 * The exact GPs are why (D-207): the Cholesky's gradient solves twice against an n × n right-hand
 * side, and at n = 300 the JavaScript kernel spent 28 ms of a 35 ms gradient there.
 *
 * Kernel-style file (loops, `let`).
 */
/** Vectors across a tile's columns, two columns each. */
const TRSM_MAX_VECTORS = 8;
// i32 locals: the column block's byte offset within a row, column blocks left, byte pointers to
// the right-hand side, the output, the diagonal entry, the earlier solution's row and the triangle
// entry, rows left, rows already solved, rows left in the inner loop
const CB = 0;
const CC = 1;
const PB = 2;
const PO = 3;
const PD = 4;
const PJ = 5;
const PT = 6;
const I = 7;
const J0 = 8;
const J = 9;
const I32_LOCALS = 10;
/** Local indices for a tile of `vectors`: the v128 accumulators, the splatted triangle entry and diagonal, then the odd column's f64s. */
function locals_of(vectors) {
    const acc = I32_LOCALS;
    const ts = acc + vectors;
    const ds = ts + 1;
    const sacc = ds + 1;
    const tr = sacc + 1;
    const dr = tr + 1;
    return { acc, ts, ds, sacc, tr, dr, v128_count: ds + 1 - acc, f64_count: dr + 1 - sacc };
}
/** The earlier solutions' pointers for row i: the next row and T[i + 1, i] (transposed), or row 0 and T[i, 0] = T[i, i] − i·col. */
function inner_pointers(layout) {
    const { t, out, transposed } = layout;
    if (transposed)
        return [
            ...OP.get(PO),
            ...OP.i32(8 * out.row),
            ...OP.i32add,
            ...OP.set(PJ),
            ...OP.get(PD),
            ...OP.i32(8 * t.row),
            ...OP.i32add,
            ...OP.set(PT),
        ];
    return [
        ...OP.i32(8 * out.at),
        ...OP.get(CB),
        ...OP.i32add,
        ...OP.set(PJ),
        ...OP.get(PD),
        ...OP.get(J0),
        ...OP.i32(8 * t.col),
        ...OP.i32mul,
        ...OP.i32sub,
        ...OP.set(PT),
    ];
}
/** One earlier solution's row: its triangle entry read once and multiplied into every lane, `acc -= T · x_j`. */
function inner_step(layout, shape) {
    const { t, out, transposed } = layout;
    const { vectors, odd } = shape;
    const L = locals_of(shape.total);
    const code = [...OP.get(PT), ...OP.f64load(0)];
    if (odd)
        code.push(...OP.tee(L.tr));
    code.push(...(vectors > 0 ? [...OP.f64x2splat, ...OP.set(L.ts)] : [OP.drop]));
    for (let v = 0; v < vectors; v++) {
        code.push(...OP.get(L.acc + v), ...OP.get(L.ts), ...OP.get(PJ), ...OP.v128load(16 * v), ...OP.f64x2mul, ...OP.f64x2sub, ...OP.set(L.acc + v));
    }
    if (odd)
        code.push(...OP.get(L.sacc), ...OP.get(L.tr), ...OP.get(PJ), ...OP.f64load(16 * vectors), ...OP.f64mul, ...OP.f64sub, ...OP.set(L.sacc));
    code.push(...OP.get(PJ), ...OP.i32(8 * out.row), ...OP.i32add, ...OP.set(PJ));
    code.push(...OP.get(PT), ...OP.i32(8 * (transposed ? t.row : t.col)), ...OP.i32add, ...OP.set(PT));
    return code;
}
/** The row's accumulators divided by the diagonal and stored. */
function row_store(shape) {
    const { vectors, odd } = shape;
    const L = locals_of(shape.total);
    const code = [...OP.get(PD), ...OP.f64load(0)];
    if (odd)
        code.push(...OP.tee(L.dr));
    code.push(...(vectors > 0 ? [...OP.f64x2splat, ...OP.set(L.ds)] : [OP.drop]));
    for (let v = 0; v < vectors; v++)
        code.push(...OP.get(PO), ...OP.get(L.acc + v), ...OP.get(L.ds), ...OP.f64x2div, ...OP.v128store(16 * v));
    if (odd)
        code.push(...OP.get(PO), ...OP.get(L.sacc), ...OP.get(L.dr), ...OP.f64div, ...OP.f64store(16 * vectors));
    return code;
}
/**
 * The rows of a tile's substitution for the column block at `CB`: one row of the output per outer
 * iteration, its accumulators loaded from the right-hand side, the earlier solutions subtracted
 * (rows i+1 … n−1 for a transposed solve, 0 … i−1 otherwise), the diagonal divided last. The
 * locals are laid out for the module's `total` vectors, so a remainder block addresses the same
 * slots as a full one.
 */
function row_pass(layout, shape) {
    const { n, t, b, out, transposed } = layout;
    const { vectors, odd } = shape;
    const L = locals_of(shape.total);
    const code = [];
    // the first row: n − 1 for a transposed solve (back substitution), 0 otherwise
    const first = transposed ? n - 1 : 0;
    const row_step = transposed ? -1 : 1;
    code.push(...OP.i32(8 * (b.at + first * b.row)), ...OP.get(CB), ...OP.i32add, ...OP.set(PB));
    code.push(...OP.i32(8 * (out.at + first * out.row)), ...OP.get(CB), ...OP.i32add, ...OP.set(PO));
    code.push(...OP.i32(8 * (t.at + first * (t.row + t.col))), ...OP.set(PD));
    code.push(...OP.i32(n), ...OP.set(I), ...OP.i32(0), ...OP.set(J0), ...OP.loop);
    for (let v = 0; v < vectors; v++)
        code.push(...OP.get(PB), ...OP.v128load(16 * v), ...OP.set(L.acc + v));
    if (odd)
        code.push(...OP.get(PB), ...OP.f64load(16 * vectors), ...OP.set(L.sacc));
    code.push(...OP.block, ...OP.get(J0), ...OP.i32eqz, ...OP.brif(0), ...inner_pointers(layout));
    code.push(...OP.get(J0), ...OP.set(J), ...OP.loop, ...inner_step(layout, shape));
    code.push(...OP.get(J), ...OP.i32(1), ...OP.i32sub, ...OP.tee(J), ...OP.brif(0), ...OP.end, ...OP.end);
    code.push(...row_store(shape));
    // the next row
    code.push(...OP.get(PB), ...OP.i32(8 * row_step * b.row), ...OP.i32add, ...OP.set(PB));
    code.push(...OP.get(PO), ...OP.i32(8 * row_step * out.row), ...OP.i32add, ...OP.set(PO));
    code.push(...OP.get(PD), ...OP.i32(8 * row_step * (t.row + t.col)), ...OP.i32add, ...OP.set(PD));
    code.push(...OP.get(J0), ...OP.i32(1), ...OP.i32add, ...OP.set(J0));
    code.push(...OP.get(I), ...OP.i32(1), ...OP.i32sub, ...OP.tee(I), ...OP.brif(0), ...OP.end);
    return code;
}
/** The kernel's body: full column blocks in a loop, then the remaining columns as one block of their own width. */
function kernel_body(layout, vectors) {
    const width = 2 * vectors;
    const blocks = Math.floor(layout.k / width);
    const rest = layout.k - blocks * width;
    const code = [];
    if (blocks > 0) {
        code.push(...OP.i32(0), ...OP.set(CB), ...OP.i32(blocks), ...OP.set(CC), ...OP.loop);
        code.push(...row_pass(layout, { total: vectors, vectors, odd: false }));
        code.push(...OP.get(CB), ...OP.i32(8 * width), ...OP.i32add, ...OP.set(CB));
        code.push(...OP.get(CC), ...OP.i32(1), ...OP.i32sub, ...OP.tee(CC), ...OP.brif(0), ...OP.end);
    }
    if (rest > 0) {
        code.push(...OP.i32(8 * blocks * width), ...OP.set(CB));
        code.push(...row_pass(layout, { total: vectors, vectors: Math.floor(rest / 2), odd: rest % 2 === 1 }));
    }
    return code;
}
/** The module's bytes: one exported function `run` over the imported memory `m.mem`. */
function trsm_module(layout, vectors, pages) {
    const L = locals_of(vectors);
    const locals = [
        [...leb(I32_LOCALS), 0x7f],
        [...leb(L.v128_count), 0x7b],
        [...leb(L.f64_count), 0x7c],
    ];
    return module_bytes(locals, kernel_body(layout, vectors), pages);
}
/** Vectors for a solve: as many as the columns allow, up to `TRSM_MAX_VECTORS`. */
function trsm_vectors(layout) {
    return Math.max(1, Math.min(TRSM_MAX_VECTORS, Math.floor(layout.k / 2)));
}
/**
 * The step that runs a triangular solve in WebAssembly on the arena.
 * @param arena  the compiled function's arena, holding every operand
 * @param layout  the solve's operands in it
 * @returns  the step's run function, or null if the module will not compile here
 */
function wasm_trsm_step(arena, layout) {
    try {
        const bytes = trsm_module(layout, trsm_vectors(layout), arena.pages);
        const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes), { m: { mem: arena.memory } });
        return instance.exports.run;
    }
    catch {
        return null;
    }
}
return wasm_trsm_step;
})();
const cholesky = (() => {
/**
 * WebAssembly f64x2 kernel for the Cholesky factorization (D-211), on the arena of `wasm_linalg.ts`.
 *
 * Right-looking, in place on the output: for each column j, the pivot's square root, the column
 * below it divided by the root, then the trailing rows updated, `A[i, k] -= L[i, j] · L[k, j]` for
 * `j < k ≤ i`, with the lanes across `k` along row `i`. That is the same arithmetic per element as
 * `numerics/linalg.ts::cholesky_into`, which subtracts the products `L[i, k'] · L[j, k']` for `k'`
 * ascending and divides or takes the root last: here the products arrive in the same order, one
 * outer step each, so the results are bit-identical. `L[k, j]` for `k` along a row is a column of
 * the factor, strided, so it is copied once per `j` into a contiguous scratch row that every
 * trailing row's lanes then read.
 *
 * A pivot that is not above zero (a matrix that is not positive definite, or a NaN) stops the
 * kernel, which returns 1, and the step fills the output with NaN as the JavaScript op does (D-136).
 *
 * Kernel-style file (loops, `let`).
 */
// i32 locals: columns left, the byte address of the pivot A[j, j], rows left below the pivot, the
// byte address of A[i, j], of A[i, j + 1] running along the row, of the scratch entry running with
// it, pairs left along a row, the flag the kernel returns, and the row's length i − j
const J = 0;
const PJ = 1;
const I = 2;
const PIJ = 3;
const PROW = 4;
const PSC = 5;
const K = 6;
const FLAG = 7;
const LEN = 8;
const I32_LOCALS = 9;
// f64 locals: the pivot, its root, the row's L[i, j]; v128: L[i, j] splatted
const D2 = I32_LOCALS;
const D = D2 + 1;
const LIJ = D + 1;
const F64_LOCALS = 3;
const LS = LIJ + 1;
const V128_LOCALS = 1;
const F64_SQRT = 0x9f;
const F64_GT = 0x64;
const I32_AND = 0x71;
const I32_SHRU = 0x76;
/** The kernel's body: the column loop, each column its root, its scaled entries, the scratch copy and the trailing update. */
function kernel_body(layout) {
    const { n, out, scratch } = layout;
    const row = 8 * out.row;
    const code = [];
    code.push(...OP.i32(1), ...OP.set(FLAG));
    code.push(...OP.i32(8 * out.at), ...OP.set(PJ), ...OP.i32(n), ...OP.set(J));
    code.push(...OP.block, ...OP.loop);
    // the pivot: stop unless it is above zero
    code.push(...OP.get(PJ), ...OP.f64load(0), ...OP.tee(D2), ...OP.f64zero, F64_GT, ...OP.i32eqz, ...OP.brif(1));
    code.push(...OP.get(PJ), ...OP.get(D2), F64_SQRT, ...OP.tee(D), ...OP.f64store(0));
    // rows below the pivot: L[i, j] = A[i, j] / d, and the scratch copy of the column
    code.push(...OP.block, ...OP.get(J), ...OP.i32(1), ...OP.i32sub, ...OP.tee(I), ...OP.i32eqz, ...OP.brif(0));
    code.push(...OP.get(PJ), ...OP.i32(row), ...OP.i32add, ...OP.set(PIJ));
    code.push(...OP.i32(8 * scratch), ...OP.set(PSC), ...OP.loop);
    code.push(...OP.get(PIJ), ...OP.get(PIJ), ...OP.f64load(0), ...OP.get(D), ...OP.f64div, ...OP.tee(LIJ), ...OP.f64store(0));
    code.push(...OP.get(PSC), ...OP.get(LIJ), ...OP.f64store(0));
    code.push(...OP.get(PIJ), ...OP.i32(row), ...OP.i32add, ...OP.set(PIJ));
    code.push(...OP.get(PSC), ...OP.i32(8), ...OP.i32add, ...OP.set(PSC));
    code.push(...OP.get(I), ...OP.i32(1), ...OP.i32sub, ...OP.tee(I), ...OP.brif(0), ...OP.end);
    // the trailing update: row i from column j + 1 to i, against the scratch row
    code.push(...OP.get(J), ...OP.i32(1), ...OP.i32sub, ...OP.set(I));
    code.push(...OP.get(PJ), ...OP.i32(row), ...OP.i32add, ...OP.set(PIJ));
    code.push(...OP.i32(1), ...OP.set(LEN)); // the row's length, counted up from 1
    code.push(...OP.loop);
    code.push(...OP.get(PIJ), ...OP.f64load(0), ...OP.f64x2splat, ...OP.set(LS));
    code.push(...OP.get(PIJ), ...OP.i32(8), ...OP.i32add, ...OP.set(PROW), ...OP.i32(8 * scratch), ...OP.set(PSC));
    // pairs: length >> 1 of them
    code.push(...OP.block, ...OP.get(LEN), ...OP.i32(1), I32_SHRU, ...OP.tee(K), ...OP.i32eqz, ...OP.brif(0), ...OP.loop);
    code.push(...OP.get(PROW), ...OP.get(PROW), ...OP.v128load(0), ...OP.get(LS), ...OP.get(PSC), ...OP.v128load(0), ...OP.f64x2mul, ...OP.f64x2sub, ...OP.v128store(0));
    code.push(...OP.get(PROW), ...OP.i32(16), ...OP.i32add, ...OP.set(PROW), ...OP.get(PSC), ...OP.i32(16), ...OP.i32add, ...OP.set(PSC));
    code.push(...OP.get(K), ...OP.i32(1), ...OP.i32sub, ...OP.tee(K), ...OP.brif(0), ...OP.end, ...OP.end);
    // the odd element
    code.push(...OP.block, ...OP.get(LEN), ...OP.i32(1), I32_AND, ...OP.i32eqz, ...OP.brif(0));
    code.push(...OP.get(PROW), ...OP.get(PROW), ...OP.f64load(0), ...OP.get(PIJ), ...OP.f64load(0), ...OP.get(PSC), ...OP.f64load(0), ...OP.f64mul, ...OP.f64sub, ...OP.f64store(0));
    code.push(...OP.end);
    code.push(...OP.get(PIJ), ...OP.i32(row), ...OP.i32add, ...OP.set(PIJ));
    code.push(...OP.get(LEN), ...OP.i32(1), ...OP.i32add, ...OP.set(LEN));
    code.push(...OP.get(I), ...OP.i32(1), ...OP.i32sub, ...OP.tee(I), ...OP.brif(0), ...OP.end);
    code.push(...OP.end); // the block around the rows below the pivot
    // the next column
    code.push(...OP.get(PJ), ...OP.i32(row + 8), ...OP.i32add, ...OP.set(PJ));
    code.push(...OP.get(J), ...OP.i32(1), ...OP.i32sub, ...OP.tee(J), ...OP.brif(0), ...OP.end);
    code.push(...OP.i32(0), ...OP.set(FLAG));
    code.push(...OP.end); // the escape block
    code.push(...OP.get(FLAG));
    return code;
}
/** The module's bytes: one exported function `run` returning 0, or 1 where a pivot was not above zero. */
function cholesky_module(layout, pages) {
    const locals = [
        [...leb(I32_LOCALS), 0x7f],
        [...leb(F64_LOCALS), 0x7c],
        [...leb(V128_LOCALS), 0x7b],
    ];
    return module_bytes(locals, kernel_body(layout), pages, [0x7f]);
}
/**
 * The step that runs a Cholesky in WebAssembly on the arena.
 * @param arena  the compiled function's arena, holding the factor
 * @param layout  the factor in it and its scratch row
 * @returns  the kernel, returning 1 where the matrix is not positive definite, or null if the module will not compile here
 */
function wasm_cholesky_kernel(arena, layout) {
    try {
        const bytes = cholesky_module(layout, arena.pages);
        const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes), { m: { mem: arena.memory } });
        return instance.exports.run;
    }
    catch {
        return null;
    }
}
return wasm_cholesky_kernel;
})();
function reduction_module({rows, cols, input, out, square}, pages) {
    const body = [];
    // Five pointers/counts, four vector accumulators, one loaded vector, two scalars.
    function tile(vectors, odd) {
        const code = [];
        for (let v = 0; v < vectors; v++) code.push(...OP.f64zero, ...OP.f64x2splat, ...OP.set(5 + v));
        if (odd) code.push(...OP.f64zero, ...OP.set(10));
        code.push(...OP.get(0), ...OP.set(3), ...OP.i32(cols), ...OP.set(4), ...OP.loop);
        for (let v = 0; v < vectors; v++) {
            code.push(...OP.get(3), ...OP.v128load(v * 16), ...OP.set(9));
            code.push(...OP.get(5 + v), ...OP.get(9));
            if (square) code.push(...OP.get(9), ...OP.f64x2mul);
            code.push(...OP.f64x2add, ...OP.set(5 + v));
        }
        if (odd) {
            code.push(...OP.get(3), ...OP.f64load(vectors * 16), ...OP.set(11));
            code.push(...OP.get(10), ...OP.get(11));
            if (square) code.push(...OP.get(11), ...OP.f64mul);
            code.push(...OP.f64add, ...OP.set(10));
        }
        code.push(...OP.get(3), ...OP.i32(rows * 8), ...OP.i32add, ...OP.set(3));
        code.push(...OP.get(4), ...OP.i32(1), ...OP.i32sub, ...OP.tee(4), ...OP.brif(0), ...OP.end);
        for (let v = 0; v < vectors; v++) code.push(...OP.get(1), ...OP.get(5 + v), ...OP.v128store(v * 16));
        if (odd) code.push(...OP.get(1), ...OP.get(10), ...OP.f64store(vectors * 16));
        return code;
    }
    body.push(...OP.i32(input * 8), ...OP.set(0), ...OP.i32(out * 8), ...OP.set(1));
    const blocks = Math.floor(rows / 8), rest = rows % 8;
    if (blocks) {
        body.push(...OP.i32(blocks), ...OP.set(2), ...OP.loop, ...tile(4, false));
        body.push(...OP.get(0), ...OP.i32(64), ...OP.i32add, ...OP.set(0));
        body.push(...OP.get(1), ...OP.i32(64), ...OP.i32add, ...OP.set(1));
        body.push(...OP.get(2), ...OP.i32(1), ...OP.i32sub, ...OP.tee(2), ...OP.brif(0), ...OP.end);
    }
    if (rest) body.push(...tile(Math.floor(rest / 2), rest % 2 === 1));
    return module_bytes([[...leb(5), 0x7f], [...leb(5), 0x7b], [...leb(2), 0x7c]], body, pages);
}
function elementwise_module({kind, count, inputs, out}, pages) {
    const binary = {
        Add: [OP.f64add, OP.f64x2add], Sub: [OP.f64sub, OP.f64x2sub],
        Mul: [OP.f64mul, OP.f64x2mul], TrueDiv: [OP.f64div, OP.f64x2div],
    };
    const body = [...OP.i32(0), ...OP.set(0)];
    function value(vector, offset) {
        const code = [];
        for (const input of inputs) {
            code.push(...(input.scalar ? OP.i32(input.at * 8) : OP.get(0)));
            code.push(...(vector && !input.scalar ? OP.v128load(input.at * 8 + offset) : OP.f64load(input.scalar ? 0 : input.at * 8 + offset)));
            if (vector && input.scalar) code.push(...OP.f64x2splat);
        }
        if (kind in binary) code.push(...binary[kind][vector ? 1 : 0]);
        else if (kind === "Neg") code.push(...(vector ? [0xfd, ...leb(0xed)] : [0x9a]));
        else code.push(...OP.tee(vector ? 2 : 3), ...OP.get(vector ? 2 : 3), ...(vector ? OP.f64x2mul : OP.f64mul));
        return code;
    }
    function store(vector, offset) {
        return [...OP.get(0), ...value(vector, offset), ...(vector ? OP.v128store(out * 8 + offset) : OP.f64store(out * 8 + offset))];
    }
    const blocks = Math.floor(count / 8), rest = count % 8;
    if (blocks) {
        body.push(...OP.i32(blocks), ...OP.set(1), ...OP.loop);
        for (let pair = 0; pair < 4; pair++) body.push(...store(true, pair * 16));
        body.push(...OP.get(0), ...OP.i32(64), ...OP.i32add, ...OP.set(0));
        body.push(...OP.get(1), ...OP.i32(1), ...OP.i32sub, ...OP.tee(1), ...OP.brif(0), ...OP.end);
    }
    for (let pair = 0; pair < Math.floor(rest / 2); pair++) body.push(...store(true, pair * 16));
    if (rest % 2) body.push(...store(false, (rest - 1) * 8));
    return module_bytes([[...leb(2), 0x7f], [...leb(1), 0x7b], [...leb(1), 0x7c]], body, pages);
}
return {make_arena, arena_elements, can_run_wasm_simd, trsm, cholesky,
    elementwise(arena, layout) {
        try {
            const module = new WebAssembly.Module(elementwise_module(layout, arena.pages));
            return new WebAssembly.Instance(module, {m: {mem: arena.memory}}).exports.run;
        } catch { return null; }
    },
    reduce(arena, layout) {
        try {
            const module = new WebAssembly.Module(reduction_module(layout, arena.pages));
            return new WebAssembly.Instance(module, {m: {mem: arena.memory}}).exports.run;
        } catch { return null; }
    },
    product(arena, layout) {
        try {
            const module = new WebAssembly.Module(matmul_module(layout, tiling_for(layout), arena.pages));
            return new WebAssembly.Instance(module, {m: {mem: arena.memory}}).exports.run;
        } catch { return null; }
    }};
})();
