// Per-node dense kernels over staging buffers; dynamic shapes use the same adapter.
const ptWasm = (() => {
    const enabled = ptWasmKernels.can_run_wasm_simd();
    const cache = new Map();
    function matrix(record, rowVector = false) {
        if (record.s.length === 2)
            return {d: record.d, rows: record.s[0], cols: record.s[1], row: record.t[0], col: record.t[1], o: record.o || 0};
        return {d: record.d, rows: rowVector ? 1 : record.s[0], cols: rowVector ? record.s[0] : 1,
            row: rowVector ? 0 : record.t[0], col: rowVector ? record.t[0] : 0, o: record.o || 0};
    }
    function transpose(value) {
        return {...value, rows: value.cols, cols: value.rows, row: value.col, col: value.row};
    }
    function stage(target, source) {
        const {d, rows, cols, row, col, o} = source;
        if ((cols === 1 || col === 1) && (rows === 1 || row === cols)) {
            target.set(d.subarray(o, o + rows * cols));
        } else {
            let at = 0;
            for (let i = 0; i < rows; i++) {
                const from = o + i * row;
                for (let j = 0; j < cols; j++) target[at++] = d[from + j * col];
            }
        }
    }
    function kernel(id, tag, sizes, build) {
        let state = cache.get(id);
        if (!state || state.tag !== tag) {
            const count = ptWasmKernels.arena_elements(sizes);
            // The assembler uses signed 32-bit byte addresses.
            const arena = count * 8 < 2 ** 31 ? ptWasmKernels.make_arena(count) : null;
            const buffers = arena ? sizes.map(size => arena.alloc(size)) : [];
            const operands = buffers.map((buffer, index) => ({at: buffer.byteOffset / 8, row: 0, col: 1}));
            state = {tag, buffers, staged: [], run: arena ? build(arena, operands) : null};
            cache.set(id, state);
        }
        return state.run ? state : null;
    }
    function dot(a, b, out, id, constantA, constantB) {
        if (!enabled) return false;
        let left = matrix(a, true), right = matrix(b);
        let {rows: m, cols: k} = left, n = right.cols;
        if (k === 1 || m * k * n < 4096 || (m < 2 && n < 2)) return false;
        if (n === 1 && !constantA && left.row !== 1) return false;
        if (n === 1) {
            [left, right] = [transpose(right), transpose(left)];
            [m, n] = [n, m];
            [constantA, constantB] = [constantB, constantA];
        }
        const state = kernel(id, `dot:${m},${k},${n}`, [m * k, k * n, m * n], (arena, [aa, bb, oo]) =>
            ptWasmKernels.product(arena, {m, k, n, a: {...aa, row: k}, b: {...bb, row: n}, out: {...oo, row: n}}));
        if (!state) return false;
        if (!constantA || !state.staged[0]) stage(state.buffers[0], left);
        if (!constantB || !state.staged[1]) stage(state.buffers[1], right);
        state.staged[0] = state.staged[1] = true;
        state.run();
        out.d.set(state.buffers[2]);
        return true;
    }
    function solve(a, b, out, id, lower, unit, negated = false) {
        const n = a.s[0], k = b.s[1];
        if (!enabled || k < 2 || n * n * k / 2 < 4096) return false;
        const state = kernel(id, `solve:${n},${k},${lower},${negated}`, [n * n, n * k, n * k], (arena, [tt, bb, oo]) => {
            const run = ptWasmKernels.trsm(arena, {n, k, t: {...tt, row: lower ? n : 1, col: lower ? 1 : n},
                b: {...bb, row: k}, out: {...oo, row: k}, transposed: !lower});
            if (!run || !negated) return run;
            const negate = ptWasmKernels.elementwise(arena, {kind: "Neg", count: n * k, inputs: [{at: bb.at, scalar: false}], out: bb.at});
            return negate ? () => {negate(); run();} : null;
        });
        if (!state) return false;
        const triangle = state.buffers[0];
        stage(triangle, matrix(a));
        for (let i = 0; i < n; i++) {
            if (unit) triangle[i * (n + 1)] = 1;
            else if (triangle[i * (n + 1)] === 0) {out.d.fill(NaN); return true;}
        }
        stage(state.buffers[1], matrix(b));
        state.run();
        out.d.set(state.buffers[2]);
        return true;
    }
    function cholesky(a, out, id, lower) {
        const n = a.s[0];
        if (!enabled || n < 8) return false;
        const state = kernel(id, `cholesky:${n}`, [n * n, n], (arena, [oo, scratch]) =>
            ptWasmKernels.cholesky(arena, {n, out: {...oo, row: n}, scratch: scratch.at}));
        if (!state) return false;
        const factor = state.buffers[0];
        const source = matrix(a);
        stage(factor, lower ? source : transpose(source));
        for (let i = 0; i < n; i++) factor.fill(0, i * n + i + 1, (i + 1) * n);
        if (state.run() !== 0) out.d.fill(NaN);
        else if (lower) out.d.set(factor);
        else for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out.d[i * n + j] = factor[j * n + i];
        return true;
    }
    function reduce(source, out, id, axis, square) {
        const kept = 1 - axis, rows = source.s[kept], cols = source.s[axis];
        if (!enabled || rows < 2 || cols < 1 || rows * cols < 4096 ||
            source.t[kept] !== 1 || source.t[axis] !== rows) return false;
        const state = kernel(id, `reduce:${square},${rows},${cols}`, [rows * cols, rows], (arena, [input, result]) =>
            ptWasmKernels.reduce(arena, {rows, cols, input: input.at, out: result.at, square}));
        if (!state) return false;
        const offset = source.o || 0;
        state.buffers[0].set(source.d.subarray(offset, offset + rows * cols));
        state.run();
        out.d.set(state.buffers[1]);
        return true;
    }
    function elementwise(kind, inputs, out, id, constants) {
        const count = size(out.s);
        if (!enabled || count < 4096 || !contiguous(out)) return false;
        const counts = inputs.map(input => size(input.s));
        if (inputs.some((input, i) => (counts[i] !== count && counts[i] !== 1) || !contiguous(input))) return false;
        const state = kernel(id, `elementwise:${kind},${count},${counts}`, [...counts, count], (arena, operands) =>
            ptWasmKernels.elementwise(arena, {kind, count, inputs: operands.slice(0, -1).map((input, i) => ({at: input.at, scalar: counts[i] === 1})), out: operands.at(-1).at}));
        if (!state) return false;
        for (let i = 0; i < inputs.length; i++) {
            if (!constants[i] || !state.staged[i]) {
                const offset = inputs[i].o || 0;
                state.buffers[i].set(inputs[i].d.subarray(offset, offset + counts[i]));
                state.staged[i] = true;
            }
        }
        state.run();
        out.d.set(state.buffers.at(-1), out.o || 0);
        return true;
    }
    return {dot, solve, cholesky, elementwise,
        norm: (source, out, id, axis) => reduce(source, out, id, axis, true),
        sum: (source, out, id, axis) => reduce(source, out, id, axis, false),
        kernelCount: () => [...cache.values()].filter(state => state.run).length};
})();
