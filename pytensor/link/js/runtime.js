function flatAddress(record, flat) {
    let offset = record.o || 0;
    for (let axis = record.s.length - 1; axis >= 0; axis--) {
        const coordinate = flat % record.s[axis];
        flat = Math.floor(flat / record.s[axis]);
        offset += coordinate * record.t[axis];
    }
    return offset;
}
function scalarValue(record) { return record.d[record.o || 0]; }
function contiguous(record) {
    const expected = strides(record.s);
    return expected.every((step, axis) => record.s[axis] === 1 || record.t[axis] === step);
}
function copyRecord(out, record) {
    if (out === record) return out;
    const count = size(record.s);
    if (contiguous(out) && contiguous(record)) {
        out.d.set(record.d.subarray(record.o || 0, (record.o || 0) + count), out.o || 0);
    } else {
        for (let i = 0; i < count; i++)
            out.d[flatAddress(out, i)] = record.d[flatAddress(record, i)];
    }
    return out;
}
function inplaceRecord(record, shape) {
    if (record.s.length !== shape.length || record.s.some((n, axis) => n !== shape[axis]))
        throw Error('inplace shape mismatch');
    return record;
}
function scalarIndex(value, length) {
    if (!Number.isSafeInteger(value)) throw Error('unsafe index');
    const index = value < 0 ? value + length : value;
    if (index < 0 || index >= length) throw Error('index out of bounds');
    return index;
}
function sliceIndices(length, start, stop, step) {
    step = step === null ? 1 : step;
    if (!Number.isSafeInteger(step) || step === 0) throw Error('invalid slice step');
    const positive = step > 0;
    function normalize(value, fallback) {
        if (value === null) return fallback;
        if (!Number.isSafeInteger(value)) throw Error('unsafe slice index');
        if (value < 0) value += length;
        return Math.max(positive ? 0 : -1, Math.min(value, positive ? length : length - 1));
    }
    start = normalize(start, positive ? 0 : length - 1);
    stop = normalize(stop, positive ? length : -1);
    return [start, Math.max(0, Math.ceil((stop - start) / step)), step];
}
function subtensorView(record, indices) {
    let offset = record.o || 0;
    const shape = [], stridesOut = [];
    for (let axis = 0; axis < record.s.length; axis++) {
        const index = axis < indices.length ? indices[axis] : [null, null, null];
        if (Array.isArray(index)) {
            const [start, count, step] = sliceIndices(record.s[axis], ...index);
            offset += start * record.t[axis];
            shape.push(count); stridesOut.push(step * record.t[axis]);
        } else {
            offset += scalarIndex(index, record.s[axis]) * record.t[axis];
        }
    }
    return {d: record.d, s: shape, t: stridesOut, o: offset};
}
function broadcastAddress(record, coordinates) {
    let offset = record.o || 0;
    const shift = coordinates.length - record.s.length;
    for (let axis = 0; axis < record.s.length; axis++)
        offset += (record.s[axis] === 1 ? 0 : coordinates[axis + shift]) * record.t[axis];
    return offset;
}
function coordinatesOf(flat, shape) {
    const result = Array(shape.length);
    for (let axis = shape.length - 1; axis >= 0; axis--) {
        result[axis] = flat % shape[axis]; flat = Math.floor(flat / shape[axis]);
    }
    return result;
}
function updateSubtensor(out, source, value, indices, set, id) {
    if (out.d === value.d) value = copyRecord(slot(id, value.s), value);
    copyRecord(out, source);
    const view = subtensorView(out, indices);
    const shape = broadcast([view.s, value.s]);
    if (shape.length !== view.s.length || shape.some((n, axis) => n !== view.s[axis]))
        throw Error('update broadcast shape mismatch');
    for (let i = 0; i < size(view.s); i++) {
        const coordinates = coordinatesOf(i, view.s);
        const target = broadcastAddress(view, coordinates);
        const input = value.d[broadcastAddress(value, coordinates)];
        if (set) out.d[target] = input; else out.d[target] += input;
    }
    return out;
}
function reshapeRecord(record, requested, id) {
    const shape = [...requested], count = size(record.s);
    let inferred = -1, known = 1;
    for (let axis = 0; axis < shape.length; axis++) {
        const length = shape[axis];
        if (length === -1 && inferred === -1) inferred = axis;
        else {
            if (!Number.isSafeInteger(length) || length < 0) throw Error('invalid reshape dimension');
            known *= length;
        }
    }
    if (inferred !== -1) {
        if (known === 0 || count % known !== 0) throw Error('cannot infer reshape dimension');
        shape[inferred] = count / known;
    }
    if (size(shape) !== count) throw Error('reshape size mismatch');
    const expected = strides(record.s);
    if (expected.every((step, axis) => record.s[axis] === 1 || record.t[axis] === step))
        return {d: record.d, s: shape, t: strides(shape), o: record.o || 0};
    return copyRecord(slot(id, shape), record);
}
function advancedPlan(source, requested) {
    const indices = [...requested];
    while (indices.length < source.s.length) indices.push([null, null, null]);
    const indexedAxes = [], slices = [];
    for (let axis = 0; axis < indices.length; axis++) {
        const index = indices[axis];
        if (index && index.index) indexedAxes.push(axis);
        else if (Array.isArray(index)) {
            const [start, count, step] = sliceIndices(source.s[axis], ...index);
            slices.push({axis, start, count, step});
        }
    }
    const advancedShape = indexedAxes.length ? broadcast(indexedAxes.map(axis => indices[axis].index.s)) : [];
    if (advancedShape.length > 1) throw Error('advanced indices must be scalars or vectors');
    const first = indexedAxes[0];
    const consecutive = indexedAxes.every((axis, k) => axis === first + k);
    const insertion = consecutive ? slices.filter(slice => slice.axis < first).length : 0;
    const shape = slices.map(slice => slice.count);
    shape.splice(insertion, 0, ...advancedShape);
    slices.forEach((slice, k) => { slice.outputAxis = k < insertion ? k : k + advancedShape.length; });
    return {indices, slices, shape, advancedAxis: advancedShape.length ? insertion : -1};
}
function advancedAddress(source, plan, coordinates) {
    let offset = source.o || 0;
    for (let axis = 0; axis < source.s.length; axis++) {
        const index = plan.indices[axis];
        let coordinate;
        if (index && index.index) {
            const record = index.index;
            const flat = record.s.length && record.s[0] !== 1 ? coordinates[plan.advancedAxis] : 0;
            coordinate = scalarIndex(record.d[flatAddress(record, flat)], source.s[axis]);
        } else if (Array.isArray(index)) {
            const slice = plan.slices.find(slice => slice.axis === axis);
            coordinate = slice.start + coordinates[slice.outputAxis] * slice.step;
        } else coordinate = scalarIndex(index, source.s[axis]);
        offset += coordinate * source.t[axis];
    }
    return offset;
}
function advancedRead(source, requested, id) {
    const plan = advancedPlan(source, requested), out = slot(id, plan.shape);
    for (let i = 0; i < out.d.length; i++)
        out.d[i] = source.d[advancedAddress(source, plan, coordinatesOf(i, plan.shape))];
    return out;
}
function advancedUpdate(source, value, requested, id, set, ignoreDuplicates, inplace) {
    const temporary = !inplace || ignoreDuplicates;
    const out = temporary ? copyRecord(slot(id, source.s), source) : source;
    if (out.d === value.d) value = copyRecord(slot(id, value.s), value);
    const plan = advancedPlan(out, requested);
    const shape = broadcast([plan.shape, value.s]);
    if (shape.length !== plan.shape.length || shape.some((n, axis) => n !== plan.shape[axis]))
        throw Error('advanced update broadcast mismatch');
    for (let i = 0; i < size(plan.shape); i++) {
        const coordinates = coordinatesOf(i, plan.shape);
        const target = advancedAddress(out, plan, coordinates);
        const input = value.d[broadcastAddress(value, coordinates)];
        if (set) out.d[target] = input;
        else if (ignoreDuplicates) out.d[target] = source.d[advancedAddress(source, plan, coordinates)] + input;
        else out.d[target] += input;
    }
    return inplace && temporary ? copyRecord(source, out) : out;
}
