"""Allocation, shape manipulation and indexing for JavaScript tensor records."""

from pytensor.graph.basic import Constant, Variable
from pytensor.link.js.dispatch.basic import JSCode, js_funcify
from pytensor.link.js.dispatch.elemwise import address, loop_lines, stride_setup
from pytensor.link.string_codegen import CODE_TOKEN
from pytensor.tensor.basic import (
    Alloc,
    AllocEmpty,
    ARange,
    Join,
    Split,
    TensorFromScalar,
)
from pytensor.tensor.extra_ops import CumOp
from pytensor.tensor.math import Dot
from pytensor.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedSubtensor,
    IncSubtensor,
    Subtensor,
    indices_from_subtensor,
)


def index_expression(op, node, inputs, start, advanced=False):
    names = dict(zip(node.inputs[start:], inputs[start:], strict=True))

    def scalar(index):
        if index is None:
            return "null"
        if isinstance(index, Variable):
            if advanced:
                if not index.type.dtype.startswith(("int", "uint")):
                    raise NotImplementedError(
                        "JS advanced indexing supports integer indices"
                    )
                return f"{{index: {names[index]}}}"
            return f"scalarValue({names[index]})"
        return str(index)

    indices = indices_from_subtensor(node.inputs[start:], op.idx_list)
    return (
        "["
        + ", ".join(
            "[" + ", ".join(map(scalar, (index.start, index.stop, index.step))) + "]"
            if isinstance(index, slice)
            else scalar(index)
            for index in indices
        )
        + "]"
    )


@js_funcify.register(Subtensor)
def js_funcify_subtensor(op, node, inputs, slot):
    name = f"v{slot}"
    return JSCode(
        (
            f"const {name} = subtensorView({inputs[0]}, {index_expression(op, node, inputs, 1)});",
        ),
        (name,),
    )


@js_funcify.register(Split)
def js_funcify_split(op, node, inputs, slot):
    source, lengths = inputs
    names = tuple(f"v{slot + i}" for i in range(op.len_splits))
    lines = [
        *(f"let {name};" for name in names),
        "{",
        CODE_TOKEN.INDENT,
        f"if ({lengths}.s[0] !== {op.len_splits}) throw Error('split count mismatch');",
        f"const lengths = Array.from({{length: {op.len_splits}}}, (_, i) => {lengths}.d[({lengths}.o || 0) + i * {lengths}.t[0]]);",
        "if (lengths.some(n => !Number.isSafeInteger(n) || n < 0)) throw Error('invalid split size');",
        f"if (lengths.reduce((a, b) => a + b, 0) !== {source}.s[{op.axis}]) throw Error('split sizes do not match axis');",
        "let offset = 0;",
    ]
    for i, name in enumerate(names):
        lines.extend(
            [
                f"{name} = {{d: {source}.d, s: {source}.s.slice(), t: {source}.t, o: ({source}.o || 0) + offset * {source}.t[{op.axis}]}};",
                f"{name}.s[{op.axis}] = lengths[{i}];",
                f"offset += lengths[{i}];",
            ]
        )
    lines.extend([CODE_TOKEN.DEDENT, "}"])
    return JSCode(tuple(lines), names)


@js_funcify.register(IncSubtensor)
def js_funcify_inc_subtensor(op, node, inputs, slot):
    name = f"v{slot}"
    source, value = inputs[:2]
    indices = index_expression(op, node, inputs, 2)
    target = source if op.inplace else f"slot({slot}, {source}.s)"
    return JSCode(
        (
            f"const {name} = updateSubtensor({target}, {source}, {value}, {indices}, {str(op.set_instead_of_inc).lower()}, {slot});",
        ),
        (name,),
    )


@js_funcify.register(AdvancedSubtensor)
def js_funcify_advanced_subtensor(op, node, inputs, slot):
    name = f"v{slot}"
    reconstructed = indices_from_subtensor(node.inputs[1:], op.idx_list)
    vectors = [
        (axis, idx)
        for axis, idx in enumerate(reconstructed)
        if isinstance(idx, Variable) and idx.ndim == 1
    ]
    if len(vectors) == 1 and all(
        isinstance(idx, slice) or (isinstance(idx, Variable) and idx.ndim == 1)
        for idx in reconstructed
    ):
        axis, index = vectors[0]
        source = inputs[0]
        index_name = inputs[node.inputs.index(index)]
        rank = node.outputs[0].ndim
        names = dict(zip(node.inputs[1:], inputs[1:], strict=True))
        shapes = []
        coordinates = {}
        setup = []
        for source_axis in range(rank):
            if source_axis == axis:
                shapes.append(f"{index_name}.s[0]")
                coordinates[source_axis] = f"idx{source_axis}"
            else:
                spec = (
                    reconstructed[source_axis]
                    if source_axis < len(reconstructed)
                    else slice(None)
                )
                parts = [
                    "null" if v is None else f"scalarValue({names[v]})"
                    for v in (spec.start, spec.stop, spec.step)
                ]
                setup.append(
                    f"const slice{source_axis} = sliceIndices({source}.s[{source_axis}], {', '.join(parts)});"
                )
                shapes.append(f"slice{source_axis}[1]")
                coordinates[source_axis] = (
                    f"(slice{source_axis}[0] + i{source_axis} * slice{source_axis}[2])"
                )
        lines = [
            f"let {name};",
            "{",
            CODE_TOKEN.INDENT,
            *setup,
            f"const shape = [{', '.join(shapes)}];",
            f"{name} = slot({slot}, shape);",
            f"const data = {source}.d;",
            *stride_setup(node.inputs[0], source, "st"),
            "let k = 0;",
        ]
        for loop_axis in range(rank):
            lines.extend(
                [
                    f"for (let i{loop_axis} = 0; i{loop_axis} < shape[{loop_axis}]; i{loop_axis}++) {{",
                    CODE_TOKEN.INDENT,
                ]
            )
            if loop_axis == axis:
                lines.append(
                    f"const idx{axis} = indexAt({index_name}, i{axis}, {source}.s[{axis}]);"
                )
        lines.append(
            f"{name}.d[k++] = data[{address(source, rank, indexed=coordinates, strides='st')}];"
        )
        for _ in range(rank):
            lines.extend([CODE_TOKEN.DEDENT, "}"])
        lines.extend([CODE_TOKEN.DEDENT, "}"])
        return JSCode(tuple(lines), (name,))
    indices = index_expression(op, node, inputs, 1, advanced=True)
    return JSCode(
        (f"const {name} = advancedRead({inputs[0]}, {indices}, {slot});",), (name,)
    )


@js_funcify.register(AdvancedIncSubtensor)
def js_funcify_advanced_inc_subtensor(op, node, inputs, slot):
    name = f"v{slot}"
    reconstructed = indices_from_subtensor(node.inputs[2:], op.idx_list)
    if (
        len(reconstructed) == 1
        and isinstance(reconstructed[0], Variable)
        and reconstructed[0].ndim == 1
        and reconstructed[0].dtype.startswith(("int", "uint"))
    ):
        source, value, index = inputs
        rank = node.inputs[0].ndim
        temporary = not op.inplace or op.ignore_duplicates
        target = (
            f"copyRecord(slot({slot}, {source}.s), {source})" if temporary else source
        )
        target_address = address(name, rank, indexed={0: "idx"}, strides="st")
        value_address = address("value", rank, node.inputs[1].ndim, strides="bs")
        if op.set_instead_of_inc:
            store = f"out[{target_address}] = data[{value_address}];"
        elif op.ignore_duplicates:
            base_address = address(source, rank, indexed={0: "idx"}, strides="baseSt")
            store = (
                f"out[{target_address}] = base[{base_address}] + data[{value_address}];"
            )
        else:
            store = f"out[{target_address}] += data[{value_address}];"
        lines = [
            f"let {name} = {target};",
            "{",
            CODE_TOKEN.INDENT,
            f"let value = {value};",
            f"if ({name}.d === value.d) value = copyRecord(slot({slot + 1}, value.s), value);",
            f"const shape = {source}.s.slice();",
            f"shape[0] = {index}.s[0];",
            "const compatible = broadcast([shape, value.s]);",
            "if (compatible.length !== shape.length || compatible.some((n, axis) => n !== shape[axis])) throw Error('advanced update broadcast mismatch');",
            f"const out = {name}.d, data = value.d;",
            *stride_setup(node.inputs[0], name, "st"),
            *stride_setup(node.inputs[1], "value", "bs"),
            *(
                [
                    f"const base = {source}.d;",
                    *stride_setup(node.inputs[0], source, "baseSt"),
                ]
                if op.ignore_duplicates and not op.set_instead_of_inc
                else []
            ),
            "if (size(shape) > 0) {",
            CODE_TOKEN.INDENT,
            *loop_lines(
                rank,
                [store],
                setup={0: [f"const idx = indexAt({index}, i0, {source}.s[0]);"]},
            ),
            CODE_TOKEN.DEDENT,
            "}",
            CODE_TOKEN.DEDENT,
            "}",
        ]
        if op.inplace and temporary:
            lines.append(f"{name} = copyRecord({source}, {name});")
        return JSCode(tuple(lines), (name,), slots=2)
    indices = index_expression(op, node, inputs, 2, advanced=True)
    flags = f"{str(op.set_instead_of_inc).lower()}, {str(op.ignore_duplicates).lower()}, {str(op.inplace).lower()}"
    return JSCode(
        (
            f"const {name} = advancedUpdate({inputs[0]}, {inputs[1]}, {indices}, {slot}, {flags});",
        ),
        (name,),
    )


@js_funcify.register(Alloc)
def js_funcify_alloc(op, node, inputs, slot):
    source = inputs[0]
    rank = node.outputs[0].ndim
    shape = "[" + ", ".join(f"scalarValue({v})" for v in inputs[1:]) + "]"
    name = f"v{slot}"
    source_address = address(source, rank, node.inputs[0].ndim)
    lines = [
        f"let {name};",
        "{",
        CODE_TOKEN.INDENT,
        f"const shape = {shape};",
        f"{name} = slot({slot}, shape);",
        f"const compatible = broadcast([shape, {source}.s]);",
        "if (compatible.length !== shape.length || compatible.some((n, a) => n !== shape[a])) throw Error('allocation broadcast mismatch');",
        "let k = 0;",
        *(
            [f"{name}.d.fill(scalarValue({source}));"]
            if all(node.inputs[0].type.broadcastable)
            else loop_lines(rank, [f"{name}.d[k++] = {source}.d[{source_address}];"])
        ),
        CODE_TOKEN.DEDENT,
        "}",
    ]
    return JSCode(tuple(lines), (name,))


@js_funcify.register(AllocEmpty)
def js_funcify_alloc_empty(op, node, inputs, slot):
    name = f"v{slot}"
    shape = "[" + ", ".join(f"scalarValue({v})" for v in inputs) + "]"
    return JSCode((f"const {name} = slot({slot}, {shape});",), (name,))


@js_funcify.register(Reshape)
def js_funcify_reshape(op, node, inputs, slot):
    name = f"v{slot}"
    source, shape = inputs
    requested = f"Array.from({{length: size({shape}.s)}}, (_, i) => {shape}.d[flatAddress({shape}, i)])"
    return JSCode(
        (f"const {name} = reshapeRecord({source}, {requested}, {slot});",), (name,)
    )


@js_funcify.register(Shape)
def js_funcify_shape(op, node, inputs, slot):
    name = f"v{slot}"
    return JSCode(
        (
            f"const {name} = slot({slot}, [{node.inputs[0].ndim}]);",
            f"{name}.d.set({inputs[0]}.s);",
        ),
        (name,),
    )


@js_funcify.register(ARange)
def js_funcify_arange(op, node, inputs, slot):
    name = f"v{slot}"
    start, stop, step = (f"scalarValue({value})" for value in inputs)
    lines = [
        f"let {name};",
        "{",
        CODE_TOKEN.INDENT,
        f"const start = {start}, stop = {stop}, step = {step};",
        "if (!Number.isFinite(start) || !Number.isFinite(stop) || !Number.isFinite(step) || step === 0) throw Error('invalid arange');",
        "const count = Math.max(0, Math.ceil((stop - start) / step));",
        f"{name} = slot({slot}, [count]);",
        f"for (let i = 0; i < count; i++) {name}.d[i] = start + i * step;",
        CODE_TOKEN.DEDENT,
        "}",
    ]
    return JSCode(tuple(lines), (name,))


@js_funcify.register(Shape_i)
def js_funcify_shape_i(op, node, inputs, slot):
    name = f"v{slot}"
    return JSCode(
        (f"const {name} = slot({slot}, []);", f"{name}.d[0] = {inputs[0]}.s[{op.i}];"),
        (name,),
    )


@js_funcify.register(SpecifyShape)
def js_funcify_specify_shape(op, node, inputs, slot):
    checks = [
        f"if ({inputs[0]}.s[{axis}] !== scalarValue({value})) throw Error('specified shape mismatch');"
        for axis, value in enumerate(inputs[1:])
        if value != "null"
    ]
    return JSCode(tuple(checks), (inputs[0],))


@js_funcify.register(CumOp)
def js_funcify_cum_op(op, node, inputs, slot):
    name, source = f"v{slot}", inputs[0]
    operator, identity = ("+", 0) if op.mode == "add" else ("*", 1)
    shape = f"[size({source}.s)]" if op.axis is None else f"{source}.s"
    previous = "i - 1" if op.axis is None else f"i - {name}.t[{op.axis}]"
    first = (
        "i === 0"
        if op.axis is None
        else f"Math.floor(i / {name}.t[{op.axis}]) % {name}.s[{op.axis}] === 0"
    )
    lines = [
        f"const {name} = slot({slot}, {shape});",
        f"for (let i = 0; i < {name}.d.length; i++) {{",
        CODE_TOKEN.INDENT,
        f"const previous = {first} ? {identity} : {name}.d[{previous}];",
        f"{name}.d[i] = previous {operator} {source}.d[flatAddress({source}, i)];",
        CODE_TOKEN.DEDENT,
        "}",
    ]
    return JSCode(tuple(lines), (name,))


@js_funcify.register(TensorFromScalar)
def js_funcify_tensor_from_scalar(op, node, inputs, slot):
    return JSCode((), (inputs[0],))


@js_funcify.register(Join)
def js_funcify_join(op, node, inputs, slot):
    name = f"v{slot}"
    arrays = inputs
    lines = [
        f"let {name};",
        "{",
        CODE_TOKEN.INDENT,
        f"const axis = {op.axis};",
        f"const shape = [...{arrays[0]}.s];",
        "shape[axis] = 0;",
        f"const arrays = [{', '.join(arrays)}];",
        "for (const source of arrays) {",
        CODE_TOKEN.INDENT,
        "if (source.s.some((n, a) => a !== axis && n !== shape[a])) throw Error('join shape mismatch');",
        "shape[axis] += source.s[axis];",
        CODE_TOKEN.DEDENT,
        "}",
        f"{name} = slot({slot}, shape);",
        "let axisOffset = 0;",
        "for (const source of arrays) {",
        CODE_TOKEN.INDENT,
        "if (contiguous(source)) {",
        CODE_TOKEN.INDENT,
        "const inner = size(shape.slice(axis + 1));",
        "const block = source.s[axis] * inner, outputBlock = shape[axis] * inner;",
        "const count = size(source.s), offset = source.o || 0;",
        f"const out = {name}.d;",
        "if (block > 0) for (let start = 0, target = axisOffset * inner; start < count; start += block, target += outputBlock)",
        CODE_TOKEN.INDENT,
        "out.set(source.d.subarray(offset + start, offset + start + block), target);",
        CODE_TOKEN.DEDENT,
        CODE_TOKEN.DEDENT,
        "} else {",
        CODE_TOKEN.INDENT,
        "for (let i = 0; i < size(source.s); i++) {",
        CODE_TOKEN.INDENT,
        "const coordinates = coordinatesOf(i, source.s);",
        "coordinates[axis] += axisOffset;",
        f"{name}.d[broadcastAddress({name}, coordinates)] = source.d[flatAddress(source, i)];",
        CODE_TOKEN.DEDENT,
        "}",
        CODE_TOKEN.DEDENT,
        "}",
        "axisOffset += source.s[axis];",
        CODE_TOKEN.DEDENT,
        "}",
        CODE_TOKEN.DEDENT,
        "}",
    ]
    return JSCode(tuple(lines), (name,))


@js_funcify.register(Dot)
def js_funcify_dot(op, node, inputs, slot):
    a, b = inputs
    left_rank, right_rank = (var.ndim for var in node.inputs)
    if left_rank not in (1, 2) or right_rank not in (1, 2):
        raise NotImplementedError("JS Dot supports vectors and matrices")
    name = f"v{slot}"
    rows = f"{a}.s[0]" if left_rank == 2 else "1"
    cols = f"{b}.s[1]" if right_rank == 2 else "1"
    extent = f"{a}.s[{left_rank - 1}]"
    output_shape = (
        "["
        + ", ".join(
            [*([rows] if left_rank == 2 else []), *([cols] if right_rank == 2 else [])]
        )
        + "]"
    )
    lines = [
        f"let {name};",
        "{",
        CODE_TOKEN.INDENT,
        f"if ({extent} !== {b}.s[0]) throw Error('dot shape mismatch');",
        f"{name} = slot({slot}, {output_shape});",
        *(
            [
                f"if (!ptWasm.dot({a}, {b}, {name}, {slot}, "
                f"{str(isinstance(node.inputs[0], Constant)).lower()}, "
                f"{str(isinstance(node.inputs[1], Constant)).lower()})) {{",
                CODE_TOKEN.INDENT,
            ]
            if node.outputs[0].dtype == "float64"
            else []
        ),
        f"const dataA = {a}.d, dataB = {b}.d, out = {name}.d;",
        f"const rows = {rows}, cols = {cols}, extent = {extent};",
        f"const ao = {a}.o || 0, bo = {b}.o || 0;",
        "const as0 = " + (f"{a}.t[0];" if left_rank == 2 else "0;"),
        f"const as1 = {a}.t[{left_rank - 1}], bs0 = {b}.t[0];",
        "const bs1 = " + (f"{b}.t[1];" if right_rank == 2 else "0;"),
        "let position = 0;",
        "let i = 0;",
        "if (cols === 1) {",
        CODE_TOKEN.INDENT,
        "for (; i + 3 < rows; i += 4) {",
        CODE_TOKEN.INDENT,
        *(f"const left{r} = ao + (i + {r}) * as0;" for r in range(4)),
        *(f"let acc{r} = 0;" for r in range(4)),
        "for (let k = 0; k < extent; k++) {",
        CODE_TOKEN.INDENT,
        "const value = dataB[bo + k * bs0], offset = k * as1;",
        *(f"acc{r} += dataA[left{r} + offset] * value;" for r in range(4)),
        CODE_TOKEN.DEDENT,
        "}",
        *(f"out[i + {r}] = acc{r};" for r in range(4)),
        CODE_TOKEN.DEDENT,
        "}",
        "for (; i < rows; i++) {",
        CODE_TOKEN.INDENT,
        "const left = ao + i * as0;",
        "let acc = 0;",
        "for (let k = 0; k < extent; k++) acc += dataA[left + k * as1] * dataB[bo + k * bs0];",
        "out[i] = acc;",
        CODE_TOKEN.DEDENT,
        "}",
        CODE_TOKEN.DEDENT,
        "} else if (extent === 1) {",
        CODE_TOKEN.INDENT,
        "for (; i < rows; i++) {",
        CODE_TOKEN.INDENT,
        "const value = dataA[ao + i * as0];",
        "let j = 0;",
        "for (; j + 3 < cols; j += 4) {",
        CODE_TOKEN.INDENT,
        *(
            f"out[position++] = 0 + value * dataB[bo + (j + {offset}) * bs1];"
            for offset in range(4)
        ),
        CODE_TOKEN.DEDENT,
        "}",
        "for (; j < cols; j++) out[position++] = 0 + value * dataB[bo + j * bs1];",
        CODE_TOKEN.DEDENT,
        "}",
        CODE_TOKEN.DEDENT,
        "} else {",
        CODE_TOKEN.INDENT,
    ]
    if left_rank == 2:
        lines.extend(
            [
                "for (; i + 1 < rows; i += 2) {",
                CODE_TOKEN.INDENT,
                "const left = ao + i * as0;",
                "let j = 0;",
                "for (; j + 3 < cols; j += 4) {",
                CODE_TOKEN.INDENT,
                "const right = bo + j * bs1;",
                *(f"let acc{r}{c} = 0;" for r in range(2) for c in range(4)),
                "for (let k = 0; k < extent; k++) {",
                CODE_TOKEN.INDENT,
                "const a0 = dataA[left + k * as1], a1 = dataA[left + as0 + k * as1];",
                "const rightK = right + k * bs0;",
                *(f"const b{c} = dataB[rightK + {c} * bs1];" for c in range(4)),
                *(f"acc{r}{c} += a{r} * b{c};" for r in range(2) for c in range(4)),
                CODE_TOKEN.DEDENT,
                "}",
                *(
                    f"out[(i + {r}) * cols + j + {c}] = acc{r}{c};"
                    for r in range(2)
                    for c in range(4)
                ),
                CODE_TOKEN.DEDENT,
                "}",
                "for (; j < cols; j++) {",
                CODE_TOKEN.INDENT,
                "const right = bo + j * bs1;",
                "let acc0 = 0, acc1 = 0;",
                "let k = 0;",
                "for (; k + 3 < extent; k += 4) {",
                CODE_TOKEN.INDENT,
                *(
                    line
                    for offset in range(4)
                    for line in (
                        f"const b{offset} = dataB[right + (k + {offset}) * bs0];",
                        f"acc0 += dataA[left + (k + {offset}) * as1] * b{offset};",
                        f"acc1 += dataA[left + as0 + (k + {offset}) * as1] * b{offset};",
                    )
                ),
                CODE_TOKEN.DEDENT,
                "}",
                "for (; k < extent; k++) {",
                CODE_TOKEN.INDENT,
                "const b = dataB[right + k * bs0];",
                "acc0 += dataA[left + k * as1] * b;",
                "acc1 += dataA[left + as0 + k * as1] * b;",
                CODE_TOKEN.DEDENT,
                "}",
                "out[i * cols + j] = acc0;",
                "out[(i + 1) * cols + j] = acc1;",
                CODE_TOKEN.DEDENT,
                "}",
                CODE_TOKEN.DEDENT,
                "}",
                "position = i * cols;",
            ]
        )
    lines.extend(
        [
            "for (; i < rows; i++) {",
            CODE_TOKEN.INDENT,
            "const left = ao + i * as0;",
            "for (let j = 0; j < cols; j++) {",
            CODE_TOKEN.INDENT,
            "const right = bo + j * bs1;",
            "let acc = 0;",
            "for (let k = 0; k < extent; k++) {",
            CODE_TOKEN.INDENT,
            "acc += dataA[left + k * as1] * dataB[right + k * bs0];",
            CODE_TOKEN.DEDENT,
            "}",
            "out[position++] = acc;",
            CODE_TOKEN.DEDENT,
            "}",
            CODE_TOKEN.DEDENT,
            "}",
            *([CODE_TOKEN.DEDENT, "}"] if node.outputs[0].dtype == "float64" else []),
            CODE_TOKEN.DEDENT,
            "}",
            CODE_TOKEN.DEDENT,
            "}",
        ]
    )
    return JSCode(tuple(lines), (name,))
