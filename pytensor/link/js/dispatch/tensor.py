"""Allocation, shape manipulation and indexing for JavaScript tensor records."""

from pytensor.graph.basic import Variable
from pytensor.link.js.dispatch.basic import JSCode, js_funcify
from pytensor.link.js.dispatch.elemwise import address, loop_lines, stride_setup
from pytensor.link.string_codegen import CODE_TOKEN
from pytensor.tensor.basic import Alloc, AllocEmpty, ARange, Join, TensorFromScalar
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
                if index.ndim > 1 or not index.type.dtype.startswith(("int", "uint")):
                    raise NotImplementedError(
                        "JS advanced indexing supports integer scalar/vector indices"
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


@js_funcify.register(IncSubtensor)
def js_funcify_inc_subtensor(op, node, inputs, slot):
    name = f"v{slot}"
    source, value = inputs[:2]
    indices = index_expression(op, node, inputs, 2)
    return JSCode(
        (
            f"const {name} = updateSubtensor(slot({slot}, {source}.s), {source}, {value}, {indices}, {str(op.set_instead_of_inc).lower()});",
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
    indices = index_expression(op, node, inputs, 2, advanced=True)
    flags = f"{str(op.set_instead_of_inc).lower()}, {str(op.ignore_duplicates).lower()}"
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
        *loop_lines(rank, [f"{name}.d[k++] = {source}.d[{source_address}];"]),
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
    rank = node.outputs[0].ndim
    name = f"v{slot}"
    arrays = inputs[1:]
    lines = [
        f"let {name};",
        "{",
        CODE_TOKEN.INDENT,
        f"const axis = scalarIndex(scalarValue({inputs[0]}), {rank});",
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
        "for (let i = 0; i < size(source.s); i++) {",
        CODE_TOKEN.INDENT,
        "const coordinates = coordinatesOf(i, source.s);",
        "coordinates[axis] += axisOffset;",
        f"{name}.d[broadcastAddress({name}, coordinates)] = source.d[flatAddress(source, i)];",
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
    left_address = (
        f"({a}.o || 0) + "
        + (f"i * {a}.t[0] + " if left_rank == 2 else "")
        + f"k * {a}.t[{left_rank - 1}]"
    )
    right_address = f"({b}.o || 0) + k * {b}.t[0]" + (
        f" + j * {b}.t[1]" if right_rank == 2 else ""
    )
    lines = [
        f"let {name};",
        "{",
        CODE_TOKEN.INDENT,
        f"if ({extent} !== {b}.s[0]) throw Error('dot shape mismatch');",
        f"{name} = slot({slot}, {output_shape});",
        "let position = 0;",
        f"for (let i = 0; i < {rows}; i++) {{",
        CODE_TOKEN.INDENT,
        f"for (let j = 0; j < {cols}; j++) {{",
        CODE_TOKEN.INDENT,
        "let acc = 0;",
        f"for (let k = 0; k < {extent}; k++) {{",
        CODE_TOKEN.INDENT,
        f"acc += {a}.d[{left_address}] * {b}.d[{right_address}];",
        CODE_TOKEN.DEDENT,
        "}",
        f"{name}.d[position++] = acc;",
        CODE_TOKEN.DEDENT,
        "}",
        CODE_TOKEN.DEDENT,
        "}",
        CODE_TOKEN.DEDENT,
        "}",
    ]
    return JSCode(tuple(lines), (name,))
