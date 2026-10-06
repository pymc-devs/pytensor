"""JavaScript loops for elementwise, reduced, and indexed tensor Ops."""

from pytensor.link.js.dispatch.basic import (
    JSCode,
    check_dtype,
    js_funcify,
    lower_subgraph,
)
from pytensor.link.js.dispatch.scalar import scalar_program
from pytensor.link.string_codegen import CODE_TOKEN
from pytensor.raise_op import CheckAndRaise
from pytensor.scalar.basic import AND, Add
from pytensor.tensor.basic import MakeVector, ScalarFromTensor
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.math import All
from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise


def shape_expr(inputs):
    return "broadcast([" + ", ".join(f"{value}.s" for value in inputs) + "])"


def loop_lines(rank, body):
    lines = []
    for axis in range(rank):
        lines.extend(
            [
                f"for (let i{axis} = 0; i{axis} < shape[{axis}]; i{axis}++) {{",
                CODE_TOKEN.INDENT,
            ]
        )
    lines.extend(body)
    for _ in range(rank):
        lines.extend([CODE_TOKEN.DEDENT, "}"])
    return lines


def address(record, rank, own_rank=None, indexed=None, strides=None):
    if own_rank is None:
        own_rank = rank
    shift = rank - own_rank
    terms = []
    for axis in range(own_rank):
        loop_axis = axis + shift
        coordinate = indexed.get(axis, f"i{loop_axis}") if indexed else f"i{loop_axis}"
        if strides is None:
            terms.append(
                f"({record}.s[{axis}] === 1 ? 0 : {coordinate}) * {record}.t[{axis}]"
            )
        else:
            terms.append(f"{coordinate} * {strides}{axis}")
    offset = f"({record}.o || 0)" if strides is None else f"{strides}offset"
    return " + ".join([offset, *terms])


def stride_setup(variable, record, prefix):
    return [f"const {prefix}offset = {record}.o || 0;"] + [
        f"const {prefix}{axis} = ({record}.s[{axis}] === 1 ? 0 : {record}.t[{axis}]);"
        for axis in range(variable.ndim)
    ]


@js_funcify.register(Elemwise)
def js_funcify_elemwise(op, node, inputs, slot):
    for output in node.outputs:
        check_dtype(output)
    rank = node.outputs[0].ndim
    statements, expressions = scalar_program(
        op.scalar_op, [f"a{index}" for index in range(len(inputs))]
    )
    if len(expressions) != len(node.outputs):
        raise ValueError("JS scalar/output count mismatch")

    names = tuple(f"v{slot + index}" for index in range(len(node.outputs)))
    lines: list[str | CODE_TOKEN] = [
        *(f"let {name};" for name in names),
        "{",
        CODE_TOKEN.INDENT,
        f"const shape = {shape_expr(inputs)};",
        *(f"{name} = slot({slot + index}, shape);" for index, name in enumerate(names)),
        "let k = 0;",
    ]
    for index, (variable, record) in enumerate(zip(node.inputs, inputs, strict=True)):
        lines.extend(stride_setup(variable, record, f"st{index}_"))

    reads = [
        f"const a{index} = {record}.d[{address(record, rank, variable.ndim, strides=f'st{index}_')}];"
        for index, (variable, record) in enumerate(
            zip(node.inputs, inputs, strict=True)
        )
    ]
    stores = [
        f"{name}.d[k] = {'Math.fround(' + expression + ')' if output.type.dtype == 'float32' else expression};"
        for name, expression, output in zip(
            names, expressions, node.outputs, strict=True
        )
    ]
    lines.extend(loop_lines(rank, [*reads, *statements, *stores, "k++; "]))
    lines.extend([CODE_TOKEN.DEDENT, "}"])
    return JSCode(tuple(lines), names)


@js_funcify.register(CAReduce)
def js_funcify_reduce(op, node, inputs, slot):
    if not isinstance(op.scalar_op, Add):
        raise NotImplementedError(f"JS reduction {op.scalar_op} is unsupported")
    source = inputs[0]
    source_rank = node.inputs[0].ndim
    axes = op.axis if op.axis is not None else tuple(range(source_rank))
    axes = tuple(axis % source_rank for axis in axes)
    kept = [axis for axis in range(source_rank) if axis not in axes]
    output_shape = "[" + ", ".join(f"{source}.s[{axis}]" for axis in kept) + "]"
    source_address = address(source, source_rank, strides="st")
    target_address = (
        " + ".join(f"i{axis} * v{slot}.t[{index}]" for index, axis in enumerate(kept))
        or "0"
    )
    name = f"v{slot}"
    lines = [
        f"let {name};",
        "{",
        CODE_TOKEN.INDENT,
        f"{name} = slot({slot}, {output_shape});",
        f"{name}.d.fill(0);",
        f"const shape = {source}.s;",
        *stride_setup(node.inputs[0], source, "st"),
        *loop_lines(
            source_rank,
            [f"{name}.d[{target_address}] += {source}.d[{source_address}];"],
        ),
        CODE_TOKEN.DEDENT,
        "}",
    ]
    return JSCode(tuple(lines), (name,))


@js_funcify.register(DimShuffle)
def js_funcify_dimshuffle(op, node, inputs, slot):
    source = inputs[0]
    shape = (
        "["
        + ", ".join(
            "1" if axis == "x" else f"{source}.s[{axis}]" for axis in op.new_order
        )
        + "]"
    )
    strides = (
        "["
        + ", ".join(
            "0" if axis == "x" else f"{source}.t[{axis}]" for axis in op.new_order
        )
        + "]"
    )
    name = f"v{slot}"
    return JSCode(
        (
            f"const {name} = {{d: {source}.d, s: {shape}, t: {strides}, o: {source}.o || 0}};",
        ),
        (name,),
    )


@js_funcify.register(FusedElemwise)
def js_funcify_fused_elemwise(op, node, inputs, slot):
    scalar_nodes = [
        inner for inner in op.fgraph.apply_nodes if isinstance(inner.op, Elemwise)
    ]
    if len(scalar_nodes) != 1:
        raise NotImplementedError("JS FusedElemwise requires one Elemwise")
    inner = scalar_nodes[0]
    scalar_input_count = len(inner.inputs)
    operands = inputs[:scalar_input_count]
    indices = inputs[scalar_input_count : scalar_input_count + len(op.indexed_inputs)]
    indexed = [{} for _ in range(scalar_input_count)]

    for index_id, spec in enumerate(op.indexed_inputs):
        if spec is None:
            continue
        sources, axis = spec
        if node.inputs[scalar_input_count + index_id].ndim != 1:
            raise NotImplementedError("JS fused gather supports vector indices")
        for source_id in sources:
            indexed[source_id][axis] = indices[index_id]
            if len(indexed[source_id]) > 1:
                return lower_subgraph(op.fgraph, inputs, slot)

    writes = {}
    target_position = scalar_input_count + len(op.indexed_inputs)
    for index_id, spec in enumerate(op.indexed_outputs):
        if spec is None:
            continue
        sources, axis, mode = spec
        for output_id in sources:
            if output_id in writes:
                return lower_subgraph(op.fgraph, inputs, slot)
            writes[output_id] = (inputs[target_position], indices[index_id], axis, mode)
        target_position += 1
    reduced = op.reduced_outputs or (None,) * len(node.outputs)
    if len(reduced) != len(node.outputs):
        raise NotImplementedError("JS fused reduction/output count mismatch")
    if any(
        spec is not None and not isinstance(spec[0], (Add, AND)) for spec in reduced
    ):
        raise NotImplementedError("JS fused reduction supports sum or all")
    rank = inner.outputs[0].ndim
    if any(node.outputs[index].ndim != rank for index in writes):
        return lower_subgraph(op.fgraph, inputs, slot)

    statements, expressions = scalar_program(
        inner.op.scalar_op, [f"a{index}" for index in range(scalar_input_count)]
    )
    if len(expressions) != len(node.outputs):
        raise NotImplementedError("JS fused scalar/output count mismatch")
    output_names = tuple(f"v{slot + index}" for index in range(len(node.outputs)))
    shapes = []
    for record, idx_axes in zip(operands, indexed, strict=True):
        if idx_axes:
            choices = " : ".join(
                f"axis === {axis} ? {index}.s[0]" for axis, index in idx_axes.items()
            )
            shapes.append(f"{record}.s.map((length, axis) => {choices} : length)")
        else:
            shapes.append(f"{record}.s")
    shape = f"broadcast([{', '.join(shapes)}])"
    lines: list[str | CODE_TOKEN] = [
        *(f"let {name};" for name in output_names),
        "{",
        CODE_TOKEN.INDENT,
        f"const shape = {shape};",
        "let k = 0;",
    ]
    for index, (variable, record) in enumerate(
        zip(inner.inputs, operands, strict=True)
    ):
        lines.append(f"const d{index} = {record}.d;")
        lines.extend(stride_setup(variable, record, f"st{index}_"))
    scalar_reductions = set()
    kept_axes = {}
    for index, (name, spec) in enumerate(zip(output_names, reduced, strict=True)):
        if index in writes:
            target, _, _, _ = writes[index]
            lines.append(
                f"{name} = copyRecord(slot({slot + index}, {target}.s), {target});"
            )
        elif spec:
            kept = tuple(axis for axis in range(rank) if axis not in spec[1])
            kept_axes[index] = kept
            output_shape = "[" + ", ".join(f"shape[{axis}]" for axis in kept) + "]"
            lines.append(f"{name} = slot({slot + index}, {output_shape});")
            identity = 1 if isinstance(spec[0], AND) else 0
            if not kept:
                scalar_reductions.add(index)
                lines.append(f"let acc{index} = {identity};")
            else:
                lines.append(f"{name}.d.fill({identity});")
        else:
            lines.append(f"{name} = slot({slot + index}, shape);")

    reads = []
    hoisted_reads = []
    for index, (variable, record, index_axes) in enumerate(
        zip(inner.inputs, operands, indexed, strict=True)
    ):
        coordinate = {
            axis: f"indexAt({index_record}, i{rank - variable.ndim + axis}, {record}.s[{axis}])"
            for axis, index_record in index_axes.items()
        }
        offset = address(record, rank, variable.ndim, coordinate, f"st{index}_")
        read = f"const a{index} = d{index}[{offset}];"
        if not index_axes and all(dim == 1 for dim in variable.type.shape):
            hoisted_reads.append(f"const a{index} = d{index}[st{index}_offset];")
        else:
            reads.append(read)
    stores = []
    for index, (name, expression, spec, output) in enumerate(
        zip(output_names, expressions, reduced, node.outputs, strict=True)
    ):
        if output.type.dtype == "float32":
            expression = f"Math.fround({expression})"
        if index in writes:
            target, index_record, axis, mode = writes[index]
            coordinate = {
                axis: f"indexAt({index_record}, i{rank - output.ndim + axis}, {target}.s[{axis}])"
            }
            target_address = address(name, rank, output.ndim, coordinate)
            operator = "+=" if mode == "inc" else "="
            stores.append(f"{name}.d[{target_address}] {operator} {expression};")
        elif spec:
            operator = "&&" if isinstance(spec[0], AND) else "+"
            if index in scalar_reductions:
                target = f"acc{index}"
            else:
                offset = " + ".join(
                    f"i{axis} * {name}.t[{position}]"
                    for position, axis in enumerate(kept_axes[index])
                )
                target = f"{name}.d[{offset}]"
            stores.append(f"{target} {operator}= {expression};")
        else:
            stores.append(f"{name}.d[k] = {expression};")
    lines.extend(hoisted_reads)
    lines.extend(loop_lines(rank, [*reads, *statements, *stores, "k++; "]))
    lines.extend(
        f"{name}.d[0] = acc{index};"
        for index, (name, spec) in enumerate(zip(output_names, reduced, strict=True))
        if index in scalar_reductions
    )
    lines.extend([CODE_TOKEN.DEDENT, "}"])
    return JSCode(tuple(lines), output_names)


@js_funcify.register(MakeVector)
def js_funcify_make_vector(op, node, inputs, slot):
    name = f"v{slot}"
    lines = [
        f"let {name};",
        "{",
        CODE_TOKEN.INDENT,
        f"{name} = slot({slot}, [{len(inputs)}]);",
    ]
    lines.extend(
        f"{name}.d[{i}] = scalarValue({value});" for i, value in enumerate(inputs)
    )
    lines.extend([CODE_TOKEN.DEDENT, "}"])
    return JSCode(tuple(lines), (name,))


@js_funcify.register(All)
def js_funcify_all(op, node, inputs, slot):
    if op.axis is not None:
        raise NotImplementedError("JS All only supports all axes")
    name = f"v{slot}"
    source = inputs[0]
    return JSCode(
        (
            f"const {name} = slot({slot}, []);",
            f"{name}.d[0] = 1;",
            f"for (let j = 0; j < size({source}.s); j++) {name}.d[0] &&= {source}.d[flatAddress({source}, j)];",
        ),
        (name,),
    )


@js_funcify.register(ScalarFromTensor)
def js_funcify_scalar_from_tensor(op, node, inputs, slot):
    return JSCode((), (inputs[0],))


@js_funcify.register(CheckAndRaise)
def js_funcify_check_and_raise(op, node, inputs, slot):
    checks = " && ".join(f"scalarValue({value})" for value in inputs[1:])
    return JSCode(
        (f"if (!({checks})) throw Error('PyTensor parameter check failed');",),
        (inputs[0],),
    )
