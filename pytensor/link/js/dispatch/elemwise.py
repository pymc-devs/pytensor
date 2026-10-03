"""JavaScript loops for elementwise, reduced, and indexed tensor Ops."""

from pytensor.link.js.dispatch.basic import JSCode, check_dtype, js_funcify
from pytensor.link.js.dispatch.scalar import scalar_program
from pytensor.link.string_codegen import CODE_TOKEN
from pytensor.scalar.basic import Add
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
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
    return " + ".join(terms) or "0"


def stride_setup(variable, record, prefix):
    return [
        f"const {prefix}{axis} = ({record}.s[{axis}] === 1 ? 0 : {record}.t[{axis}]);"
        for axis in range(variable.ndim)
    ]


@js_funcify.register(Elemwise)
def js_funcify_elemwise(op, node, inputs, slot):
    if len(node.outputs) != 1:
        raise NotImplementedError("JS Elemwise with multiple outputs")
    check_dtype(node.outputs[0])
    rank = node.outputs[0].ndim
    statements, expressions = scalar_program(
        op.scalar_op, [f"a{index}" for index in range(len(inputs))]
    )
    if len(expressions) != 1:
        raise NotImplementedError("JS multi-output scalar Op")

    name = f"v{slot}"
    lines: list[str | CODE_TOKEN] = [
        f"let {name};",
        "{",
        CODE_TOKEN.INDENT,
        f"const shape = {shape_expr(inputs)};",
        f"{name} = slot({slot}, shape);",
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
    lines.extend(
        loop_lines(
            rank,
            [*reads, *statements, f"{name}.d[k++] = {expressions[0]};"],
        )
    )
    lines.extend([CODE_TOKEN.DEDENT, "}"])
    return JSCode(tuple(lines), (name,))


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
    source_address = (
        " + ".join(f"i{axis} * {source}.t[{axis}]" for axis in range(source_rank))
        or "0"
    )
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
        (f"const {name} = {{d: {source}.d, s: {shape}, t: {strides}}};",), (name,)
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
    indexed = [None] * scalar_input_count

    for index_id, spec in enumerate(op.indexed_inputs):
        if spec is None:
            continue
        sources, axis = spec
        if axis != 0 or node.inputs[scalar_input_count + index_id].ndim != 1:
            raise NotImplementedError("JS fused gather supports vector index on axis 0")
        for source_id in sources:
            if indexed[source_id] is not None:
                raise NotImplementedError("JS fused input with multiple index axes")
            indexed[source_id] = indices[index_id]

    if any(spec is not None for spec in op.indexed_outputs):
        raise NotImplementedError("JS fused indexed write")
    reduced = op.reduced_outputs or (None,) * len(node.outputs)
    if len(reduced) != len(node.outputs):
        raise NotImplementedError("JS fused reduction/output count mismatch")
    if any(spec is not None and not isinstance(spec[0], Add) for spec in reduced):
        raise NotImplementedError("JS fused reduction supports sum")
    rank = inner.outputs[0].ndim
    if rank != 1:
        raise NotImplementedError("JS FusedElemwise currently supports rank one")

    statements, expressions = scalar_program(
        inner.op.scalar_op, [f"a{index}" for index in range(scalar_input_count)]
    )
    if len(expressions) != len(node.outputs):
        raise NotImplementedError("JS fused scalar/output count mismatch")
    output_names = tuple(f"v{slot + index}" for index in range(len(node.outputs)))
    shape = f"{indices[0]}.s" if indices else shape_expr(operands)
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
        lines.extend(stride_setup(variable, record, f"st{index}_"))
    for index, (name, spec) in enumerate(zip(output_names, reduced, strict=True)):
        lines.append(f"{name} = slot({slot + index}, {'[]' if spec else 'shape'});")
        if spec:
            lines.append(f"{name}.d[0] = 0;")

    reads = []
    for index, (variable, record, index_record) in enumerate(
        zip(inner.inputs, operands, indexed, strict=True)
    ):
        coordinate = (
            {0: f"indexAt({index_record}, i0, {record}.s[0])"}
            if index_record is not None
            else None
        )
        offset = address(record, rank, variable.ndim, coordinate, f"st{index}_")
        reads.append(f"const a{index} = {record}.d[{offset}];")
    stores = [
        f"{name}.d[0] += {expression};" if spec else f"{name}.d[k] = {expression};"
        for name, expression, spec in zip(
            output_names, expressions, reduced, strict=True
        )
    ]
    lines.extend(loop_lines(rank, [*reads, *statements, *stores, "k++; "]))
    lines.extend([CODE_TOKEN.DEDENT, "}"])
    return JSCode(tuple(lines), output_names)
