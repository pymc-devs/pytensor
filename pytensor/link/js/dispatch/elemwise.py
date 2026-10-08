"""JavaScript loops for elementwise, reduced, and indexed tensor Ops."""

from collections import Counter
from itertools import product
from math import prod

from pytensor.link.js.dispatch.basic import (
    JSCode,
    check_dtype,
    js_funcify,
    lower_subgraph,
)
from pytensor.link.js.dispatch.scalar import scalar_program
from pytensor.link.string_codegen import CODE_TOKEN
from pytensor.raise_op import CheckAndRaise
from pytensor.scalar.basic import AND, Add, Composite, Switch
from pytensor.tensor.basic import MakeVector, ScalarFromTensor
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.math import All
from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise


def shape_expr(inputs):
    return "broadcast([" + ", ".join(f"{value}.s" for value in inputs) + "])"


def loop_lines(rank, body, setup=None, axes=None, static_shape=None):
    axes = tuple(range(rank)) if axes is None else axes
    unrolled = {}
    copies = 1
    if static_shape is not None:
        for axis in reversed(axes):
            length = static_shape[axis]
            if length is not None and 0 < length <= 4 and copies * length <= 16:
                unrolled[axis] = length
                copies *= length
    lines = list(body)
    for axis in reversed(axes):
        inner = [*(setup.get(axis, ()) if setup else ()), *lines]
        if axis in unrolled:
            lines = [
                line
                for coordinate in range(unrolled[axis])
                for line in (
                    "{",
                    CODE_TOKEN.INDENT,
                    f"const i{axis} = {coordinate};",
                    *inner,
                    CODE_TOKEN.DEDENT,
                    "}",
                )
            ]
        else:
            lines = [
                f"const n{axis} = shape[{axis}];",
                f"for (let i{axis} = 0; i{axis} < n{axis}; i{axis}++) {{",
                CODE_TOKEN.INDENT,
                *inner,
                CODE_TOKEN.DEDENT,
                "}",
            ]
    return lines


def fused_scalar_loop(
    rank,
    scalar_op,
    variables,
    reads,
    stores,
    setup=None,
    axes=None,
    increment=True,
    static_shape=None,
    tiles=None,
):
    """Version a loop on one broadcast scalar condition, without changing its graph."""
    arguments = [f"a{index}" for index in range(len(variables))]
    choices = []
    if rank and isinstance(scalar_op, Composite):
        switches = Counter(
            node.inputs[0]
            for node in scalar_op.fgraph.toposort()
            if isinstance(node.op, Switch)
        )
        choices = [
            (switches[condition], index, condition)
            for index, (condition, variable) in enumerate(
                zip(scalar_op.inputs, variables, strict=True)
            )
            if switches[condition] and all(dim == 1 for dim in variable.type.shape)
        ]

    def emit(switch_choices=None):
        statements, expressions = scalar_program(
            scalar_op, arguments, switch_choices=switch_choices
        )
        writes = [
            template.replace("{value}", expression)
            for template, expression in zip(stores, expressions, strict=True)
        ]
        if tiles is not None:
            shared_reads = {}
            tiled_body = []
            for tile, coordinates in enumerate(tiles):
                aliases = []
                for identifier, expression in reads:
                    for axis, coordinate in coordinates:
                        expression = expression.replace(f"i{axis} *", f"{coordinate} *")
                    if expression not in shared_reads:
                        shared_reads[expression] = f"tileRead{len(shared_reads)}"
                    aliases.append(f"const {identifier} = {shared_reads[expression]};")
                tiled_body.extend(
                    [
                        "{",
                        CODE_TOKEN.INDENT,
                        *aliases,
                        *statements,
                        *(line.replace("{tile}", str(tile)) for line in writes),
                        CODE_TOKEN.DEDENT,
                        "}",
                    ]
                )
            body = [
                *(
                    f"const {name} = {expression};"
                    for expression, name in shared_reads.items()
                ),
                *tiled_body,
            ]
            return loop_lines(rank, body, setup, axes, static_shape)
        counter = ["k++; "] if increment else []
        input_reads = [f"const {name} = {expression};" for name, expression in reads]
        return loop_lines(
            rank,
            [*input_reads, *statements, *writes, *counter],
            setup,
            axes,
            static_shape,
        )

    if not choices:
        return emit()
    _, index, condition = max(choices, key=lambda choice: choice[0])
    return [
        f"if (a{index} !== 0 && a{index} !== false) {{",
        CODE_TOKEN.INDENT,
        *emit({condition: True}),
        CODE_TOKEN.DEDENT,
        "} else {",
        CODE_TOKEN.INDENT,
        *emit({condition: False}),
        CODE_TOKEN.DEDENT,
        "}",
    ]


def address(
    record, rank, own_rank=None, indexed=None, strides=None, broadcastable=None
):
    if own_rank is None:
        own_rank = rank
    shift = rank - own_rank
    terms = []
    for axis in range(own_rank):
        if broadcastable is not None and broadcastable[axis]:
            continue
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
        f"const {prefix}{axis} = "
        + (
            "0;"
            if variable.type.broadcastable[axis]
            else f"({record}.s[{axis}] === 1 ? 0 : {record}.t[{axis}]);"
        )
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
    allocations = []
    for index, name in enumerate(names):
        if index in op.inplace_pattern:
            target = inputs[op.inplace_pattern[index]]
            allocations.append(f"{name} = inplaceRecord({target}, shape);")
        else:
            allocations.append(f"{name} = slot({slot + index}, shape);")
    lines: list[str | CODE_TOKEN] = [
        *(f"let {name};" for name in names),
        "{",
        CODE_TOKEN.INDENT,
        f"const shape = {shape_expr(inputs)};",
        *allocations,
        "let k = 0;",
    ]
    for index, (variable, record) in enumerate(zip(node.inputs, inputs, strict=True)):
        lines.append(f"const d{index} = {record}.d;")
        lines.extend(stride_setup(variable, record, f"st{index}_"))

    reads = [
        f"const a{index} = d{index}[{address(record, rank, variable.ndim, strides=f'st{index}_', broadcastable=variable.type.broadcastable)}];"
        for index, (variable, record) in enumerate(
            zip(node.inputs, inputs, strict=True)
        )
    ]
    stores = []
    for index, (name, expression, output) in enumerate(
        zip(names, expressions, node.outputs, strict=True)
    ):
        lines.append(f"const out{index} = {name}.d;")
        if output.type.dtype == "float32":
            expression = f"Math.fround({expression})"
        if index in op.inplace_pattern:
            lines.extend(stride_setup(output, name, f"outst{index}_"))
            offset = address(name, rank, strides=f"outst{index}_")
        else:
            offset = "k"
        stores.append(f"out{index}[{offset}] = {expression};")
    lines.extend(
        loop_lines(
            rank,
            [*reads, *statements, *stores, "k++; "],
            static_shape=node.outputs[0].type.shape,
        )
    )
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
    axes = tuple(axis for axis in range(source_rank) if axis in axes)
    kept = [axis for axis in range(source_rank) if axis not in axes]
    output_shape = "[" + ", ".join(f"{source}.s[{axis}]" for axis in kept) + "]"
    source_address = address(source, source_rank, strides="st")
    name = f"v{slot}"
    lines = [
        f"let {name};",
        "{",
        CODE_TOKEN.INDENT,
        f"{name} = slot({slot}, {output_shape});",
        f"const out = {name}.d, data = {source}.d;",
        f"const shape = {source}.s;",
        *stride_setup(node.inputs[0], source, "st"),
        "let k = 0;",
        *loop_lines(
            source_rank,
            [
                "let acc = 0;",
                *loop_lines(
                    source_rank,
                    [f"acc += data[{source_address}];"],
                    axes=axes,
                    static_shape=node.inputs[0].type.shape,
                ),
                "out[k++] = acc;",
            ],
            axes=kept,
            static_shape=node.inputs[0].type.shape,
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

    if len(inner.outputs) != len(node.outputs):
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
    local_axes = None
    if reduced and all(reduced) and not writes and not any(op.indexed_inputs):
        axes = reduced[0][1]
        if 0 < len(axes) < rank and all(spec[1] == axes for spec in reduced):
            local_axes = axes
    tiles = None
    if local_axes is not None:
        kept = tuple(axis for axis in range(rank) if axis not in local_axes)
        dimensions = [inner.outputs[0].type.shape[axis] for axis in kept]
        if (
            all(length is not None and length > 0 for length in dimensions)
            and prod(dimensions) <= 16
        ):
            tiles = [
                tuple(zip(kept, coordinates, strict=True))
                for coordinates in product(*(range(length) for length in dimensions))
            ]
    local_initializers = []
    for index, (name, spec) in enumerate(zip(output_names, reduced, strict=True)):
        if index in writes:
            target, _, _, _ = writes[index]
            if index in op.destroy_map:
                lines.append(f"{name} = {target};")
            else:
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
            elif local_axes is not None:
                if tiles is None:
                    local_initializers.append(f"let acc{index} = {identity};")
                else:
                    lines.extend(
                        f"let acc{index}_{tile} = {identity};"
                        for tile in range(len(tiles))
                    )
            else:
                lines.append(f"{name}.d.fill({identity});")
        elif index in op.destroy_map:
            target = inputs[op.destroy_map[index][0]]
            lines.append(f"{name} = inplaceRecord({target}, shape);")
            lines.extend(stride_setup(node.outputs[index], name, f"outst{index}_"))
        else:
            lines.append(f"{name} = slot({slot + index}, shape);")

    reads = []
    hoisted_reads = []
    index_loads = {}
    loop_setup = {}

    def indexed_coordinate(index_record, axis, bound):
        key = (index_record, axis, bound)
        if key not in index_loads:
            name = f"idx{len(index_loads)}"
            index_loads[key] = name
            trailing = " && ".join(
                f"shape[{later}] > 0" for later in range(axis + 1, rank)
            )
            expression = f"indexAt({index_record}, i{axis}, {bound})"
            if trailing:
                expression = f"({trailing}) ? {expression} : 0"
            loop_setup.setdefault(axis, []).append(f"const {name} = {expression};")
        return index_loads[key]

    for index, (variable, record, index_axes) in enumerate(
        zip(inner.inputs, operands, indexed, strict=True)
    ):
        coordinate = {
            axis: indexed_coordinate(
                index_record, rank - variable.ndim + axis, f"{record}.s[{axis}]"
            )
            for axis, index_record in index_axes.items()
        }
        offset = address(
            record,
            rank,
            variable.ndim,
            coordinate,
            f"st{index}_",
            variable.type.broadcastable,
        )
        if not index_axes and all(dim == 1 for dim in variable.type.shape):
            hoisted_reads.append(f"const a{index} = d{index}[st{index}_offset];")
        else:
            reads.append((f"a{index}", f"d{index}[{offset}]"))
    stores = []
    lines.extend(
        f"const out{index} = {name}.d;" for index, name in enumerate(output_names)
    )
    for index, (name, spec, output) in enumerate(
        zip(output_names, reduced, node.outputs, strict=True)
    ):
        expression = "{value}"
        if output.type.dtype == "float32":
            expression = f"Math.fround({expression})"
        if index in writes:
            target, index_record, axis, mode = writes[index]
            coordinate = {
                axis: indexed_coordinate(
                    index_record, rank - output.ndim + axis, f"{target}.s[{axis}]"
                )
            }
            lines.extend(stride_setup(output, name, f"outst{index}_"))
            target_address = address(
                name, rank, output.ndim, coordinate, f"outst{index}_"
            )
            operator = "+=" if mode == "inc" else "="
            stores.append(f"out{index}[{target_address}] {operator} {expression};")
        elif spec:
            operator = "&&" if isinstance(spec[0], AND) else "+"
            if index in scalar_reductions or local_axes is not None:
                target = f"acc{index}"
                if tiles is not None:
                    target += "_{tile}"
            else:
                offset = " + ".join(
                    f"i{axis} * {name}.t[{position}]"
                    for position, axis in enumerate(kept_axes[index])
                )
                target = f"out{index}[{offset}]"
            stores.append(f"{target} {operator}= {expression};")
        else:
            offset = (
                address(name, rank, strides=f"outst{index}_")
                if index in op.destroy_map
                else "k"
            )
            stores.append(f"out{index}[{offset}] = {expression};")
    lines.extend(hoisted_reads)
    if local_axes is None:
        lines.extend(
            fused_scalar_loop(
                rank,
                inner.op.scalar_op,
                inner.inputs,
                reads,
                stores,
                loop_setup,
                static_shape=inner.outputs[0].type.shape,
            )
        )
    elif tiles is not None:
        lines.extend(
            fused_scalar_loop(
                rank,
                inner.op.scalar_op,
                inner.inputs,
                reads,
                stores,
                axes=local_axes,
                increment=False,
                static_shape=inner.outputs[0].type.shape,
                tiles=tiles,
            )
        )
        lines.extend(
            f"out{index}[{tile}] = acc{index}_{tile};"
            for index in range(len(output_names))
            for tile in range(len(tiles))
        )
    else:
        kept = tuple(axis for axis in range(rank) if axis not in local_axes)
        post = []
        for index, name in enumerate(output_names):
            offset = " + ".join(
                f"i{axis} * {name}.t[{position}]" for position, axis in enumerate(kept)
            )
            post.append(f"out{index}[{offset}] = acc{index};")
        reduced_loop = fused_scalar_loop(
            rank,
            inner.op.scalar_op,
            inner.inputs,
            reads,
            stores,
            axes=local_axes,
            increment=False,
            static_shape=inner.outputs[0].type.shape,
        )
        lines.extend(
            loop_lines(
                rank,
                [*local_initializers, *reduced_loop, *post],
                axes=kept,
                static_shape=inner.outputs[0].type.shape,
            )
        )
    lines.extend(
        f"out{index}[0] = acc{index};"
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
