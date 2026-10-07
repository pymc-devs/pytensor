"""Graph-level JavaScript generation and input conversion."""

from dataclasses import dataclass
from functools import singledispatch
from pathlib import Path

import numpy as np

from pytensor.compile.ops import DeepCopyOp
from pytensor.graph.basic import Constant
from pytensor.link.string_codegen import CODE_TOKEN, build_source_code


SUPPORTED_DTYPES = frozenset(
    {
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "bool",
    }
)


def check_dtype(var):
    dtype = var.type.dtype
    if dtype not in SUPPORTED_DTYPES:
        raise NotImplementedError(f"JS backend dtype {dtype} is unsupported")
    return dtype


@singledispatch
def js_typify(data, dtype):
    """Represent a supported PyTensor value in the JS float64 transport."""
    return js_typify(np.asarray(data), dtype)


@js_typify.register(np.ndarray)
def js_typify_ndarray(data, dtype):
    if dtype not in SUPPORTED_DTYPES:
        raise NotImplementedError(f"JS backend dtype {dtype} is unsupported")
    if dtype == "int64" and (np.any(data > 2**53 - 1) or np.any(data < -(2**53 - 1))):
        raise ValueError("JS int64 value exceeds the exact Number range")
    return np.asarray(data, dtype="float64", order="C")


def js_constant(data, dtype):
    """Use compact integer storage for constants without changing input transport."""
    array = js_typify(data, dtype)
    if (
        array.ndim > 0
        and np.dtype(dtype).kind in "iu"
        and np.all(array >= -(2**31))
        and np.all(array <= 2**31 - 1)
    ):
        return array.astype("int32")
    return array


@dataclass(frozen=True)
class JSCode:
    lines: tuple[str | CODE_TOKEN, ...]
    names: tuple[str, ...]
    slots: int | None = None


@singledispatch
def js_funcify(op, node, inputs, slot):
    """Emit code for one Apply node using already assigned input names."""
    raise NotImplementedError(f"JS Op {op} ({type(op).__name__}) is unsupported")


@js_funcify.register(DeepCopyOp)
def js_funcify_deep_copy(op, node, inputs, slot):
    name = f"v{slot}"
    source = inputs[0]
    return JSCode(
        (f"const {name} = copyRecord(slot({slot}, {source}.s), {source});",),
        (name,),
    )


@dataclass(frozen=True)
class JSProgram:
    source: str
    constants: tuple
    inputs: tuple
    outputs: tuple


_RUNTIME = """
function strides(shape) {
    let stride = 1;
    const result = Array(shape.length);
    for (let axis = shape.length - 1; axis >= 0; axis--) {
        result[axis] = stride;
        stride *= shape[axis];
    }
    return result;
}
function size(shape) { return shape.reduce((a, b) => a * b, 1); }
function broadcast(shapes) {
    const rank = Math.max(...shapes.map(s => s.length));
    const out = Array(rank).fill(1);
    for (const shape of shapes) {
        for (let axis = 0; axis < shape.length; axis++) {
            const k = rank - shape.length + axis;
            const value = shape[axis];
            if (out[k] !== 1 && value !== 1 && out[k] !== value)
                throw Error('broadcast shape mismatch');
            if (value !== 1) out[k] = value;
        }
    }
    return out;
}
function indexAt(index, i, bound) {
    const value = index.d[(index.o || 0) + (index.s[0] === 1 ? 0 : i) * index.t[0]];
    if (!Number.isSafeInteger(value)) throw Error('unsafe index');
    const coordinate = value < 0 ? value + bound : value;
    if (coordinate < 0 || coordinate >= bound) throw Error('index out of bounds');
    return coordinate;
}
const pool = [];
function slot(id, shape) {
    const n = size(shape);
    let record = pool[id];
    if (!record || record.d.length !== n) {
        record = {d: new Float64Array(n), s: shape, t: strides(shape), o: 0};
        pool[id] = record;
    } else {
        record.s = shape;
        record.t = strides(shape);
    }
    return record;
}
const inputs = [];
function setInputShape(i, shape) { inputs[i] = slot(-1 - i, shape); }
let outputs = [];
"""


def lower(fgraph):
    """Lower a rewritten FunctionGraph, leaving axis lengths dynamic."""
    names = {}
    constants = []
    lines: list[str | CODE_TOKEN] = []
    kernels: list[str | CODE_TOKEN] = []
    for index, variable in enumerate(fgraph.inputs):
        check_dtype(variable)
        names[variable] = f"inputs[{index}]"

    next_slot = 0
    for node_index, node in enumerate(fgraph.toposort()):
        for variable in node.inputs:
            if isinstance(variable, Constant) and variable not in names:
                if variable.data is None:
                    names[variable] = "null"
                    continue
                check_dtype(variable)
                constant_id = len(constants)
                constants.append(js_constant(variable.data, variable.type.dtype))
                names[variable] = f"constants[{constant_id}]"

        arguments = [f"arg{index}" for index in range(len(node.inputs))]
        fragment = js_funcify(node.op, node, arguments, next_slot)
        if len(fragment.names) != len(node.outputs):
            raise ValueError("JS codegen returned the wrong number of outputs")
        function = f"kernel{node_index}"
        if len(node.outputs) > 1:
            kernels.append(f"const {function}Outputs = [];")
        kernels.extend(
            [
                f"function {function}({', '.join(arguments)}) {{",
                CODE_TOKEN.INDENT,
                *fragment.lines,
            ]
        )
        if len(node.outputs) == 1:
            kernels.append(f"return {fragment.names[0]};")
        else:
            kernels.extend(
                f"{function}Outputs[{i}] = {value};"
                for i, value in enumerate(fragment.names)
            )
            kernels.append(f"return {function}Outputs;")
        kernels.extend([CODE_TOKEN.DEDENT, "}"])
        outputs = [f"v{node_index}_{index}" for index in range(len(node.outputs))]
        assignment = outputs[0] if len(outputs) == 1 else f"[{', '.join(outputs)}]"
        values = ", ".join(names[variable] for variable in node.inputs)
        lines.append(f"const {assignment} = {function}({values});")
        names.update(zip(node.outputs, outputs, strict=True))
        next_slot += fragment.slots if fragment.slots is not None else len(node.outputs)

    for variable in fgraph.outputs:
        if isinstance(variable, Constant) and variable not in names:
            check_dtype(variable)
            constant_id = len(constants)
            constants.append(js_constant(variable.data, variable.type.dtype))
            names[variable] = f"constants[{constant_id}]"

    output_names = ", ".join(names[variable] for variable in fgraph.outputs)
    source = build_source_code(
        [
            _RUNTIME,
            Path(__file__).parents[1].joinpath("special.js").read_text(),
            Path(__file__).parents[1].joinpath("runtime.js").read_text(),
            "const constants = [];",
            *kernels,
            "function execute() {",
            CODE_TOKEN.INDENT,
            *lines,
            f"outputs = [{output_names}];",
            CODE_TOKEN.DEDENT,
            "}",
            "function run() { execute(); return JSON.stringify(outputs.map(o => o.s)); }",
            "function repeat(n) {",
            CODE_TOKEN.INDENT,
            "for (let i = 0; i < n; i++) execute();",
            "return outputs[0].d[0];",
            CODE_TOKEN.DEDENT,
            "}",
            "globalThis.__ptjs = {inputs, constants, outputs: () => outputs, setInputShape, execute, run, repeat};",
        ]
    )
    return JSProgram(
        source, tuple(constants), tuple(fgraph.inputs), tuple(fgraph.outputs)
    )


def lower_subgraph(fgraph, inputs, slot):
    """Inline an inner graph when an indexed fusion pattern has no fused emitter."""
    from pytensor.link.js.dispatch.scalar import literal

    names = dict(zip(fgraph.inputs, inputs, strict=True))
    start = slot
    lines = []
    for node in fgraph.toposort():
        for variable in node.inputs:
            if variable in names or not isinstance(variable, Constant):
                continue
            if variable.data is None:
                names[variable] = "null"
                continue
            value = js_constant(variable.data, check_dtype(variable))
            name = f"c{slot}_{len(names)}"
            shape = list(value.shape)
            strides = [int(step // value.itemsize) for step in value.strides]
            data = ", ".join(literal(v) for v in value.ravel())
            constructor = "Int32Array" if value.dtype == "int32" else "Float64Array"
            lines.append(
                f"const {name} = {{d: new {constructor}([{data}]), s: {shape}, t: {strides}, o: 0}};"
            )
            names[variable] = name
        fragment = js_funcify(node.op, node, [names[v] for v in node.inputs], slot)
        lines.extend(fragment.lines)
        names.update(zip(node.outputs, fragment.names, strict=True))
        slot += fragment.slots if fragment.slots is not None else len(node.outputs)
    return JSCode(tuple(lines), tuple(names[v] for v in fgraph.outputs), slot - start)
