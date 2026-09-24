"""Graph-level JavaScript generation and input conversion."""

from dataclasses import dataclass
from functools import singledispatch

import numpy as np

from pytensor.compile.ops import DeepCopyOp
from pytensor.graph.basic import Constant
from pytensor.link.string_codegen import CODE_TOKEN, build_source_code


SUPPORTED_DTYPES = frozenset({"float64", "int32", "int64", "bool"})


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


@dataclass(frozen=True)
class JSCode:
    lines: tuple[str | CODE_TOKEN, ...]
    names: tuple[str, ...]


@singledispatch
def js_funcify(op, node, inputs, slot):
    """Emit code for one Apply node using already assigned input names."""
    raise NotImplementedError(f"JS Op {op} ({type(op).__name__}) is unsupported")


@js_funcify.register(DeepCopyOp)
def js_funcify_deep_copy(op, node, inputs, slot):
    # Node transport creates fresh output storage.
    return JSCode((), (inputs[0],))


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
    const value = index.d[i];
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
        record = {d: new Float64Array(n), s: shape, t: strides(shape)};
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
    for index, variable in enumerate(fgraph.inputs):
        check_dtype(variable)
        names[variable] = f"inputs[{index}]"

    next_slot = 0
    for node in fgraph.toposort():
        for variable in node.inputs:
            if isinstance(variable, Constant) and variable not in names:
                check_dtype(variable)
                constant_id = len(constants)
                constants.append(js_typify(variable.data, variable.type.dtype))
                names[variable] = f"constants[{constant_id}]"

        fragment = js_funcify(
            node.op, node, [names[variable] for variable in node.inputs], next_slot
        )
        if len(fragment.names) != len(node.outputs):
            raise ValueError("JS codegen returned the wrong number of outputs")
        lines.extend(fragment.lines)
        names.update(zip(node.outputs, fragment.names, strict=True))
        next_slot += len(node.outputs)

    for variable in fgraph.outputs:
        if isinstance(variable, Constant) and variable not in names:
            check_dtype(variable)
            constant_id = len(constants)
            constants.append(js_typify(variable.data, variable.type.dtype))
            names[variable] = f"constants[{constant_id}]"

    output_names = ", ".join(names[variable] for variable in fgraph.outputs)
    source = build_source_code(
        [
            _RUNTIME,
            "const constants = [];",
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
