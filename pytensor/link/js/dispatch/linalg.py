"""Small dense linear algebra kernels for the JavaScript compatibility backend."""

from pytensor.link.js.dispatch.basic import JSCode, js_funcify
from pytensor.link.string_codegen import CODE_TOKEN
from pytensor.tensor.basic import AllocDiag, ExtractDiag
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.solvers.triangular import SolveTriangular


@js_funcify.register(Cholesky)
def js_funcify_cholesky(op, node, inputs, slot):
    source = inputs[0]
    if node.inputs[0].ndim != 2:
        raise NotImplementedError("JS Cholesky supports matrices")
    name = f"v{slot}"

    def out(i, j):
        return f"{name}.d[{i} * n + {j}]" if op.lower else f"{name}.d[{j} * n + {i}]"

    matrix_address = (
        f"i * {source}.t[0] + j * {source}.t[1]"
        if op.lower
        else f"j * {source}.t[0] + i * {source}.t[1]"
    )
    failure = f"{name}.d.fill(NaN);"
    lines = [
        f"let {name};",
        "{",
        CODE_TOKEN.INDENT,
        f"const n = {source}.s[0];",
        f"if (n !== {source}.s[1]) throw Error('Cholesky requires a square matrix');",
        f"{name} = slot({slot}, {source}.s);",
        f"{name}.d.fill(0);",
        "let failed = false;",
        "for (let i = 0; i < n && !failed; i++) {",
        CODE_TOKEN.INDENT,
        "for (let j = 0; j <= i; j++) {",
        CODE_TOKEN.INDENT,
        f"let value = {source}.d[({source}.o || 0) + {matrix_address}];",
        f"for (let k = 0; k < j; k++) value -= {out('i', 'k')} * {out('j', 'k')};",
        "if (i === j) {",
        CODE_TOKEN.INDENT,
        f"if (!(value > 0)) {{ {failure} failed = true; break; }}",
        f"{out('i', 'j')} = Math.sqrt(value);",
        CODE_TOKEN.DEDENT,
        "} else {",
        CODE_TOKEN.INDENT,
        f"{out('i', 'j')} = value / {out('j', 'j')};",
        CODE_TOKEN.DEDENT,
        "}",
        CODE_TOKEN.DEDENT,
        "}",
        CODE_TOKEN.DEDENT,
        "}",
        CODE_TOKEN.DEDENT,
        "}",
    ]
    return JSCode(tuple(lines), (name,))


@js_funcify.register(SolveTriangular)
def js_funcify_solve_triangular(op, node, inputs, slot):
    a, b = inputs
    if node.inputs[0].ndim != 2 or node.inputs[1].ndim not in (1, 2):
        raise NotImplementedError(
            "JS triangular solve supports matrix/vector right-hand sides"
        )
    name = f"v{slot}"
    rank = node.inputs[1].ndim
    cols = f"{b}.s[1]" if rank == 2 else "1"
    b_address = f"({b}.o || 0) + i * {b}.t[0]" + (
        f" + j * {b}.t[1]" if rank == 2 else ""
    )
    lower = op.lower
    i = "step" if lower else "n - 1 - step"
    range_k = "let k = 0; k < i; k++" if lower else "let k = i + 1; k < n; k++"
    diagonal = (
        "1" if op.unit_diagonal else f"{a}.d[({a}.o || 0) + i * ({a}.t[0] + {a}.t[1])]"
    )
    lines = [
        f"let {name};",
        "{",
        CODE_TOKEN.INDENT,
        f"const n = {a}.s[0], cols = {cols};",
        f"if (n !== {a}.s[1] || n !== {b}.s[0]) throw Error('triangular solve shape mismatch');",
        f"{name} = slot({slot}, {b}.s);",
        f"const out = {name}.d;",
        "for (let step = 0; step < n; step++) {",
        CODE_TOKEN.INDENT,
        f"const i = {i};",
        "const row = i * cols;",
        f"for (let j = 0; j < cols; j++) out[row + j] = {b}.d[{b_address}];",
        f"for ({range_k}) {{",
        CODE_TOKEN.INDENT,
        f"const coefficient = {a}.d[({a}.o || 0) + i * {a}.t[0] + k * {a}.t[1]];",
        "const previous = k * cols;",
        "for (let j = 0; j < cols; j++) out[row + j] -= coefficient * out[previous + j];",
        CODE_TOKEN.DEDENT,
        "}",
        f"const diagonal = {diagonal};",
        "if (diagonal === 0) { out.fill(NaN); break; }",
        "for (let j = 0; j < cols; j++) out[row + j] /= diagonal;",
        CODE_TOKEN.DEDENT,
        "}",
        CODE_TOKEN.DEDENT,
        "}",
    ]
    return JSCode(tuple(lines), (name,))


@js_funcify.register(ExtractDiag)
def js_funcify_extract_diag(op, node, inputs, slot):
    source = inputs[0]
    rank = node.inputs[0].ndim
    a, b = op.axis1 % rank, op.axis2 % rank
    if a == b:
        raise NotImplementedError("JS diagonal requires distinct axes")
    kept = [axis for axis in range(rank) if axis not in (a, b)]
    offset = max(0, -op.offset)
    second = max(0, op.offset)
    length = (
        f"Math.max(0, Math.min({source}.s[{a}] - {offset}, {source}.s[{b}] - {second}))"
    )
    shape = "[" + ", ".join([*(f"{source}.s[{axis}]" for axis in kept), length]) + "]"
    strides = (
        "["
        + ", ".join(
            [
                *(f"{source}.t[{axis}]" for axis in kept),
                f"{source}.t[{a}] + {source}.t[{b}]",
            ]
        )
        + "]"
    )
    view = f"{{d: {source}.d, s: {shape}, t: {strides}, o: ({source}.o || 0) + {offset} * {source}.t[{a}] + {second} * {source}.t[{b}]}}"
    name = f"v{slot}"
    if op.view:
        return JSCode((f"const {name} = {view};",), (name,))
    return JSCode(
        (f"const {name} = copyRecord(slot({slot}, {shape}), {view});",), (name,)
    )


@js_funcify.register(AllocDiag)
def js_funcify_alloc_diag(op, node, inputs, slot):
    if node.inputs[0].ndim != 1 or (op.axis1, op.axis2) != (0, 1):
        raise NotImplementedError("JS AllocDiag supports vector input")
    source = inputs[0]
    name = f"v{slot}"
    row, col = max(0, -op.offset), max(0, op.offset)
    return JSCode(
        (
            f"const {name} = slot({slot}, [{source}.s[0] + {abs(op.offset)}, {source}.s[0] + {abs(op.offset)}]);",
            f"{name}.d.fill(0);",
            f"for (let i = 0; i < {source}.s[0]; i++) {name}.d[(i + {row}) * {name}.s[1] + i + {col}] = {source}.d[({source}.o || 0) + i * {source}.t[0]];",
        ),
        (name,),
    )
