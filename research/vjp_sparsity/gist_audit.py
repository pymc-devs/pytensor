"""Run the advanced gist's case constructions against this prototype.

Only case-construction prefixes and the numeric oracle are loaded from the
notebook. Its sparsity implementation, registered rewrites, plots, and marginal
dependence examples are not executed.
"""

# ruff: noqa: T201

import argparse
import ast
import json
import time
import traceback
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

import pytensor
import pytensor.tensor as pt
from pytensor.graph.basic import Constant
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import graph_inputs
from research.vjp_sparsity.prototype import jacobian_sparsity_vjp


@dataclass
class Case:
    cell: int
    group: str
    title: str
    outputs: list
    wrt: list
    inputs: list
    points: list
    reference_nnz: int | None


class Table(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = []
        self.text = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"}:
            self.text = ""

    def handle_data(self, data):
        if self.text is not None:
            self.text += data

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.text is not None:
            self.row.append(self.text.strip())
            self.text = None
        elif tag == "tr":
            self.rows.append(self.row)


def source(cell):
    text = cell["source"]
    return "".join(text) if isinstance(text, list) else text


def saved_counts(cell):
    for output in cell.get("outputs", []):
        html = output.get("data", {}).get("text/html")
        if not html:
            continue
        parser = Table()
        parser.feed("".join(html) if isinstance(html, list) else html)
        if parser.rows and "detected nnz" in parser.rows[0]:
            column = parser.rows[0].index("detected nnz")
            return {row[0]: int(row[column]) for row in parser.rows[1:]}
    return {}


def load_cases(notebook):
    ns = {
        "np": np,
        "pt": pt,
        "pytensor": pytensor,
        "Constant": Constant,
        "graph_inputs": graph_inputs,
        "graph_replace": graph_replace,
        "differentiable": lambda variable: variable.dtype.startswith("float"),
    }
    for cell, functions in (
        (3, {"test_point", "node_inputs", "local_cases"}),
        (56, {"numeric_nonzero"}),
    ):
        tree = ast.parse(source(notebook["cells"][cell]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in functions:
                exec(
                    compile(
                        ast.Module(body=[node], type_ignores=[]),
                        "<gist helpers>",
                        "exec",
                    ),
                    ns,
                )

    # cell: (group, case dictionary, input list or names, value list or shapes)
    groups = {
        31: ("masked_reductions", "masked_reduction_cases", ["matrix"], [(4, 3)]),
        35: (
            "dot_graphs",
            "dot_graph_cases",
            "inputs",
            [(2, 2), (3, 3), (5,), (4, 3), (3, 2)],
        ),
        43: (
            "solve_graphs",
            "solve_graph_cases",
            ["first_block", "second_block", "rhs"],
            [(2, 2), (2, 2), (4,)],
        ),
        51: ("kron", "kron_cases", ["small", "other_small"], [(2, 2), (2, 3)]),
        58: ("single_nodes", "single_node_cases", "inputs", "values"),
        59: ("batched", "blockwise_cases", "inputs", "values"),
        60: ("solvers", "solver_cases", "inputs", "values"),
        62: ("cholesky", "cholesky_graph_cases", ["square"], "values"),
        63: ("piecewise", "piecewise_cases", ["matrix", "row"], "values"),
        64: ("sharing", "shared_cases", ["x"], "values"),
        65: ("diamond", "diamond_cases", ["x"], "values"),
        67: ("qr", "qr_cases", "inputs", "values"),
        70: ("two_shapes", "shape_cases", "inputs", None),
        73: ("broadcast", "broadcast_cases", ["a", "b"], "values"),
        76: ("nonlinear", "gist_cases", ["x"], "values"),
        79: ("hessian", "hessian_cases", ["x"], "values"),
        84: ("cge", "cge_cases", "cge_variables", "cge_values"),
        89: ("car", "car_cases", "car_parameters", "car_values"),
        95: ("derivative_semantics", "semantics_cases", ["x", "y"], "values"),
        106: ("coloring_patterns", "pattern_cases", "inputs", "values"),
    }
    prerequisites = {48, 82, 83, 87, 88}
    cases = []
    for cell in sorted(set(groups) | prerequisites):
        text = source(notebook["cells"][cell]).split("\ngraphs,", 1)[0]
        if cell == 95:
            text = text.split("\nsemantics_tables", 1)[0]
        exec(compile(text, f"<gist cell {cell}>", "exec"), ns)
        if cell not in groups:
            continue
        group, case_name, inputs, values = groups[cell]
        inputs = (
            list(ns[inputs])
            if isinstance(inputs, str)
            else [ns[name] for name in inputs]
        )
        if cell == 70:
            points = [
                ("small", ns["test_point"]([(4, 3), (3,), (3, 3), (3,), (2, 3, 3)])),
                ("large", ns["test_point"]([(2, 5), (5,), (6, 6), (6,), (2, 3, 6)])),
            ]
        else:
            values = ns[values] if isinstance(values, str) else ns["test_point"](values)
            points = [("default", list(values))]
        expected = saved_counts(notebook["cells"][cell]) if cell not in {70, 95} else {}
        for title, (outputs, wrt) in ns[case_name].items():
            cases.append(
                Case(
                    cell,
                    group,
                    title,
                    list(outputs),
                    list(wrt),
                    inputs,
                    points,
                    expected.get(title),
                )
            )
    return cases, ns["numeric_nonzero"]


def run_case(case, numeric_nonzero):
    record = {
        "cell": case.cell,
        "group": case.group,
        "case": case.title,
        "gist_nnz": case.reference_nnz,
    }
    started = time.monotonic()
    stage = "build"
    try:
        pattern = jacobian_sparsity_vjp(case.outputs, case.wrt)
        stage = "compile"
        function = pytensor.function(case.inputs, pattern, on_unused_input="ignore")
        evaluations = []
        for label, values in case.points:
            stage = "evaluate"
            detected = function(*values)
            stage = "numeric_reference"
            numeric = numeric_nonzero(case.outputs, case.wrt, case.inputs, values)
            if detected.shape != numeric.shape:
                raise AssertionError(
                    f"pattern shape {detected.shape} != numeric {numeric.shape}"
                )
            missed = int((numeric & ~detected).sum())
            evaluations.append(
                {
                    "point": label,
                    "shape": list(detected.shape),
                    "detected_nnz": int(detected.sum()),
                    "numeric_nnz": int(numeric.sum()),
                    "missed": missed,
                    "extra": int((detected & ~numeric).sum()),
                }
            )
        record.update(
            status="unsound"
            if any(value["missed"] for value in evaluations)
            else "sound",
            evaluations=evaluations,
        )
    except Exception as error:
        record.update(
            status="error",
            stage=stage,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    record["seconds"] = round(time.monotonic() - started, 3)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", action="append")
    args = parser.parse_args()
    cases, numeric = load_cases(json.loads(args.notebook.read_text()))
    results = []
    for case in cases:
        if args.group and case.group not in args.group:
            continue
        result = run_case(case, numeric)
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "traceback"}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
