"""
Tests of printing functionality
"""

import logging
from io import StringIO
from textwrap import dedent

import numpy as np
import pytest

import pytensor
from pytensor import config
from pytensor.compile.debug.profiling import ProfileStats
from pytensor.compile.mode import get_mode
from pytensor.compile.ops import deep_copy_op
from pytensor.printing import (
    PatternPrinter,
    PPrinter,
    Print,
    _try_pydot_import,
    char_from_number,
    debugprint,
    default_printer,
    get_node_by_id,
    min_informative_str,
    pp,
    pydotprint,
)
from pytensor.tensor import as_tensor_variable
from pytensor.tensor.type import dmatrix, dvector, matrix
from tests.graph.utils import MyInnerGraphOp, MyOp, MyVariable


try:
    _try_pydot_import()
    pydot_imported = True
except Exception:
    pydot_imported = False


@pytest.mark.parametrize(
    "number,s",
    [
        (0, "A"),
        (1, "B"),
        (25, "Z"),
        (26, "BA"),
        (27, "BB"),
        (3 * 26**2 + 2 * 26 + 0, "DCA"),
        (42421337, "DOVPLX"),
    ],
)
def test_char_from_number(number: int, s: str):
    assert char_from_number(number) == s


@pytest.mark.skipif(not pydot_imported, reason="pydot not available")
def test_pydotprint_cond_highlight():
    # This is a REALLY PARTIAL TEST.
    # I did them to help debug stuff.
    x = dvector()
    f = pytensor.function([x], x * 2)
    f([1, 2, 3, 4])

    s = StringIO()
    new_handler = logging.StreamHandler(s)
    new_handler.setLevel(logging.DEBUG)
    orig_handler = pytensor.logging_default_handler

    pytensor.pytensor_logger.removeHandler(orig_handler)
    pytensor.pytensor_logger.addHandler(new_handler)
    try:
        pydotprint(f, cond_highlight=True, print_output_file=False)
    finally:
        pytensor.pytensor_logger.addHandler(orig_handler)
        pytensor.pytensor_logger.removeHandler(new_handler)

    assert (
        s.getvalue() == "pydotprint: cond_highlight is set but there"
        " is no IfElse node in the graph\n"
    )


@pytest.mark.skipif(not pydot_imported, reason="pydot not available")
def test_pydotprint_return_image():
    x = dvector()
    ret = pydotprint(x * 2, return_image=True)
    assert isinstance(ret, str | bytes)


@pytest.mark.skipif(not pydot_imported, reason="pydot not available")
def test_pydotprint_long_name():
    # This is a REALLY PARTIAL TEST.
    # It prints a graph where there are variable and apply nodes whose long
    # names are different, but not the shortened names.
    # We should not merge those nodes in the dot graph.
    x = dvector()
    mode = pytensor.compile.mode.get_default_mode().excluding("fusion")
    f = pytensor.function([x], [x * 2, x + x], mode=mode)
    f([1, 2, 3, 4])

    pydotprint(f, max_label_size=5, print_output_file=False)
    pydotprint([x * 2, x + x], max_label_size=5, print_output_file=False)


@pytest.mark.skipif(
    not pydot_imported or pytensor.config.mode in ("DebugMode", "DEBUG_MODE"),
    reason="Can't profile in DebugMode",
)
def test_pydotprint_profile():
    A = matrix()
    prof = ProfileStats(atexit_print=False, gpu_checks=False)
    f = pytensor.function([A], A + 1, profile=prof)
    pydotprint(f, print_output_file=False)
    f([[1]])
    pydotprint(f, print_output_file=False)


def test_min_informative_str():
    # evaluates a reference output to make sure the
    # min_informative_str function works as intended

    A = matrix(name="A")
    B = matrix(name="B")
    C = A + B
    C.name = "C"
    D = matrix(name="D")
    E = matrix(name="E")

    F = D + E
    G = C + F

    mis = min_informative_str(G).replace("\t", "        ")

    reference = """A. Add
 B. C
 C. Add
  D. D
  E. E"""

    # if mis != reference:
    #     print("--" + mis + "--")
    #     print("--" + reference + "--")

    assert mis == reference


def test_debugprint():
    with pytest.raises(TypeError):
        debugprint("blah")

    A = dmatrix(name="A")
    B = dmatrix(name="B")
    C = A + B
    C.name = "C"
    D = dmatrix(name="D")
    E = dmatrix(name="E")

    F = D + E
    G = C + F
    g = pytensor.function([A, B, D, E], G)

    # just test that it work
    s = StringIO()
    debugprint(G, file=s)

    s = StringIO()
    debugprint(G, file=s, id_type="int")
    s = s.getvalue()
    reference = dedent(
        r"""
        Add [id 0]
         ├─ Add [id 1] 'C'
         │  ├─ A [id 2]
         │  └─ B [id 3]
         └─ Add [id 4]
            ├─ D [id 5]
            └─ E [id 6]
        """
    ).lstrip()

    assert s == reference

    s = StringIO()
    debugprint(G, file=s, id_type="CHAR")
    s = s.getvalue()
    # The additional white space are needed!
    reference = dedent(
        r"""
        Add [id A]
         ├─ Add [id B] 'C'
         │  ├─ A [id C]
         │  └─ B [id D]
         └─ Add [id E]
            ├─ D [id F]
            └─ E [id G]
        """
    ).lstrip()

    assert s == reference

    s = StringIO()
    debugprint(G, file=s, id_type="CHAR", stop_on_name=True)
    s = s.getvalue()
    # The additional white space are needed!
    reference = dedent(
        r"""
        Add [id A]
         ├─ Add [id B] 'C'
         │  └─ ···
         └─ Add [id C]
            ├─ D [id D]
            └─ E [id E]
        """
    ).lstrip()

    assert s == reference

    s = StringIO()
    debugprint(G, file=s, id_type="")
    s = s.getvalue()
    reference = dedent(
        r"""
        Add
         ├─ Add 'C'
         │  ├─ A
         │  └─ B
         └─ Add
            ├─ D
            └─ E
        """
    ).lstrip()

    assert s == reference

    s = StringIO()
    debugprint(g, file=s, id_type="", print_storage=True)
    s = s.getvalue()
    reference = dedent(
        r"""
        Add 0 [None]
         ├─ A [None]
         ├─ B [None]
         ├─ D [None]
         └─ E [None]
        """
    ).lstrip()

    assert s == reference

    # Test the `profile` handling when profile data is missing
    g = pytensor.function([A, B, D, E], G, profile=True)

    s = StringIO()
    debugprint(g, file=s, id_type="", print_storage=True)
    s = s.getvalue()
    reference = dedent(
        r"""
        Add 0 [None]
         ├─ A [None]
         ├─ B [None]
         ├─ D [None]
         └─ E [None]
        """
    ).lstrip()

    assert s == reference

    # Add profile data
    g(np.c_[[1.0]], np.c_[[1.0]], np.c_[[1.0]], np.c_[[1.0]])

    s = StringIO()
    debugprint(g, file=s, id_type="", print_storage=True)
    s = s.getvalue()
    reference = dedent(
        r"""
        Add 0 [None]
         ├─ A [None]
         ├─ B [None]
         ├─ D [None]
         └─ E [None]
        """
    ).lstrip()

    assert reference in s

    A = dmatrix(name="A")
    B = dmatrix(name="B")
    D = dmatrix(name="D")
    J = dvector()
    s = StringIO()
    debugprint(
        pytensor.function([A, B, D, J], A + (B.dot(J) - D), mode="CVM"),
        file=s,
        id_type="",
        print_destroy_map=True,
        print_view_map=True,
    )
    s = s.getvalue()
    Gemv_op_name = "CGemv" if pytensor.config.blas__ldflags else "Gemv"
    exp_res = dedent(
        r"""
        Composite{(i0 + (i1 - i2))} 4
        ├─ A
        ├─ ExpandDims{axis=0} v={0: [0]} 3
        """
        f"        │  └─ {Gemv_op_name}{{inplace}} d={{0: [0]}} 2"
        r"""
        │     ├─ AllocEmpty{dtype='float64'} 1
        │     │  └─ Shape_i{0} 0
        │     │     └─ B
        │     ├─ 1.0
        │     ├─ B
        │     ├─ <Vector(float64, shape=(?,))>
        │     └─ 0.0
        └─ D

        Inner graphs:

        Composite{(i0 + (i1 - i2))}
        ← add
            ├─ i0
            └─ sub
            ├─ i1
            └─ i2
        """
    ).lstrip()

    assert [l.strip() for l in s.split("\n")] == [
        l.strip() for l in exp_res.split("\n")
    ]


def test_debugprint_id_type():
    a_at = dmatrix()
    b_at = dmatrix()

    d_at = b_at.dot(a_at)
    e_at = d_at + a_at

    s = StringIO()
    debugprint(e_at, id_type="auto", file=s)
    s = s.getvalue()

    exp_res = f"""Add [id {e_at.auto_name}]
 ├─ Dot [id {d_at.auto_name}]
 │  ├─ <Matrix(float64, shape=(?, ?))> [id {b_at.auto_name}]
 │  └─ <Matrix(float64, shape=(?, ?))> [id {a_at.auto_name}]
 └─ <Matrix(float64, shape=(?, ?))> [id {a_at.auto_name}]
    """

    assert [l.strip() for l in s.split("\n")] == [
        l.strip() for l in exp_res.split("\n")
    ]


def test_pprint():
    x = dvector()
    y = x[1]
    assert pp(y) == "<Vector(float64, shape=(?,))>[1]"


def test_debugprint_inner_graph():
    r1, r2 = MyVariable("1"), MyVariable("2")
    o1 = MyOp("op1")(r1, r2)
    o1.name = "o1"

    # Inner graph
    igo_in_1 = MyVariable("4")
    igo_in_2 = MyVariable("5")
    igo_out_1 = MyOp("op2")(igo_in_1, igo_in_2)
    igo_out_1.name = "igo1"

    igo = MyInnerGraphOp([igo_in_1, igo_in_2], [igo_out_1])

    r3, r4 = MyVariable("3"), MyVariable("4")
    out = igo(r3, r4)

    output_str = debugprint(out, file="str")
    lines = output_str.split("\n")

    exp_res = """MyInnerGraphOp [id A]
 ├─ 3 [id B]
 └─ 4 [id C]

Inner graphs:

MyInnerGraphOp [id A]
 ← op2 [id D] 'igo1'
    ├─ i0 [id E]
    └─ i1 [id F]
    """

    for exp_line, res_line in zip(exp_res.split("\n"), lines, strict=True):
        assert exp_line.strip() == res_line.strip()

    # Test nested inner-graph `Op`s
    igo_2 = MyInnerGraphOp([r3, r4], [out])

    r5 = MyVariable("5")
    out_2 = igo_2(r5)

    output_str = debugprint(out_2, file="str")
    lines = output_str.split("\n")

    exp_res = """MyInnerGraphOp [id A]
 └─ 5 [id B]

Inner graphs:

MyInnerGraphOp [id A]
 ← MyInnerGraphOp [id C]
    ├─ i0 [id D]
    └─ i1 [id E]

MyInnerGraphOp [id C]
 ← op2 [id F] 'igo1'
    ├─ i0 [id D]
    └─ i1 [id E]
    """

    for exp_line, res_line in zip(exp_res.split("\n"), lines, strict=True):
        assert exp_line.strip() == res_line.strip()


def test_get_var_by_id():
    r1, r2 = MyVariable("v1"), MyVariable("v2")
    o1 = MyOp("op1")(r1, r2)
    o1.name = "o1"

    igo_in_1 = MyVariable("v4")
    igo_in_2 = MyVariable("v5")
    igo_out_1 = MyOp("op2")(igo_in_1, igo_in_2)
    igo_out_1.name = "igo1"

    igo = MyInnerGraphOp([igo_in_1, igo_in_2], [igo_out_1])

    r3 = MyVariable("v3")
    o2 = igo(r3, o1)

    res = get_node_by_id(o1, "blah")

    assert res is None

    res = get_node_by_id([o1, o2], "C")

    assert res == r2

    res = get_node_by_id([o1, o2], "F")

    exp_res = igo.fgraph.outputs[0].owner
    assert res == exp_res


def test_PatternPrinter():
    r1, r2 = MyVariable("1"), MyVariable("2")
    op1 = MyOp("op1")
    o1 = op1(r1, r2)
    o1.name = "o1"

    pprint = PPrinter()
    pprint.assign(op1, PatternPrinter(("|%(0)s - %(1)s|", -1000)))
    pprint.assign(lambda pstate, r: True, default_printer)

    res = pprint(o1)

    assert res == "|1 - 2|"


def test_Print(capsys):
    r"""Make sure that `Print` `Op`\s are present in compiled graphs with constant folding."""
    x = as_tensor_variable(1.0) * as_tensor_variable(3.0)
    print_op = Print("hello")
    x_print = print_op(x)

    # Just to be more sure that we'll have constant folding...
    mode = get_mode("FAST_RUN").including("topo_constant_folding")

    fn = pytensor.function([], x_print, mode=mode)

    nodes = fn.maker.fgraph.toposort()
    assert len(nodes) == 2
    assert nodes[0].op == print_op
    assert nodes[1].op == deep_copy_op

    fn()

    stdout, _stderr = capsys.readouterr()
    assert "hello" in stdout


def test_summary_with_profile_optimizer():
    with config.change_flags(profile_optimizer=True):
        f = pytensor.function(inputs=[], outputs=[], profile=True)

    s = StringIO()
    f.profile.summary(file=s)
    assert "Rewriter Profile" in s.getvalue()


class TestDebugprintRich:
    """Tests for debugprint(..., file="rich").

    We test PyTensor's tree-building contract only — not Rich's rendering.
    Rich's own test suite covers rendering; our job is to verify that we
    construct the right tree structure and don't crash on various graph shapes.
    """

    rich = pytest.importorskip("rich")

    def test_return_type(self):
        import rich.tree

        x = dvector("x")
        tree = debugprint(x.sum(), file="rich")
        assert isinstance(tree, rich.tree.Tree)

    def test_single_output_has_one_child(self):
        # One output variable → the hidden root should have exactly one child.
        x = dvector("x")
        tree = debugprint(x.sum(), file="rich")
        assert len(tree.children) == 1

    def test_multiple_outputs_have_multiple_children(self):
        # Two output variables → the hidden root has two children.
        x = dvector("x")
        mean = x.mean()
        std = x.std()
        tree = debugprint([mean, std], file="rich")
        assert len(tree.children) == 2

    def test_linear_graph(self):
        # sum(x * 2): root → sum_node → mul_node → [x_leaf, 2_leaf]
        x = dvector("x")
        y = (x * 2).sum()
        tree = debugprint(y, file="rich")
        sum_node = tree.children[0]
        mul_node = sum_node.children[0]
        assert len(mul_node.children) == 2

    def test_named_var(self):
        # Named intermediate: the label of the mul node contains its name.
        x = dvector("x")
        y = x * 2
        y.name = "doubled"
        tree = debugprint(y.sum(), file="rich")
        mul_node = tree.children[0].children[0]
        assert "doubled" in str(mul_node.label)

    def test_dag_shared_leaf(self):
        # x + x: the add node has two children, both representing x.
        x = dvector("x")
        tree = debugprint(x + x, file="rich")
        add_node = tree.children[0]
        assert len(add_node.children) == 2

    def test_diamond_dag(self):
        # (x*2) + (x+1): add has two children; each has x as a child,
        # but the second occurrence of x is a repeat (still a child node).
        x = dvector("x")
        a = x * 2
        b = x + 1
        tree = debugprint(a + b, file="rich")
        add_node = tree.children[0]
        assert len(add_node.children) == 2
        # Each branch (mul, add) has x as a child.
        assert len(add_node.children[0].children) >= 1
        assert len(add_node.children[1].children) >= 1

    def test_depth_limit(self):
        # depth=1: the root op node is present but its inputs are not expanded.
        x = dvector("x")
        y = (x * 2).sum()
        tree = debugprint(y, depth=1, file="rich")
        sum_node = tree.children[0]
        assert len(sum_node.children) == 0

    def test_stop_on_name(self):
        # stop_on_name=True: traversal stops when it hits the named leaf x,
        # so the mul node's child (x) has no further children.
        x = dvector("x")
        x.name = "x"
        tree = debugprint((x * 2).sum(), stop_on_name=True, file="rich")
        sum_node = tree.children[0]
        mul_node = sum_node.children[0]
        x_node = mul_node.children[0]
        assert len(x_node.children) == 0

    def test_print_type(self):
        # print_type=True: the type annotation appears in the root op's label.
        x = dvector("x")
        tree = debugprint(x.sum(), print_type=True, file="rich")
        sum_node = tree.children[0]
        assert "<" in str(sum_node.label)

    def test_function_graph(self):
        # FunctionGraph: one output → one child under the hidden root.
        from pytensor.graph.fg import FunctionGraph

        x = dvector("x")
        y = x.sum()
        fg = FunctionGraph([x], [y])
        tree = debugprint(fg, file="rich")
        assert len(tree.children) == 1

    def test_inner_graph_op(self):
        # HasInnerGraph op: root has two children — the op node and the
        # "Inner graphs:" section. The op node has the two outer inputs as children.
        igo_in_1, igo_in_2 = MyVariable("x"), MyVariable("y")
        igo_out = MyOp("op")(igo_in_1, igo_in_2)
        op = MyInnerGraphOp([igo_in_1, igo_in_2], [igo_out])
        a, b = MyVariable("a"), MyVariable("b")
        out = op(a, b)
        tree = debugprint(out, file="rich")
        assert len(tree.children) == 2  # op node + "Inner graphs:" section
        op_node = tree.children[0]
        assert len(op_node.children) == 2

    def test_opfromgraph_expands_inner_graph(self):
        # OpFromGraph should produce an "Inner graphs:" section as a second
        # top-level child of the hidden root, matching the text renderer.
        from pytensor.compile.builders import OpFromGraph

        x = dvector("x")
        out = OpFromGraph([x], [x.std()])(x)
        tree = debugprint(out, file="rich")
        # root child 0: the op node; root child 1: "Inner graphs:" section
        assert len(tree.children) == 2
        inner_section = tree.children[1]
        assert len(inner_section.children) >= 1

    def test_repeated_node_no_duplication(self):
        # A repeated node renders canonically the first time (with full children)
        # and as a colored stub the second time, with ··· as its only child.
        x = dvector("x")
        shared = x * 2
        tree = debugprint(shared + shared, file="rich")
        add_node = tree.children[0]
        # The add has two children: canonical Mul and second Mul occurrence
        assert len(add_node.children) == 2
        second_mul = add_node.children[1]
        # The second occurrence has exactly one child: the ··· sentinel
        assert len(second_mul.children) == 1, (
            f"Second occurrence should have exactly one (sentinel) child, got: {second_mul.children}"
        )
        sentinel = second_mul.children[0]
        assert "···" in str(sentinel.label), (
            f"Expected '···' in sentinel label, got: {sentinel.label!r}"
        )
        assert len(sentinel.children) == 0, (
            f"Sentinel should be a leaf, but has children: {sentinel.children}"
        )

    def test_repeated_nodes_same_color(self):
        # Both the canonical occurrence and the second (stub) occurrence of a
        # shared node should carry the same Rich color markup so the user can
        # visually trace where the shared subgraph comes from.
        import re

        x = dvector("x")
        shared = x * 2
        tree = debugprint(shared + shared, file="rich")
        add_node = tree.children[0]
        canonical_mul = add_node.children[0]  # Mul [id B]  (colored, full children)
        second_mul = add_node.children[1]  # Mul [id B]  (colored, ··· child only)
        color_re = re.compile(r"\[(\w+)\]")
        canonical_colors = color_re.findall(str(canonical_mul.label))
        second_colors = color_re.findall(str(second_mul.label))
        assert canonical_colors, "canonical shared node should have a color tag"
        assert second_colors, "second occurrence of shared node should have a color tag"
        assert canonical_colors[0] == second_colors[0], (
            "canonical and second occurrences of the same node should share a color"
        )
        # The sentinel child beneath the second occurrence uses a bright_ variant.
        sentinel = second_mul.children[0]
        assert "bright_" in str(sentinel.label), (
            f"Sentinel should use a bright_ color, got: {sentinel.label!r}"
        )
