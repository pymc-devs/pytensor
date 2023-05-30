"""
Tests of printing functionality
"""
import logging
from io import StringIO
from textwrap import dedent

import numpy as np
import pytest

import pytensor
from pytensor.compile.mode import get_mode
from pytensor.compile.ops import deep_copy_op
from pytensor.printing import (
    PatternPrinter,
    PPrinter,
    Print,
    debugprint,
    default_printer,
    get_node_by_id,
    min_informative_str,
    pp,
    pydot_imported,
    pydotprint,
)
from pytensor.tensor import as_tensor_variable
from pytensor.tensor.type import dmatrix, dvector, matrix
from tests.graph.utils import MyInnerGraphOp, MyOp, MyVariable


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
    assert isinstance(ret, (str, bytes))


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
    prof = pytensor.compile.ProfileStats(atexit_print=False, gpu_checks=False)
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

    if mis != reference:
        print("--" + mis + "--")
        print("--" + reference + "--")

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
    mode = pytensor.compile.get_default_mode().including("fusion")
    g = pytensor.function([A, B, D, E], G, mode=mode)

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
    g = pytensor.function([A, B, D, E], G, mode=mode, profile=True)

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
        pytensor.function([A, B, D, J], A + (B.dot(J) - D), mode="FAST_RUN"),
        file=s,
        id_type="",
        print_destroy_map=True,
        print_view_map=True,
    )
    s = s.getvalue()
    exp_res = dedent(
        r"""
        Composite{(i2 + (i0 - i1))} 4
         ├─ ExpandDims{axis=0} v={0: [0]} 3
         │  └─ CGemv{inplace} d={0: [0]} 2
         │     ├─ AllocEmpty{dtype='float64'} 1
         │     │  └─ Shape_i{0} 0
         │     │     └─ B
         │     ├─ 1.0
         │     ├─ B
         │     ├─ <Vector(float64, shape=(?,))>
         │     └─ 0.0
         ├─ D
         └─ A

        Inner graphs:

        Composite{(i2 + (i0 - i1))}
         ← add 'o0'
            ├─ i2
            └─ sub
               ├─ i0
               └─ i1
        """
    ).lstrip()

    assert [l.strip() for l in s.split("\n")] == [
        l.strip() for l in exp_res.split("\n")
    ]


def test_debugprint_id_type():
    a_at = dvector()
    b_at = dmatrix()

    d_at = b_at.dot(a_at)
    e_at = d_at + a_at

    s = StringIO()
    debugprint(e_at, id_type="auto", file=s)
    s = s.getvalue()

    exp_res = f"""Add [id {e_at.auto_name}]
 ├─ dot [id {d_at.auto_name}]
 │  ├─ <Matrix(float64, shape=(?, ?))> [id {b_at.auto_name}]
 │  └─ <Vector(float64, shape=(?,))> [id {a_at.auto_name}]
 └─ <Vector(float64, shape=(?,))> [id {a_at.auto_name}]
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
    ├─ *0-<MyType()> [id E]
    └─ *1-<MyType()> [id F]
    """

    for exp_line, res_line in zip(exp_res.split("\n"), lines):
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
    ├─ *0-<MyType()> [id D]
    └─ *1-<MyType()> [id E]

MyInnerGraphOp [id C]
 ← op2 [id F] 'igo1'
    ├─ *0-<MyType()> [id D]
    └─ *1-<MyType()> [id E]
    """

    for exp_line, res_line in zip(exp_res.split("\n"), lines):
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

    stdout, stderr = capsys.readouterr()
    assert "hello" in stdout
