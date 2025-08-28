import pytest

import pytensor.tensor as pt
from pytensor.graph import rewrite_graph
from pytensor.graph.basic import equal_computations
from pytensor.graph.features import Feature, FullHistory, ReplaceValidate
from pytensor.graph.fg import FunctionGraph
from tests.graph.utils import MyVariable, op1


class TestReplaceValidate:
    def test_verbose(self, capsys):
        var1 = MyVariable("var1")
        var2 = MyVariable("var2")
        var3 = op1(var2, var1)
        fg = FunctionGraph([var1, var2], [var3], clone=False)

        rv_feature = ReplaceValidate()
        fg.attach_feature(rv_feature)
        rv_feature.replace_all_validate(
            fg, [(var3, var1)], reason="test-reason", verbose=True
        )

        capres = capsys.readouterr()
        assert capres.err == ""
        assert (
            "rewriting: rewrite test-reason replaces Op1.0 of Op1(var2, var1) with var1 of None"
            in capres.out
        )

        class TestFeature(Feature):
            def validate(self, *args):
                raise Exception()

        fg.attach_feature(TestFeature())

        with pytest.raises(Exception):
            rv_feature.replace_all_validate(
                fg, [(var3, var1)], reason="test-reason", verbose=True
            )

        capres = capsys.readouterr()
        assert "rewriting: validate failed on node Op1.0" in capres.out


def test_full_history():
    x = pt.scalar("x")
    out = pt.log(pt.exp(x) / pt.sum(pt.exp(x)))
    fg = FunctionGraph(outputs=[out], clone=True, copy_inputs=False)
    history = FullHistory()
    fg.attach_feature(history)
    rewrite_graph(fg, clone=False, include=("canonicalize", "stabilize"))

    history.start()
    assert equal_computations(fg.outputs, [out])

    history.end()
    assert equal_computations(fg.outputs, [pt.special.log_softmax(x)])

    history.prev()
    assert equal_computations(fg.outputs, [pt.log(pt.special.softmax(x))])

    for i in range(10):
        history.prev()
    assert equal_computations(fg.outputs, [out])

    history.goto(2)
    assert equal_computations(fg.outputs, [pt.log(pt.special.softmax(x))])

    for i in range(10):
        history.next()

    assert equal_computations(fg.outputs, [pt.special.log_softmax(x)])
