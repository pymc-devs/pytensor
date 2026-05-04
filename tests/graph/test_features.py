import pickle

import pytest

import pytensor.tensor as pt
from pytensor.graph import rewrite_graph
from pytensor.graph.basic import equal_computations
from pytensor.graph.destroyhandler import DestroyHandler
from pytensor.graph.features import (
    Feature,
    FullHistory,
    ReplaceValidate,
    register_feature_callback,
)
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
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
            def on_validate(self, *args):
                raise Exception()

        fg.attach_feature(TestFeature())

        with pytest.raises(Exception):
            rv_feature.replace_all_validate(
                fg, [(var3, var1)], reason="test-reason", verbose=True
            )

        capres = capsys.readouterr()
        assert "rewriting: validate failed on node Op1.0" in capres.out


def test_full_history():
    x = pt.vector("x")
    out = pt.log(pt.exp(x) / pt.sum(pt.exp(x), keepdims=True))
    fg = FunctionGraph(outputs=[out], clone=True, copy_inputs=False)
    history = FullHistory()
    fg.attach_feature(history)
    rewrite_graph(fg, clone=False, include=("canonicalize", "stabilize"))

    history.start()
    assert equal_computations(fg.outputs, [out])

    history.end()
    assert equal_computations(fg.outputs, [pt.special.log_softmax(x, axis=None)])

    history.prev()
    assert equal_computations(fg.outputs, [pt.log(pt.special.softmax(x, axis=None))])

    for i in range(10):
        history.prev()
    assert equal_computations(fg.outputs, [out])

    history.goto(2)
    assert equal_computations(fg.outputs, [pt.log(pt.special.softmax(x, axis=None))])

    for i in range(10):
        history.next()

    assert equal_computations(fg.outputs, [pt.special.log_softmax(x, axis=None)])


class TestPickleRoundTrip:
    """A pickled FunctionGraph must support the same Feature operations as a fresh one.

    Regression: ``ReplaceValidate``'s history dict, the checkpoint counter,
    and the ``execute_callbacks_times`` accumulator all used to be silently
    dropped on pickle without being re-initialized on unpickle, so the next
    rewrite would crash with ``'ReplaceValidate' object has no attribute 'history'``
    or similar.
    """

    def _round_trip(self, fg):
        return pickle.loads(pickle.dumps(fg))

    def test_checkpoint_after_pickle(self):
        x = pt.vector("x")
        fg = FunctionGraph([x], [x * 2 + 1])
        fg2 = self._round_trip(fg)
        chk = fg2.checkpoint()
        fg2.revert(chk)

    def test_replace_all_validate_after_pickle(self):
        x = pt.vector("x")
        fg = FunctionGraph([x], [x * 2 + 1])
        fg2 = self._round_trip(fg)
        out = fg2.outputs[0]
        fg2.replace_all_validate([(out, out + 0)], reason="test")

    def test_execute_callbacks_after_pickle(self):
        x = pt.vector("x")
        fg = FunctionGraph([x], [x * 2 + 1])
        fg2 = self._round_trip(fg)
        fg2.execute_callbacks("on_import", next(iter(fg2.apply_nodes)), "test")
        assert fg2.execute_callbacks_times

    def test_destroy_handler_after_pickle(self):
        x = pt.vector("x")
        fg = FunctionGraph([x], [x * 2 + 1])
        fg.attach_feature(DestroyHandler())
        fg2 = self._round_trip(fg)
        assert fg2.destroyers(fg2.inputs[0]) == []
        assert fg2.has_destroyers([fg2.inputs[0]]) is False
        assert isinstance(fg2.destroy_handler, DestroyHandler)

    def test_history_reason_is_stringified_on_pickle(self):
        """Decorated rewriters captured as History `reason` must not block pickling.

        ``replace_all_validate(reason=node_rewriter)`` (e.g. basic.py:process_node)
        stores the rewriter on every ``HistoryEntry``. Decorated rewriters
        aren't picklable — the decorator rebinds the name to the wrapper, so
        pickle's qualname lookup can't roundtrip ``self.fn``. A function-local
        rewriter is unpicklable for an even simpler reason (``<locals>`` in
        qualname), so this test would fail outright without the stringify.
        """

        @node_rewriter(None)
        def local_rewriter(fgraph, node):
            return None

        x = pt.vector("x")
        fg = FunctionGraph([x], [x * 2 + 1])
        out = fg.outputs[0]
        fg.replace_all_validate([(out, out + 0)], reason=local_rewriter)

        fg2 = self._round_trip(fg)

        replace_validate = next(
            f for f in fg2._features if isinstance(f, ReplaceValidate)
        )
        entries = replace_validate.history[fg2]
        assert entries, "expected at least one HistoryEntry post-replace"
        assert all(isinstance(e.reason, str) for e in entries)
        assert any(e.reason == "local_rewriter" for e in entries)


def test_provides_callback_collision_rejected_at_class_time():
    """A name cannot appear in both ``provides`` and the callback registry."""
    with pytest.raises(TypeError, match="appear in both `provides` and as callbacks"):

        class Bad(Feature):
            provides = ("on_validate",)

            @register_feature_callback
            def on_validate(self, fgraph):
                pass


@pytest.mark.parametrize(
    "feature_factory",
    [ReplaceValidate, DestroyHandler],
    ids=["ReplaceValidate", "DestroyHandler"],
)
def test_feature_provides_dispatch_contract(feature_factory):
    """Names listed in ``Feature.provides`` are reachable as ``fgraph.<name>``
    via ``__getattr__`` dispatch, both fresh and after a pickle round-trip."""
    x = pt.vector("x")
    fg = FunctionGraph([x], [x * 2 + 1])
    feature = feature_factory()
    fg.attach_feature(feature)

    for name in feature.provides:
        assert callable(getattr(fg, name)), name

    fg2 = pickle.loads(pickle.dumps(fg))
    for name in feature.provides:
        assert callable(getattr(fg2, name)), name
