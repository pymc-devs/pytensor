import pytest

from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import GraphRewriter, SequentialGraphRewriter
from pytensor.graph.rewriting.db import (
    EquilibriumDB,
    LocalGroupDB,
    ProxyDB,
    RewriteDatabase,
    SequenceDB,
)


class TestRewriter(GraphRewriter):
    name = "blah"

    def apply(self, fgraph):
        pass


class NewTestRewriter(GraphRewriter):
    name = "bleh"

    def apply(self, fgraph):
        pass


counter1 = 0

counter2 = 0


class TestOverwrite1(GraphRewriter):
    def apply(self, fgraph):
        global counter1
        counter1 += 1


class TestOverwrite2(GraphRewriter):
    def apply(self, fgraph):
        global counter2
        counter2 += 1


class TestDB:
    def test_register(self):
        db = RewriteDatabase()
        db.register("a", TestRewriter())

        db.register("b", TestRewriter())

        db.register("c", TestRewriter(), "z", "asdf")

        assert "a" in db
        assert "b" in db
        assert "c" in db

        with pytest.raises(ValueError, match=r"The tag.*"):
            db.register("c", NewTestRewriter())  # name taken

        db.register("c", NewTestRewriter(), overwrite_existing=True)

        with pytest.raises(ValueError, match=r"The tag.*"):
            db.register("z", TestRewriter())  # name collides with tag

        with pytest.raises(ValueError, match=r"The tag.*"):
            db.register("u", TestRewriter(), "b")  # name new but tag collides with name

        with pytest.raises(TypeError, match=r".* is not a valid.*"):
            db.register("d", 1)

    def test_overwrite(self):
        db = RewriteDatabase()
        fg = FunctionGraph([], [])

        db.register("a", TestRewriter())
        Rewriter = db.__getitem__("a")
        Rewriter.rewrite(fg)

        db.register("a", TestOverwrite1(), overwrite_existing=True)
        Rewriter = db.__getitem__("a")
        Rewriter.rewrite(fg)
        assert counter1 == 1 and counter2 == 0

        db.register("a", TestOverwrite2(), overwrite_existing=True)
        Rewriter = db.__getitem__("a")
        Rewriter.rewrite(fg)
        assert counter1 == 1 and counter2 == 1

    def test_EquilibriumDB(self):
        eq_db = EquilibriumDB()

        with pytest.raises(ValueError, match=r"`final_rewriter` and.*"):
            eq_db.register("d", TestRewriter(), final_rewriter=True, cleanup=True)

    def test_SequenceDB(self):
        seq_db = SequenceDB(failure_callback=None)

        res = seq_db.query("+a")

        assert isinstance(res, SequentialGraphRewriter)
        assert res.data == []

        seq_db.register("b", TestRewriter(), position=1)

        from io import StringIO

        out_file = StringIO()
        seq_db.print_summary(stream=out_file)

        res = out_file.getvalue()

        assert str(id(seq_db)) in res
        assert "names {'b'}" in res

        with pytest.raises(TypeError, match=r"`position` must be.*"):
            seq_db.register("c", TestRewriter(), position=object())

    def test_LocalGroupDB(self):
        lg_db = LocalGroupDB()

        lg_db.register("a", TestRewriter(), 1)

        assert "a" in lg_db.__position__

        with pytest.raises(TypeError, match=r"`position` must be.*"):
            lg_db.register("b", TestRewriter(), position=object())

    def test_ProxyDB(self):
        with pytest.raises(TypeError, match=r"`db` must be.*"):
            ProxyDB(object())
