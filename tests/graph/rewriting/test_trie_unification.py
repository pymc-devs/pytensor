# TODO: Use simpler test ops
import pytensor.tensor as pt
from pytensor.graph.rewriting.trie_unification import (
    Asterisk,
    Literal,
    MatchPattern,
    Trie,
)
from pytensor.graph.rewriting.unify import OpInstance
from pytensor.tensor.basic import Join
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import CAReduce, DimShuffle
from pytensor.tensor.slinalg import Cholesky, Solve


def test_solve_of_cholesky():
    def blockwise_of(core_op):
        return OpInstance(Blockwise, {"core_op": core_op})

    MatrixTransposePattern = OpInstance(DimShuffle, {"is_matrix_transpose": True})
    GenSolvePattern = blockwise_of(OpInstance(Solve, {"assume_a": Literal("gen")}))
    CholeskyPattern = blockwise_of(OpInstance(Cholesky, {"lower": "lower"}))

    A = pt.matrix("A")
    b = pt.vector("b")
    out1 = pt.linalg.solve(pt.linalg.cholesky(A), b)
    out2 = pt.linalg.solve(pt.linalg.cholesky(A).mT, b)

    P1 = MatchPattern(
        "GenSolve(Cholesky(A), b)",
        (GenSolvePattern, (CholeskyPattern, "A"), "b"),
    )
    P2 = MatchPattern(
        "GenSolve(Cholesky(A).mT, b)",
        (GenSolvePattern, (MatrixTransposePattern, (CholeskyPattern, "A")), "b"),
    )

    trie = Trie()
    trie.add_pattern(P1)
    trie.add_pattern(P2)

    r1 = dict(trie.match(out1))
    assert list(r1) == [P1]
    assert r1[P1] == {"A": A, "b": b, "lower": True}

    r2 = dict(trie.match(out2))
    assert list(r2) == [P2]
    assert r2[P2] == {"A": A, "b": b, "lower": True}


def test_mixed_blockwise_types():
    blockwise_unary = MatchPattern("Blockwise(x)", (Blockwise, "x"))
    blockwise_lower_cholesky = MatchPattern(
        "Blockwise(Cholesky(lower=True))(x)", (Blockwise(Cholesky(lower=True)), "x")
    )
    blockwise_cholesky = MatchPattern(
        "Blockwise(Cholesky)(x)", (OpInstance(Blockwise, core_op=Cholesky), "x")
    )
    alt_blockwise_lower_cholesky = MatchPattern(
        "[Alt]Blockwise(Cholesky)(lower=True)(x)",
        (
            OpInstance(Blockwise, {"core_op": OpInstance(Cholesky, [("lower", True)])}),
            "x",
        ),
    )
    solve_gen_var = MatchPattern(
        "Blockwise(Solve(assume_a=?gen))(A, b)",
        (
            OpInstance(Blockwise, core_op=OpInstance(Solve, assume_a="?gen")),
            "A",
            "b",
        ),
    )
    solve_gen_literal = MatchPattern(
        "Blockwise(Solve(assume_a=gen))(A, b)",
        (
            OpInstance(Blockwise, core_op=OpInstance(Solve, assume_a=Literal("gen"))),
            "A",
            "b",
        ),
    )

    trie = Trie()
    for pattern in (
        blockwise_unary,
        blockwise_lower_cholesky,
        blockwise_cholesky,
        alt_blockwise_lower_cholesky,
        solve_gen_var,
        solve_gen_literal,
    ):
        trie.add_pattern(pattern)

    X = pt.matrix("X")
    out = pt.linalg.cholesky(X)
    res = dict(trie.match(out))
    assert set(res) == {
        blockwise_unary,
        blockwise_lower_cholesky,
        blockwise_cholesky,
        alt_blockwise_lower_cholesky,
    }
    for subs in res.values():
        assert subs == {"x": X}

    out = pt.linalg.cholesky(X, lower=False)
    res = dict(trie.match(out))
    assert set(res) == {
        blockwise_unary,
        blockwise_cholesky,
    }
    for subs in res.values():
        assert subs == {"x": X}

    A, b = pt.matrix("A"), pt.vector("b")
    out = pt.linalg.solve(A, b)
    res = dict(trie.match(out))
    assert set(res) == {solve_gen_var, solve_gen_literal}
    assert res[solve_gen_literal] == {"A": A, "b": b}
    assert res[solve_gen_var] == {"?gen": "gen", "A": A, "b": b}


def test_asterisk():
    P1 = MatchPattern(
        "Reduce(Join(*entries))",
        (CAReduce, (Join, "axis", Asterisk("entries"))),
    )
    P2 = MatchPattern(
        "Pow(Reduce(Join(*entries)), y)",
        (pt.pow, (CAReduce, (Join, "axis", Asterisk("entries"))), "y"),
    )

    trie = Trie()
    trie.add_pattern(P1)
    trie.add_pattern(P2)

    x = pt.vector("x")
    y = pt.vector("y")
    z = pt.vector("z")
    zeroth_axis = pt.constant(0, dtype="int64")
    sum_of_join = pt.sum(pt.join(zeroth_axis, x, y, z))
    res = dict(trie.match(sum_of_join))
    assert set(res) == {P1}
    assert res[P1] == {"axis": zeroth_axis, "entries": [x, y, z]}

    exponent = pt.scalar("exponent", dtype="int64")
    pow_of_sum = pt.pow(sum_of_join, exponent)
    res = dict(trie.match(pow_of_sum))
    assert set(res) == {P2}
    assert res[P2] == {"axis": zeroth_axis, "entries": [x, y, z], "y": exponent}


def test_repeated_vars():
    P = MatchPattern(
        "Join(x, x)",
        (Join, "axis", "x", "x", Asterisk("xs")),
    )

    trie = Trie()
    trie.add_pattern(P)

    x, y = pt.vectors("xy")
    zeroth_axis = pt.constant(0, dtype="int64")

    join_xx = pt.join(zeroth_axis, x, x)
    res = dict(trie.match(join_xx))
    assert set(res) == {P}
    assert res[P] == {"axis": zeroth_axis, "x": x, "xs": []}

    join_xy = pt.join(zeroth_axis, x, y)
    res = dict(trie.match(join_xy))
    assert set(res) == set()

    join_xxy = pt.join(zeroth_axis, x, x, y)
    res = dict(trie.match(join_xxy))
    assert set(res) == {P}
    assert res[P] == {"axis": zeroth_axis, "x": x, "xs": [y]}

    join_xxyx = pt.join(zeroth_axis, x, x, y, x)
    res = dict(trie.match(join_xxyx))
    assert set(res) == {P}
    assert res[P] == {"axis": zeroth_axis, "x": x, "xs": [y, x]}
