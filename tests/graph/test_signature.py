import dataclasses
from functools import partial

import kanren as K
import pytest

from pytensor.graph.signature import Arg, F, OArg, S, expand_dims_broadcast


@dataclasses.dataclass(frozen=True)
class MyDim:
    name: str

    def __str__(self) -> str:
        # for fancy dispaly in args
        return f"~{self.name}~"


@pytest.mark.parametrize(
    ("spec", "match", "fail"),
    [
        ((3, 3), [(3, 3), (1, 3, 3)], [(2, 2)]),
        ((None, 3), [(4, 3), (MyDim("b"), 3)], [(4, 5)]),
        ((MyDim("a"), 3), [(MyDim("a"), 3)], [(3, 5)]),
        ((S("a"), S("a")), [(1, 1), (1, 2, 2), (MyDim("a"), MyDim("a"))], [(1, 2)]),
        ((S("a"), F(-1), S("a")), [(1, 1), (2, 1, 2)], [(1, 2, 2)]),
        ((S("a"), F(2), S("a")), [(1, 3, 4, 1), (2, 2, 4, 2)], [(1, 2, 2, 2)]),
        ((S("a"), F(2), S("a")), [(1, 3, 4, 1), (2, 2, 4, 2)], [(1, 2, 2, 2)]),
    ],
)
def test_expand_dims_broadcast(spec, match, fail):
    specn = partial(expand_dims_broadcast, spec=spec)
    for m in match:
        sp = specn(len(m), S=dict())
        assert len(K.run(0, sp, K.eq(sp, m))) == 1
    for f in fail:
        sp = specn(len(f), S=dict())
        assert len(K.run(0, sp, K.eq(sp, f))) == 0


def test_expand_dims_broadcast_value_errors():
    spec = (None, 3)
    with pytest.raises(
        ValueError, match="requested 4 ndims but the spec only provides maximum 3"
    ):
        expand_dims_broadcast(4, spec, bmax=1, S=dict())
    with pytest.raises(
        ValueError, match="requested 1 ndims but the spec only provides minimum 2"
    ):
        expand_dims_broadcast(1, spec, S=dict())
    with pytest.raises(
        ValueError, match="more than two Fill's with -1 or trailing range found"
    ):
        expand_dims_broadcast(5, (F(-1), 1, F(-1)), S=dict())
    with pytest.raises(
        ValueError, match="more than two Fill's with -1 or trailing range found"
    ):
        expand_dims_broadcast(5, (F(-1), 1, F(2, trailing=True)), S=dict())


@pytest.mark.parametrize(
    ["arg", "display", "match", "fail"],
    [
        (Arg(3, 3), "(.+[L:]*.,3,3)", [(3, 3), (1, 3, 3)], [(2, 2)]),
        (Arg(3, 3, broadcast="="), "(.=[L:]*.,3,3)", [(3, 3), (1, 3, 3)], [(2, 2)]),
        (
            Arg(3, 3, bmax=0),
            "(3,3)",
            [(3, 3)],
            [
                [
                    (1, 3, 3),
                    ValueError,
                    "requested 3 ndims but the spec only provides maximum 2",
                ],
                (2, 2),
                [
                    (1,),
                    ValueError,
                    "requested 1 ndims but the spec only provides minimum 2",
                ],
            ],
        ),
        (
            Arg(3, 3, bmax=1),
            "(.+[L:]1.,3,3)",
            [(3, 3), (1, 3, 3)],
            [
                (2, 2),
                [
                    (1, 1, 3, 3),
                    ValueError,
                    "requested 4 ndims but the spec only provides maximum 3",
                ],
            ],
        ),
        (Arg(None, 3), "(.+[L:]*.,None,3)", [(4, 3), (MyDim("b"), 3)], [(4, 5)]),
        (Arg(MyDim("a"), 3), "(.+[L:]*.,~a~,3)", [(MyDim("a"), 3)], [(3, 5)]),
        (
            Arg(S("a"), S("a")),
            "(.+[L:]*.,a,a)",
            [(1, 1), (1, 2, 2), (MyDim("a"), MyDim("a"))],
            [(1, 2)],
        ),
        (Arg(S("a"), F(-1), S("a")), "(a,.+*.,a)", [(1, 1), (2, 1, 2)], [(1, 2, 2)]),
        (
            Arg(S("a"), F(2), S("a")),
            "(.+[L:]*.,a,.+2.,a)",
            [(1, 3, 4, 1), (2, 2, 4, 2)],
            [(1, 2, 2, 2)],
        ),
        (
            Arg(S("a"), F(2, trailing=True), S("a")),
            "(a,.+:2.,a)",
            [(1, 3, 1), (2, 2, 4, 2)],
            [
                [
                    (1, 2, 1, 2, 1),
                    ValueError,
                    "requested 5 ndims but the spec only provides maximum 4",
                ]
            ],
        ),
    ],
)
def test_single_arg(arg, display, match, fail):
    assert str(arg) == display
    for m in match:
        sp = arg(len(m), S=dict())
        assert len(K.run(0, sp, K.eq(sp, m))) == 1
    for f in fail:
        if isinstance(f, list):
            f, e, match = f
            with pytest.raises(e, match=match):
                print(f)
                sp = arg(len(f), S=dict())
                print(sp)
        else:
            sp = arg(len(f), S=dict())
            assert len(K.run(0, sp, K.eq(sp, f))) == 0


def test_prevent_dim_mismatch():
    # both do not trail
    a1 = OArg()
    a2 = OArg()
    S = dict()
    _ = a1(2, S=S)
    with pytest.raises(
        ValueError,
        match=(
            r"The expansion pattern that does not trail i.e. .* "
            "does not match in length with other similar groups"
        ),
    ):
        _ = a2(3, S=S)


def test_no_infer_dim_with_trailing():
    a = Arg()
    S = {"L": [(3, True)]}
    with pytest.raises(
        ValueError, match="no traling Fill's are allowed when ndim is None"
    ):
        _ = a(None, S=S)


def test_infer_ndim():
    with pytest.raises(
        ValueError, match=r"fill_size is None but no group\[L\] information is found"
    ):
        OArg()(None, S=dict())

    ret = OArg()(None, S={"L": [(1, True)]})
    assert len(ret) == 1


def test_leading_dim_always_a_separate_group():
    arg = Arg(S("a"), S("a"))
    s = dict()
    arg(6, S=s)
    # none should be the key for leading dims
    assert len(s["L"]) == 1
