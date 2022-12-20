import dataclasses
from functools import partial

import kanren as K
import pytest

from pytensor.graph.signature import F, S, expand_dims_broadcast


@dataclasses.dataclass(frozen=True)
class MyDim:
    name: str


@pytest.mark.parametrize(
    ("spec", "match", "fail"),
    [
        ((3, 3), [(3, 3), (1, 3, 3)], [(2, 2)]),
        ((None, 3), [(4, 3), (MyDim("b"), 3)], [(4, 5)]),
        ((MyDim("a"), 3), [(MyDim("a"), 3)], [(3, 5)]),
        ((S("a"), S("a")), [(1, 1), (1, 2, 2), (MyDim("a"), MyDim("a"))], [(1, 2)]),
        ((S("a"), F(-1), S("a")), [(1, 1), (2, 1, 2)], [(1, 2, 2)]),
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
    with pytest.raises(ValueError, match="size of spec is less than required ndim"):
        expand_dims_broadcast(4, spec, bmax=1, S=dict())
    with pytest.raises(ValueError, match="size of spec is greater than required ndim"):
        expand_dims_broadcast(1, spec, S=dict())
    with pytest.raises(ValueError):
        expand_dims_broadcast(5, (F(-1), 1, F(-1)), S=dict())
