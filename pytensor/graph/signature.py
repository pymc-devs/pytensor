from dataclasses import dataclass
from enum import Enum
from functools import singledispatch
from itertools import count, islice
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import kanren as K
from typing_extensions import Literal


class Broadcast(Enum):
    on = "+"
    off = "="


class Fill:
    __slots__ = ("fill", "kind", "group", "trailing")
    fill: int
    kind: Broadcast
    group: int
    trailing: bool

    def __init__(
        self,
        fill: int,
        kind: Union[Literal["+", "="], Broadcast] = Broadcast.on,
        group: int = -1,
        trailing=False,
    ):
        self.kind = Broadcast(kind)
        if fill < -1:
            raise ValueError("fill is less than -1")
        self.fill = fill
        if group < -1:
            raise ValueError("group is less than -1")
        self.group = group
        self.trailing = trailing

    def __str__(self):
        if self.group != -1 and self.trailing:
            g = f"[{self.group}:]"
        elif self.group != -1:
            g = f"[{self.group}]"
        elif self.trailing:
            g = ":"
        else:
            g = ""
        if self.fill != -1:
            return f".{self.kind.value}{g}{self.fill}."
        else:
            return f".{self.kind.value}{g}*."

    __repr__ = __str__

    def expand_inf(self, S: Dict) -> Iterator[K.Var]:
        # the infinite expansion of the group pattern
        if self.kind == Broadcast.off:
            for i in count():
                # all dims equal
                yield _default_get(S, (self.group, i), K.var)
        else:
            for i in count():
                # all dims have to broadcast
                # append them to list to remember which ones
                bcast = _default_get(S, (self.group, i), list)
                dim = K.var()
                bcast.append(dim)
                yield dim

    def expand(self, fill_size: int, S: Dict) -> List[K.Var]:
        seen_n = _default_get(S, self.group, list)
        n = fill_size
        if not self.trailing and self.fill >= 0:
            n = self.fill
        elif self.trailing and self.fill >= 0 and self.fill < n:
            raise ValueError(
                f"trailing is True and asked to expand to {n} greater than {self.fill}"
            )
        self.check_integrity(seen_n, n, self.trailing)
        seen_n.append((n, self.trailing))
        if self.fill == -1 or self.trailing:
            return list(islice(self.expand_inf(S), 0, n))
        elif n == self.fill:
            return list(islice(self.expand_inf(S), 0, self.fill))
        else:
            raise ValueError(
                f"trailing is False and asked to expand to {n} instead of {self.fill}"
            )

    @staticmethod
    def check_integrity(seen_n: List[Tuple[int, bool]], n: int, trailing: bool):
        not_trailed = [s for (s, t) in seen_n if not t]
        trailed = [s for (s, t) in seen_n if t]
        if not trailing and not_trailed and n != not_trailed[0]:
            raise ValueError(
                "The expansion pattern that do not trail i.e. "
                "'...' or '.[g].' do not match in lengths"
            )
        if (not trailing and trailed and max(trailed) > n) or (
            trailing and not_trailed and max(not_trailed) < n
        ):
            raise ValueError(
                "The expansion pattern that trails i.e. '.[:].' or '.[g:].' is "
                "greater than the pattern that does not trail, i.e. '...' or .[g]."
            )


F = Fill


@dataclass
class Symbol:
    name: str

    def __hash__(self) -> int:
        return hash((self.__class__, self.name))

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__


S = Symbol


def _default_get(S, key, cls):
    if key in S:
        return S[key]
    else:
        S[key] = ret = cls()
        return ret


@singledispatch
def expand_arg(a, S: Dict, fill_size: int) -> List:
    return [a]


@expand_arg.register(Fill)
def _(a: Fill, S: Dict, fill_size: int) -> List:
    return a.expand(fill_size, S)


@expand_arg.register(Symbol)
def _(a: Symbol, S: Dict, fill_size: int):
    return [_default_get(S, a, K.var)]


@expand_arg.register(type(None))
def _(a: Symbol, S: Dict, fill_size: int):
    return [K.var()]


def length_hint(a, max=False):
    if isinstance(a, Fill) and a.fill == -1:
        if max:
            return float("inf")
        else:
            return 0
    elif isinstance(a, Fill) and a.trailing:
        if max:
            return a.fill
        else:
            return 0
    elif isinstance(a, Fill):
        return a.fill
    else:
        return 1


def expand_dims_broadcast(
    ndim: int,
    spec: Sequence[Any],
    broadcast: Union[Literal["+", "="], Broadcast] = "+",
    *,
    S: Dict,
    bmax: int = -1,
    complete=False,
) -> Tuple[Any, ...]:
    sizes = list(map(length_hint, spec))
    n_fills = sum(s == 0 for s in sizes)
    if n_fills > 1:
        raise ValueError("more than two fills with -1 or trailing range found")
    elif n_fills == 0 and not complete:
        # prepend global broadcasting dimensions
        spec = [Fill(bmax, broadcast, trailing=True)] + list(spec)
    max_size = sum(length_hint(s, max=True) for s in spec)
    if max_size < ndim:
        raise ValueError(
            f"requested {ndim} ndims but the spec only provides maximum {max_size}"
        )
    min_size = sum(map(length_hint, spec))
    fill_size = ndim - min_size
    if fill_size < 0:
        raise ValueError(
            f"requested {ndim} ndims but the spec only provides minimum {min_size}"
        )
    dims: List[Any] = []
    for arg in reversed(spec):
        args = expand_arg(arg, S, fill_size)
        dims.extend(args)
    return tuple(reversed(dims))


class Arg:
    __slots__ = ("spec",)
    spec: Tuple[Any, ...]

    def __init__(
        self,
        *spec: Any,
        broadcast: Union[Literal["+", "="], Broadcast] = "+",
        bmax: Optional[int] = None,
    ) -> None:
        sizes = list(map(length_hint, spec))
        n_expand = sum(s == 0 for s in sizes)
        extra: Tuple[Fill, ...]
        if n_expand > 1:
            raise ValueError("More than two fills with -1 or trailing range found")
        elif n_expand == 0 and bmax is None:
            extra = (Fill(-1, broadcast, trailing=True),)
        elif n_expand == 0 and bmax is not None and bmax != 0:
            extra = (Fill(bmax, broadcast, trailing=True),)
        else:
            extra = ()
        self.spec = extra + tuple(spec)

    def __str__(self) -> str:
        return "(" + ",".join(map(str, self.spec)) + ")"

    __repr__ = __str__

    def __call__(self, ndim, *, S) -> Tuple[Any, ...]:
        """Bound an argument to dims.

        Parameters
        ----------
        ndim : int
            the desired ndim
        S : Dict
            context where argument information is stored

        Returns
        -------
        Tuple
            kanren variables or concrete values
        """
        return expand_dims_broadcast(ndim, self.spec, S=S, complete=True)
