from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import singledispatch
from itertools import count, islice
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

import kanren as K
import numpy as np
from typing_extensions import Literal

from pytensor.scalar.basic import upcast
from pytensor.utils import flatten


StateType = Dict[
    Union[Tuple[int, int], str, int, Tuple[str, int]],
    Union[List[Tuple[int, bool]], K.var],
]


class Broadcast(Enum):
    on = "+"
    off = "="


class Fill:
    __slots__ = ("fill", "kind", "group", "trailing")
    fill: int
    kind: Broadcast
    group: Optional[Union[int, Literal["L"]]]
    trailing: bool

    def __init__(
        self,
        fill: int,
        kind: Union[Literal["+", "="], Broadcast] = Broadcast.on,
        group: Optional[Union[int, Literal["L"]]] = None,
        trailing=False,
    ):
        self.kind = Broadcast(kind)
        if fill < -1:
            raise ValueError("fill is less than -1")
        self.fill = fill
        if group is not None and group != "L" and group < 0:
            raise ValueError("group is less than 0")
        self.group = group
        self.trailing = trailing

    def __str__(self):
        if self.group is not None and self.trailing:
            g = f"[{self.group}:]"
        elif self.group is not None:
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

    def expand_inf(self, S: StateType) -> Iterator[K.Var]:
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

    def expand(self, fill_size: Optional[int], S: StateType) -> List[K.Var]:
        if self.group is not None:
            seen_n = _default_get(S, self.group, list)
        else:
            seen_n = []
        if fill_size is not None:
            n = fill_size
            if not self.trailing and self.fill >= 0:
                n = self.fill
            elif self.trailing and self.fill >= 0 and self.fill < n:
                raise ValueError(
                    f"trailing is True and asked to expand to {n} greater than {self.fill}"
                )
        else:
            if self.trailing:
                raise ValueError(
                    "trailing is True and trying to infer n with fill_size None"
                )
            elif not seen_n:
                raise ValueError(
                    f"fill_size is None but no group[{self.group}] information is found"
                )
            n = max(s for s, _ in seen_n)
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
    def check_integrity(
        seen_n: List[Tuple[int, bool]], n: Optional[int], trailing: bool
    ):
        if n is None:
            return
        not_trailed = [s for (s, t) in seen_n if not t]
        trailed = [s for (s, t) in seen_n if t]
        if not trailing and not_trailed and n != not_trailed[0]:
            raise ValueError(
                "The expansion pattern that does not trail i.e. "
                "'...' or '.[g].' does not match in length with other similar groups"
            )
        if (not trailing and trailed and max(trailed) > n) or (
            trailing and not_trailed and max(not_trailed) < n
        ):
            raise ValueError(
                "The expansion pattern that trails i.e. '.[:].' or '.[g:].' is "
                "greater than the pattern that does not trail, i.e. '...' or .[g]."
            )


F = Fill


@dataclass(frozen=True, unsafe_hash=True)
class Symbol:
    name: str

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
def expand_arg(a, S: StateType, fill_size: Optional[int]) -> List:
    return [a]


@expand_arg.register(Fill)
def _(a: Fill, S: StateType, fill_size: Optional[int]) -> List:
    return a.expand(fill_size, S)


@expand_arg.register(Symbol)
def _(a: Symbol, S: StateType, fill_size: Optional[int]):
    return [_default_get(S, a, K.var)]


@expand_arg.register(type(None))
def _(a: Symbol, S: StateType, fill_size: Optional[int]):
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
    ndim: Optional[int],
    spec: Sequence[Any],
    broadcast: Union[Literal["+", "="], Broadcast] = "+",
    *,
    S: StateType,
    bmax: int = -1,
    complete=False,
) -> Tuple[Any, ...]:
    """_summary_

    Parameters
    ----------
    ndim : Optional[int]
        set as int to iver trailing or expanding dimensions, set to none to infer vice versa,
        trailing dimensions are not allowerd in this case because the whole group dimension
        information is used.
    spec : Sequence[Any]
        Sequence of dimension specifiers. Could be `Fill`, `Symbol`, `int`, `None` or a custom object
    S : Dict
        Context for the dimension expansion, should be shared within a group of related variables
    broadcast : Union[Literal["+", "="], Broadcast]
        Action on broadcasting leading dimension, default "+" which is regular broadcast
    bmax : int, optional
        Number of dimensions to support in broadcast expansion, by default -1
    complete : bool, optional
        Is the `spec` complete and already set all the additional broadcasting patterns, by default False

    Returns
    -------
    Tuple[Any, ...]
       Sequence of dimension specifiers with
       `Fill`, `Symbol` and `None` expanded to `kanren.Var`,
       could also contain `int`, `None` or a custom object

    Raises
    ------
    ValueError
        If the constuction failed
    """
    sizes = list(map(length_hint, spec))
    n_fills = sum(s == 0 for s in sizes)
    if n_fills > 1:
        raise ValueError("more than two Fill's with -1 or trailing range found")
    elif n_fills == 0 and not complete:
        # prepend global broadcasting dimensions
        # trailing=False when the dimension should be instead inferred
        spec = [Fill(bmax, broadcast, trailing=ndim is not None, group="L")] + list(
            spec
        )
    elif ndim is None and sum(a.trailing for a in spec if isinstance(a, Fill)):
        raise ValueError("no traling Fill's are allowed when ndim is None")
    if ndim is not None:
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
    else:
        fill_size = None
    dims: List[Any] = []
    for arg in reversed(spec):
        args = expand_arg(arg, S, fill_size)
        dims.extend(args)
    return tuple(reversed(dims))


class SpecifyDtype(ABC):
    ...

    @abstractmethod
    def __call__(self, *args: np.dtype) -> np.dtype:
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class ConstDtype(SpecifyDtype):
    __slots__ = ("dtype",)
    dtype: np.dtype

    def __init__(self, dtype: Union[str, np.dtype]) -> None:
        self.dtype = np.dtype(dtype)

    def __call__(self, *args: np.dtype) -> np.dtype:
        return self.dtype

    def __str__(self) -> str:
        return str(self.dtype)


D = ConstDtype


class PromoteDtype(SpecifyDtype):
    __slots__ = ("pos",)
    pos: Tuple[int, ...]
    default: Optional[np.dtype]

    def __init__(
        self, p: int, *ps: int, default: Optional[Union[str, np.dtype]] = None
    ) -> None:
        self.pos = (p, *ps)
        if default is not None:
            self.default = np.dtype(default)
        else:
            self.default = None

    def __call__(self, *args: np.dtype) -> Any:
        if self.default is None:
            return upcast(*(args[p] for p in self.pos))
        else:
            return upcast(self.default, *(args[p] for p in self.pos))

    def __str__(self) -> str:
        args: Tuple[Union[np.dtype, int], ...] = self.pos
        if self.default is not None:
            args = (self.default, *args)
        return "|".join(map(str, args))


P = PromoteDtype


class Arg:
    __slots__ = ("spec",)
    spec: Tuple[Any, ...]

    def __init__(
        self,
        *spec: Any,
        broadcast: Union[Literal["+", "="], Broadcast] = "+",
        bmax: Optional[int] = None,
        trailing=True,
    ) -> None:
        sizes = list(map(length_hint, spec))
        n_expand = sum(s == 0 for s in sizes)
        extra: Tuple[Fill, ...]
        if n_expand > 1:
            raise ValueError("More than two fills with -1 or trailing range found")
        elif n_expand == 0 and bmax is None:
            extra = (Fill(-1, broadcast, trailing=trailing, group="L"),)
        elif n_expand == 0 and bmax is not None and bmax != 0:
            extra = (Fill(bmax, broadcast, trailing=trailing, group="L"),)
        else:
            extra = ()
        self.spec = extra + tuple(spec)

    def bound(self, ndim, *, S: StateType) -> Tuple[Any, ...]:
        """Bound an argument to dims and dtype.

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

    def __str__(self) -> str:
        return "(" + ",".join(map(str, self.spec)) + ")"

    def __repr__(self) -> str:
        return str(self)

    @property
    def min_ndim(self):
        return sum(map(length_hint, self.spec))

    @property
    def max_ndim(self):
        return sum(length_hint(s, max=True) for s in self.spec)


@dataclass(frozen=True)
class AbstractTensor:
    dtype: np.dtype
    dims: Tuple[Any, ...]


@singledispatch
def as_abstract(arg) -> AbstractTensor:
    raise NotImplementedError("can't create an abstract tensor")


@as_abstract.register(tuple)
def _(arg: Tuple):
    return AbstractTensor(*arg)


@as_abstract.register(AbstractTensor)
def _(arg):
    return arg


class IArg(Arg):
    __slots__ = ("spec", "dtype")
    dtype: Tuple[Union[np.dtype, Type]]

    def __init__(
        self,
        dtype: Union[Union[np.dtype, Type], Tuple[Union[np.dtype, Type]], None],
        *spec: Any,
        broadcast: Union[Literal["+", "="], Broadcast] = "+",
        bmax: Optional[int] = None,
        trailing=True,
    ) -> None:
        super().__init__(*spec, broadcast=broadcast, bmax=bmax, trailing=trailing)
        if dtype is None:
            dtype = (np.generic,)
        tdtype = cast(Tuple[Union[np.dtype, Type]], tuple(flatten(dtype)))
        for d in tdtype:
            if not np.issubdtype(d, np.generic):
                raise ValueError(
                    "dtype should be a subclass from np.generic, see "
                    "https://numpy.org/doc/stable/reference/arrays.scalars.html"
                )
        self.dtype = tdtype

    def __call__(self, dtype: np.dtype, ndim: int, *, S: StateType) -> AbstractTensor:
        if not any(np.issubdtype(dtype, target) for target in self.dtype):
            raise ValueError(f"not issubdtype({dtype}, {self.dtype})")
        return AbstractTensor(dtype, super().bound(ndim, S=S))


class OArg(Arg):
    __slots__ = ("spec", "dtype_promotion")

    dtype_promotion: SpecifyDtype

    def __init__(
        self,
        dtype: Union[SpecifyDtype, str, np.dtype],
        *spec: Any,
        broadcast: Union[Literal["+", "="], Broadcast] = "+",
        bmax: Optional[int] = None,
    ) -> None:
        super().__init__(*spec, broadcast=broadcast, bmax=bmax, trailing=False)
        if not isinstance(dtype, SpecifyDtype):
            self.dtype_promotion = ConstDtype(dtype)
        else:
            self.dtype_promotion = dtype

    def __call__(self, *input_dtypes: np.dtype, S: StateType) -> AbstractTensor:
        shapes = super().bound(None, S=S)
        dtype = self.dtype_promotion(*input_dtypes)
        return AbstractTensor(dtype, shapes)


class Signature:
    def __init__(self, inputs: Sequence[IArg], outputs: Sequence[OArg]) -> None:
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)

    def __call__(
        self, *inputs: Any
    ) -> Tuple[Tuple[AbstractTensor, ...], Tuple[AbstractTensor, ...]]:
        ainputs: Tuple[AbstractTensor, ...] = tuple(map(as_abstract, inputs))
        dtypes = [a.dtype for a in ainputs]
        state: StateType = dict()
        kinputs = [
            arg(inp.dtype, len(inp.dims), S=state)
            for arg, inp in zip(self.inputs, ainputs)
        ]
        koutputs = [arg(*dtypes, S=state) for arg in self.outputs]
        res = K.run(2, (kinputs, koutputs), K.eq(kinputs, ainputs))
        assert len(res) == 1
        return cast(
            Tuple[Tuple[AbstractTensor, ...], Tuple[AbstractTensor, ...]], res[0]
        )
