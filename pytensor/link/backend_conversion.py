"""Cross-backend conversion registry for shared-variable representations.

Some backends (e.g. JAX) cannot operate on the host representation of certain
shared variables (a numpy ``Generator``, a scipy sparse matrix) and require a
backend-native representation instead. A backend registers how to convert a host
value *to* its native form and, when possible, back *from* it.

A compiled function keeps a per-backend view of such a shared value and
reconciles it lazily, so a single-backend loop never pays a conversion: the
conversion runs only when a value crosses backends (see
`SharedVariable._reconcile_into`).
"""

from collections.abc import Callable
from dataclasses import dataclass


HOST = "host"


@dataclass(frozen=True)
class BackendConversion:
    tag: str
    handles: Callable[[object], bool]
    to_native: Callable[[object], object]
    from_native: Callable[[object], object]
    lossy: bool = False


_CONVERSIONS: dict[str, BackendConversion] = {}


def register_backend_conversion(conversion: BackendConversion) -> None:
    _CONVERSIONS[conversion.tag] = conversion


def backend_conversion(tag: str) -> BackendConversion | None:
    return _CONVERSIONS.get(tag)


def backend_handles(tag: str, type_) -> bool:
    """Whether ``tag`` needs a native view for this Type (a type-and-backend property).

    Whether the Type is divergent for *some* backend at all is the intrinsic
    ``Type.is_backend_divergent`` property (import-order-independent); this asks
    whether the specific, already-imported ``tag`` provides a conversion for it.
    """
    conversion = _CONVERSIONS.get(tag)
    return conversion is not None and conversion.handles(type_)
