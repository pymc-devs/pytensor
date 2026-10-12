"""Deprecated: use ``pytensor.conditional`` instead."""

import warnings

from pytensor.conditional import op as _op
from pytensor.conditional import rewriting as _rewriting


_MOVED_NAMES = {
    "IfElse": _op,
    "LazyIfElse": _op,
    "ifelse": _op,
    "cond_make_inplace": _rewriting,
    "ifelse_lift_single_if_through_acceptable_ops": _rewriting,
    "cond_merge_ifs_true": _rewriting,
    "cond_merge_ifs_false": _rewriting,
    "CondMerge": _rewriting,
    "cond_remove_identical": _rewriting,
    "cond_merge_random_op": _rewriting,
}
__all__ = list(_MOVED_NAMES)


def __getattr__(name):
    if name in _MOVED_NAMES:
        module = _MOVED_NAMES[name]
        warnings.warn(
            f"Importing {name!r} from 'pytensor.ifelse' is deprecated. "
            f"Use 'from {module.__name__} import {name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
