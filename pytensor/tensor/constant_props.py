"""Cached static predicates over the *data* of constant variables.

Each helper memoizes its result on the variable's ``tag`` so repeated rewrite
passes do not re-scan the same constant. These only inspect constant data, so
this stays a dependency-free leaf module (graph-walking sign analysis such as
``is_provably_positive`` lives in ``subtensor`` with the ops it recurses over).
"""

import numpy as np

from pytensor.graph.basic import Constant


def constant_is_all_negative(var) -> bool:
    """Whether ``var`` is a constant whose entries are all negative, cached on its tag."""
    if not isinstance(var, Constant):
        return False
    cached: bool | None = getattr(var.tag, "all_negative", None)
    if cached is not None:
        return cached
    result = bool(np.all(np.asarray(var.data) < 0))
    var.tag.all_negative = result
    return result


def constant_indices_are_unique(idx) -> bool:
    """Check whether a constant index has no duplicate entries.

    Boolean indices, scalars, and single-element arrays are trivially unique.
    For larger integer arrays, indices that mix positive and negative values
    may alias, so those are treated as potentially duplicated.  The result
    is cached on ``idx.tag``.
    """
    if not isinstance(idx, Constant):
        return False
    cached = getattr(idx.tag, "unique_indices", None)
    if cached is not None:
        return bool(cached)
    idx_val = np.asarray(idx.data)
    if idx_val.dtype == bool:
        result = True
    elif idx_val.size <= 1:
        result = True
    else:
        has_pos = (idx_val >= 0).any()
        has_neg = (idx_val < 0).any()
        result = not (has_pos and has_neg) and np.unique(idx_val).size == idx_val.size
    idx.tag.unique_indices = result
    return result


def constant_is_arange(idx) -> tuple[int, int, int] | None:
    """Match ``idx`` to ``np.arange(offset, offset + d * step, step)``
    and return ``(d, offset, step)``, else ``None``.

    Single-element constants return ``(1, value, 1)``.  The result is cached
    on ``idx.tag.is_arange`` (``False`` sentinels a no-match).
    """
    if not isinstance(idx, Constant):
        return None
    cached = getattr(idx.tag, "is_arange", None)
    if cached is not None:
        return cached if cached is not False else None
    idx_val = np.asarray(idx.data)
    if idx_val.ndim != 1 or idx_val.size == 0 or idx_val.dtype.kind not in "iu":
        result: tuple[int, int, int] | None = None
    elif idx_val.size == 1:
        result = (1, int(idx_val[0]), 1)
    else:
        diffs = np.diff(idx_val)
        step = int(diffs[0])
        if step != 0 and np.all(diffs == step):
            result = (int(idx_val.size), int(idx_val[0]), step)
        else:
            result = None
    idx.tag.is_arange = result if result is not None else False
    return result
