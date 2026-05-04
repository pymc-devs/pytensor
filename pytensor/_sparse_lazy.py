"""Lazy registration of scipy.sparse handlers on pytensor's dispatchers.

Imported by `pytensor/__init__.py` so `pytensor.sparse` doesn't have to be
loaded eagerly at startup. The fallbacks match `type(x).__module__` against
`scipy.sparse` as a string, so this module itself doesn't import scipy.sparse;
the real handlers (and their scipy.sparse dependency) are pulled in only when
an actual scipy.sparse value is passed to `as_symbolic` / `shared`.
"""

from pytensor.basic import _as_symbolic
from pytensor.compile.sharedvalue import shared_constructor


def _lazy_as_symbolic_sparse(x):
    if type(x).__module__.startswith("scipy.sparse"):
        from pytensor.sparse.basic import as_symbolic_sparse

        return as_symbolic_sparse
    return None


def _lazy_shared_sparse(x):
    if type(x).__module__.startswith("scipy.sparse"):
        from pytensor.sparse.sharedvar import sparse_constructor

        return sparse_constructor
    return None


_as_symbolic.register_lazy(_lazy_as_symbolic_sparse)  # type: ignore[attr-defined]
shared_constructor.register_lazy(_lazy_shared_sparse)  # type: ignore[attr-defined]
