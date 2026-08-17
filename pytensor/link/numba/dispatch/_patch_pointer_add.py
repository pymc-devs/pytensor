"""Patch ``numba.core.cgutils.pointer_add`` to preserve pointer provenance.

Fixes a numba miscompile (numba issue `#10605
<https://github.com/numba/numba/issues/10605>`_). ``pointer_add`` forms the
result pointer with a ``ptrtoint``/``inttoptr`` round-trip, whose
provenance-less result LLVM miscompiles once the access is inlined (as it is
inside a scan), producing wrong results or a bogus ``MemoryError`` from
``_empty_nd_impl``. Non-contiguous (``'A'``-layout) element access reaches this
through ``get_item_pointer2``.

Every numba caller of ``pointer_add`` adds an offset that stays within the
object the pointer points into (see the audit in `numba#10696
<https://github.com/numba/numba/pull/10696>`_), so forming the result with a
byte-wise ``getelementptr`` -- which preserves provenance -- is always valid.

Imported for its side effect from ``pytensor.link.numba.dispatch``; applied only
on numba < 0.67, where the fix landed. Drop this module once that is our minimum.
"""

from llvmlite import ir
from numba import version_info as numba_version_info
from numba.core import cgutils


_int8_ptr = ir.IntType(8).as_pointer()


def _pointer_add_gep(builder, ptr, offset, return_type=None):
    if isinstance(offset, int):
        offset = cgutils.intp_t(offset)
    base = builder.bitcast(ptr, _int8_ptr)
    addr = builder.gep(base, [offset])
    return builder.bitcast(addr, return_type or ptr.type)


# Fixed upstream in numba 0.67.0 (numba/numba#10696)
if numba_version_info.short < (0, 67):
    # Every numba caller resolves ``pointer_add`` through the module at call time,
    # so reassigning the attribute reaches them all (including ``get_item_pointer2``).
    cgutils.pointer_add = _pointer_add_gep
