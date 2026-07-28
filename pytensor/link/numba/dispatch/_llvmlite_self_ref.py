"""Import-time shim giving stock llvmlite the ability to emit self-referential
metadata nodes (``!0 = !{ !0 }``).

Self-referential nodes are how LLVM makes ``!alias.scope``/``!noalias`` domains and
scopes globally unique. llvmlite only supports them once this PR lands, adding
``Module.add_metadata(operands, self_ref=True)``:

    https://github.com/numba/llvmlite/pull/895

This shim provides the same ``self_ref`` keyword on older llvmlite so the alias-scope
markers emitted by ``vectorize_codegen`` work without a patched llvmlite. It is a
no-op when the native API is present.
"""

import inspect

from llvmlite import ir
from llvmlite.ir import module as _ll_module
from llvmlite.ir import values as _ll_values


def ensure_self_ref_metadata_support() -> None:
    """Patch ``Module.add_metadata`` to accept ``self_ref=True`` if it doesn't already."""
    if "self_ref" in inspect.signature(ir.Module.add_metadata).parameters:
        return
    if getattr(_ll_module.Module, "_pytensor_self_ref_patched", False):
        return

    base_add_metadata = _ll_module.Module.add_metadata

    class _SelfRefMDValue(_ll_values.MDValue):
        """Metadata node whose first operand is itself.

        The self-reference is kept out of the hashed/compared state: a self-ref
        scope is routinely used as an operand of another metadata node (e.g. an
        ``alias.scope`` set), and hashing a tuple that transitively contains the
        node would otherwise recurse forever. Equality falls back to identity,
        matching the uniqueness guarantee self-referential nodes exist to provide.
        """

        def __init__(self, parent, operands, name):
            super().__init__(parent, operands, name)
            self._self_ref_tail = tuple(operands)
            self.operands = (self, *self._self_ref_tail)

        def __hash__(self):
            return hash(self._self_ref_tail)

        def __eq__(self, other):
            return self is other

        def __ne__(self, other):
            return self is not other

    def add_metadata(self, operands, *, self_ref=False):
        if not self_ref:
            return base_add_metadata(self, operands)
        if not isinstance(operands, list | tuple):
            raise TypeError(
                f"expected a list or tuple of metadata values, got {operands!r}"
            )
        operands = self._fix_metadata_operands(operands)
        return _SelfRefMDValue(self, operands, name=str(len(self.metadata)))

    _ll_module.Module.add_metadata = add_metadata
    _ll_module.Module._pytensor_self_ref_patched = True
