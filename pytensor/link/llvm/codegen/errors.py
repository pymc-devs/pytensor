"""The exception a failed call raises: error codes.

A failing check reports an error code with its message (`Codegen.check`), and the
call raises the exception of that code. The common exceptions are fixed kinds of the
runtime (`ERR_VALUE` etc. in `pytensor_llvm.runtime`). Any other exception class (the
`exc_type` of a `CheckAndRaise`) gets a code of the program being compiled: the codes
from `ERR_CUSTOM` on index its `ExceptionTable`, which its function keeps.
"""

from dataclasses import dataclass, field
from functools import cache

import numpy as np


@cache
def _fixed_codes() -> dict[type[BaseException], int]:
    # (the extension is optional: imported when code is generated)
    from pytensor_llvm import runtime as rt

    return {
        RuntimeError: rt.ERR_RUNTIME,
        ValueError: rt.ERR_VALUE,
        IndexError: rt.ERR_INDEX,
        np.linalg.LinAlgError: rt.ERR_LINALG,
        AssertionError: rt.ERR_ASSERTION,
    }


@dataclass(eq=False)
class ExceptionTable:
    """The exception classes of a program's error codes beyond the fixed kinds: code
    `ERR_CUSTOM + k` raises `classes[k]`. Everything generated for one program shares
    one (`KernelModule.exceptions`)."""

    classes: list[type[BaseException]] = field(default_factory=list)

    def code(self, exc: type[BaseException]) -> int:
        """The error code that raises `exc`."""
        fixed = _fixed_codes().get(exc)
        if fixed is not None:
            return fixed
        if exc not in self.classes:
            self.classes.append(exc)
        from pytensor_llvm import runtime as rt

        return rt.ERR_CUSTOM + self.classes.index(exc)
