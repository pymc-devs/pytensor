"""Types of `pytensor_llvm`, the Rust extension of PyTensor's LLVM backend."""

from . import interp as interp
from . import ir as ir

ABI_VERSION: int

def llvm_version() -> str: ...
def c_header() -> str:
    """The C header of the C API (`pytensor_llvm.h`)."""

def threads() -> int:
    """The number of threads parallel loops use."""

def set_threads(n: int) -> None:
    """Set the number of threads (before the pool is first used)."""
