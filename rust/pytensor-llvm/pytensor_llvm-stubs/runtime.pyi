"""`pytensor_llvm.runtime`: compiled functions, Rust kernels, the C API (not typed yet)."""

from typing import Any

def __getattr__(name: str) -> Any: ...
