"""`pytensor_llvm.runtime`: compiled functions, Rust kernels, the C API (the kernels
aren't typed yet)."""

from collections.abc import Sequence
from typing import Any

import numpy as np

class Function:
    """A compiled program's function, callable from Python and (`c_function`) from C."""

    def __init__(
        self,
        address: int,
        config: int,
        inputs: Sequence[tuple[np.dtype, Sequence[int | None], bool]],
        outputs: Sequence[tuple[np.dtype, Sequence[int | None], bool]],
        keepalive: Sequence[Any],
        uses_pool: bool,
    ) -> None:
        """`inputs`: `(dtype, shape, contiguous)` per graph input (shape: None where
        only known at run time; contiguous: the generated code assumes C-contiguous
        memory); `outputs`: `(dtype, shape, provided)` per graph output (provided:
        written into memory the caller provides, else returned in a buffer).
        `keepalive` is kept as long as the function: what `address` and `config` belong
        to. `uses_pool`: may the function run work on the runtime's thread pool?"""
    def __call__(self, *args: Any) -> list[Any]: ...
    def seed(self, seed: int) -> None:
        """Restart the random state of Python calls from `seed`."""
    def set_parallel(self, on: bool) -> None:
        """Allow or forbid parallel loops for Python calls."""
    def c_function(self) -> int:
        """A new reference to this function for the C API, as the address of a
        `PtFunction` (release it with `function_decref` from `c_api_table()`)."""
    @staticmethod
    def c_api_table() -> int:
        """The address of the `PtFunctionApi` table for functions from `c_function()`."""

def c_api_table() -> int:
    """The address of the `PtFunctionApi` table for functions from
    `Function.c_function()` (header: `pytensor_llvm.c_header()`)."""

def __getattr__(name: str) -> Any: ...
