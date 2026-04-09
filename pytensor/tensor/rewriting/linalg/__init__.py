"""Linear algebra rewrites, organized by category.

Importing this package registers all linalg rewrites with the optimizer.
"""

from pytensor.tensor.rewriting.linalg import (
    decomposition,
    inverse,
    products,
    solve,
    summary,
    utils,
)

# Re-export for backwards compatibility
from pytensor.tensor.rewriting.linalg.inverse import inv_to_solve


__all__ = [
    "decomposition",
    "inv_to_solve",
    "inverse",
    "products",
    "solve",
    "summary",
    "utils",
]
