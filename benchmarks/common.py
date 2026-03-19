"""Common utilities for ASV benchmarks."""

import sys
from pathlib import Path


# ASV doesn't add the repo root to sys.path, so `tests` isn't importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from fixtures import create_radon_model


__all__ = ["create_radon_model"]
