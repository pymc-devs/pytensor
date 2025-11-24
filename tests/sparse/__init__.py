import pytest

from pytensor.compile import get_default_mode
from pytensor.link.numba import NumbaLinker


if isinstance(get_default_mode().linker, NumbaLinker):
    pytest.skip(
        reason="Numba does not support Sparse Ops yet",
        allow_module_level=True,
    )
