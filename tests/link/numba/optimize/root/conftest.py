import pytest

from pytensor.tensor.optimize import RootOp


@pytest.fixture
def native_only(monkeypatch):
    def perform(*args):
        raise AssertionError("Numba root used the full Python fallback")

    monkeypatch.setattr(RootOp, "perform", perform)
