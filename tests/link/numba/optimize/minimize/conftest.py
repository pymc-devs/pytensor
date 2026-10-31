import pytest

from pytensor.tensor.optimize import MinimizeOp


@pytest.fixture
def native_only(monkeypatch):
    def perform(*args):
        raise AssertionError("Numba minimize used the full Python fallback")

    monkeypatch.setattr(MinimizeOp, "perform", perform)
