import pytest


@pytest.fixture(scope="session", autouse=True)
def mlx_cpu_default():
    # GitHub's macOS runners ship an older Metal stack that aborts when a
    # CPU-produced array (mlx.linalg is CPU-only) feeds an op on the default GPU
    # stream.
    mx = pytest.importorskip("mlx.core")
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)
