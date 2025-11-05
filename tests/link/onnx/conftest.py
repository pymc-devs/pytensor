"""Pytest configuration and fixtures for ONNX backend tests."""

import numpy as np
import pytest

from pytensor.configdefaults import config

# Import hypothesis if available
try:
    from hypothesis import HealthCheck, Phase, Verbosity, settings

    # Hypothesis profiles for different testing scenarios
    settings.register_profile("dev", max_examples=10, deadline=None)
    settings.register_profile(
        "ci",
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    settings.register_profile(
        "debug",
        max_examples=10,
        verbosity=Verbosity.verbose,
        phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target],
    )

    # Load dev profile by default
    settings.load_profile("dev")
except ImportError:
    # Hypothesis not available, tests will skip
    pass


@pytest.fixture(scope="module", autouse=True)
def set_pytensor_flags():
    """Module-level PyTensor configuration."""
    with config.change_flags(cxx="", compute_test_value="ignore", floatX="float32"):
        yield


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def float32_vector(rng):
    """Sample float32 vector for testing."""
    return rng.normal(size=10).astype("float32")


@pytest.fixture
def float32_matrix(rng):
    """Sample float32 matrix for testing."""
    return rng.normal(size=(5, 5)).astype("float32")
