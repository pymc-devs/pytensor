"""Tests for the deprecation shims in `pytensor.tensor.{nlinalg,slinalg}`.

After the linalg reorganization, the old `nlinalg` and `slinalg` module
namespaces survive only as `__getattr__`-based shims that emit a
`DeprecationWarning` and forward attribute access to the new
`pytensor.tensor.linalg` locations.
"""

import importlib

import pytest

from pytensor.tensor import linalg as new_linalg


@pytest.mark.parametrize(
    ("module_name", "attr"),
    [
        ("pytensor.tensor.nlinalg", "svd"),
        ("pytensor.tensor.nlinalg", "MatrixInverse"),
        ("pytensor.tensor.nlinalg", "norm"),
        ("pytensor.tensor.slinalg", "cholesky"),
        ("pytensor.tensor.slinalg", "Solve"),
        ("pytensor.tensor.slinalg", "solve_continuous_lyapunov"),
    ],
)
def test_shim_emits_deprecation_warning_and_forwards(module_name, attr):
    module = importlib.import_module(module_name)

    with pytest.warns(DeprecationWarning, match=module_name):
        obj = getattr(module, attr)

    # The shim should forward to the same object exposed by the new public API.
    assert obj is getattr(new_linalg, attr)


def test_shim_raises_on_unknown_attribute():
    import pytensor.tensor.nlinalg as nlinalg
    import pytensor.tensor.slinalg as slinalg

    with pytest.raises(AttributeError):
        nlinalg.this_name_does_not_exist

    with pytest.raises(AttributeError):
        slinalg.this_name_does_not_exist
