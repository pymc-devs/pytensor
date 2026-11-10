import importlib

import pytest

import pytensor
from pytensor.conditional import op, rewriting


@pytest.mark.parametrize(
    "name, module",
    [
        ("ifelse", op),
        ("IfElse", op),
        ("cond_make_inplace", rewriting),
    ],
)
def test_deprecated_ifelse_import(name, module):
    legacy = importlib.import_module("pytensor.ifelse")
    with pytest.warns(DeprecationWarning, match="pytensor.ifelse"):
        assert getattr(legacy, name) is getattr(module, name)
    # Importing the old module must preserve the root helper with the same name.
    assert pytensor.ifelse is op.ifelse


def test_unknown_attribute():
    legacy = importlib.import_module("pytensor.ifelse")
    with pytest.raises(AttributeError):
        legacy.unknown_attribute
