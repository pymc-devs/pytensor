"""TUTORIAL.md: its code runs."""

import re
from pathlib import Path

import pytest


pytest.importorskip("pytensor_llvm")

import pytensor.link.llvm  # noqa: E402


TUTORIAL = Path(pytensor.link.llvm.__file__).parent / "TUTORIAL.md"


def test_the_tutorials_code_runs(capsys):
    """Every ```python block, in order, in one namespace. (Blocks marked ```py are
    fragments shown for illustration and are not run.)"""
    text = TUTORIAL.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```$", text, flags=re.M | re.S)
    assert len(blocks) > 10
    namespace = {"__name__": "tutorial"}
    for k, block in enumerate(blocks):
        exec(compile(block, f"TUTORIAL.md block {k + 1}", "exec"), namespace)
    printed = capsys.readouterr().out
    # what the tutorial shows as output is what is printed
    for shown in re.findall(r"^```text\n(.*?)^```$", text, flags=re.M | re.S):
        for line in shown.splitlines():
            if line.strip() and line.strip() != "...":
                assert line in printed, line
