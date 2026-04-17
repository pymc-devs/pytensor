#!/usr/bin/env python
"""Bump the numba upper bound in pyproject.toml when a new numba minor is out.

Queries PyPI for the latest stable numba release. If it is >= the current
upper bound pinned in pyproject.toml, rewrite the pin to ``<major.minor+1``
relative to that release.

Exit code:
    0  - no bump needed (already at or above latest)
    0  - bump written (emits GITHUB_OUTPUT for the workflow)
    1  - unexpected state (no pin found, conflicting pins, etc.)
"""

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

from packaging.version import Version


PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
# Matches pins of the form  "numba>=X.Y,<A.B"  with either a main-dep or
# optional-extra quote style. The cap (group 2) is what we bump.
PIN_RE = re.compile(r'("numba>=\d+\.\d+,<)(\d+\.\d+)(")')
PYPI_URL = "https://pypi.org/pypi/numba/json"


def fetch_latest_numba() -> Version:
    with urllib.request.urlopen(PYPI_URL, timeout=30) as resp:
        data = json.load(resp)
    candidates: list[Version] = []
    for raw, files in data["releases"].items():
        if not files:
            continue
        v = Version(raw)
        if v.is_prerelease or v.is_devrelease:
            continue
        candidates.append(v)
    if not candidates:
        raise RuntimeError("no stable numba release found on PyPI")
    return max(candidates)


def emit_output(key: str, value: str) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        return
    with Path(out).open("a") as f:
        f.write(f"{key}={value}\n")


def main() -> int:
    text = PYPROJECT.read_text()
    pins = PIN_RE.findall(text)
    if not pins:
        print(f"no numba pin matching {PIN_RE.pattern} in {PYPROJECT}", file=sys.stderr)
        return 1
    caps = {Version(cap) for _, cap, _ in pins}
    if len(caps) != 1:
        print(f"expected one numba upper bound, found {caps}", file=sys.stderr)
        return 1
    current_cap = caps.pop()
    latest = fetch_latest_numba()
    print(f"current upper bound: <{current_cap}")
    print(f"latest numba release: {latest}")

    if latest < current_cap:
        print("no bump needed")
        emit_output("bumped", "false")
        return 0

    new_cap = Version(f"{latest.major}.{latest.minor + 1}")
    new_text = PIN_RE.sub(rf"\g<1>{new_cap}\g<3>", text)
    PYPROJECT.write_text(new_text)
    print(f"bumped upper bound to <{new_cap}")
    emit_output("bumped", "true")
    emit_output("latest", str(latest))
    emit_output("new_cap", str(new_cap))
    return 0


if __name__ == "__main__":
    sys.exit(main())
