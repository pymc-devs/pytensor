#!/usr/bin/env python
"""Bump the numba upper bound in pyproject.toml when a new numba release is out.

Queries PyPI for the latest stable numba release and compares it against the
current ``<=`` cap pinned in pyproject.toml. The pin is rewritten to the
canonical ``"numba>=X.Y,<=A.B.C"`` form whenever:

* the latest release is newer than the cap; or
* the existing pin is in a non-canonical shape (``<`` instead of ``<=``, or
  no upper bound at all). Form drift triggers a PR so the inconsistency is
  surfaced for review rather than silently smoothed over.

Exit code:
    0  - no bump needed (pin is canonical and at the latest release)
    0  - bump written (emits GITHUB_OUTPUT for the workflow)
    1  - unexpected state (no pin found, inconsistent pins, etc.)
"""

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

from packaging.version import Version


PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
# Match any quoted numba spec: the bare name, or numba followed by a version
# operator. The spec body (group "spec") is what we re-parse and rewrite.
PIN_RE = re.compile(r'"numba(?P<spec>(?:[<>=!~][^"]*)?)"')
LOWER_RE = re.compile(r">=\d+\.\d+(?:\.\d+)?")
UPPER_RE = re.compile(r"<(=?)(\d+\.\d+(?:\.\d+)?)")
DEFAULT_LOWER = ">=0.58"
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


def parse_spec(spec: str) -> tuple[str, str | None, Version | None]:
    """Return ``(lower_clause, upper_op, upper_cap)`` for a numba pin body.

    ``upper_op`` is ``"<"``, ``"<="``, or ``None`` when no upper bound is
    present. A missing lower bound falls back to ``DEFAULT_LOWER`` so the
    rewritten pin still constrains both sides.
    """
    lower_m = LOWER_RE.search(spec)
    lower = lower_m.group(0) if lower_m else DEFAULT_LOWER
    upper_m = UPPER_RE.search(spec)
    if upper_m is None:
        return lower, None, None
    op = "<=" if upper_m.group(1) == "=" else "<"
    return lower, op, Version(upper_m.group(2))


def main() -> int:
    text = PYPROJECT.read_text()
    matches = list(PIN_RE.finditer(text))
    if not matches:
        print(f"no numba pin found in {PYPROJECT}", file=sys.stderr)
        return 1

    parsed = [parse_spec(m.group("spec")) for m in matches]
    if len(set(parsed)) != 1:
        print(f"inconsistent numba pins: {parsed}", file=sys.stderr)
        return 1
    lower, upper_op, current_cap = parsed[0]

    latest = fetch_latest_numba()
    latest_str = f"{latest.major}.{latest.minor}.{latest.micro}"
    print(f"current pin spec: numba{matches[0].group('spec')}")
    print(f"latest numba release: {latest}")

    if upper_op == "<=" and current_cap == latest:
        print("no bump needed")
        emit_output("bumped", "false")
        return 0

    new_spec = f"{lower},<={latest_str}"
    new_text = PIN_RE.sub(f'"numba{new_spec}"', text)
    PYPROJECT.write_text(new_text)

    if upper_op is None:
        reason = "pin had no upper bound"
    elif upper_op != "<=":
        reason = "pin used '<' instead of '<='"
    else:
        reason = f"upper bound was {current_cap}, latest is {latest}"
    print(f"rewrote pin ({reason}); new spec: numba{new_spec}")
    emit_output("bumped", "true")
    emit_output("latest", str(latest))
    emit_output("new_cap", latest_str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
