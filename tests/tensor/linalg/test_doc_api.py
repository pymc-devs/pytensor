"""Guard against drift between `pytensor.tensor.linalg.__all__` and the
`doc/library/tensor/linalg.rst` API page.

Both sides have to be edited together: if you add a new function to
`__all__`, this test will fail until you also document it in the rst,
and vice-versa. The doc page is hand-grouped into logical sections
(decomposition, solve, ...), which is why we don't generate it from
`__all__` automatically — but we do want a hard guarantee that the two
stay in sync.
"""

import re
from pathlib import Path

import pytensor.tensor.linalg as linalg


DOC_PATH = (
    Path(__file__).resolve().parents[3] / "doc" / "library" / "tensor" / "linalg.rst"
)

_AUTOFUNCTION_RE = re.compile(
    r"^\.\.\s+autofunction::\s+pytensor\.tensor\.linalg\.(\w+)\s*$",
    re.MULTILINE,
)


def _documented_names() -> set[str]:
    return set(_AUTOFUNCTION_RE.findall(DOC_PATH.read_text()))


def test_linalg_doc_matches_all():
    documented = _documented_names()
    exported = set(linalg.__all__)

    missing_from_docs = exported - documented
    missing_from_all = documented - exported

    assert not missing_from_docs, (
        f"{len(missing_from_docs)} name(s) in pytensor.tensor.linalg.__all__ "
        f"are not documented in {DOC_PATH.name}: {sorted(missing_from_docs)}. "
        "Add an `.. autofunction::` entry to the appropriate section."
    )
    assert not missing_from_all, (
        f"{len(missing_from_all)} name(s) documented in {DOC_PATH.name} "
        f"are not in pytensor.tensor.linalg.__all__: {sorted(missing_from_all)}. "
        "Either add them to `__all__` or remove the rst entry."
    )


def test_linalg_doc_has_no_duplicate_entries():
    text = DOC_PATH.read_text()
    documented = _AUTOFUNCTION_RE.findall(text)
    duplicates = {name for name in documented if documented.count(name) > 1}
    assert not duplicates, (
        f"Duplicate `.. autofunction::` entries in {DOC_PATH.name}: "
        f"{sorted(duplicates)}"
    )
