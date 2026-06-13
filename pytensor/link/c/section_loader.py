import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from re import Pattern


C_CODE_SECTIONS: frozenset[str] = frozenset(
    {
        "init_code",
        "init_code_apply",
        "init_code_struct",
        "support_code",
        "support_code_apply",
        "support_code_struct",
        "cleanup_code_struct",
        "code",
        "code_cleanup",
    }
)

SECTION_RE: Pattern = re.compile(r"^#section ([a-zA-Z0-9_]+)$", re.MULTILINE)
BACKWARD_RE: Pattern = re.compile(
    r"^PYTENSOR_(APPLY|SUPPORT)_CODE_SECTION$", re.MULTILINE
)


def read_c_code_files(func_files: Iterable[Path | str]) -> list[str]:
    """Read the text of each C source file.

    Parameters
    ----------
    func_files : iterable of Path or str
        Paths to the files to read. Relative paths resolve against the current
        working directory; callers that support class-relative paths (e.g.
        `ExternalCOp.get_path`) must resolve them before calling.

    Returns
    -------
    list of str
        The contents of each file, in order.
    """
    return [Path(func_file).read_text(encoding="utf-8") for func_file in func_files]


def split_c_code_sections(
    func_codes: Sequence[str], func_files: Sequence[Path | str]
) -> dict[str, str]:
    """Split C source texts on ``#section`` markers into a section dict.

    Each source must consist of ``#section <name>`` markers (names from
    `C_CODE_SECTIONS`) followed by the section's code. When several sources define
    the same section, later sources append to earlier ones. The legacy
    ``PYTENSOR_(APPLY|SUPPORT)_CODE_SECTION`` markers are still recognized, but may
    not be mixed with ``#section`` markers across the provided sources.

    Parameters
    ----------
    func_codes : sequence of str
        The C source texts to split.
    func_files : sequence of Path or str
        The path each source was read from, used only in error messages.

    Returns
    -------
    dict mapping str to str
        Section name to accumulated section code.

    Raises
    ------
    ValueError
        If old- and new-style markers are mixed, a source has code before its first
        marker, a section name is unknown, or a source has no markers at all.
    """
    old_markers_present = any(BACKWARD_RE.search(code) for code in func_codes)
    new_markers_present = any(SECTION_RE.search(code) for code in func_codes)

    if old_markers_present and new_markers_present:
        raise ValueError(
            "Both the new and the old syntax for "
            "identifying code sections are present in the "
            "provided C code. These two syntaxes should not "
            "be used at the same time."
        )

    code_sections: dict[str, str] = {}

    for func_file, code in zip(func_files, func_codes, strict=True):
        if BACKWARD_RE.search(code):
            # Legacy markers, still accepted for back-compat.
            split = BACKWARD_RE.split(code)
            n = 1
            while n < len(split):
                if split[n] == "APPLY":
                    code_sections["support_code_apply"] = split[n + 1]
                elif split[n] == "SUPPORT":
                    code_sections["support_code"] = split[n + 1]
                n += 2
            continue

        elif SECTION_RE.search(code):
            # Check for code outside of the supported sections
            split = SECTION_RE.split(code)
            if split[0].strip() != "":
                raise ValueError(
                    "Stray code before first #section "
                    f"statement (in file {func_file}): {split[0]}"
                )

            # Separate the code into the proper sections
            n = 1
            while n < len(split):
                if split[n] not in C_CODE_SECTIONS:
                    raise ValueError(
                        f"Unknown section type (in file {func_file}): {split[n]}"
                    )
                if split[n] not in code_sections:
                    code_sections[split[n]] = ""
                code_sections[split[n]] += split[n + 1]
                n += 2

        else:
            raise ValueError(f"No valid section marker was found in file {func_file}")

    return code_sections


def load_c_code_sections(
    func_files: Sequence[Path | str],
) -> tuple[list[str], dict[str, str]]:
    """Read C source files and split them into ``#section``-marked sections.

    Parameters
    ----------
    func_files : sequence of Path or str
        Paths to the files to read, also used in error messages. Relative paths
        resolve against the current working directory.

    Returns
    -------
    func_codes : list of str
        The raw text of each file, needed separately for cache-version hashing.
    code_sections : dict mapping str to str
        Section name to accumulated section code, as returned by
        `split_c_code_sections`.
    """
    func_codes = read_c_code_files(func_files)
    return func_codes, split_c_code_sections(func_codes, func_files)
