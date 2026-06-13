import pytest

from pytensor.link.c.section_loader import (
    C_CODE_SECTIONS,
    load_c_code_sections,
    split_c_code_sections,
)


def test_sections_append_across_files():
    code_a = "#section support_code\nint helper_a();\n#section code\nrun_a();\n"
    code_b = "#section support_code\nint helper_b();\n"

    sections = split_c_code_sections([code_a, code_b], ["a.c", "b.c"])

    assert sections["support_code"] == "\nint helper_a();\n\nint helper_b();\n"
    assert sections["code"] == "\nrun_a();\n"


def test_all_section_names_accepted():
    code = "".join(f"#section {name}\n// {name}\n" for name in sorted(C_CODE_SECTIONS))

    sections = split_c_code_sections([code], ["all.c"])

    assert set(sections) == C_CODE_SECTIONS


def test_stray_code_before_first_section_raises():
    code = "int stray();\n#section code\nrun();\n"

    with pytest.raises(
        ValueError, match=r"Stray code before first #section statement \(in file a\.c\)"
    ):
        split_c_code_sections([code], ["a.c"])


def test_unknown_section_raises():
    code = "#section not_a_section\nrun();\n"

    with pytest.raises(ValueError, match=r"Unknown section type \(in file a\.c\)"):
        split_c_code_sections([code], ["a.c"])


def test_no_section_marker_raises():
    with pytest.raises(
        ValueError, match=r"No valid section marker was found in file a\.c"
    ):
        split_c_code_sections(["int orphan();\n"], ["a.c"])


def test_mixed_old_and_new_markers_raise():
    old_style = "PYTENSOR_SUPPORT_CODE_SECTION\nint helper();\n"
    new_style = "#section code\nrun();\n"

    with pytest.raises(ValueError, match="Both the new and the old syntax"):
        split_c_code_sections([old_style, new_style], ["old.c", "new.c"])


def test_legacy_markers_still_parse():
    code = (
        "PYTENSOR_SUPPORT_CODE_SECTION\n"
        "int helper();\n"
        "PYTENSOR_APPLY_CODE_SECTION\n"
        "int apply_helper();\n"
    )

    sections = split_c_code_sections([code], ["legacy.c"])

    assert sections["support_code"] == "\nint helper();\n"
    assert sections["support_code_apply"] == "\nint apply_helper();\n"


def test_load_c_code_sections_reads_files(tmp_path):
    file_a = tmp_path / "a.c"
    file_b = tmp_path / "b.c"
    file_a.write_text("#section support_code\nint helper();\n")
    file_b.write_text("#section code\nrun();\n")

    func_codes, sections = load_c_code_sections([file_a, file_b])

    assert func_codes == [file_a.read_text(), file_b.read_text()]
    assert sections == {"support_code": "\nint helper();\n", "code": "\nrun();\n"}
