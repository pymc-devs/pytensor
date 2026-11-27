from pathlib import Path
from types import SimpleNamespace

import pytest

from pytensor import config
from pytensor.compile import compiledir
from pytensor.compile.compiledir import (
    CompiledirContent,
    basecompiledir_ls,
    collect_compiledir_content,
    print_compiledir_content,
    remove_extra_caches,
    render_basecompiledir_ls,
    render_compiledir_content,
)
from pytensor.link.utils import write_generated_src


@pytest.fixture
def fake_config(tmp_path, monkeypatch):
    cfg = SimpleNamespace(compiledir=tmp_path, base_compiledir=tmp_path)
    monkeypatch.setattr(compiledir, "config", cfg)
    return cfg


def test_remove_extra_caches(tmp_path):
    for name in ("numba", "compiledir_keep"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "file").write_text("x")
    src_dir = tmp_path / "src"

    with config.change_flags(generated_src_dir=str(src_dir)):
        generated = Path(write_generated_src("def one():\n    return 1\n"))
        other = src_dir / "other.py"
        other.write_text("x")

        remove_extra_caches(tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "compiledir_keep",
            "src",
        ]
        assert not generated.exists()
        assert other.exists()


def test_collect_compiledir_content_empty(fake_config):
    content = collect_compiledir_content()
    assert isinstance(content, CompiledirContent)
    assert content.compiledir == fake_config.compiledir
    assert content.table == []
    assert content.table_multiple_ops == []
    assert content.zeros_op == 0
    assert content.unreadable_keys == 0


def test_collect_compiledir_content_counts_unreadable_keys(fake_config):
    (fake_config.compiledir / "no_key").mkdir()  # no key.pkl: skipped
    bad = fake_config.compiledir / "bad_key"
    bad.mkdir()
    (bad / "key.pkl").write_bytes(b"")  # empty file: EOFError
    content = collect_compiledir_content()
    assert content.unreadable_keys == 1


def test_render_compiledir_content(capsys, fake_config):
    render_compiledir_content(collect_compiledir_content())
    out = capsys.readouterr().out
    assert f"PyTensor cache: {fake_config.compiledir}" in out
    assert "List of 0 compiled individual ops" in out
    assert "Skipped 0 files that contained 0 op" in out


def test_print_compiledir_content_matches_collect_and_render(capsys, fake_config):
    print_compiledir_content()
    printed = capsys.readouterr().out
    render_compiledir_content(collect_compiledir_content())
    assert capsys.readouterr().out == printed


def test_basecompiledir_ls_returns_sorted_lists(fake_config):
    (fake_config.base_compiledir / "b_dir").mkdir()
    (fake_config.base_compiledir / "a_dir").mkdir()
    (fake_config.base_compiledir / "file.txt").write_text("x")
    subdirs, others = basecompiledir_ls()
    assert subdirs == [
        fake_config.base_compiledir / "a_dir",
        fake_config.base_compiledir / "b_dir",
    ]
    assert others == [fake_config.base_compiledir / "file.txt"]


def test_render_basecompiledir_ls(capsys, fake_config):
    render_basecompiledir_ls([], [])
    out = capsys.readouterr().out
    assert "Sub-directories (possible compile caches):" in out
    assert "(None)" in out
