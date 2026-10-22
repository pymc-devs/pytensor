from pathlib import Path

from pytensor import config
from pytensor.bin.pytensor_cache import remove_extra_caches
from pytensor.link.utils import write_generated_src


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


def _write_key(compiledir, name, payload):
    entry = compiledir / name
    entry.mkdir()
    (entry / "key.pkl").write_bytes(payload)


def test_list_skips_keys_from_missing_modules_and_unreadable_keys(
    tmp_path, capsys, monkeypatch
):
    import pickle
    from types import SimpleNamespace

    from pytensor.compile import compiledir

    # A key pickled in an environment that had a package this one lacks: the
    # pickle names a module that cannot be imported here (protocol-0 GLOBAL).
    missing = b"cno_such_pkg_xyz.ops\nSomeOp\n."
    _write_key(tmp_path, "tmp_missing_1", missing)
    _write_key(tmp_path, "tmp_missing_2", missing)
    _write_key(tmp_path, "tmp_truncated", pickle.dumps({"a": 1})[:5])
    _write_key(tmp_path, "tmp_garbage", b"not a pickle")

    # compiledir is not a mutable flag, so point the module at tmp_path directly.
    monkeypatch.setattr(compiledir, "config", SimpleNamespace(compiledir=tmp_path))
    compiledir.print_compiledir_content()

    out = capsys.readouterr().out
    assert (
        "Skipped 2 key files whose Ops come from modules not installed in this "
        "environment: no_such_pkg_xyz (2)"
    ) in out
    assert "Skipped 2 key files that could not be read" in out
