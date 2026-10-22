import sys
from pathlib import Path

import pytest

from pytensor import config
from pytensor.bin.pytensor_cache import main, remove_extra_caches
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


@pytest.mark.parametrize(
    "argv, exit_status",
    [(["pytensor-cache", "bogus"], 1), (["pytensor-cache", "help"], 0)],
)
def test_usage_goes_to_stderr_only_on_error(argv, exit_status, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == exit_status

    out, err = capsys.readouterr()
    usage, silent = (err, out) if exit_status else (out, err)
    assert silent == ""
    assert 'Type "pytensor-cache clear" to erase the cache' in usage
    assert ('command "pytensor-cache bogus" not recognized' in usage) == bool(
        exit_status
    )
