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

        # A second call, with nothing left to remove, is a no-op.
        remove_extra_caches(tmp_path)
