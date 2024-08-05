import os
import shutil
from pathlib import Path
from tempfile import mkdtemp

from pytensor.misc.pkl_utils import StripPickler
from pytensor.tensor.type import matrix


# FIXME: this test looks weird
class TestStripPickler:
    def setup_method(self):
        # Work in a temporary directory to avoid cluttering the repository
        self.origdir = Path.cwd()
        self.tmpdir = mkdtemp()
        os.chdir(self.tmpdir)

    def teardown_method(self):
        # Get back to the original dir, and delete the temporary one
        os.chdir(self.origdir)
        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir)

    def test_basic(self):
        with Path("test.pkl").open("wb"):
            m = matrix()
            dest_pkl = "my_test.pkl"
            with Path(dest_pkl).open("wb") as f:
                strip_pickler = StripPickler(f, protocol=-1)
                strip_pickler.dump(m)
