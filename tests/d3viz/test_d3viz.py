import filecmp
import tempfile
from pathlib import Path

import numpy as np
import pytest

import pytensor.d3viz as d3v
from pytensor import compile
from pytensor.compile.function import function
from pytensor.configdefaults import config
from pytensor.printing import _try_pydot_import
from tests.d3viz import models


try:
    _try_pydot_import()
except Exception as e:
    pytest.skip(f"pydot not available: {e!s}", allow_module_level=True)


class TestD3Viz:
    def setup_method(self):
        self.rng = np.random.default_rng(0)
        self.data_dir = Path("data") / "test_d3viz"

    def check(self, f, reference=None, verbose=False):
        tmp_dir = Path(tempfile.mkdtemp())
        html_file = tmp_dir / "index.html"
        if verbose:
            print(html_file)  # noqa: T201
        d3v.d3viz(f, html_file)
        assert html_file.stat().st_size > 0
        if reference:
            assert filecmp.cmp(html_file, reference)

    def test_mlp(self):
        m = models.Mlp()
        f = function(m.inputs, m.outputs)
        self.check(f)

    def test_mlp_profiled(self):
        if config.mode in ("DebugMode", "DEBUG_MODE"):
            pytest.skip("Can't profile in DebugMode")
        m = models.Mlp()
        profile = compile.profiling.ProfileStats(False)
        f = function(m.inputs, m.outputs, profile=profile)
        x_val = self.rng.normal(0, 1, (1000, m.nfeatures))
        f(x_val)
        self.check(f)

    def test_ofg(self):
        m = models.Ofg()
        f = function(m.inputs, m.outputs)
        self.check(f)

    def test_ofg_nested(self):
        m = models.OfgNested()
        f = function(m.inputs, m.outputs)
        self.check(f)

    def test_ofg_simple(self):
        m = models.OfgSimple()
        f = function(m.inputs, m.outputs)
        self.check(f)
