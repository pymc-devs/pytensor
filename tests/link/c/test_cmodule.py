"""
We don't have real tests for the cache, but it would be great to make them!

But this one tests a current behavior that isn't good: the c_code isn't
deterministic based on the input type and the op.
"""

import multiprocessing
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.function import function
from pytensor.compile.ops import DeepCopyOp
from pytensor.configdefaults import config
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.link.c.basic import CLinker
from pytensor.link.c.cmodule import GCC_compiler, ModuleCache, default_blas_ldflags
from pytensor.link.c.exceptions import CompileError
from pytensor.link.c.op import COp
from pytensor.tensor.type import dvectors, vector


class MyOp(DeepCopyOp):
    nb_called = 0

    def c_code_cache_version(self):
        return ()

    def c_code(self, node, name, inames, onames, sub):
        MyOp.nb_called += 1
        (iname,) = inames
        (oname,) = onames
        fail = sub["fail"]
        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, version = self.c_code_and_version[itype]
            rand = np.random.random()
            return f'printf("{rand}\\n");{code % locals()}'
        # Else, no C code
        return super(DeepCopyOp, self).c_code(node, name, inames, onames, sub)


class MyAdd(COp):
    __props__ = ()

    def make_node(self, *inputs):
        outputs = [vector()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, out_):
        (out,) = out_
        out[0] = inputs[0][0] + 1

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        return f"{z} = {x} + 1;"


class MyAddVersioned(MyAdd):
    def c_code_cache_version(self):
        return (1,)


def test_compiler_error():
    with pytest.raises(CompileError), tempfile.TemporaryDirectory() as dir_name:
        GCC_compiler.compile_str("module_name", "blah", location=dir_name)


def test_inter_process_cache():
    """
    TODO FIXME: This explanation is very poorly written.

    When a `COp` with `COp.c_code`, but no version. If we have two `Apply`
    nodes in a graph with distinct inputs variable, but the input variables
    have the same `Type`, do we reuse the same module? Even if they would
    generate different `COp.c_code`?  Currently this test show that we generate
    the `COp.c_code` only once.

    This is to know if the `COp.c_code` can add information specific to the
    ``node.inputs[*].owner`` like the name of the variable.

    """

    x, y = dvectors("xy")
    f = function([x, y], [MyOp()(x), MyOp()(y)])
    f(np.arange(60), np.arange(60))
    if config.mode == "FAST_COMPILE" or config.cxx == "":
        assert MyOp.nb_called == 0
    else:
        assert MyOp.nb_called == 1

    # What if we compile a new function with new variables?
    x, y = dvectors("xy")
    f = function([x, y], [MyOp()(x), MyOp()(y)])
    f(np.arange(60), np.arange(60))
    if config.mode == "FAST_COMPILE" or config.cxx == "":
        assert MyOp.nb_called == 0
    else:
        assert MyOp.nb_called == 1


@pytest.mark.filterwarnings("error")
def test_cache_versioning():
    """Make sure `ModuleCache._add_to_cache` is working."""

    my_add = MyAdd()
    with pytest.warns(match=".*specifies no C code cache version.*"):
        assert my_add.c_code_cache_version() == ()

    my_add_ver = MyAddVersioned()
    assert my_add_ver.c_code_cache_version() == (1,)

    assert len(MyOp.__props__) == 0
    assert len(MyAddVersioned.__props__) == 0

    x = vector("x")

    z = my_add(x)
    z_v = my_add_ver(x)

    with tempfile.TemporaryDirectory() as dir_name:
        cache = ModuleCache(dir_name)

        lnk = CLinker().accept(FunctionGraph(outputs=[z]))
        with pytest.warns(match=".*specifies no C code cache version.*"):
            key = lnk.cmodule_key()
        assert key[0] == ()

        with pytest.warns(match=".*c_code_cache_version.*"):
            cache.module_from_key(key, lnk)

        lnk_v = CLinker().accept(FunctionGraph(outputs=[z_v]))
        key_v = lnk_v.cmodule_key()
        assert len(key_v[0]) > 0

        assert key_v not in cache.entry_from_key

        stats_before = cache.stats[2]
        cache.module_from_key(key_v, lnk_v)
        assert stats_before < cache.stats[2]


def test_flag_detection():
    """
    TODO FIXME: This is a very poor test.

    Check that the code detecting blas flags does not raise any exception.
    It used to happen on Python 3 because of improper string handling,
    but was not detected because that path is not usually taken,
    so we test it here directly.
    """
    res = GCC_compiler.try_flags(["-lblas"])
    assert isinstance(res, bool)


@pytest.fixture(
    scope="module",
    params=["mkl_intel", "mkl_gnu", "openblas", "lapack", "blas", "no_blas"],
)
def blas_libs(request):
    key = request.param
    libs = {
        "mkl_intel": ["mkl_core", "mkl_rt", "mkl_intel_thread", "iomp5", "pthread"],
        "mkl_gnu": ["mkl_core", "mkl_rt", "mkl_gnu_thread", "gomp", "pthread"],
        "openblas": ["openblas", "gfortran", "gomp", "m"],
        "lapack": ["lapack", "blas", "cblas", "m"],
        "blas": ["blas", "cblas"],
        "no_blas": [],
    }
    return libs[key]


@pytest.fixture(scope="function", params=["Linux", "Windows", "Darwin"])
def mock_system(request):
    with patch("platform.system", return_value=request.param):
        yield request.param


@pytest.fixture()
def cxx_search_dirs(blas_libs, mock_system):
    libext = {"Linux": "so", "Windows": "dll", "Darwin": "dylib"}
    libtemplate = f"{{lib}}.{libext[mock_system]}"
    libraries = []
    with tempfile.TemporaryDirectory() as d:
        flags = None
        for lib in blas_libs:
            lib_path = Path(d) / libtemplate.format(lib=lib)
            lib_path.write_bytes(b"1")
            libraries.append(lib_path)
            if flags is None:
                flags = f"-l{lib}"
            else:
                flags += f" -l{lib}"
        if "gomp" in blas_libs and "mkl_gnu_thread" not in blas_libs:
            flags += " -fopenmp"
        if len(blas_libs) == 0:
            flags = ""
        yield f"libraries: ={d}".encode(sys.stdout.encoding), flags


@pytest.fixture(
    scope="function", params=[False, True], ids=["Working_CXX", "Broken_CXX"]
)
def cxx_search_dirs_status(request):
    return request.param


@patch("pytensor.link.c.cmodule.std_lib_dirs", return_value=[])
@patch("pytensor.link.c.cmodule.check_mkl_openmp", return_value=None)
def test_default_blas_ldflags(
    mock_std_lib_dirs, mock_check_mkl_openmp, cxx_search_dirs, cxx_search_dirs_status
):
    cxx_search_dirs, expected_blas_ldflags = cxx_search_dirs
    mock_process = MagicMock()
    if cxx_search_dirs_status:
        error_message = ""
        mock_process.communicate = lambda *args, **kwargs: (cxx_search_dirs, b"")
        mock_process.returncode = 0
    else:
        error_message = "Unsupported argument -print-search-dirs"
        error_message_bytes = error_message.encode(sys.stderr.encoding)
        mock_process.communicate = lambda *args, **kwargs: (b"", error_message_bytes)
        mock_process.returncode = 1
    with patch("pytensor.link.c.cmodule.subprocess_Popen", return_value=mock_process):
        with patch.object(
            pytensor.link.c.cmodule.GCC_compiler,
            "try_compile_tmp",
            return_value=(True, True),
        ):
            if cxx_search_dirs_status:
                assert set(default_blas_ldflags().split(" ")) == set(
                    expected_blas_ldflags.split(" ")
                )
            else:
                expected_warning = re.escape(
                    "Pytensor cxx failed to communicate its search dirs. As a consequence, "
                    "it might not be possible to automatically determine the blas link flags to use.\n"
                    f"Command that was run: {config.cxx} -print-search-dirs\n"
                    f"Output printed to stderr: {error_message}"
                )
                with pytest.warns(
                    UserWarning,
                    match=expected_warning,
                ):
                    assert default_blas_ldflags() == ""


def test_default_blas_ldflags_no_cxx():
    with pytensor.config.change_flags(cxx=""):
        assert default_blas_ldflags() == ""


@pytest.fixture()
def windows_conda_libs(blas_libs):
    libtemplate = "{lib}.dll"
    libraries = []
    with tempfile.TemporaryDirectory() as d:
        subdir = Path(d) / "Library" / "bin"
        subdir.mkdir(exist_ok=True, parents=True)
        flags = f'-L"{subdir}"'
        for lib in blas_libs:
            lib_path = subdir / libtemplate.format(lib=lib)
            lib_path.write_bytes(b"1")
            libraries.append(lib_path)
            flags += f" -l{lib}"
        if "gomp" in blas_libs and "mkl_gnu_thread" not in blas_libs:
            flags += " -fopenmp"
        if len(blas_libs) == 0:
            flags = ""
        yield d, flags


@patch("pytensor.link.c.cmodule.std_lib_dirs", return_value=[])
@patch("pytensor.link.c.cmodule.check_mkl_openmp", return_value=None)
def test_default_blas_ldflags_conda_windows(
    mock_std_lib_dirs, mock_check_mkl_openmp, windows_conda_libs
):
    mock_sys_prefix, expected_blas_ldflags = windows_conda_libs
    mock_process = MagicMock()
    mock_process.communicate = lambda *args, **kwargs: (b"", b"")
    mock_process.returncode = 0
    with patch("sys.platform", "win32"):
        with patch("sys.prefix", mock_sys_prefix):
            with patch(
                "pytensor.link.c.cmodule.subprocess_Popen", return_value=mock_process
            ):
                with patch.object(
                    pytensor.link.c.cmodule.GCC_compiler,
                    "try_compile_tmp",
                    return_value=(True, True),
                ):
                    assert set(default_blas_ldflags().split(" ")) == set(
                        expected_blas_ldflags.split(" ")
                    )


@patch(
    "os.listdir", return_value=["mkl_core.1.dll", "mkl_rt.1.0.dll", "mkl_rt.1.1.lib"]
)
@patch("sys.platform", "win32")
def test_patch_ldflags(listdir_mock):
    mkl_path = Path("some_path")
    flag_list = ["-lm", "-lopenblas", f"-L {mkl_path}", "-l mkl_core", "-lmkl_rt"]
    assert GCC_compiler.patch_ldflags(flag_list) == [
        "-lm",
        "-lopenblas",
        f"-L {mkl_path}",
        '"' + str(mkl_path / "mkl_core.1.dll") + '"',
        '"' + str(mkl_path / "mkl_rt.1.0.dll") + '"',
    ]


@patch(
    "os.listdir",
    return_value=[
        "libopenblas.so",
        "libm.a",
        "mkl_core.1.dll",
        "mkl_rt.1.0.dll",
        "mkl_rt.1.1.dll",
    ],
)
@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
def test_linking_patch(listdir_mock, platform):
    libs = ["openblas", "m", "mkl_core", "mkl_rt"]
    lib_dirs = ['"mock_dir"']
    with patch("sys.platform", platform):
        if platform == "win32":
            assert GCC_compiler.linking_patch(lib_dirs, libs) == [
                "-lopenblas",
                "-lm",
                '"' + str(Path(lib_dirs[0].strip('"')) / "mkl_core.1.dll") + '"',
                '"' + str(Path(lib_dirs[0].strip('"')) / "mkl_rt.1.1.dll") + '"',
            ]
        else:
            GCC_compiler.linking_patch(lib_dirs, libs) == [
                "-lopenblas",
                "-lm",
                "-lmkl_core",
                "-lmkl_rt",
            ]


def test_cache_race_condition():
    with tempfile.TemporaryDirectory() as dir_name:

        @config.change_flags(on_opt_error="raise", on_shape_error="raise")
        def f_build(factor):
            # Some of the caching issues arise during constant folding within the
            # optimization passes, so we need these config changes to prevent the
            # exceptions from being caught
            a = pt.vector()
            f = pytensor.function([a], factor * a)
            return f(np.array([1], dtype=config.floatX))

        ctx = multiprocessing.get_context()
        compiledir_prop = pytensor.config._config_var_dict["compiledir"]

        # The module cache must (initially) be `None` for all processes so that
        # `ModuleCache.refresh` is called
        with (
            patch.object(compiledir_prop, "val", dir_name, create=True),
            patch.object(pytensor.link.c.cmodule, "_module_cache", None),
        ):
            assert pytensor.config.compiledir == dir_name

            num_procs = 30
            rng = np.random.default_rng(209)

            for i in range(10):
                # A random, constant input to prevent caching between runs
                factor = rng.random()
                procs = [
                    ctx.Process(target=f_build, args=(factor,))
                    for i in range(num_procs)
                ]
                for proc in procs:
                    proc.start()
                for proc in procs:
                    proc.join()

                assert not any(
                    exit_code != 0 for exit_code in [proc.exitcode for proc in procs]
                )
