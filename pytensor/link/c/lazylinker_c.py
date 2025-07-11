import logging
import sys
import warnings
from importlib import reload
from pathlib import Path
from types import ModuleType

import pytensor
from pytensor.compile.compilelock import lock_ctx
from pytensor.configdefaults import config
from pytensor.link.c.cmodule import GCC_compiler


_logger = logging.getLogger(__file__)

force_compile = False
version = 0.31  # must match constant returned in function get_version()
lazylinker_ext: ModuleType | None = None


def try_import():
    global lazylinker_ext
    sys.path[0:0] = [str(config.compiledir)]
    import lazylinker_ext

    del sys.path[0]


def try_reload():
    sys.path[0:0] = [str(config.compiledir)]
    reload(lazylinker_ext)
    del sys.path[0]


try:
    # See gh issue #728 for why these lines are here. Summary: compiledir must
    # be at the beginning of the path to avoid conflicts with any other
    # lazylinker_ext modules that might exist (this step handled in try_import
    # and try_reload). An __init__.py file must be created for the same reason.
    # Note that these lines may seem redundant (they are repeated in
    # compile_str()) but if another lazylinker_ext does exist then it will be
    # imported and compile_str won't get called at all.
    location = config.compiledir / "lazylinker_ext"
    location.mkdir(exist_ok=True)

    init_file = location / "__init__.py"
    if not init_file.exists():
        try:
            with init_file.open("w"):
                pass
        except OSError as e:
            if init_file.exists():
                pass  # has already been created
            else:
                e.args += (f"{location} exist? {location.exists()}",)
                raise

    _need_reload = False
    if force_compile:
        raise ImportError()
    else:
        try_import()
        _need_reload = True
        actual_version = getattr(lazylinker_ext, "_version", None)
        if version != actual_version:
            raise ImportError(
                "Version check of the existing lazylinker compiled file."
                f" Looking for version {version}, but found {actual_version}. "
                f"Extra debug information: force_compile={force_compile}, _need_reload={_need_reload}"
            )
except ImportError:
    with lock_ctx():
        # Maybe someone else already finished compiling it while we were
        # waiting for the lock?
        try:
            if force_compile:
                raise ImportError()
            if _need_reload:
                # The module was successfully imported earlier: we need to
                # reload it to check if the version was updated.
                try_reload()
            else:
                try_import()
                _need_reload = True
            actual_version = getattr(lazylinker_ext, "_version", None)
            if version != actual_version:
                raise ImportError(
                    "Version check of the existing lazylinker compiled file."
                    f" Looking for version {version}, but found {actual_version}. "
                    f"Extra debug information: force_compile={force_compile}, _need_reload={_need_reload}"
                )
        except ImportError:
            # It is useless to try to compile if there isn't any
            # compiler!  But we still want to try to load it, in case
            # the cache was copied from another computer.
            if not config.cxx:
                raise
            _logger.info("Compiling new CVM")
            dirname = "lazylinker_ext"
            cfile = Path(pytensor.__path__[0]) / "link/c/c_code/lazylinker_c.c"
            if not cfile.exists():
                # This can happen in not normal case. We just
                # disable the c clinker. If we are here the user
                # didn't disable the compiler, so print a warning.
                warnings.warn(
                    "The file lazylinker_c.c is not available. This do"
                    "not happen normally. You are probably in a strange"
                    "setup. This mean PyTensor can not use the cvm:"
                    "our c execution engine for PyTensor function. If you"
                    "want to remove this warning, use the PyTensor flag"
                    "'cxx=' (set to an empty string) to disable all c"
                    "code generation."
                )
                raise ImportError("The file lazylinker_c.c is not available.")

            code = cfile.read_text("utf-8")

            loc = config.compiledir / dirname
            loc.mkdir(exist_ok=True)

            args = GCC_compiler.compile_args()
            GCC_compiler.compile_str(dirname, code, location=loc, preargs=args)
            # Save version into the __init__.py file.
            init_py = loc / "__init__.py"

            init_py.write_text(f"_version = {version}\n")

            # If we just compiled the module for the first time, then it was
            # imported at the same time: we need to make sure we do not
            # reload the now outdated __init__.pyc below.
            init_pyc = loc / "__init__.pyc"
            init_pyc.unlink(missing_ok=True)

            try_import()
            try_reload()
            from lazylinker_ext import lazylinker_ext as lazy_c

            assert (
                lazylinker_ext is not None
                and lazylinker_ext._version == lazy_c.get_version()
            )
            _logger.info(f"New version {lazylinker_ext._version}")

from lazylinker_ext.lazylinker_ext import CLazyLinker, get_version  # noqa
from lazylinker_ext.lazylinker_ext import *  # noqa

assert force_compile or (version == get_version())
