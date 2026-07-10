import os
import sys
import sysconfig

import pytest


# Using pytest_plugins causes `tests/link/c/test_cmodule.py::test_cache_versioning` to fail
# pytest_plugins = ["tests.fixtures"]


def pytest_sessionstart(session):
    os.environ["PYTENSOR_FLAGS"] = ",".join(
        [
            os.environ.setdefault("PYTENSOR_FLAGS", ""),
            "on_opt_error=raise,on_shape_error=raise,cmodule__warn_no_version=True",
        ]
    )
    os.environ["NUMBA_BOUNDSCHECK"] = "1"


def pytest_sessionfinish(session, exitstatus):
    # On a free-threaded build, importing any single-phase-init C extension
    # re-enables the GIL. The default (numba) backend must not, so if the GIL is
    # back on, some test pulled in PyTensor's own C extensions. Checked after the
    # whole session so it is independent of collection order.
    if sysconfig.get_config_var("Py_GIL_DISABLED") == 1 and sys._is_gil_enabled():
        print(  # noqa: T201
            "\nERROR: the GIL was re-enabled during the test session. "
            "A C-extension was imported on a free-threaded build.",
            file=sys.stderr,
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
