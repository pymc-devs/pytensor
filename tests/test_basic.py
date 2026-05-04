import subprocess
import sys

import pytensor


def test_root_module_not_polluted():
    import types

    # Filter out submodules since other tests may have imported them
    module_items = sorted(
        i
        for i in dir(pytensor)
        if not i.startswith("__")
        and not isinstance(getattr(pytensor, i), types.ModuleType)
    )
    assert module_items == [
        "In",
        "Lop",
        "Mode",
        "OpFromGraph",
        "Out",
        "Rop",
        "config",
        "dprint",
        "function",
        "get_mode",
        "grad",
        "ifelse",
        "map",
        "pullback",
        "pushforward",
        "scan",
        "shared",
        "wrap_jax",
        "wrap_py",
    ]


def test_eager_import_scipy_submodules():
    expensive_scipy_modules = (
        "scipy.linalg",
        "scipy.sparse",
        "scipy.special",
        "scipy.signal",
        "scipy.stats",
    )

    code = (
        "import sys\n"
        "import pytensor.tensor  # noqa: F401\n"
        f"loaded = [m for m in {expensive_scipy_modules!r} if m in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = [m for m in result.stdout.strip().split(",") if m]
    assert not loaded, (
        f"`import pytensor.tensor` eagerly loaded scipy submodules: {loaded}. "
        "These should be deferred to first use."
    )
