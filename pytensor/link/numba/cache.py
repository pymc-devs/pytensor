from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from numba.core.caching import CacheImpl, _CacheLocator

from pytensor import config


NUMBA_PYTENSOR_CACHE_ENABLED = True
NUMBA_CACHE_PATH = config.base_compiledir / "numba"
NUMBA_CACHE_PATH.mkdir(exist_ok=True)
CACHED_SRC_FUNCTIONS = {}


def compile_and_cache_numba_function_src(
    src: str,
    function_name: str,
    global_env: dict[Any, Any] | None = None,
    local_env: dict[Any, Any] | None = None,
    key: str | None = None,
) -> Callable:
    if key is not None:
        filename = NUMBA_CACHE_PATH / key
        with filename.open("wb") as f:
            f.write(src.encode())
    else:
        with NamedTemporaryFile(delete=False) as f:
            filename = f.name
            f.write(src.encode())

    if global_env is None:
        global_env = {}

    if local_env is None:
        local_env = {}

    mod_code = compile(src, filename, mode="exec")
    exec(mod_code, global_env, local_env)

    res = local_env[function_name]
    res.__source__ = src  # type: ignore

    if key is not None:
        CACHED_SRC_FUNCTIONS[res] = key
    return res


def cache_numba_function(
    fn,
    key: str | None = None,
) -> Callable:
    if key is not None:
        CACHED_SRC_FUNCTIONS[fn] = key
    return fn


class NumbaPyTensorCacheLocator(_CacheLocator):
    def __init__(self, py_func, py_file, hash):
        # print(f"New locator {py_func=}, {py_file=}, {hash=}")
        self._py_func = py_func
        self._py_file = py_file
        self._hash = hash
        # src_hash = hash(pytensor_loader._module_sources[self._py_file])
        # self._hash = hash((src_hash, py_file, pytensor.__version__))

    def ensure_cache_path(self):
        pass

    def get_cache_path(self):
        """
        Return the directory the function is cached in.
        """
        return NUMBA_CACHE_PATH

    def get_source_stamp(self):
        """
        Get a timestamp representing the source code's freshness.
        Can return any picklable Python object.
        """
        return 0

    def get_disambiguator(self):
        """
        Get a string disambiguator for this locator's function.
        It should allow disambiguating different but similarly-named functions.
        """
        return self._hash

    @classmethod
    def from_function(cls, py_func, py_file):
        """
        Create a locator instance for the given function located in the given file.
        """
        # py_file = Path(py_file).parent
        # if py_file == (config.base_compiledir / "numba"):
        if NUMBA_PYTENSOR_CACHE_ENABLED and py_func in CACHED_SRC_FUNCTIONS:
            # print(f"Applies to {py_file}")
            return cls(py_func, Path(py_file).parent, CACHED_SRC_FUNCTIONS[py_func])


CacheImpl._locator_classes.insert(0, NumbaPyTensorCacheLocator)
