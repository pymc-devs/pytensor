import importlib
import os
import sys
import tempfile
from collections.abc import Callable
from typing import Any

import numba
import numba.core.caching
from numba.core.caching import CacheImpl


class PyTensorLoader(importlib.abc.SourceLoader):
    def __init__(self):
        # Key is "pytensor_generated_" + hash of pytensor graph
        self._module_sources = {}
        self._module_globals = {}
        self._module_locals = {}

    def get_source(self, fullname):
        if fullname not in self._module_sources:
            raise ImportError()
        return self._module_sources[fullname]

    def get_data(self, path):
        if path not in self._module_sources:
            raise ImportError()
        return self._module_sources[path].encode("utf-8")

    def get_filename(self, path):
        if path not in self._module_sources:
            raise ImportError()
        return path

    def add_module(self, name, src, global_env, local_env):
        self._module_sources[name] = src
        self._module_globals[name] = global_env
        self._module_locals[name] = local_env

    def exec_module(self, module):
        name = module.__name__
        variables = module.__dict__
        variables.update(self._module_globals[name])
        variables.update(self._module_locals[name])
        code = compile(self._module_sources[name], name, "exec")
        exec(code, variables)

    def create_module(self, spec):
        return None


pytensor_loader = PyTensorLoader()


def load_module(key, src, global_env, local_env):
    pytensor_loader.add_module(key, src, global_env, local_env)
    spec = importlib.util.spec_from_loader(key, pytensor_loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[key] = module
    return module


class NumbaPyTensorCacheLocator(numba.core.caching._CacheLocator):
    def __init__(self, py_func, py_file):
        # print(f"New locator {py_func=}, {py_file=}")
        self._py_func = py_func
        self._py_file = py_file
        self._hash = py_file
        # src_hash = hash(pytensor_loader._module_sources[self._py_file])
        # self._hash = hash((src_hash, py_file, pytensor.__version__))

    def ensure_cache_path(self):
        path = self.get_cache_path()
        os.makedirs(path, exist_ok=True)
        # Ensure the directory is writable by trying to write a temporary file
        tempfile.TemporaryFile(dir=path).close()

    def get_cache_path(self):
        """
        Return the directory the function is cached in.
        """
        return "~/.cache/pytensor"

    def get_source_stamp(self):
        """
        Get a timestamp representing the source code's freshness.
        Can return any picklable Python object.
        """

        return self._hash

    def get_disambiguator(self):
        """
        Get a string disambiguator for this locator's function.
        It should allow disambiguating different but similarly-named functions.
        """
        return None

    @classmethod
    def from_function(cls, py_func, py_file):
        """
        Create a locator instance for the given function located in the
        given file.
        """
        if py_func.__module__ in pytensor_loader._module_sources:
            return cls(py_func, py_file)


CacheImpl._locator_classes.append(NumbaPyTensorCacheLocator)


def compile_function_src2(
    key: str,
    src: str,
    function_name: str,
    global_env: dict[Any, Any] | None = None,
    local_env: dict[Any, Any] | None = None,
) -> Callable:
    # with NamedTemporaryFile(delete=False) as f:
    #     filename = f.name
    #     f.write(src.encode())

    if global_env is None:
        global_env = {}

    if local_env is None:
        local_env = {}

    # mod_code = compile(src, filename, mode="exec")
    # exec(mod_code, global_env, local_env)
    # print(key, src)
    module = load_module(key, src, global_env, local_env)
    res = getattr(module, function_name)

    # res = cast(Callable, res)
    # res.__source__ = src  # type: ignore
    return res
