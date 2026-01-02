from collections.abc import Callable
from hashlib import sha256
from pickle import dump
from tempfile import NamedTemporaryFile
from typing import Any
from weakref import WeakKeyDictionary

import numba
from llvmlite import ir
from numba.core import cgutils
from numba.core.caching import CacheImpl, _CacheLocator

from pytensor.configdefaults import config


NUMBA_CACHE_PATH = config.base_compiledir / "numba"
NUMBA_CACHE_PATH.mkdir(parents=True, exist_ok=True)
CACHED_SRC_FUNCTIONS: WeakKeyDictionary[Callable, str] = WeakKeyDictionary()


class NumbaPyTensorCacheLocator(_CacheLocator):
    """Locator for Numba functions defined from PyTensor-generated source code.

    It uses an internally-defined hash to disambiguate functions.

    Functions returned by the PyTensor dispatchers are cached in the CACHED_SRC_FUNCTIONS
    weakref dictionary when `compile_numba_function_src` is called with a `cache_key`.
    When numba later attempts to find a cache for such a function, this locator gets triggered
    and directs numba to the PyTensor Numba cache directory, using the provided hash as disambiguator.

    It is not necessary that the python functions be cached by the dispatchers.
    As long as the key is the same, numba will be directed to the same cache entry, even if the function is fresh.
    Conversely, if the function changed but the key is the same, numba will still use the old cache.
    """

    def __init__(self, py_func, py_file, hash):
        self._py_func = py_func
        self._py_file = py_file
        self._hash = hash

    def ensure_cache_path(self):
        """We ensured this when the module was loaded.

        It's too slow to run every time a cache is needed.
        """
        pass

    def get_cache_path(self):
        """Return the directory the function is cached in."""
        return NUMBA_CACHE_PATH

    def get_source_stamp(self):
        """Get a timestamp representing the source code's freshness.
        Can return any picklable Python object.

        This can be used to invalidate all caches from previous PyTensor releases.
        """
        return 0

    def get_disambiguator(self):
        """Get a string disambiguator for this locator's function.
        It should allow disambiguating different but similarly-named functions.
        """
        return self._hash

    @classmethod
    def from_function(cls, py_func, py_file):
        """Create a locator instance for functions stored in CACHED_SRC_FUNCTIONS."""
        if py_func in CACHED_SRC_FUNCTIONS and config.numba__cache:
            return cls(py_func, py_file, CACHED_SRC_FUNCTIONS[py_func])


# Register our locator at the front of Numba's locator list
CacheImpl._locator_classes.insert(0, NumbaPyTensorCacheLocator)


def hash_from_pickle_dump(obj: Any) -> str:
    """Create a sha256 hash from the pickle dump of an object."""

    # Stream pickle directly into the hasher to avoid a large temporary bytes object
    hasher = sha256()

    class HashFile:
        def write(self, b):
            hasher.update(b)

    dump(obj, HashFile())
    return hasher.hexdigest()


def compile_numba_function_src(
    src: str,
    function_name: str,
    global_env: dict[Any, Any] | None = None,
    local_env: dict[Any, Any] | None = None,
    write_to_disk: bool = False,
    cache_key: str | None = None,
) -> Callable:
    """Compile (and optionally cache) a function from source code for use with Numba.

    This function compiles the provided source code string into a Python function
    with the specified name. If `store_to_disk` is True, the source code is written
    to a temporary file before compilation. The compiled function is then executed
    in the provided global and local environments.

    If a `cache_key` is provided the function is registered in a `CACHED_SRC_FUNCTIONS`
    weak reference dictionary, to be used by the `NumbaPyTensorCacheLocator` for caching.

    """
    if write_to_disk:
        with NamedTemporaryFile(delete=False) as f:
            filename = f.name
            f.write(src.encode())
    else:
        filename = "<string>"

    if global_env is None:
        global_env = {}

    if local_env is None:
        local_env = {}

    mod_code = compile(src, filename, mode="exec")
    exec(mod_code, global_env, local_env)

    res = local_env[function_name]
    res.__source__ = src

    if cache_key is not None:
        CACHED_SRC_FUNCTIONS[res] = cache_key

    return res  # type: ignore


@numba.extending.intrinsic(prefer_literal=True)
def _call_cached_ptr(typingctx, get_ptr_func, func_type_ref, unique_func_name_lit):
    """
    Enable caching of function pointers returned by `get_ptr_func`.

    When one of our Numba-dispatched functions depends on a pointer to a compiled function function (e.g. when we call
    cython_lapack routines), numba will refuse to cache the function, because the pointer may change between runs.

    This intrinsic allows us to cache the pointer ourselves, by storing it in a global variable keyed by a literal
    `unique_func_name_lit`. The first time the intrinsic is called, it will call `get_ptr_func` to get the pointer, store it
    in the global variable, and return it. Subsequent calls will load the pointer from the global variable.
    """
    func_type = func_type_ref.instance_type
    cache_key = unique_func_name_lit.literal_value

    def codegen(context, builder, signature, args):
        ptr_ty = ir.PointerType(ir.IntType(8))
        null = ptr_ty(None)
        align = 64

        mod = builder.module
        var = cgutils.add_global_variable(mod, ptr_ty, f"_ptr_cache_{cache_key}")
        var.align = align
        var.linkage = "private"
        var.initializer = null

        var_val = builder.load_atomic(var, "acquire", align)
        result_ptr = cgutils.alloca_once_value(builder, var_val)

        with builder.if_then(builder.icmp_signed("==", var_val, null), likely=False):
            sig = typingctx.resolve_function_type(get_ptr_func, [], {})
            f = context.get_function(get_ptr_func, sig)
            new_ptr = f(builder, [])
            new_ptr = builder.inttoptr(new_ptr, ptr_ty)
            builder.store_atomic(new_ptr, var, "release", align)
            builder.store(new_ptr, result_ptr)

        sfunc = cgutils.create_struct_proxy(func_type)(context, builder)
        sfunc.c_addr = builder.load(result_ptr)

        return sfunc._getvalue()

    sig = func_type(get_ptr_func, func_type_ref, unique_func_name_lit)
    return sig, codegen
