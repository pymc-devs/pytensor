import warnings
import weakref
from collections.abc import Callable
from functools import singledispatch, wraps
from hashlib import sha256
from pathlib import Path
from pickle import dumps
from tempfile import NamedTemporaryFile
from typing import Any

from numba.core.caching import CacheImpl, _CacheLocator

from pytensor import config
from pytensor.link.numba.compile import numba_funcify, numba_njit


NUMBA_PYTENSOR_CACHE_ENABLED = True
NUMBA_CACHE_PATH = config.base_compiledir / "numba"
NUMBA_CACHE_PATH.mkdir(exist_ok=True)
CACHED_SRC_FUNCTIONS = weakref.WeakKeyDictionary()


class NumbaPyTensorCacheLocator(_CacheLocator):
    def __init__(self, py_func, py_file, hash):
        self._py_func = py_func
        self._py_file = py_file
        self._hash = hash

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


@singledispatch
def numba_funcify_default_op_cache_key(
    op, node=None, **kwargs
) -> Callable | tuple[Callable, Any]:
    """Funcify an Op and implement a default cache key.

    The default cache key is based on the op class and its properties.
    It does not take into account the node inputs or other context.
    Note that numba will use the array dtypes, rank and layout as part of the cache key,
    but not the static shape or constant values.
    If the funcify implementation exploits this information, then this method should not be used.
    Instead dispatch directly on `numba_funcify_and_cache_key` (or just numba_funcify)
    which won't use any cache key.
    """
    # Default cache key of None which means "don't try to do directly cache this function"
    raise NotImplementedError()


def register_funcify_default_op_cache_key(op_type):
    """Register a funcify implementation for both cache and non-cache versions."""

    def decorator(dispatch_func):
        # Register with the cache key dispatcher
        numba_funcify_default_op_cache_key.register(op_type)(dispatch_func)

        # Create a wrapper for the non-cache dispatcher
        @wraps(dispatch_func)
        def dispatch_func_wrapper(*args, **kwargs):
            func, _key = dispatch_func(*args, **kwargs)
            # Discard the key for the non-cache version
            return func

        # Register the wrapper with the non-cache dispatcher
        numba_funcify.register(op_type)(dispatch_func_wrapper)

        return dispatch_func

    return decorator


@singledispatch
def numba_funcify_and_cache_key(op, node=None, **kwargs) -> tuple[Callable, str | None]:
    # Default cache key of None which means "don't try to do directly cache this function"
    if hasattr(op, "_props"):
        try:
            func_and_salt = numba_funcify_default_op_cache_key(op, node=node, **kwargs)
        except NotImplementedError:
            pass
        else:
            if isinstance(func_and_salt, tuple):
                func, salt = func_and_salt
            else:
                func, salt = func_and_salt, "0"
            props_dict = op._props_dict()
            if not props_dict:
                # Simple op, just use the type string as key
                key_bytes = str((type(op), salt)).encode()
            else:
                # Simple props, can use string representation of props as key
                simple_types = (str, bool, int, type(None), float)
                container_types = (tuple, frozenset)
                if all(
                    isinstance(v, simple_types)
                    or (
                        isinstance(v, container_types)
                        and all(isinstance(i, simple_types) for i in v)
                    )
                    for v in props_dict.values()
                ):
                    key_bytes = str(
                        (type(op), tuple(props_dict.items()), salt)
                    ).encode()
                else:
                    # Complex props, use pickle to serialize them
                    key_bytes = dumps((str(type(op)), tuple(props_dict.items()), salt))
            return func, sha256(key_bytes).hexdigest()

    # Fallback
    return numba_funcify(op, node=node, **kwargs), None


def register_funcify_and_cache_key(op_type):
    """Register a funcify implementation for both cache and non-cache versions."""

    def decorator(dispatch_func):
        # Register with the cache key dispatcher
        numba_funcify_and_cache_key.register(op_type)(dispatch_func)

        # Create a wrapper for the non-cache dispatcher
        @wraps(dispatch_func)
        def dispatch_func_wrapper(*args, **kwargs):
            func, _key = dispatch_func(*args, **kwargs)
            # Discard the key for the non-cache version
            return func

        # Register the wrapper with the non-cache dispatcher
        numba_funcify.register(op_type)(dispatch_func_wrapper)

        return dispatch_func_wrapper

    return decorator


def numba_njit_and_cache(op, *args, **kwargs):
    jitable_func, key = numba_funcify_and_cache_key(op, *args, **kwargs)

    if key is not None:
        # To force numba to use our cache, we must compile the function so that any closure
        # becomes a global variable...
        op_name = op.__class__.__name__
        cached_func = compile_numba_function_src(
            src=f"def {op_name}(*args): return jitable_func(*args)",
            function_name=op_name,
            global_env=globals() | {"jitable_func": jitable_func},
            cache_key=key,
        )
        return numba_njit(cached_func, final_function=True, cache=True), key
    else:
        if config.numba__cache and config.compiler_verbose:
            warnings.warn(
                f"Custom numba cache disabled for {op} of type {type(op)}. "
                f"Even if the function is cached by numba, larger graphs using this function cannot be cached.\n"
                "To enable custom caching, register a numba_funcify_and_cache_key implementation for this Op, with a proper cache key."
            )

        return numba_njit(
            lambda *args: jitable_func(*args), final_function=True, cache=False
        ), None


def compile_numba_function_src(
    src: str,
    function_name: str,
    global_env: dict[Any, Any] | None = None,
    local_env: dict[Any, Any] | None = None,
    store_to_disk: bool = False,
    cache_key: str | None = None,
) -> Callable:
    if store_to_disk:
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
    res.__source__ = src  # type: ignore

    if cache_key is not None:
        CACHED_SRC_FUNCTIONS[res] = cache_key
    return res
