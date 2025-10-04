import weakref
from hashlib import sha256
from pathlib import Path

from numba.core.caching import CacheImpl, _CacheLocator

from pytensor import config
from pytensor.graph.basic import Apply


NUMBA_PYTENSOR_CACHE_ENABLED = True
NUMBA_CACHE_PATH = config.base_compiledir / "numba"
NUMBA_CACHE_PATH.mkdir(exist_ok=True)
CACHED_SRC_FUNCTIONS = weakref.WeakKeyDictionary()


class NumbaPyTensorCacheLocator(_CacheLocator):
    def __init__(self, py_func, py_file, hash):
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


def cache_node_key(node: Apply, extra_key="") -> str:
    op = node.op
    return sha256(
        str(
            (
                # Op signature
                (type(op), op._props_dict() if hasattr(op, "_props_dict") else ""),
                # Node signature
                tuple((type(inp_type := inp.type), inp_type) for inp in node.inputs),
                # Extra key given by the caller
                extra_key,
            ),
        ).encode()
    ).hexdigest()
