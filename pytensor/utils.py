"""Utility functions that only depend on the standard library."""

import hashlib
import logging
import os
import struct
import subprocess
import sys
from collections.abc import Iterable, Sequence
from functools import partial
from pathlib import Path

import numpy as np


__all__ = [
    "LOCAL_BITWIDTH",
    "NDARRAY_C_VERSION",
    "NPY_RAVEL_AXIS",
    "PYTHON_INT_BITWIDTH",
    "NoDuplicateOptWarningFilter",
    "call_subprocess_Popen",
    "get_unbound_function",
    "maybe_add_to_os_environ_pathlist",
    "output_subprocess_Popen",
    "subprocess_Popen",
]


__excepthooks: list = []


LOCAL_BITWIDTH = struct.calcsize("P") * 8
"""
32 for 32bit arch, 64 for 64bit arch.
By "architecture", we mean the size of memory pointers (size_t in C),
*not* the size of long int, as it can be different.

Note that according to Python documentation, `platform.architecture()` is
not reliable on OS X with universal binaries.
'P' denotes a void*, and the size is expressed in bytes.
"""

PYTHON_INT_BITWIDTH = struct.calcsize("l") * 8
"""
The bit width of Python int (C long int).

Note that it can be different from the size of a memory pointer.
'l' denotes a C long int, and the size is expressed in bytes.
"""

NPY_RAVEL_AXIS = np.iinfo(np.int32).min
"""
The value of the numpy C API NPY_RAVEL_AXIS.
"""

NDARRAY_C_VERSION = np._core._multiarray_umath._get_ndarray_c_version()  # type: ignore[attr-defined]


def __call_excepthooks(type, value, trace):
    """
    This function is meant to replace excepthook and do some
    special work if the exception value has a __thunk_trace__
    field.
    In that case, it retrieves the field, which should
    contain a trace as returned by L{traceback.extract_stack},
    and prints it out on L{stderr}.

    The normal excepthook is then called.

    Parameters:
    ----------
    type
        Exception class
    value
        Exception instance
    trace
        Traceback object

    Notes
    -----
    This hook replaced in testing, so it does not run.

    """
    for hook in __excepthooks:
        hook(type, value, trace)
    sys.__excepthook__(type, value, trace)


def add_excepthook(hook):
    """Adds an excepthook to a list of excepthooks that are called
    when an unhandled exception happens.

    See https://docs.python.org/3/library/sys.html#sys.excepthook for signature info.
    """
    __excepthooks.append(hook)
    sys.excepthook = __call_excepthooks


def get_unbound_function(unbound):
    # Op.make_thunk isn't bound, so don't have a __func__ attr.
    # But bound method, have a __func__ method that point to the
    # not bound method. That is what we want.
    if hasattr(unbound, "__func__"):
        return unbound.__func__
    return unbound


def maybe_add_to_os_environ_pathlist(var: str, newpath: Path | str) -> None:
    """
    Unfortunately, Conda offers to make itself the default Python
    and those who use it that way will probably not activate envs
    correctly meaning e.g. mingw-w64 g++ may not be on their PATH.

    This function ensures that, if `newpath` is an absolute path,
    and it is not already in os.environ[var] it gets added to the
    front.

    The reason we check first is because Windows environment vars
    are limited to 8191 characters and it is easy to hit that.

    `var` will typically be 'PATH'.
    """
    if not Path(newpath).is_absolute():
        return

    try:
        oldpaths = os.environ[var].split(os.pathsep)
        if str(newpath) not in oldpaths:
            newpaths = os.pathsep.join([str(newpath), *oldpaths])
            os.environ[var] = newpaths
    except Exception:
        pass


def subprocess_Popen(command: list[str], **params) -> subprocess.Popen:
    """
    Utility function to work around windows behavior that open windows.

    :see: call_subprocess_Popen and output_subprocess_Popen
    """
    startupinfo = None
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
        try:
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
        except AttributeError:
            startupinfo.dwFlags |= subprocess._subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]

        # "If shell is True, it is recommended to pass args as a string rather than as a sequence." (cite taken from https://docs.python.org/2/library/subprocess.html#frequently-used-arguments)
        # In case when command arguments have spaces, passing a command as a list will result in incorrect arguments break down, and consequently
        # in "The filename, directory name, or volume label syntax is incorrect" error message.
        # Passing the command as a single string solves this problem.
        if isinstance(command, list):
            command = " ".join(command)  # type: ignore[assignment]

    return subprocess.Popen(command, startupinfo=startupinfo, **params)


def call_subprocess_Popen(command: list[str], **params) -> int:
    """
    Calls subprocess_Popen and discards the output, returning only the
    exit code.
    """
    if "stdout" in params or "stderr" in params:
        raise TypeError("don't use stderr or stdout with call_subprocess_Popen")
    with Path(os.devnull).open("wb") as null:
        # stdin to devnull is a workaround for a crash in a weird Windows
        # environment where sys.stdin was None
        params.setdefault("stdin", null)
        params["stdout"] = null
        params["stderr"] = null
        p = subprocess_Popen(command, **params)
        returncode = p.wait()
    return returncode


def output_subprocess_Popen(command: list[str], **params) -> tuple[bytes, bytes, int]:
    """
    Calls subprocess_Popen, returning the output, error and exit code
    in a tuple.
    """
    if "stdout" in params or "stderr" in params:
        raise TypeError("don't use stderr or stdout with output_subprocess_Popen")
    if "encoding" in params:
        raise TypeError(
            "adjust the output_subprocess_Popen type annotation to support str"
        )
    params["stdout"] = subprocess.PIPE
    params["stderr"] = subprocess.PIPE
    p = subprocess_Popen(command, **params)
    # we need to use communicate to make sure we don't deadlock around
    # the stdout/stderr pipe.
    out = p.communicate()
    return (*out, p.returncode)


def hash_from_code(msg: str | bytes) -> str:
    """Return the SHA256 hash of a string or bytes."""
    # hashlib.sha256() requires an object that supports buffer interface,
    # but Python 3 (unicode) strings don't.
    if isinstance(msg, str):
        msg = msg.encode()
    # Python 3 does not like module names that start with a digit.
    return f"m{hashlib.sha256(msg).hexdigest()}"


def uniq(seq: Sequence) -> list:
    """
    Do not use set, this must always return the same value at the same index.
    If we just exchange other values, but keep the same pattern of duplication,
    we must keep the same order.

    """
    # TODO: consider building a set out of seq so that the if condition
    # is constant time -JB
    return [x for i, x in enumerate(seq) if seq.index(x) == i]


def difference(seq1: Iterable, seq2: Iterable):
    r"""
    Returns all elements in seq1 which are not in seq2: i.e ``seq1\seq2``.

    """
    seq2 = list(seq2)
    try:
        # try to use O(const * len(seq1)) algo
        if len(seq2) < 4:  # I'm guessing this threshold -JB
            raise Exception("not worth it")
        set2 = set(seq2)
        return [x for x in seq1 if x not in set2]
    except Exception:
        # maybe a seq2 element is not hashable
        # maybe seq2 is too short
        # -> use O(len(seq1) * len(seq2)) algo
        return [x for x in seq1 if x not in seq2]


def to_return_values(values):
    if len(values) == 1:
        return values[0]
    else:
        return values


def from_return_values(values):
    if isinstance(values, list | tuple):
        return values
    else:
        return [values]


def flatten(a) -> list:
    """
    Recursively flatten tuple, list and set in a list.

    """
    if isinstance(a, tuple | list | set):
        l = []
        for item in a:
            l.extend(flatten(item))
        return l
    else:
        return [a]


def apply_across_args(*fns):
    """Create new functions that distributes the wrapped functions across iterable arguments.

    For example, a function, `fn`, that uses this decorator satisfies
    `fn("hi") == [fn("h"), fn("i")]`.
    """

    def f2(f, *names):
        if names and isinstance(names[0], int):
            if names == 1:
                return f()
            else:
                return [f() for i in range(names[0])]
        if isinstance(names, tuple):
            if len(names) == 1:
                names = names[0]
        if len(names) == 1:
            return f(names)
        else:
            return [f(name) for name in names]

    if len(fns) == 1:
        return partial(f2, fns[0])
    else:
        return [partial(f2, f) for f in fns]


class NoDuplicateOptWarningFilter(logging.Filter):
    """Filter to avoid duplicating optimization warnings."""

    prev_msgs: set = set()

    def filter(self, record):
        msg = record.getMessage()
        if msg.startswith("Optimization Warning: "):
            if msg in self.prev_msgs:
                return False
            else:
                self.prev_msgs.add(msg)
                return True
        return True


class Singleton:
    """Convenient base class for a singleton.

    It saves having to implement __eq__ and __hash__.

    """

    __instance = None

    def __new__(cls):
        # If sub-subclass of SingletonType don't redeclare __instance
        # when we look for it, we will find it in the subclass.  We
        # don't want that, so we check the class.  When we add one, we
        # add one only to the current class, so all is working
        # correctly.
        if not (cls.__instance and isinstance(cls.__instance, cls)):
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __str__(self):
        return self.__class__.__name__

    def __eq__(self, other):
        if self is other:
            return True
        if type(self) is type(other):
            return True
        return False

    def __hash__(self):
        return hash(type(self))
