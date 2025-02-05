import linecache
import sys
import traceback
from abc import ABCMeta
from collections.abc import Sequence
from io import StringIO
from typing import TYPE_CHECKING, Any, TypeVar, Union


if TYPE_CHECKING:
    from pytensor.graph.basic import Apply, Variable

T = TypeVar("T", bound=Union["Apply", "Variable"])


def simple_extract_stack(
    f=None, limit: int | None = None, skips: Sequence[str] | None = None
) -> list[tuple[str | None, int, str, str | None]]:
    """This is traceback.extract_stack from python 2.7 with this change:

    - Comment the update of the cache.
    - Skip internal stack trace level.

    The update of the cache call os.stat to verify is the cache is up
    to date.  This take too much time on cluster.

    limit - The number of stack level we want to return. If None, mean
    all what we can.

    skips - partial path of stack level we don't want to keep and count.
        When we find one level that isn't skipped, we stop skipping.

    """
    if skips is None:
        skips = []

    if f is None:
        f = sys._getframe().f_back

    if limit is None:
        if hasattr(sys, "tracebacklimit"):
            limit = sys.tracebacklimit
    trace: list[tuple[str | None, int, str, str | None]] = []
    n = 0
    while f is not None and (limit is None or n < limit):
        lineno = f.f_lineno
        co = f.f_code
        filename = co.co_filename
        name = co.co_name
        #        linecache.checkcache(filename)
        line: str | None = linecache.getline(filename, lineno, f.f_globals)
        if line:
            line = line.strip()
        else:
            line = None
        f = f.f_back

        # Just skip inner level
        if len(trace) == 0:
            rm = False
            for p in skips:
                # The 'tests' exception was added; otherwise, we'd lose the
                # stack trace during in our test cases. We're not sure this is
                # the right way to do it, though.
                if p in filename and "tests" not in filename:
                    rm = True
                    break
            if rm:
                continue
        trace.append((filename, lineno, name, line))
        n = n + 1
    trace.reverse()
    return trace


def add_tag_trace(thing: T, user_line: int | None = None) -> T:
    """Add tag.trace to a node or variable.

    The argument is returned after being affected (inplace).

    Parameters
    ----------
    thing
        The object where we add .tag.trace.
    user_line
        The max number of user line to keep.

    Notes
    -----
    We also use config.traceback__limit for the maximum number of stack level
    we look.

    """
    from pytensor.configdefaults import config

    if user_line is None:
        user_line = config.traceback__limit

    if user_line == -1:
        user_line = None
    skips = [
        "pytensor/tensor/",
        "pytensor\\tensor\\",
        "pytensor/compile/",
        "pytensor\\compile\\",
        "pytensor/graph/",
        "pytensor\\graph\\",
        "pytensor/scalar/basic.py",
        "pytensor\\scalar\\basic.py",
        "pytensor/scan/",
        "pytensor\\scan\\",
        "pytensor/sparse/",
        "pytensor\\sparse\\",
        "pytensor/typed_list/",
        "pytensor\\typed_list\\",
    ]

    if config.traceback__compile_limit > 0:
        skips = []

    tr = simple_extract_stack(limit=user_line, skips=skips)
    # Different python version use different sementic for
    # limit. python 2.7 include the call to extrack_stack. The -1 get
    # rid of it.

    if tr:
        thing.tag.trace = [tr]
    else:
        thing.tag.trace = tr
    return thing


def get_variable_trace_string(v):
    sio = StringIO()
    # For backward compatibility with old trace
    tr = getattr(v.tag, "trace", [])
    if isinstance(tr, list) and len(tr) > 0:
        print(" \nBacktrace when that variable is created:\n", file=sio)
        # The isinstance is needed to handle old pickled trace
        if isinstance(tr[0], tuple):
            traceback.print_list(v.tag.trace, sio)
        else:
            # Print separate message for each element in the list of
            # backtraces
            for idx, subtr in enumerate(tr):
                if len(tr) > 1:
                    print(f"trace {int(idx)}", file=sio)
                traceback.print_list(subtr, sio)
    return sio.getvalue()


class InconsistencyError(Exception):
    """
    This exception should be thrown by listeners to FunctionGraph when the
    graph's state is invalid.

    """


class MissingInputError(Exception):
    """
    A symbolic input needed to compute the outputs is missing.

    """

    def __init__(self, *args, **kwargs):
        if kwargs:
            # The call to list is needed for Python 3
            assert list(kwargs) == ["variable"]
            error_msg = get_variable_trace_string(kwargs["variable"])
            if error_msg:
                args = (*args, error_msg)
        s = "\n".join(args)  # Needed to have the new line print correctly
        super().__init__(s)


class TestValueError(Exception):
    """Base exception class for all test value errors."""


class MethodNotDefined(Exception):
    """
    To be raised by functions defined as part of an interface.

    When the user sees such an error, it is because an important interface
    function has been left out of an implementation class.

    """


class MetaType(ABCMeta):
    def __new__(cls, name, bases, dct):
        props = dct.get("__props__", None)
        if props is not None:
            if not isinstance(props, tuple):
                raise TypeError("__props__ has to be a tuple")
            if not all(isinstance(p, str) for p in props):
                raise TypeError("elements of __props__ have to be strings")

            def _props(self):
                """
                Tuple of properties of all attributes
                """
                return tuple(getattr(self, a) for a in props)

            dct["_props"] = _props

            def _props_dict(self):
                """This return a dict of all ``__props__`` key-> value.

                This is useful in optimization to swap op that should have the
                same props. This help detect error that the new op have at
                least all the original props.

                """
                return {a: getattr(self, a) for a in props}

            dct["_props_dict"] = _props_dict

            if "__hash__" not in dct:

                def __hash__(self):
                    return hash((type(self), tuple(getattr(self, a) for a in props)))

                dct["__hash__"] = __hash__

            if "__eq__" not in dct:

                def __eq__(self, other):
                    return type(self) is type(other) and tuple(
                        getattr(self, a) for a in props
                    ) == tuple(getattr(other, a) for a in props)

                dct["__eq__"] = __eq__

            # FIXME: This overrides __str__ inheritance when props are provided
            if "__str__" not in dct:
                if len(props) == 0:

                    def __str__(self):
                        return f"{self.__class__.__name__}"

                else:

                    def __str__(self):
                        classname = self.__class__.__name__
                        args = ", ".join(f"{p}={getattr(self, p)!r}" for p in props)
                        return f"{classname}{{{args}}}"

                dct["__str__"] = __str__

        return super().__new__(cls, name, bases, dct)


class MetaObject(metaclass=MetaType):
    __slots__: list = []

    def __ne__(self, other):
        return not self == other


class Scratchpad:
    def clear(self):
        self.__dict__.clear()

    def __update__(self, other):
        self.__dict__.update(other.__dict__)
        return self

    def __str__(self):
        return "scratchpad" + str(self.__dict__)

    def __repr__(self):
        return "scratchpad" + str(self.__dict__)

    def info(self):
        print(f"<pytensor.graph.utils.scratchpad instance at {id(self)}>")  # noqa: T201
        for k, v in self.__dict__.items():
            print(f"  {k}: {v}")  # noqa: T201

    # These two methods have been added to help Mypy
    def __getattribute__(self, name):
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value


class ValidatingScratchpad(Scratchpad):
    """This `Scratchpad` validates attribute values."""

    def __init__(self, attr, attr_filter):
        super().__init__()

        object.__setattr__(self, "attr", attr)
        object.__setattr__(self, "attr_filter", attr_filter)

    def __setattr__(self, attr, obj):
        if getattr(self, "attr", None) == attr:
            obj = self.attr_filter(obj)

        return object.__setattr__(self, attr, obj)


class AssocList:
    """An associative list.

    This class is like a `dict` that accepts unhashable keys by using an
    assoc list for internal use only
    """

    def __init__(self):
        self._dict = {}
        self._list = []

    def __getitem__(self, item):
        return self.get(item, None)

    def __setitem__(self, item, value):
        try:
            self._dict[item] = value
        except Exception:
            for i, (key, val) in enumerate(self._list):
                if key == item:
                    self._list[i] = (item, value)
                    return
            self._list.append((item, value))

    def __delitem__(self, item):
        try:
            if item in self._dict:
                del self._dict[item]
                return
        except TypeError as e:
            assert "unhashable type" in str(e)
        for i, (key, val) in enumerate(self._list):
            if key == item:
                del self._list[i]
                return
            raise KeyError(item)

    def discard(self, item):
        try:
            if item in self._dict:
                del self._dict[item]
                return
        except TypeError as e:
            assert "unhashable type" in str(e)
        for i, (key, val) in enumerate(self._list):
            if key == item:
                del self._list[i]
                return

    def get(self, item, default):
        try:
            return self._dict[item]
        except Exception:
            for item2, value in self._list:
                try:
                    if item == item2:
                        return value
                    if item.equals(item2):
                        return value
                except Exception:
                    if item is item2:
                        return value
            return default

    def clear(self):
        self._dict = {}
        self._list = []

    def __repr__(self):
        return f"AssocList({self._dict}, {self._list})"
